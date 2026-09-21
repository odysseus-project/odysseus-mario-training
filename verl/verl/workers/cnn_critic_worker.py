# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CNN Critic Worker for vision-based RL tasks.

This worker manages a CNN-based critic model for PPO training in game environments
where observations are images rather than text tokens.
"""

import datetime
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import optim

from verl import DataProto
from verl.base_config import BaseConfig
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend
from verl.utils.profiler import ProfilerConfig
from verl.workers.critic import CNNCriticModel, NatureCNN, DataParallelCNNCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["CNNCriticWorker", "CNNCriticConfig"]


@dataclass
class CNNCriticModelConfig(BaseConfig):
    """Configuration for CNN critic model architecture."""

    # Model architecture type: "simple" (CNNCriticModel) or "nature" (NatureCNN)
    # - "simple": 3 conv layers with batch norm, good for smaller images
    # - "nature": Nature DQN architecture, designed for 84x84 images
    architecture: str = "nature"

    # Input image dimensions
    # For Super Mario Land: 160x144 (or resize_ratio applied)
    input_channels: int = 3
    input_height: int = 144
    input_width: int = 160

    # Hidden layer dimension for fully connected layers
    hidden_dim: int = 256

    # Whether to use batch normalization (only for "simple" architecture)
    use_batch_norm: bool = True

    # Key to access images in the data batch
    # Common options: "pixel_values", "images", "image"
    image_key: str = "pixel_values"


@dataclass
class CNNCriticConfig(BaseConfig):
    """Configuration for CNN critic worker."""

    # Number of rollouts per update (mirrors actor rollout_n)
    rollout_n: int = 1

    # Strategy for the critic, default "cnn"
    strategy: str = "cnn"

    # Whether to enable the critic worker.
    # By default it is only enabled if advantage estimator is gae
    # Set it to True manually if you always want to enable critic worker
    enable: Optional[bool] = None

    # NCCL timeout in seconds
    nccl_timeout: int = 600

    # ============================================================
    # Model Configuration
    # ============================================================
    model: CNNCriticModelConfig = field(default_factory=CNNCriticModelConfig)

    # ============================================================
    # Optimizer Configuration
    # ============================================================
    # Learning rate
    lr: float = 3e-4

    # Weight decay for regularization
    weight_decay: float = 0.0

    # Warmup steps ratio; total steps will be injected at runtime
    lr_warmup_steps_ratio: float = 0.0

    # Total training steps (must be overridden at runtime)
    total_training_steps: int = -1

    # ============================================================
    # PPO Configuration
    # ============================================================

    # PPO mini-batch size per update
    ppo_mini_batch_size: int = 256

    # [Deprecated] Global micro batch size
    ppo_micro_batch_size: Optional[int] = None

    # Local per-GPU micro batch size
    ppo_micro_batch_size_per_gpu: Optional[int] = None

    # Whether to automatically adjust batch size at runtime
    use_dynamic_bsz: Optional[bool] = False

    # Max tokens per GPU in one PPO batch (doubled for critic)
    ppo_max_token_len_per_gpu: int = 32768

    # Max token length per GPU in forward pass
    forward_max_token_len_per_gpu: Optional[int] = 32768

    # Number of PPO epochs per batch
    ppo_epochs: int = 1

    # ============================================================
    # Training Configuration
    # ============================================================
    # Gradient clipping max norm
    grad_clip: float = 1.0

    # Value loss clipping range (PPO value clipping)
    cliprange_value: float = 0.2

    # Loss aggregation mode: "mean" or "sum"
    loss_agg_mode: str = "mean"

    # Profiler configuration
    profiler: Optional[ProfilerConfig] = None

    def validate(self, n_gpus: int, train_batch_size: int):
        """Validate critic configuration with runtime parameters.

        Args:
            n_gpus: Total number of GPUs available
            train_batch_size: Training batch size from data config
        """
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"critic.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )


class CNNCriticWorker(Worker):
    """
    Worker for CNN-based critic in PPO training.
    
    This worker handles:
    - Initialization of the CNN critic model
    - Computing value estimates for observations
    - Updating the critic during training
    
    Unlike the LLM-based critic, this worker processes image observations
    directly through a CNN rather than token sequences through a transformer.
    """
    
    def __init__(self, config: Union[CNNCriticConfig, DictConfig]):
        Worker.__init__(self)
        
        # Handle both dataclass and OmegaConf configs
        if isinstance(config, DictConfig):
            # Store the raw config for attribute access
            self.config = config
            # Extract values using OmegaConf's get method
            self._nccl_timeout = config.get("nccl_timeout", 600)
            self._rollout_n = config.get("rollout_n", 1)
            self._ppo_mini_batch_size = config.get("ppo_mini_batch_size", 256)
        else:
            self.config = config
            self._nccl_timeout = config.nccl_timeout
            self._rollout_n = config.rollout_n
            self._ppo_mini_batch_size = config.ppo_mini_batch_size
        
        # Initialize distributed training if not already done
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self._nccl_timeout),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        
        # Set device
        rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(rank)
        
        self.device = get_device_id()
        self.device_name = get_device_name()
        
        # Normalize batch sizes for distributed training
        world_size = torch.distributed.get_world_size()
        self._ppo_mini_batch_size *= self._rollout_n
        self._ppo_mini_batch_size //= world_size
    
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value, handling both dataclass and OmegaConf."""
        if isinstance(self.config, DictConfig):
            return self.config.get(key, default)
        return getattr(self.config, key, default)
    
    def _get_model_config_value(self, key: str, default: Any = None) -> Any:
        """Get a model configuration value."""
        model_config = self._get_config_value("model", {})
        if isinstance(model_config, BaseConfig):
            return model_config.get(key, default)
        return getattr(model_config, key, default)
    
    def _build_critic_model(self) -> nn.Module:
        """Build the CNN critic model based on configuration."""
        architecture = self._get_model_config_value("architecture", "simple")
        input_channels = self._get_model_config_value("input_channels", 3)
        input_height = self._get_model_config_value("input_height", 144)
        input_width = self._get_model_config_value("input_width", 160)
        hidden_dim = self._get_model_config_value("hidden_dim", 256)
        use_batch_norm = self._get_model_config_value("use_batch_norm", True)
        
        if architecture == "simple":
            model = CNNCriticModel(
                input_channels=input_channels,
                input_height=input_height,
                input_width=input_width,
                hidden_dim=hidden_dim,
                use_batch_norm=use_batch_norm,
            )
        elif architecture == "nature":
            model = NatureCNN(
                input_channels=input_channels,
                input_height=input_height,
                input_width=input_width,
                hidden_dim=hidden_dim,
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        return model.to(self.device)
    
    def _build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Build optimizer for the critic model."""
        lr = self._get_config_value("lr", 3e-4)
        weight_decay = self._get_config_value("weight_decay", 0.0)
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    
    def _build_lr_scheduler(self, optimizer: optim.Optimizer):
        """Build learning rate scheduler with warmup."""
        total_steps = self._get_config_value("total_training_steps", -1)
        warmup_ratio = self._get_config_value("lr_warmup_steps_ratio", 0.0)
        warmup_steps = int(warmup_ratio * total_steps) if total_steps > 0 else 0
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the CNN critic model, optimizer, and scheduler."""
        self.critic_module = self._build_critic_model()
        self.critic_optimizer = self._build_optimizer(self.critic_module)
        self.critic_lr_scheduler = self._build_lr_scheduler(self.critic_optimizer)
        
        # Get image key from model config
        image_key = self._get_model_config_value("image_key", "pixel_values")
        
        # Wrap in DataParallelCNNCritic
        # Create a config-like object that DataParallelCNNCritic can use
        self.critic = DataParallelCNNCritic(
            config=self._create_critic_config(),
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            image_key=image_key,
        )
        
        # Log model info
        if self.rank == 0:
            n_params = sum(p.numel() for p in self.critic_module.parameters())
            architecture = self._get_model_config_value("architecture", "simple")
            input_channels = self._get_model_config_value("input_channels", 3)
            input_height = self._get_model_config_value("input_height", 144)
            input_width = self._get_model_config_value("input_width", 160)
            print(f"CNN Critic initialized with {n_params:,} parameters")
            print(f"Architecture: {architecture}")
            print(f"Input shape: ({input_channels}, {input_height}, {input_width})")
    
    def _create_critic_config(self):
        """Create a config object for DataParallelCNNCritic."""
        class CriticConfigWrapper:
            """Simple wrapper to provide config attributes."""
            def __init__(self, worker):
                self._worker = worker
                # PPO config
                self.ppo_mini_batch_size = worker._ppo_mini_batch_size
                self.ppo_micro_batch_size_per_gpu = worker._get_config_value("ppo_micro_batch_size_per_gpu", 32)
                self.ppo_epochs = worker._get_config_value("ppo_epochs", 4)
                self.grad_clip = worker._get_config_value("grad_clip", 1.0)
                self.cliprange_value = worker._get_config_value("cliprange_value", 0.2)
                self.loss_agg_mode = worker._get_config_value("loss_agg_mode", "mean")
                self.use_dynamic_bsz = worker._get_config_value("use_dynamic_bsz", False)
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        return CriticConfigWrapper(self)
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto) -> DataProto:
        """
        Compute value estimates for observations.
        
        Args:
            data: DataProto containing image observations
        
        Returns:
            DataProto with computed values added to batch
        """
        data = data.to("cpu")
        values = self.critic.compute_values(data)
        
        # Add values to the data batch
        output = DataProto.from_dict(tensors={"values": values})
        output = output.to("cpu")
        return output
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto) -> dict:
        """
        Update the critic model using PPO value loss.
        
        Args:
            data: DataProto containing training data with values and returns
        
        Returns:
            Dictionary of training metrics
        """
        data = data.to("cpu")
        metrics = self.critic.update_critic(data)
        
        # Add learning rate to metrics
        lr = self.critic_optimizer.param_groups[0]["lr"]
        metrics["critic/lr"] = lr
        # Step the learning rate scheduler
        self.critic_lr_scheduler.step()

        output = DataProto(batch=None, meta_info={"metrics": metrics})
        output = output.to("cpu")
        
        return output
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """Save model checkpoint."""
        if self.rank == 0:
            checkpoint = {
                "model_state_dict": self.critic_module.state_dict(),
                "optimizer_state_dict": self.critic_optimizer.state_dict(),
                "scheduler_state_dict": self.critic_lr_scheduler.state_dict(),
                "config": self.config,
            }
            torch.save(checkpoint, local_path)
            print(f"CNN Critic checkpoint saved to {local_path}")
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        """Load model checkpoint."""
        # Convert device ID to proper device string for map_location
        if self.device_name == "cpu":
            map_location = "cpu"
        else:
            map_location = f"{self.device_name}:{self.device}"
        checkpoint = torch.load(local_path, map_location=map_location, weights_only=False)
        self.critic_module.load_state_dict(checkpoint["model_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.critic_lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.rank == 0:
            print(f"CNN Critic checkpoint loaded from {local_path}")


def create_cnn_critic_from_config(config_dict: dict) -> CNNCriticWorker:
    """
    Factory function to create a CNN critic worker from a configuration dictionary.
    
    Args:
        config_dict: Dictionary containing configuration parameters
    
    Returns:
        Configured CNNCriticWorker instance
    """
    from omegaconf import OmegaConf
    
    # Create config object
    config = OmegaConf.create(config_dict)
    
    # Convert to dataclass
    model_config = CNNCriticModelConfig(
        architecture=config.get("model", {}).get("architecture", "simple"),
        input_channels=config.get("model", {}).get("input_channels", 3),
        input_height=config.get("model", {}).get("input_height", 144),
        input_width=config.get("model", {}).get("input_width", 160),
        hidden_dim=config.get("model", {}).get("hidden_dim", 256),
        use_batch_norm=config.get("model", {}).get("use_batch_norm", True),
        image_key=config.get("model", {}).get("image_key", "pixel_values"),
    )
    
    critic_config = CNNCriticConfig(
        model=model_config,
        lr=config.get("lr", 3e-4),
        weight_decay=config.get("weight_decay", 0.0),
        lr_warmup_steps_ratio=config.get("lr_warmup_steps_ratio", 0.0),
        total_training_steps=config.get("total_training_steps", -1),
        ppo_mini_batch_size=config.get("ppo_mini_batch_size", 256),
        ppo_micro_batch_size_per_gpu=config.get("ppo_micro_batch_size_per_gpu", 32),
        ppo_epochs=config.get("ppo_epochs", 4),
        grad_clip=config.get("grad_clip", 1.0),
        cliprange_value=config.get("cliprange_value", 0.2),
        loss_agg_mode=config.get("loss_agg_mode", "mean"),
    )
    
    return CNNCriticWorker(critic_config)

