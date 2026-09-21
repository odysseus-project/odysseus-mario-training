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
CNN-based PPO Critic for vision-based RL tasks (e.g., game agents)
"""

import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.device import get_device_id, get_device_name
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.workers.critic import BasePPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["CNNCriticModel", "DataParallelCNNCritic"]


class CNNCriticModel(nn.Module):
    """
    A simple CNN-based critic model for vision-based RL tasks.
    
    Takes image observations as input and outputs value estimates.
    This is designed for game environments where the observation is a screenshot.
    
    Architecture:
        - 3 convolutional layers with batch normalization
        - 2 fully connected layers
        - Outputs a single value estimate per image
    
    Args:
        input_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
        input_height (int): Height of input image
        input_width (int): Width of input image
        hidden_dim (int): Dimension of hidden layers
        use_batch_norm (bool): Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        input_height: int = 144,  # Default for Super Mario Land
        input_width: int = 160,   # Default for Super Mario Land
        hidden_dim: int = 256,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.hidden_dim = hidden_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Batch normalization layers (optional)
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(64)
        
        # Calculate the size after convolutions
        self._conv_output_size = self._get_conv_output_size(input_channels, input_height, input_width)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._conv_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output single value
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_conv_output_size(self, channels: int, height: int, width: int) -> int:
        """Calculate the output size after convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).size(1)
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN critic.
        
        Args:
            images: Tensor of shape (batch_size, channels, height, width)
                   Values should be normalized to [0, 1] range
        
        Returns:
            values: Tensor of shape (batch_size,) containing value estimates
        """
        # Convolutional layers
        x = self.conv1(images)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        
        # Flatten and pass through fully connected layers
        # Ensure tensor is contiguous before view (batch norm can make it non-contiguous)
        x = x.contiguous().view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        values = self.fc2(x).squeeze(-1)  # (batch_size,)
        
        return values


class NatureCNN(nn.Module):
    """
    Nature DQN-style CNN architecture.
    
    Reference: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        input_height: int = 84,
        input_width: int = 84,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.hidden_dim = hidden_dim
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate output size
        self._conv_output_size = self._get_conv_output_size(input_channels, input_height, input_width)
        
        self.fc = nn.Linear(self._conv_output_size, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self._initialize_weights()
    
    def _get_conv_output_size(self, channels: int, height: int, width: int) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).size(1)
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous().view(x.size(0), -1)
        x = F.relu(self.fc(x))
        values = self.value_head(x).squeeze(-1)
        return values


class DataParallelCNNCritic(BasePPOCritic):
    """
    Data Parallel CNN Critic for PPO training.
    
    This class wraps the CNN model and provides the interface required by verl's
    PPO trainer, including compute_values and update_critic methods.
    
    Args:
        config: Configuration object containing training parameters
        critic_module: The CNN model (CNNCriticModel or NatureCNN)
        critic_optimizer: Optimizer for the critic
        image_key: Key to access images in the data batch (default: "pixel_values")
        image_transform: Optional transform to apply to images before passing to model
    """
    
    def __init__(
        self,
        config,
        critic_module: nn.Module,
        critic_optimizer: optim.Optimizer,
        image_key: str = "pixel_values",
        image_transform: Optional[nn.Module] = None,
    ):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.image_key = image_key
        self.image_transform = image_transform
        self.device_name = get_device_name()
        self.scale_factor = self.config.get("scale_factor", 100.0)
    
    def _extract_images(self, batch_data: dict, non_tensor_batch: dict) -> torch.Tensor:
        """
        Extract images from the batch data.
        
        Priority order:
        1. multi_modal_data from non_tensor_batch (PIL images from agent_loop)
        2. Direct tensor under image_key in batch_data
        3. pixel_values in batch_data
        4. multi_modal_inputs in batch_data
        
        Args:
            batch_data: Dictionary containing tensor batch data
            non_tensor_batch: Dictionary containing non-tensor batch data
        
        Returns:
            images: Tensor of shape (batch_size, C, H, W)
        """
        images = None
        import numpy as np
        from PIL import Image
        
        # First priority: multi_modal_data from agent_loop (PIL images)
        if "multi_modal_data" in non_tensor_batch:
            multi_modal_data = non_tensor_batch["multi_modal_data"]
            # multi_modal_data is a numpy array of objects, each is a dict like {"image": PIL.Image}
            image_list = []
            for mmd in multi_modal_data:
                if mmd is not None and isinstance(mmd, dict) and "image" in mmd:
                    img = mmd["image"][-1]
                    if isinstance(img, Image.Image):
                        # PIL resize expects (width, height), not (height, width)
                        target_size = (self.critic_module.input_width, self.critic_module.input_height)
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                        # Convert PIL Image to numpy array, then to tensor
                        img_array = np.array(img)
                        # Handle grayscale images (add channel dimension)
                        if len(img_array.shape) == 2:
                            img_array = img_array[:, :, np.newaxis]
                        # Handle RGB images
                        if img_array.shape[2] == 3:
                            pass  # Already RGB
                        elif img_array.shape[2] == 4:
                            # RGBA -> RGB
                            img_array = img_array[:, :, :3]
                        else:
                            raise ValueError(f"Unexpected image shape: {img_array.shape}")
                        image_list.append(img_array)
                    else:
                        raise ValueError(f"Expected PIL Image, got {type(img)}")
                else:
                    raise ValueError(f"Invalid multi_modal_data entry: {mmd}")
            
            if image_list:
                images = torch.from_numpy(np.stack(image_list)).permute(0, 3, 1, 2).float() / 255.0
                print(f"images shape: {images.shape}")
        
        # Check images shape (batch_size, channels, height, width)
        if images is not None:
            if images.shape[0] != batch_data["response_mask"].shape[0]:
                raise ValueError(f"Expected batch_size {batch_data['response_mask'].shape[0]}, got {images.shape[0]}")
            if images.shape[1] != 3:
                raise ValueError(f"Expected 3 channels, got {images.shape[1]}")
            if images.shape[2] != self.critic_module.input_height:
                raise ValueError(f"Expected height {self.critic_module.input_height}, got {images.shape[2]}")
            if images.shape[3] != self.critic_module.input_width:
                raise ValueError(f"Expected width {self.critic_module.input_width}, got {images.shape[3]}")
        
        if images is None:
            raise ValueError(
                f"Could not find images in batch. "
                f"Available batch keys: {list(batch_data.keys())}. "
                f"Available non_tensor keys: {list(non_tensor_batch.keys())}. "
                f"Expected image_key: {self.image_key}"
            )
        
        # Handle list of tensors
        if isinstance(images, list):
            images = torch.stack(images)
        
        # Ensure correct device
        if not images.is_cuda:
            images = images.to(get_device_id())
        
        # Apply optional transform
        if self.image_transform is not None:
            images = self.image_transform(images)
        
        # Ensure float and proper range [0, 1]
        if images.dtype != torch.float32:
            images = images.float()
        if images.max() > 1.0:
            images = images / 255.0
        
        return images
    
    def _forward_batch(self, batch_data: dict, non_tensor_batch: dict) -> torch.Tensor:
        """
        Forward pass for the entire batch (no micro batching needed for small CNN).
        
        Args:
            batch_data: Dictionary containing tensor batch data
            non_tensor_batch: Dictionary containing non-tensor batch data
        
        Returns:
            values: Tensor of shape (batch_size,) containing value estimates
        """
        images = self._extract_images(batch_data, non_tensor_batch)
        
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            values = self.critic_module(images)
        
        return values
    
    def _optimizer_step(self) -> torch.Tensor:
        """Perform optimizer step with gradient clipping."""
        assert self.config.grad_clip is not None
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic_module.parameters(), 
            max_norm=self.config.grad_clip
        )
        
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        
        return grad_norm
    
    @GPUMemoryLogger(role="cnn critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor:
        """
        Compute value estimates for a batch of data.
        
        No micro batching needed since CNN is small - process entire batch at once.
        Returns per-token values (expanded from batch-level) for GAE compatibility.
        The CNN outputs one value per image, which is expanded to match response length.
        
        Args:
            data: DataProto containing the batch
        
        Returns:
            values: Tensor of shape (batch_size, response_length) for GAE compatibility
        """
        self.critic_module.eval()
        
        # Select relevant keys - we need multi_modal_data from non_tensor_batch
        select_keys = ["response_mask"]  # Need responses to get response_length
        non_tensor_select_keys = []
        
        # Priority: multi_modal_data (PIL images from agent_loop)
        if "multi_modal_data" in data.non_tensor_batch:
            non_tensor_select_keys.append("multi_modal_data")
        # Fallback options
        elif "multi_modal_inputs" in data.non_tensor_batch:
            non_tensor_select_keys.append("multi_modal_inputs")
        
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        data = data.to(get_device_id())
        
        with torch.no_grad():
            # Get batch-level values (one per image)
            values_batch = self._forward_batch(data.batch, data.non_tensor_batch)  # (batch_size,)
        
        return values_batch * self.scale_factor
    
    @GPUMemoryLogger(role="cnn critic", logger=logger)
    def update_critic(self, data: DataProto) -> dict:
        """
        Update the critic using PPO value loss.
        
        Args:
            data: DataProto containing training data
        
        Returns:
            metrics: Dictionary of training metrics
        """
        self.critic_module.train()
        metrics = {}
        
        # Select keys needed for training
        select_keys = ["response_mask", "values", "token_level_rewards"]  # Need response_mask to aggregate returns
        non_tensor_select_keys = []
        
        # Priority: multi_modal_data (PIL images from agent_loop)
        if "multi_modal_data" in data.non_tensor_batch:
            non_tensor_select_keys.append("multi_modal_data")
        # Fallback options
        elif "multi_modal_inputs" in data.non_tensor_batch:
            non_tensor_select_keys.append("multi_modal_inputs")
        
        mini_batch = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        for _ in range(self.config.ppo_epochs):
            mini_batch = mini_batch.to(get_device_id())
            mini_batch_metrics = {}
            
            returns_batch = mini_batch.batch["token_level_rewards"].sum(-1) / self.scale_factor  # (batch_size,)
            # values_batch = mini_batch.batch["values"]  # (batch_size,)
            
            # Forward pass - get batch-level predictions
            vpreds = self._forward_batch(mini_batch.batch, mini_batch.non_tensor_batch)  # (batch_size,)

            vf_loss = F.smooth_l1_loss(vpreds, returns_batch)
            
            # Compute value loss using batch-level values with PPO clipping
            # vpreds_clipped = values_batch + torch.clamp(
            #     vpreds - values_batch, 
            #     -self.config.cliprange_value, 
            #     self.config.cliprange_value
            # )
            # vf_loss1 = (vpreds - returns_batch) ** 2
            # vf_loss = vf_loss1.mean()
            # vf_loss2 = (vpreds_clipped - returns_batch) ** 2
            # vf_loss = torch.max(vf_loss1, vf_loss2).mean()
            
            # Compute clip fraction
            # vf_clipfrac = ((vpreds - values_batch).abs() > self.config.cliprange_value).float().mean()
            
            # Backward pass
            self.critic_optimizer.zero_grad()
            loss = vf_loss
            loss.backward()
            
            mini_batch_metrics.update({
                "critic/vf_loss": vf_loss.detach().item(),
                # "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                "critic/vpred_mean": vpreds.detach().mean().item(),
                "critic/returns_mean": returns_batch.detach().mean().item(),
            })
            
            grad_norm = self._optimizer_step()
            mini_batch_metrics["critic/grad_norm"] = grad_norm.detach().item()
            
            append_to_dict(metrics, mini_batch_metrics)
    
        self.critic_optimizer.zero_grad()
        return metrics

