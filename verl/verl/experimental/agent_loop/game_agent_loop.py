import ast
import copy
from dataclasses import dataclass, field
from enum import Enum
import importlib
import json
import logging
import os
import random
import re
from typing import Any, Tuple, List
from uuid import uuid4
from datetime import datetime

from transformers import AutoProcessor, AutoTokenizer
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register, _DummyConfig, AsyncLLMServerManager
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.game_envs.super_mario_land.env_prompts import SYSTEM_PROMPT
from verl.experimental.agent_loop.utils import compute_return_to_go, resize_image

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class GameState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    STEPPING_ENV = "stepping_env"
    TERMINATED = "terminated"


@dataclass
class GameData:
    """Mutable state for one game rollout and its per-turn snapshots."""

    messages: list[dict[str, Any]]
    image_data: Any
    metrics: dict[str, Any]
    request_id: str
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    turn_scores: list[float] = field(default_factory=list)
    level_progress: list[float] = field(default_factory=list)
    game_score: list[float] = field(default_factory=list)
    user_turns: int = 0
    assistant_turns: int = 0


def parse_button_sequence(message: str) -> List[str]:
    """Parse button sequence from assistant message with <answer> tags.
    
    Args:
        message: Assistant message containing <answer>["button1", "button2", ...]</answer>
        or <answer>['button1', 'button2', ...]</answer>
        
    Returns:
        List of button strings to press
    """
    # Extract content from <answer> tags
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, message, re.DOTALL)
    
    if not match:
        raise ValueError(f"No <answer> tag found in the assistant response")
    
    answer_content = match.group(1).strip()
    
    # Try to parse as Python list literal (handles both single and double quotes)
    try:
        buttons = ast.literal_eval(answer_content)
        if isinstance(buttons, list):
            if len(buttons) > 2:
                raise ValueError(f"Too many buttons: {buttons}. Maximum 2 buttons are allowed per turn.")
            return [str(btn).lower() for btn in buttons]
        else:
            raise ValueError(f"Answer content is not a list: {answer_content}")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse answer as list: {answer_content}, error: {e}")


def get_class_from_name(class_name: str):
    """Dynamically import a class from a string."""

    module_name, cls_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls


@register("game_agent")
class GameAgentLoop(AgentLoopBase):
    """Generate button actions and collect per-turn game training outputs."""

    def __init__(
        self,
        trainer_config: _DummyConfig,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)

        # Initialize game environment from config file
        game_config_path = trainer_config.config.actor_rollout_ref.rollout.multi_turn.game_config_path
        if not game_config_path:
            raise ValueError("game_config_path must be specified in the config file for GameAgentLoop")
        game_config = OmegaConf.load(game_config_path)
        game_class_name = game_config.get("class_name", None)
        if not game_class_name:
            raise ValueError("class_name must be specified in the game config file")
        game_class = get_class_from_name(game_class_name)
        game_env_config = OmegaConf.to_container(game_config.get("config", {}), resolve=True)
        # world_level_choices: list of [world, level] e.g. [[1,2], [1,3], [2,1]]
        world_level_choices = game_env_config.pop("world_level_choices", None)
        validate_world_level_choices = game_env_config.pop("validate_world_level_choices", None)
        world_level_weights = kwargs.pop("world_level_weights", None)  # dict "world{W}_level{L}" -> weight for dynamic sampling
        validate = kwargs.pop("validate", False)  # when True, always sample world/level uniformly for validation
        # Convert OmegaConf DictConfig (e.g. from batch.meta_info) to plain dict so isinstance(..., dict) and .get() work
        if world_level_weights is not None and not isinstance(world_level_weights, dict):
            try:
                world_level_weights = OmegaConf.to_container(world_level_weights, resolve=True)
            except Exception:
                world_level_weights = dict(world_level_weights) if hasattr(world_level_weights, "items") else None

        def _to_pairs(raw: Any) -> List[Tuple[int, int]]:
            """Convert config list [[world, level], ...] to list of (world, level) tuples."""
            if raw is None:
                return []
            out = []
            for item in raw:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.append((int(item[0]), int(item[1])))
                else:
                    raise ValueError(f"world_level_choices entry must be [world, level], got: {item!r}")
            return out

        pairs = _to_pairs(world_level_choices) if world_level_choices is not None else []

        self._chosen_world_level = None  # default for init_state_choices and other scenarios
        if pairs:
            assert game_env_config.get("init_state", None) is None, "init_state must be None when world/level choices are specified"
            logger.warning(f"[world_level_sampling] world_level_weights in game_agent_loop: {world_level_weights}, validate: {validate}")
            if validate and validate_world_level_choices is not None:
                validate_pairs = _to_pairs(validate_world_level_choices)
                if not validate_pairs:
                    validate_pairs = pairs  # fallback to training pairs if validate list is empty
                world, level = random.choice(validate_pairs)
                logger.warning(f"[world_level_sampling] Selected world: {world}, level: {level} (validation, validate_world_level_choices)")
            elif validate:
                world, level = random.choice(pairs)
                logger.warning(f"[world_level_sampling] Selected world: {world}, level: {level} (validation, uniform sampling)")
            elif world_level_weights and isinstance(world_level_weights, dict):
                existing_weights = [world_level_weights.get(f"world{w}_level{l}") for w, l in pairs]
                known = [x for x in existing_weights if x is not None]
                default_weight = (sum(known) / len(known)) if known else 1.0
                weights = [world_level_weights.get(f"world{w}_level{l}", default_weight) for w, l in pairs]
                total = sum(weights)
                if total > 0:
                    probs = [w / total for w in weights]
                    logger.warning(f"[world_level_sampling] weights: {weights}, probs: {probs}")
                    world, level = random.choices(pairs, weights=probs, k=1)[0]
                    logger.warning(
                        f"[world_level_sampling] Selected world: {world}, level: {level} (weighted sampling)"
                    )
                else:
                    world, level = random.choice(pairs)
                    logger.warning(f"[world_level_sampling] Selected world: {world}, level: {level} (fallback uniform)")
            else:
                world, level = random.choice(pairs)
                logger.warning(f"[world_level_sampling] Selected world: {world}, level: {level} (uniform sampling)")
            game_env_config["world"] = world
            game_env_config["level"] = level
            self._chosen_world_level = f"world{world}_level{level}"

        init_state_choices = game_env_config.pop("init_state_choices", None)
        if init_state_choices:
            assert game_env_config.get("init_state", None) is None, "init_state must be None if init_state_choices are specified"
            state_paths = [f for f in os.listdir(init_state_choices) if f.endswith(".state")]
            state_pool = [os.path.join(init_state_choices, f) for f in state_paths]
            init_state = random.choice(state_pool)
            game_env_config["init_state"] = init_state
            logger.warning(f"Randomly selected init_state: {init_state}")
        self.game_env = game_class(**game_env_config)
        self.log_dir = trainer_config.config.trainer.get("validation_data_dir")

        if not self.markov_context_turns:
            raise ValueError("markov_context_turns is not specified in the config file for GameAgentLoop")

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        multi_turn = config.actor_rollout_ref.rollout.multi_turn
        if multi_turn.get("tool_config_path"):
            raise NotImplementedError("Tools are not supported for GameAgentLoop")
        if multi_turn.get("interaction_config_path"):
            raise NotImplementedError("Interactions are not supported for GameAgentLoop")
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level GameAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.markov_context_turns = config.actor_rollout_ref.rollout.multi_turn.markov_context_turns
        cls.markov_discount_factor = config.actor_rollout_ref.rollout.multi_turn.markov_discount_factor
        cls.group_interval = config.actor_rollout_ref.rollout.multi_turn.get("group_interval", 1)
        cls.norm_in_agent_loop = config.actor_rollout_ref.rollout.multi_turn.get("norm_in_agent_loop", False)
        cls.format_penalty = config.actor_rollout_ref.rollout.multi_turn.get("format_penalty", -100.0)
        if cls.norm_in_agent_loop:
            norm_adv_by_std_in_grpo = config.algorithm.get(
                "norm_adv_by_std_in_grpo", True
            )  # GRPO adv normalization factor
            assert not norm_adv_by_std_in_grpo, "norm_adv_by_std_in_grpo must be False when norm_in_agent_loop is True"
        cls.image_resize_ratio = config.actor_rollout_ref.rollout.multi_turn.get("image_resize_ratio", 8)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> List[AgentLoopOutput]:
        try:
            return await self._run_impl(sampling_params, **kwargs)
        finally:
            # Clean up game environment to prevent resource leaks
            if hasattr(self, 'game_env') and self.game_env is not None:
                try:
                    self.game_env.close()
                except Exception as e:
                    logger.warning(f"Error when closing game environment: {e}")

    async def _run_impl(self, sampling_params: dict[str, Any], **kwargs) -> List[AgentLoopOutput]:
        """Internal implementation of run method."""
        # Each observation carries the canonical prompt; begin with an empty history.
        messages = []
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = kwargs.get("request_id", uuid4().hex)
        sampling_params["max_tokens"] = self.response_length

        # Create GameData instance to encapsulate all state
        agent_data = GameData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
        )
        agent_data_list = []
        reward_list = []
        level_progress_list = []
        game_score_list = []
        # Reset game environment
        self.game_env.reset(seed=0)

        # State machine loop
        state = GameState.STEPPING_ENV
        while state != GameState.TERMINATED:
            if state == GameState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == GameState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
                agent_data.assistant_turns += 1
                if state != GameState.TERMINATED:
                    agent_data_list.append(copy.deepcopy(agent_data))
            elif state == GameState.STEPPING_ENV:
                state = await self._handle_environment_step(agent_data, first_turn=(len(agent_data_list) == 0))
                if len(agent_data_list) > 0: # align the reward list with the agent_data_list
                    reward_list.append(agent_data.turn_scores[-1])
                    level_progress_list.append(agent_data.level_progress[-1])
                    game_score_list.append(agent_data.game_score[-1])
            else:
                logger.error(f"Invalid state: {state}")
                state = GameState.TERMINATED

        # compute return-to-go for each turn
        assert len(agent_data_list) == len(reward_list), "The number of agent_data_list and reward_list should be equal"
        if self.norm_in_agent_loop:
            rtg_list = reward_list
        else:
            rtg_list = compute_return_to_go(reward_list, self.markov_discount_factor)

        # Finalize output
        output_list = []
        if len(rtg_list) > 5:
            agent_data_list = agent_data_list[:-5]
            rtg_list = rtg_list[:-5]
            level_progress_list = level_progress_list[:-5]
            game_score_list = game_score_list[:-5]
        for turn_idx, (agent_data, rtg, level_progress, game_score) in enumerate(
            zip(agent_data_list, rtg_list, level_progress_list, game_score_list)
        ):
            response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
            prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
            multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
            output = AgentLoopOutput(
                prompt_ids=prompt_ids[-self.prompt_length :],
                response_ids=response_ids[: self.response_length],
                response_mask=agent_data.response_mask[: self.response_length],
                multi_modal_data=multi_modal_data,
                response_logprobs=agent_data.response_logprobs[: self.response_length]
                if agent_data.response_logprobs
                else None,
                num_turns=len(agent_data.messages),
                metrics=agent_data.metrics,
                extra_fields={},
            )
            output.extra_fields.update({"turn_scores": rtg})
            output.extra_fields.update({"level_progress": level_progress})
            output.extra_fields.update({"discretized_level_progress": level_progress // self.group_interval})
            output.extra_fields.update({"game_score": game_score})
            output.extra_fields.update({
                "chosen_world_level": self._chosen_world_level if self._chosen_world_level is not None else "unknown"
            })
            output.extra_fields.update({"request_id": agent_data.request_id})
            output.extra_fields.update({"turn_idx": turn_idx})
            output.extra_fields.update({"step_reward": reward_list[turn_idx]})
            output_list.append(output)

        # Save image data
        if kwargs.get("validate", False) and agent_data.image_data is not None:
            global_step = kwargs.get("step", "none")
            now = datetime.now()
            time_format = now.strftime("%Y%m%d_%H%M%S.%f")
            if self.game_env.init_state:
                state_name = self.game_env.init_state.split("/")[-1].split(".")[0]
            else:
                state_name = f"world{self.game_env.world}_level{self.game_env.level}"
            score = sum(reward_list)
            save_dir = f"{self.log_dir}/globalstep{global_step}/{time_format}_{state_name}_score{score}"
            print(f"Logging validation rollout to {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            for idx, agent_data in enumerate(agent_data_list):
                img = agent_data.image_data[-1]
                img.convert("RGB").save(f"{save_dir}/screen{idx}.jpg")
                with open(f"{save_dir}/messages{idx}.json", "w") as f:
                    json.dump(agent_data.messages, f)
        return output_list

    async def _handle_pending_state(self, agent_data: GameData, sampling_params: dict[str, Any]) -> GameState:
        """Prepare the prompt, preserving the original empty tools list in the chat template."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=[],
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Resize before processor so prompt encoding matches generate path (and agent_loop)
            images_for_processor = (
                [resize_image(img, self.image_resize_ratio) for img in agent_data.image_data]
                if self.image_resize_ratio != 1 else agent_data.image_data
            )
            model_inputs = self.processor(text=[raw_prompt], images=images_for_processor, return_tensors="pt")
            # Debug: log the resized image shape to verify VLM receives correct size (only on first turn)
            if "pixel_values" in model_inputs and agent_data.assistant_turns == 0:
                pixel_values = model_inputs["pixel_values"]
                orig_size = agent_data.image_data[0].size if agent_data.image_data and hasattr(agent_data.image_data[0], 'size') else 'N/A'
                if hasattr(pixel_values, "shape"):
                    logger.warning(f"[VLM Image Check] Original image size: {orig_size}, "
                                   f"pixel_values shape: {pixel_values.shape}, "
                                   f"pixel_values dtype: {pixel_values.dtype}, "
                                   f"image_grid_thw: {model_inputs.get('image_grid_thw', 'N/A')}")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=[],
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return GameState.GENERATING

    async def _handle_generating_state(self, agent_data: GameData, sampling_params: dict[str, Any]) -> GameState:
        """Generate a model response and advance to the environment step."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            images_for_generate = (
                [resize_image(img, self.image_resize_ratio) for img in agent_data.image_data]
                if self.image_resize_ratio != 1 else agent_data.image_data
            )
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=images_for_generate if self.processor is not None else None,
            )

        if len(output.token_ids) > self.response_length:
            logger.warning(f"Response length exceeded: {len(output.token_ids)} > {self.response_length}, truncating...")
            output.token_ids = output.token_ids[: self.response_length]
            if output.log_probs:
                output.log_probs = output.log_probs[: self.response_length]

        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Preserve the original turn-limit ordering before applying the generated action.
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            logger.warning(f"Max assistant turns exceeded: {agent_data.assistant_turns} >= {self.max_assistant_turns}")
            return GameState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            logger.warning(f"Max user turns exceeded: {agent_data.user_turns} >= {self.max_user_turns}")
            return GameState.TERMINATED

        assistant_message = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.response_ids)
        )
        add_messages.append({"role": "assistant", "content": assistant_message})
        agent_data.messages.extend(add_messages)

        return GameState.STEPPING_ENV

    async def _handle_environment_step(self, agent_data: GameData, first_turn: bool = False) -> GameState:
        """Observe the initial state or apply the generated button action."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        if first_turn:
            obs = self.game_env.get_observation()
            reward = 0.0
            done = False
            level_progress = self.game_env.mario.level_progress
            game_score = self.game_env.mario.score
            text_obs = obs["state"]
            image_obs = [obs["image"]]
        else:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids)
            )

            try:
                # Parse button sequence from assistant message
                button_sequence = parse_button_sequence(assistant_message)
                obs, reward, terminated, truncated, info = self.game_env.step(button_sequence)
                done = terminated or truncated
                level_progress = info.get("level_progress", self.game_env.mario.level_progress)
                game_score = info.get("game_score", self.game_env.mario.score)
                if info.get("invalid_buttons", []):
                    text_obs = f"Invalid buttons: {info['invalid_buttons']}\n" + obs["state"]
                else:
                    text_obs = obs["state"]
                image_obs = [obs["image"]]
            except Exception as e:
                obs = self.game_env.get_observation()
                reward = self.format_penalty
                done = False
                level_progress = self.game_env.mario.level_progress
                game_score = self.game_env.mario.score
                text_obs = f"Error when interacting with game environment: {e}.\n" + obs["state"]
                image_obs = [obs["image"]]

        text_obs = text_obs + SYSTEM_PROMPT

        # Only add image placeholders if we have a processor (VLM), otherwise use plain text (LLM)
        if self.processor is not None and image_obs:
            user_content = []
            if text_obs:
                user_content.append({"type": "text", "text": text_obs})
            user_content.append({"type": "image"}) # NOTE: change the order to match SFT training data
            add_messages.append({"role": "user", "content": user_content})
        else:
            add_messages.append({"role": "user", "content": text_obs})
        agent_data.messages.extend(add_messages)

        # Handle image data
        if image_obs:
            for img in image_obs:
                if img is not None:
                    new_images_this_turn.append(img)  # No deep copy needed here

        # Process information returned from game enviornment and update into agent_data
        agent_data.turn_scores.extend([reward])
        agent_data.level_progress.extend([level_progress])
        agent_data.game_score.extend([game_score])
        if done:
            return GameState.TERMINATED

        # Add image data to agent_data here
        # No deep copy needed - images from get_observation() are already new objects
        if agent_data.image_data is None:
            agent_data.image_data = []
        elif not isinstance(agent_data.image_data, list):
            agent_data.image_data = [agent_data.image_data]
        for img in new_images_this_turn:
            agent_data.image_data.append(img)

        # Manage the markov context: no system prompt
        if len(agent_data.messages) > self.markov_context_turns * 2 - 1: # user, assistant, user, assistant, ..., user
            assert len(agent_data.messages) == 2 * len(agent_data.image_data) - 1, "The number of user messages and image data should be equal"
            # Truncate oldest user and assistant messages, and image data
            agent_data.messages = agent_data.messages[-self.markov_context_turns * 2 + 1:]
            agent_data.image_data = agent_data.image_data[-self.markov_context_turns:]
            # reset state variables
            agent_data.prompt_ids = []
            agent_data.response_ids = []
            agent_data.response_mask = []
            agent_data.response_logprobs = []

        agent_data.user_turns += 1
        return GameState.PENDING
