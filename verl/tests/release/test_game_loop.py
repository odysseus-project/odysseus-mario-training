"""Scripted CPU rollouts preserve the game loop's state-machine behavior."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
from PIL import Image
import torch

from verl.experimental.agent_loop import game_agent_loop as game_module
from verl.experimental.agent_loop.agent_loop import _DummyConfig


def normalize(value):
    """Serialize rollout values deterministically, including image content."""
    if isinstance(value, Image.Image):
        return {"size": list(value.size), "mode": value.mode,
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


class ScriptedTokenizer:
    """Encode characters so tests need no downloaded tokenizer or weights."""

    def __init__(self):
        self.templates = []

    def apply_chat_template(self, messages, **kwargs):
        record = copy.deepcopy({"messages": messages, "kwargs": kwargs})
        # The old tool-only startup calculation does not feed a rollout prompt.
        if kwargs.get("add_generation_prompt"):
            self.templates.append(record)
        text = json.dumps(record, sort_keys=True)
        return [ord(char) for char in text] if kwargs.get("tokenize") else text

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class ScriptedProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.inputs = []

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(self, text, images, return_tensors):
        self.inputs.append(normalize({"text": text, "images": images, "return_tensors": return_tensors}))
        return {"input_ids": torch.tensor([[ord(char) for char in text[0]]])}


class InlineExecutor:
    """Keep scripted tokenization synchronous; worker scheduling is outside this test."""

    async def run_in_executor(self, executor, function):
        return function()


class ScriptedServer:
    def __init__(self, scenario):
        self.scenario = scenario
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(normalize(copy.deepcopy(kwargs)))
        index = len(self.calls) - 1
        if self.scenario.get("fail_generation") == index:
            raise RuntimeError("scripted generation failure")
        responses = self.scenario.get("responses", ['<answer>["right"]</answer>'])
        response = responses[min(index, len(responses) - 1)]
        tokens = [ord(char) for char in response]
        log_probs = [-0.125 * (i + 1) for i in range(len(tokens))] if self.scenario.get("log_probs", True) else []
        return SimpleNamespace(token_ids=tokens, log_probs=log_probs)


class ScriptedEnvironment:
    def __init__(self, scenario, **kwargs):
        self.scenario = scenario
        self.kwargs = kwargs
        self.world = kwargs.get("world", 1)
        self.level = kwargs.get("level", 1)
        self.init_state = kwargs.get("init_state")
        self.mario = SimpleNamespace(level_progress=10, score=0)
        self.steps = 0
        self.actions = []
        self.resets = []
        self.close_count = 0

    def reset(self, seed):
        self.resets.append(seed)
        return self.get_observation(), {}

    def get_observation(self):
        return {"state": f"step={self.steps}\n",
                "image": Image.new("RGB", (16, 12), (self.steps * 20, 40, 80))}

    def step(self, buttons):
        self.actions.append(list(buttons))
        if self.scenario.get("fail_step") == len(self.actions):
            raise RuntimeError("scripted environment failure")
        self.steps += 1
        self.mario.level_progress += 3
        self.mario.score += 10
        info = {"level_progress": self.mario.level_progress, "game_score": self.mario.score,
                "invalid_buttons": [button for button in buttons if button not in ("right", "a", "noop")]}
        terminated = self.steps == self.scenario.get("terminate_at", 3)
        truncated = self.steps == self.scenario.get("truncate_at", -1)
        return self.get_observation(), 3.0 * self.steps - 1.0, terminated, truncated, info

    def close(self):
        self.close_count += 1


SCENARIOS = {
    "text_termination": {},
    "multimodal_validation": {"multimodal": True, "validate": True},
    "pipe_validation": {"multimodal": True, "validate": True, "pipe": True},
    "two_turn_context": {"multimodal": True, "context": 2},
    "malformed_actions": {"multimodal": True, "terminate_at": 2, "responses": [
        "missing answer", "<answer>['right', 'a', 'b']</answer>", "<answer>'right'</answer>",
        "<answer>['invalid']</answer>", "<answer>['right']</answer>"]},
    "response_truncation": {"response_length": 8, "assistant_limit": 3, "terminate_at": 99},
    "assistant_limit": {"assistant_limit": 2, "terminate_at": 99},
    "user_limit": {"user_limit": 2, "terminate_at": 99},
    "trailing_five_turns": {"multimodal": True, "terminate_at": 8, "validate": True},
    "deferred_return_normalization": {"normalize_rewards": True},
    "without_logprobs": {"log_probs": False},
    "generation_failure": {"fail_generation": 1},
    "environment_failure": {"fail_step": 1, "terminate_at": 2},
    "environment_truncation": {"truncate_at": 2, "terminate_at": 99},
    "weighted_training_level": {"weighted": True},
}


def make_config(directory, scenario):
    game_config = {"init_state": "/assets/pipe_gray.state"} if scenario.get("pipe") else {
        "world_level_choices": [[1, 1], [2, 2]], "validate_world_level_choices": [[3, 1], [4, 2]],
    }
    path = directory / "environment.yaml"
    OmegaConf.save(OmegaConf.create({"class_name": "scripted.Environment", "config": game_config}), path)
    return OmegaConf.create({
        "actor_rollout_ref": {"rollout": {
            "prompt_length": 128, "response_length": scenario.get("response_length", 256),
            "multi_turn": {"game_config_path": str(path), "tool_config_path": None,
                "interaction_config_path": None, "format": "hermes", "max_parallel_calls": 1,
                "max_tool_response_length": 64, "tool_response_truncate_side": "right",
                "max_user_turns": scenario.get("user_limit"),
                "max_assistant_turns": scenario.get("assistant_limit", 12),
                "markov_context_turns": scenario.get("context", 1), "markov_discount_factor": 0.5,
                "group_interval": 5, "norm_in_agent_loop": scenario.get("normalize_rewards", False),
                "format_penalty": -100.0, "image_resize_ratio": 0.5}}},
        "data": {"apply_chat_template_kwargs": {"enable_thinking": False}},
        "algorithm": {"norm_adv_by_std_in_grpo": False},
        "trainer": {"validation_data_dir": str(directory / "validation")},
    })


async def collect_rollout(scenario):
    """Run the real loop with scripted model/environment boundaries."""
    class IsolatedGameLoop(game_module.GameAgentLoop):
        _class_initialized = False

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        config = make_config(directory, scenario)
        tokenizer = ScriptedTokenizer()
        processor = ScriptedProcessor(tokenizer) if scenario.get("multimodal") else None
        server = ScriptedServer(scenario)
        environments = []

        def make_environment(**kwargs):
            environment = ScriptedEnvironment(scenario, **kwargs)
            environments.append(environment)
            return environment

        with patch.object(game_module, "get_class_from_name", return_value=make_environment), \
                patch.object(game_module.random, "choice", side_effect=lambda values: values[0]), \
                patch.object(game_module.random, "choices", side_effect=lambda values, **kwargs: [values[-1]]):
            weights = {"world1_level1": 1.0, "world2_level2": 2.0} if scenario.get("weighted") else None
            loop = IsolatedGameLoop(_DummyConfig(config), server, tokenizer, processor,
                                    validate=scenario.get("validate", False), world_level_weights=weights)
            loop.loop = InlineExecutor()
            error = None
            outputs = []
            try:
                outputs = await loop.run({"temperature": 0.7}, request_id="scripted",
                                         validate=scenario.get("validate", False), step=7)
            except RuntimeError as exception:
                if not scenario.get("fail_generation"):
                    raise
                error = str(exception)

        artifacts = {}
        for path in sorted((directory / "validation").rglob("*")):
            if path.is_file():
                name = re.sub(r"\d{8}_\d{6}\.\d{6}", "TIMESTAMP", str(path.relative_to(directory)))
                artifacts[name] = json.loads(path.read_text()) if path.suffix == ".json" else hashlib.sha256(path.read_bytes()).hexdigest()
        environment = environments[0]
        return normalize({
            "outputs": [output.model_dump(exclude={"metrics"}) for output in outputs],
            "templates": tokenizer.templates, "processor_inputs": processor.inputs if processor else [],
            "server_calls": server.calls, "actions": environment.actions, "steps": environment.steps,
            "environment_kwargs": environment.kwargs, "resets": environment.resets,
            "close_count": environment.close_count, "error": error, "validation_artifacts": artifacts,
        })


def summarize(result):
    """Keep a reviewable summary plus a hash covering all deterministic fields."""
    return {"sha256": hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(),
            "outputs": len(result["outputs"]), "actions": result["actions"], "steps": result["steps"],
            "close_count": result["close_count"], "error": result["error"],
            "turn_scores": [output["extra_fields"]["turn_scores"] for output in result["outputs"]],
            "validation_files": len(result["validation_artifacts"])}


class ScriptedGameLoopTests(unittest.TestCase):
    def test_rollouts_match_pre_cleanup_behavior(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/game_loop_rollouts.json").read_text())
        for name, scenario in SCENARIOS.items():
            with self.subTest(scenario=name):
                result = asyncio.run(collect_rollout(scenario))
                self.assertEqual(summarize(result), fixture["scenarios"][name])
                self.assertEqual(result["close_count"], 1)
                self.assertTrue(all(call["kwargs"]["tools"] == [] for call in result["templates"]))

    def test_unsupported_configs_fail_before_initialization_and_after_reuse(self):
        for field, message in (("tool_config_path", "Tools"), ("interaction_config_path", "Interactions")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                class IsolatedGameLoop(game_module.GameAgentLoop):
                    _class_initialized = False

                config = make_config(Path(temporary), {})
                multi_turn = config.actor_rollout_ref.rollout.multi_turn
                tokenizer = ScriptedTokenizer()
                multi_turn[field] = "unsupported.yaml"
                with self.assertRaisesRegex(NotImplementedError, message):
                    IsolatedGameLoop.init_class(config, tokenizer, None)
                self.assertFalse(IsolatedGameLoop._class_initialized)
                multi_turn[field] = None
                IsolatedGameLoop.init_class(config, tokenizer, None)
                self.assertTrue(IsolatedGameLoop._class_initialized)
                multi_turn[field] = "unsupported.yaml"
                with self.assertRaisesRegex(NotImplementedError, message):
                    IsolatedGameLoop.init_class(config, tokenizer, None)


if __name__ == "__main__":
    unittest.main()
