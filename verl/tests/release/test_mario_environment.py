"""Regression coverage for the fixed Mario training environment."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from verl.utils.game_envs.super_mario_land import pyboy_super_mario_land_env as environment


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = Path(__file__).parent / "fixtures/mario_environment_traces.json"
ACTIONS = [
    ["right"], ["right"], ["a", "right"], ["a"], ["noop"], [],
    ["invalid"], ["RIGHT", "invalid"], ["left"], ["b", "right"],
    ["up"], ["down"], ["a", "b"], ["right", "noop"], ["left", "right"],
] * 2
LEVELS = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2),
          (3, 1), (3, 2), (3, 3), (4, 1), (4, 2)]


def capture_traces(rom_path, factory=environment.PyBoySuperMarioLandEnv):
    """Record actual emulator outputs while isolating the test's global RNG use."""
    cases = [(f"world{world}_level{level}", None, world, level) for world, level in LEVELS]
    cases.append(("pipe", str(REPO_ROOT / "rom/pipe_gray.state"), 1, 1))
    traces = {}
    rng_state = random.getstate()
    try:
        for name, state, world, level in cases:
            random.seed(719)
            env = factory(rom_path=rom_path, init_state=state, world=world, level=level, headless=True)
            try:
                def snapshot(observation):
                    image = observation["image"]
                    return {"image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                            "image_size": list(image.size), "image_mode": image.mode,
                            "state": observation["state"], "world": list(env.mario.world),
                            "level_progress": int(env.mario.level_progress), "score": int(env.mario.score),
                            "lives": int(env.mario.lives_left), "step_count": env.step_count}

                observation, info = env.reset(seed=0)
                records = [{"reset": snapshot(observation), "info": info}]
                for action in ACTIONS:
                    frame_before = env.pyboy.frame_count
                    observation, reward, done, truncated, info = env.step(action)
                    records.append({"action": action, "observation": snapshot(observation),
                                    "reward": reward, "done": bool(done), "truncated": bool(truncated),
                                    "info": info, "frames": env.pyboy.frame_count - frame_before})
                    if done or truncated:
                        break
                observation, info = env.reset(seed=0)
                records.append({"reset_after_actions": snapshot(observation), "info": info})
                # The worker uses the same Python RNG for later world/level selection.
                records.append({"next_random": random.random()})
                traces[name] = records
            finally:
                env.close()
    finally:
        random.setstate(rng_state)
    return traces


class FixedEnvironmentTests(unittest.TestCase):
    def test_reward_ignores_score_and_suppresses_progress_on_life_changes(self):
        env = object.__new__(environment.PyBoySuperMarioLandEnv)
        env.pyboy = SimpleNamespace(memory={0xFFA6: 0})
        env.mario = SimpleNamespace(world=(1, 1), coins=0, lives_left=2,
                                    score=100, time_left=300, level_progress=10)
        env.set_prev_state()
        for progress, score, lives, death_timer, expected in (
            (13, 900, 2, 0, 30.0),
            (7, 900, 2, 0, -30.0),
            (10, 900, 2, 0, 0.0),
            (13, 900, 1, 0, 0.0),
            (13, 900, 3, 0, 0.0),
            (13, 900, 2, 0x81, 0.0),
        ):
            with self.subTest(progress=progress, lives=lives, death_timer=death_timer):
                env.mario.level_progress, env.mario.score, env.mario.lives_left = progress, score, lives
                env.pyboy.memory[0xFFA6] = death_timer
                self.assertEqual(env.compute_reward(), expected)

    def test_step_limits_and_death_detection(self):
        env = object.__new__(environment.PyBoySuperMarioLandEnv)
        env.pyboy = SimpleNamespace(memory={0xFFA6: 0})
        env.mario = SimpleNamespace(world=(1, 1), coins=0, lives_left=3,
                                    score=100, time_left=300, level_progress=10, game_over=lambda: False)
        env.press_buttons = lambda buttons: None
        env.get_observation = lambda: {"state": ""}
        env.set_prev_state()
        with patch.object(environment.random, "random", return_value=0.5):
            for count, lives, initial_lives, death_timer, game_over, expected in (
                (0, 3, 3, 0, False, False),
                (env.MAX_STEP - 1, 3, 3, 0, False, True),
                (0, 3, 3, 0x81, False, True),
                (0, 1, 2, 0, False, True),
                (0, 2, 3, 0, False, False),
                (0, 3, 3, 0, True, True),
            ):
                with self.subTest(count=count, lives=lives, death_timer=death_timer, game_over=game_over):
                    env.step_count, env.init_lives_left = count, initial_lives
                    env.mario.lives_left = lives
                    env.mario.game_over = lambda: game_over
                    env.pyboy.memory[0xFFA6] = death_timer
                    observation, reward, done, truncated, info = env.step(["right"])
                    self.assertEqual(done, expected)
                    self.assertFalse(truncated)

    def test_retired_options_fail_before_starting_emulator(self):
        with patch.object(environment, "PyBoy") as emulator:
            for name in ("with_text_info", "with_game_area", "use_old_tick", "use_simple_reward",
                         "epsilon", "cgb", "sound", "resize_ratio"):
                with self.subTest(option=name), self.assertRaisesRegex(TypeError, name):
                    environment.PyBoySuperMarioLandEnv(rom_path="unused.gb", **{name: False})
            emulator.assert_not_called()


@unittest.skipUnless(os.environ.get("ODYSSEUS_TEST_ROM"), "Set ODYSSEUS_TEST_ROM for emulator trace checks")
class EmulatorTraceTests(unittest.TestCase):
    def test_matches_training_environment_before_option_removal(self):
        fixture = json.loads(FIXTURE_PATH.read_text())
        rom_path = os.environ["ODYSSEUS_TEST_ROM"]
        if hashlib.sha256(Path(rom_path).read_bytes()).hexdigest() != fixture["rom_sha256"]:
            self.skipTest("Golden emulator traces require the reference Super Mario Land ROM revision")
        self.assertEqual(importlib.metadata.version("pyboy"), fixture["pyboy_version"])
        actual = capture_traces(rom_path)
        self.assertEqual(set(actual), set(fixture["cases"]))
        for name, expected in fixture["cases"].items():
            with self.subTest(case=name):
                self.assertEqual(len(actual[name]) - 3, expected["action_steps"])
                digest = hashlib.sha256(json.dumps(actual[name], sort_keys=True).encode()).hexdigest()
                self.assertEqual(digest, expected["sha256"])


if __name__ == "__main__":
    unittest.main()
