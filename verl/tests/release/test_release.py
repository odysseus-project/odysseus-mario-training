"""CPU release checks. Set ODYSSEUS_TEST_ROM to include emulator integration."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq

from jobs.train import REPO_ROOT, VERL_ROOT, save_run_metadata, validate_inputs
from jobs.generate_dataset import generate_dataset
from verl.utils.game_envs.super_mario_land.env_prompts import FORMAT_PROMPT, SYSTEM_PROMPT

FIXTURE = json.loads((Path(__file__).parent / "fixtures/original_experiments.json").read_text())


def config_for(name, overrides=()):
    with initialize_config_dir(config_dir=str(VERL_ROOT / "jobs/config"), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


class ConfigTests(unittest.TestCase):
    def test_all_experiments_match_release_settings(self):
        # The fixture was captured from the original shell scripts before migration.
        # The authors subsequently standardized pipe prompt/response limits to 2048/1024.
        pipe_token_limits = {"data.max_prompt_length": 2048, "data.max_response_length": 1024}
        with patch.dict(os.environ, {"ODYSSEUS_MODEL_PATH": "Qwen/Qwen3-VL-8B-Instruct"}):
            for name, original in FIXTURE["experiments"].items():
                config = config_for(name)
                OmegaConf.resolve(config)
                for key, expected in original["overrides"].items():
                    with self.subTest(experiment=name, setting=key):
                        if key == "critic":
                            self.assertEqual(config.critic.strategy, "cnn")
                            continue
                        if key == "data.custom_cls.path":
                            expected = str(VERL_ROOT / expected)
                        if name.startswith("pipe_") and key in pipe_token_limits:
                            expected = pipe_token_limits[key]
                        self.assertEqual(OmegaConf.select(config, key), expected)
                with self.subTest(experiment=name, setting="actor_rollout_ref.rollout.prompt_length"):
                    self.assertEqual(config.actor_rollout_ref.rollout.prompt_length,
                                     2048 if name.startswith("pipe_") else 4096)
                with self.subTest(experiment=name, setting="actor_rollout_ref.rollout.response_length"):
                    self.assertEqual(config.actor_rollout_ref.rollout.response_length,
                                     1024 if name.startswith("pipe_") else 4096)

    def test_smoke_profiles_preserve_algorithm_and_topology(self):
        for name in FIXTURE["experiments"]:
            with self.subTest(experiment=name):
                multi = name == "train_1024_base"
                base = config_for(name)
                smoke = config_for(name, ["smoke=" + ("multi_level" if multi else "pipe")])
                for key in ("algorithm", "critic.enable", "critic.strategy", "trainer.nnodes",
                            "trainer.n_gpus_per_node", "data.max_prompt_length",
                            "actor_rollout_ref.rollout.multi_turn.markov_discount_factor"):
                    self.assertEqual(OmegaConf.select(base, key), OmegaConf.select(smoke, key))
                self.assertEqual(smoke.data.train_batch_size, 16 if multi else 8)
                self.assertEqual(smoke.actor_rollout_ref.actor.ppo_mini_batch_size, 16 if multi else 8)
                self.assertEqual(smoke.trainer.total_training_steps, 2)
                self.assertEqual(smoke.trainer.save_freq, 1)
                self.assertEqual(smoke.trainer.test_freq, 1)
                self.assertEqual(smoke.trainer.critic_warmup, 0)
                self.assertEqual(smoke.actor_rollout_ref.rollout.multi_turn.max_assistant_turns, 16)
                self.assertEqual(smoke.data.max_response_length, 1024)

    def test_retained_state_integrity(self):
        expected = {
            "pipe_gray.state": "e35958b4ae644fd20f89d5ee24554ec54369289fc8fe2855de0a1f771ba86c0a",
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((REPO_ROOT / "rom" / name).read_bytes()).hexdigest(), digest)


class DatasetTests(unittest.TestCase):
    def test_canonical_prompt_and_format(self):
        self.assertEqual(hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), FIXTURE["prompt_sha256"])
        self.assertIn("noop", SYSTEM_PROMPT)
        self.assertIn(FORMAT_PROMPT, SYSTEM_PROMPT)

    def test_generation_counts_schema_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, val = generate_dataset(tmp, train_size=16, test_size=8)
            self.assertEqual(pq.read_table(train).num_rows, 16)
            self.assertEqual(pq.read_table(val).num_rows, 8)
            for path in (train, val):
                for index, row in enumerate(pq.read_table(path).to_pylist()):
                    self.assertEqual(row["sample_index"], index)
                    self.assertEqual(row["agent_name"], "game_agent")
                    self.assertEqual(row["prompt"], [{"role": "system", "content": SYSTEM_PROMPT}])
            original = train.read_bytes()
            with self.assertRaises(FileExistsError):
                generate_dataset(tmp, train_size=1, test_size=1)
            self.assertEqual(train.read_bytes(), original)
            generate_dataset(tmp, train_size=2, test_size=1, overwrite=True)
            self.assertEqual(pq.read_table(train).num_rows, 2)

    def test_invalid_sizes_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            for train_size, test_size in ((0, 1), (1, 0), (-1, 1)):
                with self.assertRaises(ValueError):
                    generate_dataset(Path(tmp) / "data", train_size, test_size)
            self.assertFalse((Path(tmp) / "data").exists())

    def test_unsupported_game_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "data"
            with self.assertRaisesRegex(ValueError, "Unknown game"):
                generate_dataset(target, game="pokemon_red")
            result = subprocess.run(
                [sys.executable, "-m", "jobs.generate_dataset", "--game", "pokemon_red",
                 "--output_dir", str(target)],
                cwd=VERL_ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid choice", result.stderr)
            self.assertFalse(target.exists())

    def test_prompt_import_does_not_load_emulator_or_other_games(self):
        code = """
import sys
import verl.utils.game_envs
import verl.utils.game_envs.super_mario_land
from verl.utils.game_envs.super_mario_land.env_prompts import SYSTEM_PROMPT
assert SYSTEM_PROMPT
assert 'pyboy' not in sys.modules
assert 'gym_super_mario_bros' not in sys.modules
assert not any(name.startswith('verl.utils.game_envs.pokemon_red') for name in sys.modules)
"""
        subprocess.run([sys.executable, "-c", code], cwd=VERL_ROOT, check=True)


class EnvironmentTests(unittest.TestCase):
    def test_presets_resolve_the_retained_environment(self):
        from hydra.utils import get_class
        from verl.utils import game_envs
        from verl.utils.game_envs import super_mario_land

        environment_class = super_mario_land.PyBoySuperMarioLandEnv
        self.assertIs(game_envs.PyBoySuperMarioLandEnv, environment_class)
        self.assertTrue(issubclass(environment_class, game_envs.BaseEnv))
        for module in (game_envs, super_mario_land):
            for removed in ("PokemonRedEnv", "SuperMarioBrosEnv", "SuperMarioLandEnv",
                            "PyBoySuperMarioLandEnvLegacy"):
                self.assertFalse(hasattr(module, removed))
        for name in FIXTURE["experiments"]:
            with self.subTest(experiment=name):
                config = config_for(name)
                multi_turn = config.actor_rollout_ref.rollout.multi_turn
                environment = OmegaConf.load(multi_turn.game_config_path)
                self.assertIs(get_class(environment.class_name), environment_class)
                self.assertIsNone(multi_turn.tool_config_path)
                self.assertTrue(set(environment.config) <= {
                    "rom_path", "init_state", "world", "level", "headless",
                    "world_level_choices", "validate_world_level_choices",
                })

    def test_mario_reward_values_and_validation_metadata(self):
        from recipe.game_agent.dataset import test_game_score as score_game
        from verl.utils.reward_score import default_compute_score

        for score in (-10.0, 0.0, 12.5):
            info = {"turn_scores": score, "level_progress": 717, "game_score": 100}
            with self.subTest(score=score):
                expected = {"score": score, "pred": "action"}
                self.assertEqual(default_compute_score("super_mario_land", "action", None,
                                                      extra_info=info), expected)
                self.assertEqual(score_game("super_mario_land", "action", None, info),
                                 dict(expected, level_progress=717, game_score=100))
        with self.assertRaises(NotImplementedError):
            default_compute_score("pokemon_red", "action", None, extra_info=info)
        with self.assertRaises(ValueError):
            score_game("pokemon_red", "action", None, info)


class AgentLoopTests(unittest.TestCase):
    def test_generated_rows_select_the_canonical_game_loop(self):
        from hydra.utils import get_class
        from jobs.generate_dataset import make_sample
        from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
        from verl.experimental.agent_loop.game_agent_loop import GameAgentLoop

        sample = make_sample("super_mario_land", SYSTEM_PROMPT, 0)
        target = _agent_loop_registry[sample["agent_name"]]["_target_"]
        self.assertIs(get_class(target), GameAgentLoop)
        self.assertEqual(GameAgentLoop.__module__, "verl.experimental.agent_loop.game_agent_loop")
        for retired in ("markov_agent", "markov_agent_sft"):
            self.assertNotIn(retired, _agent_loop_registry)

    def test_discounted_returns(self):
        from verl.experimental.agent_loop.utils import compute_return_to_go

        for rewards, discount, expected in (
            ([], 0.9, []),
            ([2.0], 0.9, [2.0]),
            ([1.0, -2.0, 4.0], 0.5, [1.0, 0.0, 4.0]),
            ([1.0, -2.0, 4.0], 1.0, [3.0, 2.0, 4.0]),
            ([1.0, -2.0, 4.0], 0.0, [1.0, -2.0, 4.0]),
        ):
            with self.subTest(rewards=rewards, discount=discount):
                original = list(rewards)
                self.assertEqual(compute_return_to_go(rewards, discount), expected)
                self.assertEqual(rewards, original)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Preflight checks file availability/schema; the optional integration test loads an actual ROM.
        rom = self.root / "test.gb"
        rom.write_bytes(b"preflight placeholder")
        env = patch.dict(os.environ, {"ODYSSEUS_ROM_PATH": str(rom), "ODYSSEUS_DATA_ROOT": str(self.root),
                                     "ODYSSEUS_PIPE_STATE": str(REPO_ROOT / "rom/pipe_gray.state"),
                                     "ODYSSEUS_RUN_DIR": str(self.root / "run"),
                                     "WANDB_DIR": str(self.root / "run/wandb")})
        env.start()
        self.addCleanup(env.stop)
        self.train, _ = generate_dataset(self.root / "smoke_pipe", train_size=16, test_size=8)
        self.config = config_for("pipe_cnn_critic_pos", ["smoke=pipe"])

    def test_preflight_and_resolved_snapshots(self):
        environment, provenance = validate_inputs(self.config)
        self.assertEqual(provenance["dataset_rows"], {"train_files": 16, "val_files": 8})
        metadata = save_run_metadata(self.config, environment, provenance)
        self.assertTrue((metadata / "provenance.json").is_file())
        training = OmegaConf.load(metadata / "training.yaml")
        self.assertEqual(training.actor_rollout_ref.rollout.multi_turn.game_config_path,
                         str(metadata / "environment.yaml"))
        self.assertNotIn("${", (metadata / "environment.yaml").read_text())
        self.assertNotIn("${", (metadata / "training.yaml").read_text())

    def test_missing_rom_and_dataset_fail_clearly(self):
        with patch.dict(os.environ, {"ODYSSEUS_ROM_PATH": ""}):
            with self.assertRaisesRegex(ValueError, "Set ODYSSEUS_ROM_PATH"):
                validate_inputs(self.config)
        self.train.unlink()
        with self.assertRaisesRegex(ValueError, "generate it using the README"):
            validate_inputs(self.config)

    def test_rejects_wrong_agent_and_prompt(self):
        rows = pq.read_table(self.train).to_pylist()
        for field, wrong in (("agent_name", "single_turn_agent"), ("agent_name", "markov_agent_sft"),
                             ("prompt", [{"role": "system", "content": "wrong"}])):
            with self.subTest(field=field):
                original = rows[0][field]
                rows[0][field] = wrong
                pq.write_table(pa.Table.from_pylist(rows), self.train)
                with self.assertRaisesRegex(ValueError, f"invalid {field}"):
                    validate_inputs(self.config)
                rows[0][field] = original

    def test_launcher_propagates_failure_through_tee(self):
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        python = bin_dir / "python"
        python.write_text("#!/bin/sh\necho simulated-training-failure\nexit 23\n")
        python.chmod(0o755)
        env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
        result = subprocess.run(["bash", str(VERL_ROOT / "jobs/run.sh"), "--config-name", "pipe_no_critic"],
                                env=env, capture_output=True, text=True, cwd=self.root)
        self.assertEqual(result.returncode, 23)
        self.assertIn("simulated-training-failure", (self.root / "run/train.log").read_text())


@unittest.skipUnless(os.environ.get("ODYSSEUS_TEST_ROM"), "Set ODYSSEUS_TEST_ROM to test actual emulator assets")
class EmulatorTests(unittest.TestCase):
    def test_pipe_and_all_multilevel_resets_and_steps(self):
        from verl.utils.game_envs.super_mario_land import PyBoySuperMarioLandEnv

        env_config = OmegaConf.load(VERL_ROOT / "jobs/config/5_levels_train_10_levels_eval_env_simple_gray_config.yaml")
        levels = [tuple(level) for level in env_config.config.validate_world_level_choices]
        cases = [(None, world, level) for world, level in levels]
        cases.append((str(REPO_ROOT / "rom/pipe_gray.state"), 1, 1))
        for state, world, level in cases:
            with self.subTest(state=state, world=world, level=level):
                env = PyBoySuperMarioLandEnv(rom_path=os.environ["ODYSSEUS_TEST_ROM"], init_state=state,
                                            world=world, level=level, headless=True)
                try:
                    env.reset(seed=0)
                    self.assertEqual(tuple(env.mario.world), (world, level))
                    if state:
                        self.assertEqual(env.mario.level_progress, 717)
                    observation, reward, done, truncated, info = env.step(["right"])
                    self.assertIsInstance(reward, (int, float))
                    self.assertIn("level_progress", info)
                    self.assertTrue(observation)
                    self.assertEqual(observation["state"], "")
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
