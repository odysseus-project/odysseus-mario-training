"""Compose an Odysseus experiment, check its inputs, and start the existing trainer."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import hydra
from omegaconf import DictConfig, OmegaConf

VERL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERL_ROOT.parent


def configure_environment():
    """Defaults also apply when invoking this module without the shell launcher."""
    os.environ["ODYSSEUS_VERL_ROOT"] = str(VERL_ROOT)
    run_name = os.environ.setdefault(
        "ODYSSEUS_RUN_NAME", "odysseus_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    )
    for key, default in {
        "ODYSSEUS_DATA_ROOT": VERL_ROOT / "jobs/dataset",
        "ODYSSEUS_PIPE_STATE": REPO_ROOT / "rom/pipe_gray.state",
        "ODYSSEUS_RUN_DIR": REPO_ROOT / "outputs" / run_name,
    }.items():
        os.environ[key] = str(Path(os.environ.get(key, str(default))).expanduser().resolve())
    if os.environ.get("ODYSSEUS_ROM_PATH"):
        os.environ["ODYSSEUS_ROM_PATH"] = str(Path(os.environ["ODYSSEUS_ROM_PATH"]).expanduser().resolve())
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", str(Path(os.environ["ODYSSEUS_RUN_DIR"]) / "wandb"))
    os.environ.setdefault("VLLM_USE_V1", "1")


configure_environment()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(value, description):
    path = Path(value).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.R_OK):
        raise ValueError(f"{description} is missing or unreadable: {path}")
    return path


def validate_inputs(config):
    """CPU-only validation. Never start Ray or load model weights here."""
    if not os.environ.get("ODYSSEUS_ROM_PATH"):
        raise ValueError("Set ODYSSEUS_ROM_PATH to your Super Mario Land ROM before launching.")
    OmegaConf.resolve(config)
    expected_package = VERL_ROOT / "verl/__init__.py"
    spec = importlib.util.find_spec("verl")
    if spec is None or Path(spec.origin).resolve() != expected_package:
        raise ValueError(f"Import verl from {VERL_ROOT}; an installation from another checkout is taking precedence.")
    env_path = require_file(config.actor_rollout_ref.rollout.multi_turn.game_config_path, "Game configuration")
    environment = OmegaConf.load(env_path)
    OmegaConf.resolve(environment)
    rom = require_file(environment.config.rom_path, "ROM (ODYSSEUS_ROM_PATH)")
    environment.config.rom_path = str(rom)
    inputs = {str(env_path): sha256_file(env_path), str(rom): sha256_file(rom)}
    if environment.config.get("init_state"):
        state = require_file(environment.config.init_state, "Save state (ODYSSEUS_PIPE_STATE)")
        environment.config.init_state = str(state)
        inputs[str(state)] = sha256_file(state)

    # Check paths before importing the dataset/emulator stack, for useful setup errors.
    splits = {}
    for key in ("train_files", "val_files"):
        values = config.data[key]
        values = [values] if isinstance(values, str) else list(values)
        if not values:
            raise ValueError(f"data.{key} must contain at least one Parquet file")
        paths = [require_file(v, f"data.{key}; generate it using the README dataset commands") for v in values]
        config.data[key] = [str(path) for path in paths]
        splits[key] = paths

    import pyarrow.parquet as pq
    from verl.utils.game_envs.super_mario_land.env_prompts import SYSTEM_PROMPT

    expected = {
        "data_source": "super_mario_land",
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}],
        "ability": "game",
        "reward_model": {"style": "rule", "ground_truth": None},
        "agent_name": "game_agent",
    }
    row_counts = {}
    for key, paths in splits.items():
        count = 0
        for path in paths:
            table = pq.read_table(path)
            missing = (set(expected) | {"sample_index"}) - set(table.column_names)
            if missing:
                raise ValueError(f"{path}: missing dataset columns {sorted(missing)}")
            for index, row in enumerate(table.to_pylist()):
                for name, value in expected.items():
                    if row[name] != value:
                        raise ValueError(
                            f"{path}: row {index} has invalid {name}; regenerate with "
                            "python -m jobs.generate_dataset using the README arguments and --overwrite."
                        )
                if row["sample_index"] != index:
                    raise ValueError(f"{path}: sample_index must be sequential starting at zero")
            if not table.num_rows:
                raise ValueError(f"Empty dataset: {path}")
            count += table.num_rows
            inputs[str(path)] = sha256_file(path)
        row_counts[key] = count
    if row_counts["train_files"] < config.data.train_batch_size:
        raise ValueError("Training dataset is smaller than data.train_batch_size; the training loader would be empty.")

    model = str(config.actor_rollout_ref.model.path)
    if model.startswith(("/", ".", "~")) and not Path(model).expanduser().is_dir():
        raise ValueError(f"Model directory does not exist: {model}")
    return environment, {"input_sha256": inputs, "dataset_rows": row_counts,
                         "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}


def save_run_metadata(config, environment, provenance):
    run_dir = Path(config.trainer.default_local_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    metadata_dir = run_dir / "metadata" / stamp
    metadata_dir.mkdir(parents=True)
    env_path = metadata_dir / "environment.yaml"
    OmegaConf.save(environment, env_path, resolve=True)
    config.actor_rollout_ref.rollout.multi_turn.game_config_path = str(env_path)
    OmegaConf.save(config, metadata_dir / "training.yaml", resolve=True)
    version = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
    provenance.update({
        "source_revision": version.stdout.strip() if version.returncode == 0 else None,
        "source_dirty": bool(status.stdout.strip()),
        "source_status": status.stdout.splitlines(),
        "source_root": str(REPO_ROOT),
        "model_path": str(config.actor_rollout_ref.model.path),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
    })
    model_config = Path(str(config.actor_rollout_ref.model.path)).expanduser() / "config.json"
    if model_config.is_file():
        provenance["model_config_sha256"] = sha256_file(model_config)
        model_dir = model_config.parent.resolve()
        provenance["model_resolved_path"] = str(model_dir)
        if model_dir.parent.name == "snapshots":
            provenance["model_revision"] = model_dir.name
    (metadata_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    return metadata_dir


@hydra.main(config_path="config", config_name="pipe_cnn_critic_pos", version_base=None)
def main(config: DictConfig):
    environment, provenance = validate_inputs(config)
    if config.release.check_only:
        print(json.dumps({"status": "input checks passed", **provenance}, indent=2))
        return
    import torch

    if torch.cuda.device_count() < config.trainer.n_gpus_per_node:
        raise RuntimeError(f"This preset requires {config.trainer.n_gpus_per_node} visible GPUs per node; "
                           f"found {torch.cuda.device_count()}. Run inside your GPU allocation.")
    # Resolve a Hub ID once so every worker and the metadata use the same snapshot.
    # The trainer's copy_to_local() also uses snapshot_download for Hub IDs.
    model_source = str(config.actor_rollout_ref.model.path)
    local_model = Path(model_source).expanduser()
    if local_model.is_dir():
        model_path = local_model.resolve()
    else:
        from huggingface_hub import snapshot_download

        model_path = Path(snapshot_download(model_source, revision=os.environ.get("ODYSSEUS_MODEL_REVISION"))).resolve()
    config.actor_rollout_ref.model.path = str(model_path)
    provenance["model_source"] = model_source
    metadata_dir = save_run_metadata(config, environment, provenance)
    print(f"Run metadata: {metadata_dir}")
    from verl.trainer.main_ppo import run_ppo

    run_ppo(config)


if __name__ == "__main__":
    main()
