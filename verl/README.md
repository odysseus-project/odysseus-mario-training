# Odysseus training runtime

This directory contains the VERL-based reinforcement learning runtime for [Odysseus](https://odysseus-project.github.io/). Start with the [main README](../README.md) for installation, game assets, dataset preparation, and training commands.

The training stack uses FSDP actors, asynchronous vLLM rollouts, a PyBoy Super Mario Land environment, and an optional turn-level CNN critic. It supports pipe ablations, five-level training, built-in validation, checkpoint resume, and actor export.

| Path | Purpose |
| --- | --- |
| [`jobs/`](jobs/) | Training launchers, dataset generation, and Slurm examples. |
| [`jobs/config/`](jobs/config/) | Experiment presets and short-run overlays. |
| [`recipe/game_agent/dataset.py`](recipe/game_agent/dataset.py) | RL dataset adapter. |
| [`verl/experimental/agent_loop/game_agent_loop.py`](verl/experimental/agent_loop/game_agent_loop.py) | Multi-turn game interaction. |
| [`verl/utils/game_envs/super_mario_land/`](verl/utils/game_envs/super_mario_land/) | Game environment and prompt. |
| [`verl/model_merger/`](verl/model_merger/) | FSDP actor checkpoint export. |
| [`tests/release/`](tests/release/) | Focused configuration, dataset, rollout, and emulator checks. |

Use [`jobs/install.sh`](jobs/install.sh) with the pinned [requirements](requirements-odysseus.txt) and [constraints](constraints-odysseus.txt) to install the training environment. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [test guide](tests/README.md) for development checks.

The runtime derives from [Chengshuai-Shi/game-agent](https://github.com/Chengshuai-Shi/game-agent) and [VERL](https://github.com/verl-project/verl). Upstream licensing and attribution are provided in [LICENSE](LICENSE) and [Notice.txt](Notice.txt).
