# Tests

## Maintained release suite

From the repository's `verl/` directory in the release environment:

```bash
python -m unittest discover -s tests/release -v
```

Set `ODYSSEUS_TEST_ROM` to the absolute path of an external Super Mario Land ROM to include emulator checks; otherwise that test is skipped. `tests/release/` covers the eight retained presets, smoke overlays, original-setting parity, canonical prompt and state hashes, datasets, preflight/provenance, shell failure propagation, canonical game-loop/environment dispatch, discounted returns, and game reward/validation metadata.

`test_game_loop.py` runs 15 deterministic scripted scenarios against a fixture captured before the tool/interaction cleanup at commit `40200aa`. It checks token IDs/masks/log-probabilities, images, actions, rewards, context and turn limits, termination, failure cleanup, weighted level selection, and validation artifacts. Model, tokenizer, processor, and environment boundaries are scripted, and tokenization runs inline; wall-clock metrics and validation timestamps are excluded from comparison. The fixture is a behavior baseline, not a model-quality result. Unsupported tool/interaction configs are also tested before initialization and after class reuse.

`test_mario_environment.py` covers progress rewards, life/death and step-limit termination, and rejection of retired options. With the reference ROM and pinned PyBoy version, it also compares 297 actual emulator action steps across the pipe state and all ten validation levels against traces captured before fixing the training options at commit `ccf617d`. These checks include screenshot hashes, frame timing, empty state text, rewards, metadata, resets, and the shared Python RNG sequence. The golden trace check is skipped for a different ROM revision; the basic emulator reset/step checks still run.

GPU training, built-in validation during training, distributed startup, checkpoint writing/resume, and actual FSDP export require separate execution with the necessary resources and model/checkpoint inputs. Follow the [release README](../../README.md) for the supported smoke commands.

## Inherited shared tests

Other test folders mirror the runtime namespaces, including trainer, model, worker, data/configuration, and checkpoint utilities. Retained coverage includes PPO/GRPO algorithms and metrics, protocol/data handling, RL datasets, FSDP workers/checkpoints, vLLM rollout, and Ray coordination. These tests support future runtime changes; they are not all validated in the release environment. Some require GPUs, multiple processes, optional backends, network access, or external model/data files. Run relevant tests deliberately rather than treating the entire directory as a CPU suite.

Mixed agent-loop and model/configuration suites still contain generic single-turn/tool examples or optional backend cases alongside checks of shared training machinery. Their presence does not make those examples or backends part of the maintained experiment scope; narrowing those suites requires a separate review.

| Directory or naming convention | Purpose |
| --- | --- |
| `release/` | Focused CPU/configuration/emulator release checks. |
| `special_sanity/` | Source, configuration, license, and layout checks. |
| `special_distributed/` | Tests requiring multiple GPUs/processes. |
| `special_standalone/` | Tests with dedicated environment requirements. |
| `*_on_cpu.py` | Inherited CPU-oriented tests; dependencies still vary by module. |

Tests and launchers tied to removed recipes/examples, the documentation site, upstream CI, and NPU setup have been retired. The unrelated SGLang/tool-resource suites, alternative Hugging Face rollout test, GSM8K interaction and GPT-OSS parser tests, inherited benchmark/SFT/generation/LoRA integration harnesses, SFT dataset tests, Megatron-only tests, and code-sandbox reward tests have also been removed. Use the release README's Mario smoke commands for training integration checks.

Vendored upstream CI workflows have also been removed. The retained sanity scripts check source/configuration documentation, not a local documentation website.

Run `python tests/special_sanity/validate_structure.py` to check the layout. See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidance.
