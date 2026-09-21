# Developing the Odysseus training release

The maintained scope is the seven pipe ablations and batch-1024 multi-level training, including dataset preparation, built-in validation, checkpoint saving/resume, and FSDP export. Follow the [release README](../README.md) for the pinned environment and experiment commands.

## Local checks

From the repository's `verl/` directory in the release environment:

```bash
python -m pip check
python -m unittest discover -s tests/release -v
python tests/special_sanity/validate_structure.py
```

Set `ODYSSEUS_TEST_ROM` to an absolute path to your external Super Mario Land ROM before running the release tests to include emulator resets/steps for the pipe state and all ten validation levels. Without it, the emulator test is skipped. ROMs are not bundled.

The focused suite checks all eight presets and smoke overlays, the canonical prompt and state hashes, dataset generation, preflight, and launcher failure propagation. For changes to shared runtime code, run relevant inherited tests as well; see [tests/README.md](tests/README.md). The inherited suite includes accelerator and optional-backend tests and is not a single CPU-only test target.

From the repository root, check whitespace with `git diff --check`. The retained pre-commit configuration can be selected explicitly after installing `pre-commit`:

```bash
pre-commit run --config verl/.pre-commit-config.yaml --files <changed-files>
```

Ruff hooks may edit files. Local hooks inspect source docstrings/licenses and regenerate shared trainer reference configs; the configuration generator requires the release environment. Whole-tree lint/docstring compliance has not been established for the inherited runtime. Review any generated changes separately from a removal-only patch.

## Changes and review

Preserve the original-setting fixture in `tests/release/fixtures/` when cleaning up files. Changes to training settings, prompt placement, reward calculation, rollout termination, or checkpoint formats require an explicit rationale and appropriate validation. Remove dependent tests and obsolete documentation together with retired components; retain coverage for shared components that remain.

Describe the resulting behavior and checks performed in each change. State validation limits: CPU/emulator checks do not exercise GPU kernels, distributed Ray startup, training updates, checkpoint resume, or numerical reproduction. The release README provides GPU smoke commands for when resources are available.

Upstream CI workflows and the broader documentation site are not part of this snapshot. Preserve the [license](LICENSE), [notice](Notice.txt), and source copyright notices.
