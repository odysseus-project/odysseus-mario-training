#!/usr/bin/env bash
# Example resource request; add your site's account/partition/time/memory settings.
#SBATCH --job-name=train_1024_base
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
: "${ODYSSEUS_REPO_ROOT:?Set ODYSSEUS_REPO_ROOT to the absolute game-agent checkout path}"
# Activate the release environment before sbatch; it is inherited by all steps.
exec bash "$ODYSSEUS_REPO_ROOT/verl/jobs/slurm/run_multinode.sh" "$@"
