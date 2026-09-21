#!/usr/bin/env bash
# Example resource request; add your site's account/partition/time/memory settings.
#SBATCH --job-name=pipe_grpo_traj_outcome_pos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
: "${ODYSSEUS_REPO_ROOT:?Set ODYSSEUS_REPO_ROOT to the absolute game-agent checkout path}"
# Activate the release environment before sbatch; it is inherited by this job.
export ODYSSEUS_RUN_NAME=${ODYSSEUS_RUN_NAME:-pipe_grpo_traj_outcome_pos_${SLURM_JOB_ID}}
exec bash "$ODYSSEUS_REPO_ROOT/verl/jobs/run.sh" --config-name pipe_grpo_traj_outcome_pos "$@"
