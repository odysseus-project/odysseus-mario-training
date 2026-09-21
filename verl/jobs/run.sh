#!/usr/bin/env bash
set -euo pipefail

odysseus_verl_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Normalize caller-supplied paths before changing the working directory.
for odysseus_key in ODYSSEUS_ROM_PATH ODYSSEUS_PIPE_STATE ODYSSEUS_DATA_ROOT ODYSSEUS_RUN_DIR; do
    if [[ -n ${!odysseus_key:-} ]]; then
        printf -v "$odysseus_key" '%s' "$(realpath -m -- "${!odysseus_key}")"
        export "$odysseus_key"
    fi
done
if [[ -n ${ODYSSEUS_MODEL_PATH:-} && -d $ODYSSEUS_MODEL_PATH ]]; then
    export ODYSSEUS_MODEL_PATH=$(realpath -- "$ODYSSEUS_MODEL_PATH")
fi
export ODYSSEUS_VERL_ROOT="$odysseus_verl_root"
export ODYSSEUS_RUN_NAME=${ODYSSEUS_RUN_NAME:-odysseus_$(date -u +%Y%m%dT%H%M%S)_${SLURM_JOB_ID:-local}_$$}
export ODYSSEUS_RUN_DIR=${ODYSSEUS_RUN_DIR:-$odysseus_verl_root/../outputs/$ODYSSEUS_RUN_NAME}
export PYTHONPATH="$odysseus_verl_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd -- "$odysseus_verl_root"

for odysseus_arg in "$@"; do
    case "$odysseus_arg" in
        --help|-h|--cfg|--cfg=*|--info|--info=*) exec python -m jobs.train "$@" ;;
    esac
done

mkdir -p -- "$ODYSSEUS_RUN_DIR"
python -m jobs.train "$@" 2>&1 | tee -a "$ODYSSEUS_RUN_DIR/train.log"
