#!/usr/bin/env bash
# Example bootstrap for a shared filesystem and an already activated environment.
set -euo pipefail
: "${ODYSSEUS_REPO_ROOT:?Set ODYSSEUS_REPO_ROOT to the absolute game-agent checkout path}"
: "${SLURM_JOB_NODELIST:?Run inside a two-node Slurm allocation}"
export ODYSSEUS_REPO_ROOT=$(realpath -- "$ODYSSEUS_REPO_ROOT")
export ODYSSEUS_RUN_NAME=${ODYSSEUS_RUN_NAME:-train_1024_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%S)}
export ODYSSEUS_RUN_DIR=${ODYSSEUS_RUN_DIR:-$ODYSSEUS_REPO_ROOT/outputs/$ODYSSEUS_RUN_NAME}
export PYTHONPATH="$ODYSSEUS_REPO_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd -- "$ODYSSEUS_REPO_ROOT/verl"
mkdir -p -- "$ODYSSEUS_RUN_DIR"
mapfile -t odysseus_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${#odysseus_nodes[@]} != 2 ]]; then
    echo "train_1024_base requires exactly two allocated nodes" >&2
    exit 1
fi
odysseus_head=${odysseus_nodes[0]}
odysseus_steps=()
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if ((${#odysseus_steps[@]})); then
        kill "${odysseus_steps[@]}" 2>/dev/null || true
        wait "${odysseus_steps[@]}" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
step() {
    local node=$1
    shift
    srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=8 --kill-on-bad-exit=1 --nodelist="$node" "$@"
}
check_steps() {
    local pid
    for pid in "${odysseus_steps[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "A Ray Slurm step exited; inspect ray-*.log in $ODYSSEUS_RUN_DIR" >&2
            exit 1
        fi
    done
}

# Validate assets/datasets before starting Ray. All nodes use the same checkout.
bash jobs/run.sh --config-name train_1024_base "$@" release.check_only=true
for odysseus_node in "${odysseus_nodes[@]}"; do
    step "$odysseus_node" python -m jobs.cluster probe --gpus 8 \
        > "$ODYSSEUS_RUN_DIR/node-$odysseus_node.json"
done
# A suffix such as -ib0 is optional and site-specific; no network interface is assumed.
odysseus_head_ip=$(step "$odysseus_head" python -m jobs.cluster address \
    "${odysseus_head}${ODYSSEUS_RAY_HOST_SUFFIX:-}")
export RAY_ADDRESS="$odysseus_head_ip:${ODYSSEUS_RAY_PORT:-6379}"
srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=8 --kill-on-bad-exit=1 --nodelist="$odysseus_head" ray start --head --node-ip-address="$odysseus_head_ip" \
    --port="${ODYSSEUS_RAY_PORT:-6379}" --num-cpus="${SLURM_CPUS_PER_TASK:-32}" --num-gpus=8 \
    --disable-usage-stats --block > "$ODYSSEUS_RUN_DIR/ray-head.log" 2>&1 &
odysseus_steps+=("$!")
step "$odysseus_head" timeout 150s python -m jobs.cluster wait --nodes 1 --gpus 8 --timeout 120
check_steps
for odysseus_node in "${odysseus_nodes[@]:1}"; do
    odysseus_node_ip=$(step "$odysseus_node" python -m jobs.cluster address \
        "${odysseus_node}${ODYSSEUS_RAY_HOST_SUFFIX:-}")
    srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=8 --kill-on-bad-exit=1 --nodelist="$odysseus_node" ray start --address="$RAY_ADDRESS" --node-ip-address="$odysseus_node_ip" \
        --num-cpus="${SLURM_CPUS_PER_TASK:-32}" --num-gpus=8 --disable-usage-stats --block \
        > "$ODYSSEUS_RUN_DIR/ray-$odysseus_node.log" 2>&1 &
    odysseus_steps+=("$!")
done
step "$odysseus_head" timeout 270s python -m jobs.cluster wait --nodes 2 --gpus 8 --timeout 240
check_steps
step "$odysseus_head" bash jobs/run.sh --config-name train_1024_base "$@"
