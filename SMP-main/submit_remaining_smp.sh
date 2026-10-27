#!/usr/bin/env bash

# Submit the eight incomplete SMP experiments as independent single-GPU runs.
# Usage: activate the peft environment, then run
#   ./submit_remaining_smp.sh 2 4
# or provide four permitted GPU IDs for four parallel queues.

set -euo pipefail

cd "$(dirname "$0")"

if (( $# == 0 )); then
    GPU_IDS=(2 4)
else
    GPU_IDS=("$@")
fi

if (( ${#GPU_IDS[@]} != 2 && ${#GPU_IDS[@]} != 4 )); then
    echo "Provide exactly two or four GPU IDs." >&2
    exit 2
fi

for gpu in "${GPU_IDS[@]}"; do
    case "$gpu" in
        0|3)
            echo "GPU $gpu is reserved and must not receive an SMP job." >&2
            exit 2
            ;;
    esac
done

if [[ "${CONDA_DEFAULT_ENV:-}" != "peft" ]]; then
    echo "Activate the peft conda environment before submitting." >&2
    exit 2
fi

export FSCIL_DATA_ROOT="${FSCIL_DATA_ROOT:-/home/team/zhaohongwei/Dataset}"
readonly RUNNER="./run_smp_queue.sh"

submit_queue() {
    local gpu="$1"
    local name="$2"
    shift 2
    local log_dir="../logs/${name}"
    local log_name="${name//1993/-1993}"
    log_name="${log_name//2025/-2025}"
    log_name="${log_name//42/-42}"
    mkdir -p "$log_dir"

    nohup setsid "$RUNNER" "$gpu" "$@" \
        > "${log_dir}/SMP-${log_name}-GPU${gpu}.out" 2>&1 < /dev/null &
    local pid=$!
    printf '%s\n' "$pid" > "../logs/SMP-${name}-GPU${gpu}.pid"
    echo "Submitted ${name} queue on GPU ${gpu} (PID: ${pid})"
}

readonly CUB="smp_cub_B100_Inc10.json"
readonly CIFAR="smp_cifar_B60_Inc5.json"
readonly INR="smp_inr_B100_Inc10.json"
readonly MINI="smp_mini_B60_Inc5.json"

if (( ${#GPU_IDS[@]} == 2 )); then
    submit_queue "${GPU_IDS[0]}" "IN1K-Shuffle2025" \
        "configs/IN1K-Shuffle2025/${CUB}" \
        "configs/IN1K-Shuffle2025/${CIFAR}" \
        "configs/IN1K-Shuffle2025/${INR}" \
        "configs/IN1K-Shuffle2025/${MINI}"
    submit_queue "${GPU_IDS[1]}" "IN1K-Shuffle42" \
        "configs/IN1K-Shuffle42/${CUB}" \
        "configs/IN1K-Shuffle42/${CIFAR}" \
        "configs/IN1K-Shuffle42/${INR}" \
        "configs/IN1K-Shuffle42/${MINI}"
else
    submit_queue "${GPU_IDS[0]}" "IN1K-Shuffle2025" \
        "configs/IN1K-Shuffle2025/${MINI}" \
        "configs/IN1K-Shuffle2025/${CUB}"
    submit_queue "${GPU_IDS[1]}" "IN1K-Shuffle2025" \
        "configs/IN1K-Shuffle2025/${CIFAR}" \
        "configs/IN1K-Shuffle2025/${INR}"
    submit_queue "${GPU_IDS[2]}" "IN1K-Shuffle42" \
        "configs/IN1K-Shuffle42/${MINI}" \
        "configs/IN1K-Shuffle42/${CUB}"
    submit_queue "${GPU_IDS[3]}" "IN1K-Shuffle42" \
        "configs/IN1K-Shuffle42/${CIFAR}" \
        "configs/IN1K-Shuffle42/${INR}"
fi
