#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

readonly GPU_ID="${GPU_ID:-0}"
case "$GPU_ID" in
    0|3)
        echo "Refusing to submit SMP on reserved GPU $GPU_ID." >&2
        exit 2
        ;;
esac
readonly SHEETS=(
    "IN1K-NoShuffle1993"
    "IN1K-Shuffle1993"
    "IN1K-Shuffle2025"
    "IN1K-Shuffle42"
)

for sheet in "${SHEETS[@]}"; do
    output_dir="../logs/${sheet}"
    log_sheet="${sheet//1993/-1993}"
    log_sheet="${log_sheet//2025/-2025}"
    log_sheet="${log_sheet//42/-42}"
    output_file="${output_dir}/SMP-${log_sheet}-GPU${GPU_ID}.out"
    mkdir -p "$output_dir"
    echo "[$(date --iso-8601=seconds)] Starting $sheet on physical GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES="$GPU_ID" ./train_smp.sh "$sheet" > "$output_file" 2>&1
done
