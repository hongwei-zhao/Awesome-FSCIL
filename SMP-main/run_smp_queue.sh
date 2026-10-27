#!/usr/bin/env bash

set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: $0 GPU_ID CONFIG [CONFIG ...]" >&2
    exit 2
fi

readonly GPU_ID="$1"
shift

case "$GPU_ID" in
    0|3)
        echo "Refusing to submit SMP on reserved GPU $GPU_ID." >&2
        exit 2
        ;;
esac

cd "$(dirname "$0")"
export FSCIL_DATA_ROOT="${FSCIL_DATA_ROOT:-/home/team/zhaohongwei/Dataset}"

for config in "$@"; do
    if [[ ! -f "$config" ]]; then
        echo "Missing configuration: $config" >&2
        exit 2
    fi

    echo "[$(date --iso-8601=seconds)] Starting ${config} on physical GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python -u main.py --config="$config"
done
