#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p ../logs

readonly GPU_ID="${GPU_ID:-2}"
case "$GPU_ID" in
    0|3)
        echo "Refusing to submit SMP on reserved GPU $GPU_ID." >&2
        exit 2
        ;;
esac

nohup env GPU_ID="$GPU_ID" bash ./run_all_with_logs.sh \
    > "../logs/SMP-submit-gpu${GPU_ID}.out" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "../logs/SMP-submit-gpu${GPU_ID}.pid"
echo "SMP submitted on GPU $GPU_ID (PID: $pid)"
echo "Driver log: ../logs/SMP-submit-gpu${GPU_ID}.out"
