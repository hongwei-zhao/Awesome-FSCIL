#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/team/zhaohongwei/Code/Research/Awesome-FSCIL
log_file="$repo_root/logs/IN1K-Shuffle1993/SD-LoRA-IN1K-Shuffle-1993-RERUN-GPU7.out"
source_session=fscil_s1993_sdlora_resume
target_session=fscil_s1993_sdlora_gpu4

while [ "$(grep -a -c '^Finished ' "$log_file" || true)" -lt 2 ]; do
  sleep 1
done

tmux kill-session -t "$source_session" 2>/dev/null || true
nvidia-smi -i 4 --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
tmux new-session -d -s "$target_session" "bash $repo_root/scripts/run_in1k_shuffle1993_method.sh sdlora 4 imagenetr >> $log_file 2>&1"
echo "SD-LoRA ImageNet-R and miniImageNet submitted to physical GPU 4."
