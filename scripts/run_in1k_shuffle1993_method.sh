#!/usr/bin/env bash
set -euo pipefail

method=${1:?method is required}
physical_gpu=${2:?physical GPU is required}
start_dataset=${3:-cub}

case "$physical_gpu" in
  1|2|4|5|6|7) ;;
  *)
    echo "Refusing disallowed physical GPU: $physical_gpu" >&2
    exit 2
    ;;
esac

case "$start_dataset" in
  cub) start_index=0 ;;
  cifar) start_index=1 ;;
  imagenetr) start_index=2 ;;
  mini) start_index=3 ;;
  *)
    echo "Unknown start dataset: $start_dataset" >&2
    exit 2
    ;;
esac

repo_root=/home/team/zhaohongwei/Code/Research/Awesome-FSCIL
source /home/team/zhaohongwei/anaconda3/etc/profile.d/conda.sh
conda activate peft
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export PYTHONUNBUFFERED=1

echo "METHOD=$method"
echo "PHYSICAL_GPU=$physical_gpu"
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
nvidia-smi -i "$physical_gpu" --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

case "$method" in
  finetune|simplecil|l2p|coda_prompt|lae)
    cd "$repo_root"
    config_root="configs/IN1K-Shuffle1993/$method"
    case "$method" in
      finetune) prefix=finetune ;;
      simplecil) prefix=simplecil ;;
      l2p) prefix=l2p ;;
      coda_prompt) prefix=coda_prompt ;;
      lae) prefix=lae ;;
    esac
    configs=(
      "$config_root/${prefix}_cub_B100_Inc10.json"
      "$config_root/${prefix}_cifar_B60_Inc5.json"
      "$config_root/${prefix}_inr_B100_Inc10.json"
      "$config_root/${prefix}_mini_B60_Inc5.json"
    )
    ;;
  inflora)
    cd "$repo_root"
    configs=(
      configs/inflora/IN1K-Shuffle1993/inflora_cub_B100_Inc10.json
      configs/inflora/IN1K-Shuffle1993/inflora_cifar_B60_Inc5.json
      configs/inflora/IN1K-Shuffle1993/inflora_inr_B100_Inc10.json
      configs/inflora/IN1K-Shuffle1993/inflora_mini_B60_Inc5.json
    )
    ;;
  sdlora)
    cd "$repo_root"
    configs=(
      configs/sdlora/IN1K-Shuffle1993/sdlora_cub_B100_Inc10.json
      configs/sdlora/IN1K-Shuffle1993/sdlora_cifar_B60_Inc5.json
      configs/sdlora/IN1K-Shuffle1993/sdlora_inr_B100_Inc10.json
      configs/sdlora/IN1K-Shuffle1993/sdlora_mini_B60_Inc5.json
    )
    ;;
  *)
    echo "Unknown method: $method" >&2
    exit 2
    ;;
esac

configs=("${configs[@]:$start_index}")

for config in "${configs[@]}"; do
  echo "RUNNING_CONFIG=$config"
  python main.py --config="$config"
done

echo "RERUN_COMPLETE"
