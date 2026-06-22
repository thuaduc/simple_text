#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=3
export MODEL_NAME="Qwen/Qwen3.5-2B"
export DATA_DIR=cochrane/data

python experiments/sentence_level/finetune.py \
  --load_in_4bit \
  --epochs 3 \
  --early_stopping_patience 1 \
  --lr 1e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --output_dir "experiments/sentence_level/lora_adapter/qwen35-2b" \
  "$@"
