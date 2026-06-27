#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=3
export MODEL_NAME="Qwen/Qwen3.5-2B"
export DATA_DIR=cochrane/data

python experiments/sentence_level/finetune.py \
  --model "$MODEL_NAME" \
  --prompt default_zero_shot \
  --rag \
  --glossary_path cochrane/data/MedSimplify.csv \
  --max_definitions 10 \
  --max_length 1024 \
  --num_epochs 3 \
  --patience 1 \
  --learning_rate 5e-5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --extra_data cochrane/data/external_rephrase_train.csv \
  --output "experiments/sentence_level/lora_adapter/qwen35-2b-zero-shot-rag" \
  "$@"
