#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

export MODEL_NAME="Qwen/Qwen3.5-2B"
export DATA_DIR=cochrane/data

python experiments/sentence_level/run_baseline.py \
  --split test \
  --prompt default_zero_shot \
  --load_in_4bit \
  --adapter_path experiments/sentence_level/lora_adapter/qwen35-2b/checkpoint-328 \
  --run_name qwen35-2b-lora \
  "$@"