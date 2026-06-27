#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

export MODEL_NAME="Qwen/Qwen3.5-2B"
export DATA_DIR=cochrane/data

python experiments/sentence_level/run_baseline.py \
  --split test \
  --prompt default_zero_shot \
  --rag \
  --rag_mode after \
  --run_name qwen35-2b-raw-model-rag-postedit \
  "$@"
