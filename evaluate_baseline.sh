#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

export MODEL_NAME="Qwen/Qwen3.5-2B"
export DATA_DIR=cochrane/data

python experiments/sentence_level/run_baseline.py \
  --split test \
  --baseline identity \
  --run_name identity-copy-test \
  "$@"