#!/bin/bash
# Candidate generation + reranking on top of the frozen QLoRA adapter.
# Implements outputs/simpletext-task1-improvement.md.
#
# Recommended workflow (tune on val, run once on test):
#   1) Measure the achievable ceiling on val:
#        ./evaluate_rerank.sh --split val --num_candidates 8 --rerank oracle \
#          --run_name qwen35-2b-lora-oracle-val
#   2) Compare deployable selectors on val:
#        ./evaluate_rerank.sh --split val --num_candidates 8 --rerank mbr \
#          --run_name qwen35-2b-lora-mbr-val
#        ./evaluate_rerank.sh --split val --num_candidates 8 --rerank readability \
#          --run_name qwen35-2b-lora-read-val
#   3) Run the winning config once on test (override --split/--run_name via "$@").
#
# Defaults below: test split, K=8, MBR self-consensus.

set -euo pipefail

cd "$(dirname "$0")"

export MODEL_NAME="Qwen/Qwen3.5-2B"
export DATA_DIR=cochrane/data

python experiments/sentence_level/run_baseline.py \
  --split test \
  --prompt default_zero_shot \
  --adapter_path experiments/sentence_level/lora_adapter/qwen35-2b-best/checkpoint-328 \
  --num_candidates 8 \
  --rerank mbr \
  --run_name qwen35-2b-lora-rerank \
  "$@"
