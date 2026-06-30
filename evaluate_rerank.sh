#!/bin/bash
# Candidate generation + reranking on top of the frozen QLoRA adapter.
#
# Reference-free selectors (mbr, readability, sari_mbr) were REMOVED after all
# lost to greedy on val (see outputs/simpletext-task1-rerank-experiment.md).
# Remaining methods:
#   --rerank oracle    max-SARI ceiling (uses gold refs; NOT deployable)
#   --rerank learned   trained reranker via --reranker_path (see below)
#
# Learned-reranker workflow:
#   1) Build labeled candidate data (GPU; once per split):
#        python experiments/sentence_level/build_reranker_data.py \
#          --split train --adapter_path experiments/sentence_level/lora_adapter/qwen35-2b-best/checkpoint-328 \
#          --output cochrane/data/reranker_train.jsonl
#        python experiments/sentence_level/build_reranker_data.py \
#          --split val   --adapter_path experiments/sentence_level/lora_adapter/qwen35-2b-best/checkpoint-328 \
#          --output cochrane/data/reranker_val.jsonl
#   2) Train + tune the reranker (CPU; needs scikit-learn):
#        python experiments/sentence_level/train_reranker.py \
#          --train cochrane/data/reranker_train.jsonl --val cochrane/data/reranker_val.jsonl \
#          --out experiments/sentence_level/reranker.pkl
#   3) Evaluate the learned reranker on val, then once on test if it beats greedy (47.42):
#        ./evaluate_rerank.sh --split val --rerank learned \
#          --reranker_path experiments/sentence_level/reranker.pkl \
#          --run_name qwen35-2b-lora-learned-val
#
# Default below: oracle ceiling on test.

set -euo pipefail

cd "$(dirname "$0")"

export MODEL_NAME="Qwen/Qwen3.5-2B"
export DATA_DIR=cochrane/data

python experiments/sentence_level/run_baseline.py \
  --split test \
  --prompt default_zero_shot \
  --adapter_path experiments/sentence_level/lora_adapter/qwen35-2b-best/checkpoint-328 \
  --num_candidates 8 \
  --rerank oracle \
  --run_name qwen35-2b-lora-oracle-test \
  "$@"
