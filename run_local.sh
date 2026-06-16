#!/bin/bash

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Create logs directory
mkdir -p logs

# Generate timestamp-based run name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="qwen35-2b-few-shot-rag-${TIMESTAMP}"

# Set environment variables
export MODEL_NAME="Qwen/Qwen3.5-2B"
export MAX_NEW_TOKENS=256
export TEMPERATURE=0.7
export DATA_DIR=cochrane/data
export BATCH_SIZE=8
export RANDOM_SEED=42

# Activate virtual environment
source .venv/bin/activate

echo "Starting run: ${RUN_NAME}"
echo "Using: few_shot prompt (3 examples) + RAG (MedSimplify glossary)"
echo "Logs will be saved to: logs/${RUN_NAME}.log"

# Run the experiment with best Week 1 prompt + Week 2 RAG
python experiments/sentence_level/run_baseline.py \
  --run_name "${RUN_NAME}" \
  --split test \
  --prompt few_shot \
  --num_shots 3 \
  --glossary_path cochrane/data/MedSimplify.csv \
  --max_definitions 10 \
  --load_in_4bit \
  "$@" \
  2>&1 | tee "logs/${RUN_NAME}.log"

echo "Run completed. Results saved to: experiments/sentence_level/results/${RUN_NAME}/"
