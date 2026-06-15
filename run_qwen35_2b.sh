#!/bin/bash

#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --job-name=simpletext-qwen35-2b
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p logs
  sbatch "$0" "$@"
  exit $?
fi

cd "${SLURM_SUBMIT_DIR}"

source .venv/bin/activate

export MODEL_NAME="Qwen/Qwen3.5-2B"
export MAX_NEW_TOKENS=256
export TEMPERATURE=0.7
export DATA_DIR=cochrane/data
export BATCH_SIZE=8
export RANDOM_SEED=42

python experiments/sentence_level/run_baseline.py --run_name "qwen35-2b-${SLURM_JOB_ID}"
