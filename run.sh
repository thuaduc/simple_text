#!/bin/bash

#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --job-name=simpletext-baseline
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
set -a && source .env && set +a

python experiments/sentence_level/run_baseline.py --load_in_4bit
