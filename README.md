# Text Simplification — CLEF 2026 SimpleText Task 1.1

Zero-shot sentence simplification on the Cochrane-auto corpus using **Qwen3.5** (2B or 4B).

## Current scope

- **Task**: 1.1 — sentence-level simplification
- **Data**: `cochraneauto_sents_rephrase_*.csv` only (rephrase operations; train / val / test)
- **Models**: `Qwen/Qwen3.5-2B` or `Qwen/Qwen3.5-4B`
- **Metric**: SARI (plus BLEU, BERTScore in `metrics.json`)

Paragraph, document, and multi-operation sentence splits are in `cochrane/data/` but not used yet.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env   # set MODEL_NAME to Qwen3.5-2B or Qwen3.5-4B
```

Key `.env` options:

```bash
MODEL_NAME=Qwen/Qwen3.5-4B
DATA_DIR=cochrane/data
BATCH_SIZE=8
LOAD_IN_4BIT=false
```

## Run evaluation

**Local:**

```bash
source .venv/bin/activate
python experiments/sentence_level/run_baseline.py --load_in_4bit
```

**SLURM** (results saved per job ID):

```bash
./run_qwen35_2b.sh   # V100, ~2B model
./run_qwen35_4b.sh   # H100, ~4B model
```

Outputs go to `experiments/sentence_level/results/<run_name>/` (e.g. `qwen35-4b-5678037`).

### Useful flags

| Flag | Default | Description |
|------|---------|-------------|
| `--run_name` | — | Subfolder under results; avoids overwriting prior runs |
| `--test_size` | all (688) | Limit test examples for quick runs |
| `--load_in_4bit` | off | 4-bit quantization on GPU |
| `--num_shots` | 0 | Few-shot examples from rephrase train set |
| `--batch_size` | 8 | Generation batch size |

```bash
# Quick dev run
python experiments/sentence_level/run_baseline.py \
  --test_size 100 \
  --run_name dev-test \
  --load_in_4bit
```

## Data

Under `cochrane/data/`:

| File | Split | Rows |
|------|-------|------|
| `cochraneauto_sents_rephrase_train.csv` | train | 5,384 |
| `cochraneauto_sents_rephrase_val.csv` | val |  | 773 |
| `cochraneauto_sents_rephrase_test.csv` | test | 667 |

Columns used: `complex` (input), `simple` (reference list), `label`, `pair_id`.

## Output

Each run writes to its results subfolder:

- `predictions.txt` — one simplification per line
- `input_output_pairs.txt` — input / output / references
- `metrics.json` — SARI, BLEU, BERTScore + run metadata

## Project layout

```
simple_text/
├── src/
│   ├── models/llama_simplifier.py
│   ├── evaluation/metrics.py
│   └── utils/data_loader.py
├── experiments/sentence_level/
│   ├── run_baseline.py
│   └── results/
├── cochrane/data/
│   └── cochraneauto_sents_rephrase_*.csv
├── run_qwen35_2b.sh
└── run_qwen35_4b.sh
```

## Troubleshooting

- **OOM**: `--load_in_4bit`, smaller model (`Qwen3.5-2B`), or `--batch_size 1`
- **Slow**: use GPU; `--test_size 100` for iteration
- **Import errors**: activate `.venv`, then run `python -m pip install -e .`
