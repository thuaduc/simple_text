# CLAUDE.md

Guidance for working in this repository.

## Project goal

[CLEF 2026 SimpleText Task 1 — Text Simplification](https://simpletext-project.com/2026/): simplify biomedical text from Cochrane systematic reviews.

## Current focus (narrow scope)

We are **only** working on **Task 1.1** (sentence-level) with the **rephrase-only** Cochrane-auto subset:

```
cochrane/data/cochraneauto_sents_rephrase_train.csv
cochrane/data/cochraneauto_sents_rephrase_val.csv
cochrane/data/cochraneauto_sents_rephrase_test.csv
```

- Default eval split: **test** (688 sentences)
- Default filter: `rephrase_only=True` in `load_cochrane_sentences()` — do not switch to `cochraneauto_sents_*.csv` or doc/para files unless explicitly asked
- `--all_labels` exists but is out of scope for now

## Models

Use **Qwen3.5** instruct models only for current experiments:

| Model | HuggingFace ID | Script |
|-------|----------------|--------|
| Qwen3.5 2B | `Qwen/Qwen3.5-2B` | `run_qwen35_2b.sh` |
| Qwen3.5 4B | `Qwen/Qwen3.5-4B` | `run_qwen35_4b.sh` |

Set `MODEL_NAME` in `.env` or export in the shell scripts. Both are ≤7B and fit the task constraint.

Default prompting: **zero-shot** (`--num_shots 0`). Use `--load_in_4bit` when GPU memory is tight.

## Running

**Local (no SLURM):**
```bash
./run_local.sh                     # default settings
./run_local.sh --load_in_4bit      # with quantization
./run_local.sh --test_size 100     # quick test
```

**Direct Python:**
```bash
source .venv/bin/activate
python experiments/sentence_level/run_baseline.py --load_in_4bit --run_name my-run
```

**SLURM:**
SLURM scripts pass `--run_name "qwen35-{2b|4b}-${SLURM_JOB_ID}"` so reruns do not overwrite `experiments/sentence_level/results/`.
Local script uses timestamp: `qwen35-2b-YYYYMMDD_HHMMSS`.

## Key files

- `experiments/sentence_level/run_baseline.py` — main eval pipeline
- `experiments/sentence_level/IMPROVEMENT_PLAN.md` — **main progress/results doc (update this, not random markdown files)**
- `src/utils/data_loader.py` — loads rephrase CSVs
- `src/models/llama_simplifier.py` — generation wrapper (model-agnostic despite the name)
- `src/config.py` — reads `.env` (`MODEL_NAME`, `DATA_DIR`, etc.)

## Documentation policy

**DO NOT create random documentation, summaries, or status markdown files.** Instead:

- **Update `experiments/sentence_level/IMPROVEMENT_PLAN.md`** with progress, results, and findings
- Add result sections at the end of IMPROVEMENT_PLAN.md for each week/experiment
- Keep implementation notes concise and focused
- Only create new docs if explicitly requested by the user

## Out of scope for now

- Document / paragraph simplification (`cochraneauto_docs_*.csv`, `cochraneauto_para_*.csv`)
- Full multi-operation sentence set (`cochraneauto_sents_*.csv` without rephrase filter)
- Fine-tuning (BART, T5, LoRA), Paper 358 MedSimplify prompts, hybrid MBR pipelines
- Models other than Qwen3.5 2B/4B unless the user changes direction

## Evaluation

Automatic metrics: SARI, BLEU, BERTScore (see `src/evaluation/metrics.py`). SARI is the primary score for comparison.
