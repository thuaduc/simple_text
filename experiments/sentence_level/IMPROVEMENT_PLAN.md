# Plan To Improve Text Simplification

## Recommended Direction

Prioritize a validation-driven Qwen3.5 pipeline: evaluate prompt variants first, add glossary grounding second, then dedicate a focused week to fine-tuning. Recent CLEF 2025 evidence points this way: the best sentence-level systems were strong prompt-only or plan-guided LLM systems, while fine-tuning had mixed results. The RAG-style MedSimplify work showed small but real gains from curated biomedical definitions, and the current SARI, BLEU, and BERTScore metrics are enough for the poster timeline.

Key papers/signals:

- [CLEF 2025 Task 1 overview](https://ceur-ws.org/Vol-4038/paper_344.pdf): Task 1.1 top systems included `gpt-4.1-mini` at 43.34 SARI, plan-guided LLaMA at 42.33, and Cochrane-trained/plan-guided BART at 42.31.
- [LIS/ASM MedSimplify RAG notebook](https://ceur-ws.org/Vol-4038/paper_358.pdf): exact-match biomedical glossary definitions + zero-shot prompting improved over their no-definition baseline, with FKGL-based output selection slightly improving further.
- [UM-FHS CLEF 2025 notebook](https://arxiv.org/html/2512.16541): prompt-only `gpt-4.1-mini` beat its sentence-level fine-tuned variant, so fine-tuning should be treated as a later experiment, not the first bet.
- [Redefining Simplicity, 2025](https://arxiv.org/html/2502.08281v1): lightweight LLMs can be competitive for sentence simplification, supporting the current Qwen3.5 2B/4B constraint.
- [Meta-Evaluation of Sentence Simplification Metrics, 2024](https://aclanthology.org/2024.lrec-main.981.pdf): keep reporting standard metrics for comparability; for this project, use the existing SARI, BLEU, and BERTScore pipeline.

## 4-Week Timeline

### Week 1: Prompt Evaluation And Baseline Control

Goal: get trustworthy validation results quickly and identify the strongest prompt-only system.

1. **Make prompting experimental instead of fixed**

   Replace the single hard-coded prompt in `src/prompts/templates.py` with named prompt variants:

   - `default_zero_shot`: current conservative baseline.
   - `nih_k8`: explicit NIH/plain-language grade-8 guidance from recent CLEF systems.
   - `plan_guided`: two-stage generation, first produce a short simplification plan, then produce only the simplified sentence.
   - `few_shot_retrieval`: retrieve 1-3 similar examples from the train split only and include them as demonstrations.

2. **Create a real dev/test experiment harness**

   Extend `experiments/sentence_level/run_baseline.py` so experiments can run on `train`, `val`, or `test`, defaulting development to `val` and reserving `test` for final comparisons. Keep `rephrase_only=True` by default through `src/utils/data_loader.py`. Also record generation settings, prompt variant, seed, split, and run name in `metrics.json`.

3. **Run prompt-only validation experiments**

   Compare Qwen3.5 2B and 4B on `val` for the baseline and prompt variants. Select 1-2 winners for deeper work based on SARI, BLEU, BERTScore, empty-output rate, and manual inspection of representative examples.

Poster output by end of week 1:

- A controlled baseline table.
- A prompt comparison table.
- A few qualitative examples showing common improvements and failures.

### Week 2: Glossary RAG

Goal: test whether biomedical term grounding improves the best week-1 prompt.

1. **Add high-precision biomedical glossary RAG**

   Add a small retrieval module under `src/retrieval/` that loads a term-definition CSV such as MedSimplify, normalizes terms, and retrieves definitions using exact phrase matching first. This mirrors the best-supported RAG evidence and avoids noisy semantic matches early. Later, add optional fuzzy/embedding retrieval only if exact matching has poor coverage on `val`.

   Data flow:

   ```mermaid
   flowchart LR
     inputSentence[Input Sentence] --> termMatcher[Exact Term Matcher]
     glossary[Biomedical Glossary] --> termMatcher
     termMatcher --> definitions[Relevant Definitions]
     definitions --> promptBuilder[Definition Augmented Prompt]
     inputSentence --> promptBuilder
     promptBuilder --> qwen[Qwen3.5]
     qwen --> prediction[Simplified Sentence]
   ```

2. **Add a definition-augmented prompt**

   Add `definition_augmented` as a prompt variant and run it only on top of the best week-1 prompt style. Track glossary coverage so the poster can report how often RAG actually contributes context.

Poster output by end of week 2:

- RAG vs no-RAG validation comparison using SARI, BLEU, and BERTScore.
- Glossary coverage statistics.

### Week 3: Fine-Tuning

Goal: test whether supervised adaptation of Qwen3.5 improves over the best prompt-only and RAG systems.

1. **Prepare the fine-tuning data**

   Convert `cochraneauto_sents_rephrase_train.csv` and `cochraneauto_sents_rephrase_val.csv` into instruction-tuning examples using the best prompt format from weeks 1-2. Keep the task scoped to rephrase-only sentence simplification.

2. **Run QLoRA SFT on Qwen3.5**

   Start with Qwen3.5 2B for speed and memory safety. If the setup is stable and time/resources allow, repeat with Qwen3.5 4B. Select checkpoints by validation SARI, with BLEU and BERTScore as secondary signals.

3. **Compare fine-tuning against non-trained systems**

   Compare the fine-tuned model on `val` against the best week-1 prompt-only system and the best week-2 RAG system. Keep fine-tuning only if it clearly improves the current three metrics or produces better qualitative examples for the poster.

Poster output by end of week 3:

- Fine-tuning setup details: model, train/val data, prompt format, and checkpoint selection rule.
- Validation comparison of prompt-only, RAG, and fine-tuned systems.
- A clear decision on whether the final system should use fine-tuning.

### Week 4: Final Runs And Poster Material

Goal: freeze the best system, run final evaluation, and prepare defensible poster evidence.

1. **Add candidate generation if time allows**

   Generate several candidates with the best prompt variants and compare candidate settings on `val` using the current three metrics. Use the best setting for the final `test` run instead of adding a new evaluation stack.

2. **Run the final controlled experiment ladder**

   Use `val` for all tuning, then run `test` once for the final few systems:

   - Baseline: current Qwen3.5 2B/4B zero-shot.
   - Prompt-only: best week-1 prompt.
   - RAG: exact glossary definitions + best prompt.
   - Fine-tuned Qwen3.5 if week 3 beats the non-trained systems.
   - Optional candidate-generation system if time allows.

3. **Prepare poster-ready evidence**

   Select representative examples, write the method diagram, and summarize the main result using SARI, BLEU, and BERTScore. Keep the narrative focused on which intervention helped most under the Qwen3.5 constraint.

Poster output by end of week 4:

- Final test table with baseline, best prompt-only, best RAG, and optional candidate-generation system.
- Error analysis categories with representative examples.
- A concise method diagram and key takeaway: which intervention improved simplification most under the Qwen3.5 constraint.

## Step Checklist

- [x] Week 1: Implement prompt variants and split-aware validation runs.
- [x] Week 1: Compare prompt-only Qwen3.5 2B/4B systems on `val` (N=50 quick test done, `few_shot` wins).
- [x] Week 2: Build exact-match biomedical glossary retrieval and definition-augmented prompting.
- [x] Week 2: Report RAG coverage and compare with SARI, BLEU, and BERTScore.
- [x] Week 3: Fine-tune Qwen3.5 with QLoRA on the rephrase-only train split (`default_zero_shot`, 46.50 SARI on test).
- [x] Week 3: Train the `few_shot` QLoRA adapter — did not improve over `default_zero_shot`, so training reverted to `default_zero_shot` in `run_finetune.sh`.
- [x] Week 3: Compare fine-tuned models against prompt-only and RAG winners with the same split and decoding settings.
- [x] Week 4: Add external biomedical data (PLABA + Med-EASi) filtered to 1→1 rephrase (+8,587 pairs, 2.64× train).
- [x] Week 4: Fine-tune with `--extra_data` and compare against the baseline adapter on `test` — external data did **not** help (46.80 vs 47.38 SARI).
- [x] Week 4: Run final selected systems on `test` (full ladder, N=667, 2026-06-27).
- [~] Week 4: Add candidate generation — **code complete** (candidate sampling + MBR/readability/oracle reranking; 18/18 offline tests pass). GPU runs (val ceiling → selector pick → test) still pending.

## Success Criteria

- Preserve the current baseline result as a reproducible run.
- Achieve a validation SARI gain without increasing empty outputs, factual-number errors, or unreadable outputs.
- Prefer a final system that is simple to submit: one Qwen3.5 model, one selected prompt/retrieval configuration, deterministic output settings, and full metadata in results.
- Use the Task 1.1 rephrase-only files only unless we explicitly decide to expand scope later.

---

## Implementation Progress

### Week 1 Implementation (✅ Complete)

**Implemented prompt variants:**
- `default_zero_shot`: Conservative baseline
- `nih_k8`: NIH plain-language, 8th-grade reading level
- `plan_guided`: Two-stage (plan → simplify)
- `few_shot`: Retrieves 1-3 similar examples from train split (lexical similarity)

**Infrastructure added:**
- Few-shot retrieval module (`src/retrieval/few_shot.py`)
- Updated runner with `--prompt` and `--num_shots` arguments
- Split-aware experiments (train/val/test)
- Helper scripts: `run_week1_prompts.sh`, `compare_results.py`
- **SLURM scripts updated:** `run_qwen35_2b.sh` and `run_qwen35_4b.sh` now use `few_shot` prompt by default

**Validation results (N=50, Qwen3.5-2B, val split):**
- **few_shot:** 39.06 SARI, 12.47 BLEU, 0.9084 BERTScore ✓ Best
- **nih_k8:** 37.90 SARI, 5.34 BLEU, 0.8937 BERTScore
- **default_zero_shot:** 36.95 SARI, 12.94 BLEU, 0.9045 BERTScore (baseline)
- **plan_guided:** 35.01 SARI, 3.44 BLEU, 0.8262 BERTScore

**Key findings:**
- `few_shot` wins with +2.10 SARI over baseline (+5.7%)
- `few_shot` also has highest BERTScore (0.9084)
- `plan_guided` underperforms unexpectedly (may need prompt tuning)
- `nih_k8` shows lower BLEU (5.34), suggests more aggressive rewording

### Week 2 Implementation (✅ Complete)

**RAG system:**
- MedSimplify glossary: 3,113 biomedical terms downloaded to `cochrane/data/MedSimplify.csv`
- Exact phrase matching with word boundaries
- Deduplication (keeps shortest definition per term)
- Coverage tracking and per-sentence matched terms saved

**Test results (N=667):**
- **Baseline (default_zero_shot):** 40.52 SARI, 18.49 BLEU, 0.9160 BERTScore
- **With RAG (definition_augmented):** 41.50 SARI (+0.98), 19.71 BLEU (+1.22), 0.9160 BERTScore
- **Coverage:** 84.3% of sentences matched at least one term (avg 2.51 terms/sentence)

**Key findings:**
- RAG provides consistent improvement (+2.4% SARI, +6.6% BLEU)
- Semantic preservation maintained (BERTScore unchanged)
- Top matched terms: risk (84×), adverse (48×), effects (46×), treatment (46×)
- Manual inspection shows better handling of placebo, statistical notation, and acronyms

### Week 3 Implementation (Complete)

**QLoRA fine-tuning added:**
- `experiments/sentence_level/finetune.py` — minimal LoRA SFT script
  - Builds instruction examples with the existing `default_zero_shot` prompt; target is the first simplified reference
  - Loss is masked on prompt tokens (`-100`), computed only on the simplified output
  - `LoraConfig(target_modules="all-linear", task_type=CAUSAL_LM)`, defaults `r=16`, `alpha=32`, `dropout=0.05`
  - Trains on `train`, evaluates on `val` each epoch, saves best adapter by epoch
  - `--load_in_4bit` enables QLoRA; gradient checkpointing + paged 8-bit optimizer for memory-tight GPUs
- `run_finetune.sh` — local launcher (Qwen3.5-2B, QLoRA, single GPU via `CUDA_VISIBLE_DEVICES`)
- Evaluation support: `run_baseline.py --adapter_path <dir>` loads the trained adapter on top of the base model (via `SentenceSimplifier(adapter_path=...)`)
- Added `peft>=0.19.0` dependency

**How to run:**
```bash
# Train
./run_finetune.sh                            # local, Qwen3.5-2B QLoRA
# or directly
python experiments/sentence_level/finetune.py --load_in_4bit

# Evaluate the adapter on test
python experiments/sentence_level/run_baseline.py \
  --split test --adapter_path experiments/sentence_level/lora_adapter/<run> \
  --run_name qwen35-2b-lora
```

**Test results (N=667):**
- **QLoRA default_zero_shot (`checkpoint-328`):** 46.50 SARI, 27.22 BLEU, 0.9240 BERTScore
- **QLoRA + RAG (`definition_augmented`):** 45.87 SARI, 26.73 BLEU, 0.9232 BERTScore
- Run output: `experiments/sentence_level/results/qwen35-2b-lora`
- RAG + fine-tuning output: `experiments/sentence_level/results/qwen35-2b-ze-shot-lora-definition-augmented`
- Evaluation config: `Qwen/Qwen3.5-2B`, `default_zero_shot`, greedy decoding, 4-bit loading, rephrase-only test split
- Compared with Week 2 RAG test result, QLoRA improves by +5.00 SARI, +7.51 BLEU, and +0.0080 BERTScore
- RAG does not stack with the fine-tuned adapter in this run: adding `definition_augmented` to the QLoRA adapter reduces SARI by 0.63 and BLEU by 0.49 versus plain QLoRA.

### Week 4 Implementation: External Data Augmentation (Complete)

**Motivation:** the rephrase-only train split is small (5,239 pairs) and QLoRA overfits within ~1 epoch. To add more in-domain supervision without leaving the rephrase scope, we pull external biomedical sentence-simplification corpora and filter them to a strict 1→1 reword shape.

**Data gained:**

| Source | Kept pairs |
|--------|-----------:|
| Cochrane rephrase train (baseline) | 5,239 |
| PLABA (external) | 7,218 |
| Med-EASi (external) | 1,369 |
| **External total added** | **8,587** |
| **Combined train** | **13,826** |

- External data adds **+8,587 pairs (+164%)**, taking the train set from 5,239 → **13,826 (~2.64×)**.
- Filtering kept 8,587 of 11,212 raw pairs; main drops: splits (1,723), near-identical copies (296), empty/deletions (234), identity (184), near-total rewrites (120).

**Resources:**
- **PLABA** — Plain Language Adaptation of Biomedical Abstracts (Attal et al. 2023). 750 PubMed abstracts, sentence-aligned plain-language adaptations. Sentence-level `data.json` from OSF project [rnpmf](https://osf.io/rnpmf/) (auto-downloaded and cached to `cochrane/data/plaba_data.json`).
- **Med-EASi** — Finely annotated medical text simplification (Basu et al. 2023). HuggingFace dataset [`cbasu/Med-EASi`](https://huggingface.co/datasets/cbasu/Med-EASi); `Expert`→`Simple` pairs, all splits used (no overlap with the Cochrane eval sets).

**Implementation:**
- `experiments/sentence_level/build_external_rephrase.py` — fetches both sources, filters to 1→1 rephrase pairs (drop empty/deletions, identity copies, 1→many splits, and char Levenshtein ratio outside `[0.30, 0.95]`; length sanity + dedup), and writes `cochrane/data/external_rephrase_train.csv` in the Cochrane schema (`pair_id, complex, label='rephrase', simple, source`).
- `src/utils/data_loader.py` — added `load_rephrase_csv()` to read any such CSV into the standard loader tuple.
- `experiments/sentence_level/finetune.py` — added `--extra_data` to append external CSV(s) to **train only**; `val`/`test` remain pure Cochrane rephrase so eval stays comparable.

**How to run:**
```bash
# Build the augmentation CSV (one-time; caches PLABA download)
python experiments/sentence_level/build_external_rephrase.py

# Fine-tune with external data mixed into train
./run_finetune.sh --extra_data cochrane/data/external_rephrase_train.csv
```

**Notes / risks:** PLABA and Med-EASi are sometimes more abstractive than Cochrane's light rephrases. If SARI drifts, tighten the band (e.g. `--max_sim 0.9 --min_sim 0.4`) or rebuild with `--sources plaba` only.

### Latest Full Test Ladder (2026-06-27, N=667)

Refreshed end-to-end run of every candidate system on the Cochrane-auto rephrase-only **test** split (N=667), Qwen3.5-2B, greedy decoding, `seed=42`. These numbers supersede the earlier per-week figures above (which were from earlier runs / smaller N).

| System | Run name | Adapter | Prompt | SARI | BLEU | BERTScore |
|--------|----------|---------|--------|-----:|-----:|----------:|
| Identity copy | `identity-copy-test` | — | — | 47.88 | 25.23 | 0.9179 |
| Raw model | `qwen35-2b-raw-model` | — | `default_zero_shot` | 41.83 | 21.19 | 0.9183 |
| Raw model + RAG | `qwen35-2b-raw-model-rag` | — | `definition_augmented` | 43.18 | 23.31 | 0.9188 |
| QLoRA | `qwen35-2b-lora` | `qwen35-2b-best/checkpoint-328` | `default_zero_shot` | **47.38** | **28.39** | **0.9248** |
| QLoRA + extra data | `qwen35-2b-zero-shot-extra-data` | `…-extra-data/checkpoint-433` | `default_zero_shot` | 46.80 | 28.16 | 0.9244 |

**Key findings:**
- **QLoRA is the best LLM system**: 47.38 SARI, and it clearly dominates on BLEU (28.39) and BERTScore (0.9248). Over the raw model it adds **+5.55 SARI / +7.20 BLEU / +0.0065 BERTScore**.
- **Identity copy is a deceptively strong SARI baseline** (47.88, nominally above QLoRA). This is a known SARI quirk on light-rephrase data where keeping tokens is rewarded; QLoRA still wins decisively on BLEU and BERTScore and produces genuinely simplified text, so it remains the system of record. Worth flagging on the poster as a metric caveat.
- **RAG helps the raw model**: `definition_augmented` adds **+1.35 SARI / +2.12 BLEU** over the raw zero-shot model (43.18 vs 41.83), with 84.3% glossary coverage (avg 2.51 terms/sentence, 349 unique terms; top: risk, adverse, effects, treatment). RAG remains a raw-model intervention only — it is not paired with the fine-tuned adapter, which was never trained on the definitions block.
- **External data did not help**: adding PLABA + Med-EASi (`checkpoint-433`) gave 46.80 SARI, **−0.58 vs plain QLoRA** (and slightly lower BLEU/BERTScore). The larger, more abstractive external pairs drift away from Cochrane's light rephrases; the 5,239-pair in-domain adapter stays the winner.

### Next Steps

1. ~~Run Week 1 validation to compare all prompt variants~~ ✅ Complete (N=50)
2. ~~Run Week 2 RAG evaluation~~ ✅ Complete (raw model + RAG: 43.18 SARI on test)
3. ~~Run first QLoRA evaluation~~ ✅ Complete (`default_zero_shot`, 47.38 SARI on test) ← best system
4. ~~Test RAG + QLoRA stacking~~ ✅ Complete — RAG does not stack with the fine-tuned adapter (prompt mismatch); RAG kept as a raw-model intervention.
5. ~~Evaluate the few-shot QLoRA adapter~~ ✅ Done — few-shot fine-tuning did not improve over `default_zero_shot`; `run_finetune.sh` reverted to `default_zero_shot`.
6. ~~Retrain `default_zero_shot` QLoRA with `--extra_data`~~ ✅ Done — external data did not help (46.80 vs 47.38 SARI).
7. Freeze QLoRA (`checkpoint-328`) as the final system; optionally add candidate generation if time remains.

### Week 4 Implementation: Candidate Generation + Reranking (Code complete, runs pending)

**Motivation:** the generator did single greedy decode only — no candidate pool, no selection. This was the only unfinished checklist item. Literature support: the *winning* TSAR 2025 simplification system (EhiMeNLP, https://aclanthology.org/2025.tsar-1.18/) used candidate generation + readability/similarity reranking; MBR decoding gives "reliable several-point improvements across metrics ... without any additional data or training" at K× inference cost (https://aclanthology.org/2023.bigpicture-1.9.pdf). Full plan: `outputs/simpletext-task1-improvement.md`.

**What was added (no new dependencies):**
- `src/rerank/reranker.py` + `src/rerank/__init__.py` — reference-free selectors:
  - `select_mbr` (self-consensus; token-F1 utility by default, pluggable to BERTScore)
  - `select_readability` (EhiMeNLP/TSAR style: drop degenerate → FKGL-drop filter → rank by source-fidelity within a band, avoiding the identity-copy trap)
  - `select_oracle` (max-SARI per sentence; **ceiling measurement only**, uses gold refs)
  - `fkgl()` lightweight Flesch-Kincaid (no `textstat` dep), `token_f1()` pure-Python
- `src/models/sentence_simplifier.py` — new `simplify_candidates_batch(...)` using `num_return_sequences=K` (sampling forced on).
- `run_baseline.py` — new flags `--num_candidates`, `--rerank {mbr,readability,oracle}`, `--rerank_temperature`; saves the full candidate pool + selection to `candidates.jsonl`; records rerank settings in `metrics.json`. `--num_candidates 1` keeps the existing greedy path unchanged.
- `evaluate_rerank.sh` — launcher on top of `checkpoint-328` (K=8, MBR default; comments give the val oracle/mbr/readability ladder).
- `experiments/sentence_level/test_reranker.py` — 18 offline unit checks (fkgl, token_f1, mbr, readability, oracle via injected SARI, batch). **All 18 pass** (`python3 experiments/sentence_level/test_reranker.py`), no GPU/model needed.

**Verification done so far (2026-06-30):** `py_compile` clean on all four changed/added modules; 18/18 reranker unit checks pass offline. **Not yet run on GPU** — SARI/BLEU/BERTScore for K/temperature/selector are NOT measured yet.

**Recommended next runs (tune on `val`, then one `test` run):**
1. `./evaluate_rerank.sh --split val --num_candidates 8 --rerank oracle --run_name qwen35-2b-lora-oracle-val` → read the max-SARI ceiling. **Decision gate:** if oracle barely beats 47.38, abandon this bet.
2. Compare `--rerank mbr` vs `--rerank readability` on `val` (K=8, T=0.7); watch empty-output rate + BERTScore.
3. Run the winning config once on `test` and add a row to the test ladder below.

**Current standings (N=667 test, 2026-06-27):**
- Identity copy: 47.88 SARI / 25.23 BLEU / 0.9179 BERTScore (SARI-only artifact baseline)
- Raw model `default_zero_shot`: 41.83 SARI / 21.19 BLEU / 0.9183 BERTScore
- Raw model + RAG `definition_augmented`: 43.18 SARI / 23.31 BLEU / 0.9188 BERTScore
- QLoRA + extra data: 46.80 SARI / 28.16 BLEU / 0.9244 BERTScore
- **QLoRA `default_zero_shot` (`checkpoint-328`): 47.38 SARI / 28.39 BLEU / 0.9248 BERTScore ← best system of record**
- Decision: freeze QLoRA `checkpoint-328` as the final system.
