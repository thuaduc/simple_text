# Candidate Generation + Reranking for Cochrane Sentence Simplification

**Project:** CLEF 2026 SimpleText Task 1.1 — sentence-level, rephrase-only Cochrane-auto subset
**Model:** Qwen3.5-2B + QLoRA adapter (`checkpoint-328`), greedy = system of record
**Date:** 2026-07-01
**Status:** Experiment complete (negative for deployable selectors); supervised reranker proposed

---

## 1. Motivation

The QLoRA generator is the best system of record (47.38 SARI on test), but it is evaluated
with a single **greedy** decode — one input → one output. Two facts motivated trying a
selection stage:

1. The generator produced only one candidate; there was no candidate pool and no reranking.
2. The identity-copy baseline (47.88 SARI) nominally outscores QLoRA (47.38) — a known SARI
   artifact on light-rephrase data — suggesting a selector that picks faithful-but-edited
   candidates might help.

**Hypothesis:** sampling K candidates from the frozen adapter and selecting the best with a
reference-free rule beats single greedy decoding, with no new training or data.

Literature support: the *winning* TSAR 2025 simplification system (EhiMeNLP) used candidate
generation + readability/similarity reranking; MBR decoding gives "reliable several-point
improvements ... without any additional data or training" at K× inference cost. (See
`outputs/simpletext-task1-improvement.md` for the full cited brief.)

---

## 2. Method

### 2.1 Candidate generation
Reuse the frozen QLoRA adapter; turn on sampling and generate **K = 8** candidates per
sentence at **temperature 0.7** (`num_return_sequences=8`). No retraining.

### 2.2 Selectors evaluated
All selectors pick **one** of the 8 candidates. Only their picking rule differs.

| Selector | Picks the candidate that… | Sees gold ref? | Deployable |
|----------|---------------------------|:--------------:|:----------:|
| `mbr` | is most similar (token-F1) to the other 7 (consensus) | no | yes |
| `readability` | drops FKGL vs source, then best fidelity-to-source within a band | no | yes |
| `sari_mbr` | scores highest SARI using the other 7 candidates as pseudo-refs | no | yes |
| `oracle` | scores highest **true SARI vs the gold reference** | **yes** | **no (ceiling)** |

The **oracle** is a measuring stick, not a system: it cheats by reading the answer key to
pick the best-possible candidate, giving the upper bound of what perfect selection could
achieve on this pool. It cannot be submitted (no gold refs at test time).

### 2.3 Implementation
- `src/rerank/reranker.py` — `select_mbr`, `select_readability`, `select_sari_mbr`,
  `select_oracle`, plus `fkgl()` (dependency-free Flesch-Kincaid) and `token_f1()`.
- `src/models/sentence_simplifier.py` — `simplify_candidates_batch(...)` (K samples/sentence).
- `experiments/sentence_level/run_baseline.py` — flags `--num_candidates`, `--rerank
  {mbr,sari_mbr,readability,oracle}`, `--rerank_temperature`; saves the full pool +
  selection to `candidates.jsonl`. `--num_candidates 1` leaves the greedy path unchanged.
- `experiments/sentence_level/test_reranker.py` — 21 offline unit checks (all pass), no GPU.

---

## 3. Results

Validation split, **N = 758**, K = 8, T = 0.7, adapter `checkpoint-328`. 0% empty outputs
in every run.

| Selector | SARI | BLEU | BERTScore | vs greedy | Verdict |
|----------|-----:|-----:|----------:|----------:|---------|
| **Greedy (baseline)** | **47.42** | 27.88 | 0.9256 | — | system of record |
| sari_mbr | 46.33 | 27.40 | (skipped) | **−1.09** | best deployable; still loses |
| mbr (token-F1) | 45.39 | 25.77 | 0.9245 | −2.03 | loses |
| readability | 41.52 | 24.36 | 0.9191 | −5.90 | loses badly |
| **Oracle (max-SARI of 8)** | **55.62** | 30.26 | 0.9279 | **+8.20** | ceiling only (not deployable) |

Test-split reference points (N = 667, prior runs): greedy QLoRA `checkpoint-328` = 47.38
SARI / 28.39 BLEU / 0.9248 BERTScore; identity-copy = 47.88 SARI (artifact).

### Source for every number
`experiments/sentence_level/results/qwen35-2b-lora-{greedy,sarimbr,mbr,read,oracle}-val/metrics.json`
(read directly from disk). Candidate pools: `…/candidates.jsonl`.

---

## 4. Findings

1. **The bet fails the decision gate.** All three deployable selectors score *below* greedy
   on val. None is promoted to test. Greedy QLoRA stays the system of record.

2. **Metric-aligned selection helps, but not enough.** Ordering the deployable selectors by
   how well their objective matches SARI tracks their performance exactly: `sari_mbr`
   (−1.09) > `mbr` (−2.03) > `readability` (−5.90). Aligning the utility with SARI closed
   roughly half the MBR gap — directionally correct, still short.

3. **The bottleneck is selection, not generation.** The oracle is **+8.20** over greedy:
   somewhere in the 8 samples is a 55.62-quality answer the model already produced. The
   failure is that no reference-free rule can identify it.

4. **Why reference-free proxies cap out.** Candidates agree with *each other* more than with
   the gold simplification, so consensus/pseudo-reference scoring (mbr, sari_mbr) converges
   on the conservative middle and loses SARI's edit reward; FKGL+fidelity (readability)
   over-edits and loses SARI's keep reward. SARI-against-pseudo-references is too weak a
   proxy for SARI-against-the-true-reference.

---

## 5. Proposed solution: supervised / learned reranker

**Scope note:** this adds a *new trained component* (excluded by the original CLAUDE.md
scope). The QLoRA generator is **not** retrained — only a small scorer is learned on top of
frozen candidate pools. The key shift: train the scorer with **true SARI as the label**
(references used only at train time) while scoring candidates at inference from
**reference-free features** only — exactly the alignment the proxies lack.

### Target & decision gate
- **Pass:** learned reranker on val pools gives corpus SARI > **47.42** (greedy), margin
  surviving a seed re-run.
- **Stretch:** recover ≥25% of the 8.20 gap → ≈49.4 SARI (genuinely clears the 47.88
  identity-copy artifact on BLEU/BERTScore too).
- Promote to a single **test** run only if val passes; otherwise freeze greedy and report
  reranking as a negative result with the oracle-ceiling caveat.

### Phases
1. **Labeled data (main GPU cost):** generate K=8 candidates per **train** sentence
   (~5,239×8); reuse existing **val** pools; same adapter/T=0.7/fixed seed so feature
   distributions match. Label each candidate with **true SARI vs its gold reference**;
   per-group oracle-best index is the listwise target. → `cochrane/data/reranker_train.jsonl`.
2. **Reference-free features** (`src/rerank/features.py`, all computable at test time):
   token-F1 / Levenshtein / length ratio / edit counts / BERTScore-to-source;
   FKGL, complex-word %, **numeric-consistency vs source**; pool-relative consensus and the
   `sari_mbr` score; and the high-value **length-normalized model log-prob** (add an
   `output_scores` path to candidate generation).
3. **Train scorer** (`experiments/sentence_level/train_reranker.py`): LambdaMART / GBDT
   listwise ranker over sentence-groups (sklearn first to avoid a new dependency; LightGBM
   only if promising). **Tune by downstream val SARI of selected outputs**, not regression
   loss.
4. **Integrate & evaluate:** `--rerank learned --reranker_path <model>` in `run_baseline.py`;
   run the val ladder (greedy 47.42 → learned → oracle 55.62); one test run only if it passes.

### Leakage controls (critical)
- Train reranker on **train pools only**; tune on **val**; touch **test once**.
- Inference features never include gold references.
- Numeric-consistency feature guards against fluent-but-wrong candidates.

### Risks
- Only ~758 val groups for tuning → overfitting risk: small regularized feature set,
  train-internal CV, report a seed re-run.
- Learned rerankers typically recover only **30–60%** of an oracle gap — realistic outcome
  is roughly +2 to +4 SARI over greedy, **not** the full +8.2.
- New dependency (LightGBM) avoidable by starting with sklearn.
- ~1 focused week; train-pool generation dominates the cost.

---

## 6. Reproduce

```bash
# Deployable selectors on val
python experiments/sentence_level/run_baseline.py --split val \
  --prompt default_zero_shot \
  --adapter_path experiments/sentence_level/lora_adapter/qwen35-2b-best/checkpoint-328 \
  --num_candidates 8 --rerank sari_mbr --rerank_temperature 0.7 \
  --run_name qwen35-2b-lora-sarimbr-val
# swap --rerank {mbr,readability,oracle}; oracle additionally uses gold refs (ceiling)

# Greedy baseline on val
python experiments/sentence_level/run_baseline.py --split val \
  --prompt default_zero_shot \
  --adapter_path experiments/sentence_level/lora_adapter/qwen35-2b-best/checkpoint-328 \
  --run_name qwen35-2b-lora-greedy-val

# Offline reranker unit tests (no GPU)
python experiments/sentence_level/test_reranker.py
```

---

## 7. Status summary

- **Done:** candidate-generation + reranking implemented and unit-tested; 4 deployable +
  oracle runs measured on val.
- **Result:** no reference-free selector beats greedy; greedy `checkpoint-328` remains the
  system of record. Oracle ceiling +8.20 confirms large, unrealized headroom in the pool.
- **Next decision:** either (a) freeze greedy and write reranking as a negative result, or
  (b) build the supervised reranker in §5 to attempt to recover the gap.

*Detailed progress log lives in `experiments/sentence_level/IMPROVEMENT_PLAN.md`; the cited
literature brief is `outputs/simpletext-task1-improvement.md`.*
