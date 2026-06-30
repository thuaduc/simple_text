# A Concrete Plan to Improve SimpleText Task 1.1: Candidate Generation + Reranking

## Executive summary

The project has exhausted its main intervention bets — prompt variants, glossary RAG, external-data augmentation, and few-shot fine-tuning — and converged on a single best system of record: **QLoRA `checkpoint-328` at 47.38 SARI / 28.39 BLEU / 0.9248 BERTScore** (N=667 test). Two facts make the *next* bet obvious:

1. The generator (`src/models/sentence_simplifier.py`) currently produces exactly **one** output via greedy decoding — no candidate pool, no reranking. This is also the only unfinished item on the project's own checklist ("Week 4: add candidate generation if time remains").
2. The identity-copy baseline scores **47.88 SARI**, nominally above the genuine QLoRA system — a known SARI artifact on light-rephrase data that the project flags as a poster caveat.

**Proposed single improvement: add candidate generation + reference-free reranking on top of the frozen QLoRA adapter.** Sample K candidates from the existing model, then select the best with (A) MBR self-consensus and/or (B) a readability-filter-then-similarity reranker. This requires **no new training, data, or model** and is strongly supported by recent literature: the *winning* system of the TSAR 2025 simplification shared task was exactly candidate-generation + readability/similarity reranking, and MBR decoding reliably yields "several-point improvements across metrics ... without any additional data or training" at the cost of K× inference.

This is a low-risk, in-scope, poster-timeline-friendly bet that directly targets the SARI-vs-identity gap.

## Why this bet (and not the others)

| Already tried | Outcome (from IMPROVEMENT_PLAN.md) |
|---|---|
| Prompt variants (nih_k8, plan_guided, few_shot) | few_shot best raw-model prompt; superseded by fine-tuning |
| Glossary RAG | helps raw model (+1.35 SARI) but does NOT stack with the adapter |
| External data (PLABA+Med-EASi) | did NOT help (46.80 vs 47.38) |
| Few-shot QLoRA | did NOT beat zero-shot QLoRA |
| **Candidate generation** | **NOT YET DONE — the only open checklist item** |

Every remaining lever except candidate generation has been pulled. Candidate generation is also the cheapest remaining lever: it reuses the frozen adapter and only changes inference.

## What the evidence says

### Candidate generation + reranking is a winning recipe in simplification
- **EhiMeNLP won the TSAR 2025 shared task** with a two-step strategy: candidate generation (multiple LLMs + prompts), then reranking by **readability-based filtering** + **semantic similarity to the original** (Miyata et al., 2025). This is essentially the proposed Option B.
- A second strong TSAR 2025 system used **CEFR (readability) filtering first**, then judge-based candidate selection. Best practice across both: **filter by readability, then rank by similarity/quality among survivors.**
- In the medical domain specifically, a **reranked beam search by readability** improved Flesch-Kincaid by up to **2.43 points** and human judgments **while maintaining meaning preservation** (arXiv:2310.11191).

### MBR self-consensus gives reliable gains with no training
- MBR decoding "provides reliable several-point improvements across metrics for a wide variety of tasks without any additional data or training," at an additional inference cost (ACL 2023 BigPicture).
- Pure sampling without selection lowers quality; MBR over a sampled pool resolves the diversity-vs-quality tradeoff (Findings ACL 2023). MBR has been demonstrated on a **text simplification task** with multi-prompt candidate banks (EMNLP 2024) and as test-time compute for instruction-following LLMs (ICLR 2025).

## The proposed method (concrete)

### Stage 1 — Candidate generation (no new training)
- Reuse the frozen QLoRA adapter (`checkpoint-328`).
- Enable sampling (`--sample`) and generate **K candidates** per sentence (e.g., K ∈ {4, 8}) at a moderate temperature (e.g., 0.7).
- Optional extension supported by the multi-prompt MBR result: draw candidates from several existing prompt variants (`default_zero_shot`, `nih_k8`, `few_shot`) to increase diversity. Keep this as a secondary experiment.

### Stage 2 — Reference-free selection (two variants to compare on `val`)
**Option A — MBR self-consensus.** Score each candidate by mean similarity (BERTScore-F1 or token-F1) to the other K−1 candidates; pick the highest. No reference needed.

**Option B — Readability-filter-then-similarity (EhiMeNLP/TSAR style).**
1. Hard-reject empty/degenerate/length-outlier candidates.
2. **Readability filter:** keep candidates whose FKGL drops vs the input (or falls in a target band).
3. **Rank survivors by semantic similarity to the source** (embedding/BERTScore) to preserve clinical fidelity; pick the top one.

Tune K, temperature, and the selection rule on **`val` only** (where gold refs exist, measured by real SARI/BLEU/BERTScore + empty-output rate). Run the single winning configuration **once** on `test`.

### Decision gate
Keep the reranking system only if, on `val`, it improves SARI over greedy QLoRA **without** raising empty-output rate or lowering BERTScore. If only BLEU/BERTScore improve while SARI is flat, report it honestly as a fidelity/fluency gain (still poster-worthy given the SARI artifact discussion).

## Minimal implementation sketch
- `src/models/sentence_simplifier.py`: add a `simplify_candidates(..., k, temperature)` path using `num_return_sequences=k` (or k independent sampled calls) returning a list per input.
- `src/retrieval/` or new `src/rerank/`: a small reranker module implementing Option A and Option B (FKGL via `textstat`, similarity via the existing BERTScore utility).
- `run_baseline.py`: add `--num_candidates`, `--rerank {mbr,readability}` flags; record them in `metrics.json`.
- Oracle/sanity checks before trusting results: (1) with k=1 the pipeline must reproduce the current greedy numbers; (2) an *oracle* reranker that picks the max-SARI candidate per sentence gives the upper bound — if the oracle barely beats greedy, the bet is capped and should be abandoned early.

## Expected outcome, honestly
- Literature supports "several-point" metric gains for MBR generally and readability rerankers improving FKGL without hurting meaning. A realistic target is a modest SARI gain plus clearer BLEU/BERTScore gains over greedy QLoRA.
- It is **not** guaranteed to beat the 47.88 identity-copy SARI artifact, because SARI rewards token retention on light-rephrase data. The honest framing: this produces a genuinely simplified output that is selected to be both faithful and adequately edited, and wins on BLEU/BERTScore — consistent with the project's existing "system of record" argument.

## Open questions
- How large is the oracle (max-SARI) ceiling on `val`? This bounds the achievable gain and should be measured first.
- Does multi-prompt candidate sourcing beat single-prompt sampling for this adapter?
- Does Option A or Option B win, and do they combine (readability filter → MBR among survivors)?
- Side-bets worth a single comparison run: 4B-QLoRA, and beam search vs sampling for candidate generation.

## Caveats and disagreements
- MBR/rerank gains cost **K× inference** (all MBR sources agree). For 667 test sentences this is cheap, but worth noting.
- Selecting on a proxy metric (FKGL/BERTScore) can game that proxy rather than true quality; mitigations: validate on held-out `val` with the full metric suite and manual inspection.
- Per-metric deltas for the EhiMeNLP winner were not extracted (PDF full-text parsing intentionally skipped); the claim rests on the abstract and shared-task ranking, not a reproduced number.
