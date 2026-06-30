# Plan: One Concrete Improvement for SimpleText Task 1.1 (Sentence Simplification)

**Slug:** `simpletext-task1-improvement`
**Date:** 2026-06-30
**Mode:** Direct search (lead-owned). No researcher subagents — this is a narrow, repo-grounded "propose one concrete improvement" task, not a broad survey.

## Context (from CLAUDE.md + IMPROVEMENT_PLAN.md)

- Scope: Task 1.1 sentence-level, rephrase-only Cochrane-auto subset; test split N=667/688; primary metric SARI (BLEU, BERTScore secondary).
- Models: Qwen3.5 2B/4B instruct only (≤7B constraint).
- Current best system of record: **QLoRA `default_zero_shot` (`checkpoint-328`) = 47.38 SARI / 28.39 BLEU / 0.9248 BERTScore** (N=667, greedy, seed=42).
- Known artifact: **identity-copy = 47.88 SARI** (higher than QLoRA) — a SARI quirk on light-rephrase data; QLoRA still wins decisively on BLEU/BERTScore.
- Tried and parked: RAG (helps raw model only, +1.35 SARI; does not stack with adapter), external data (PLABA+Med-EASi, did NOT help: 46.80 vs 47.38), few-shot QLoRA (did not beat zero-shot).
- **Only open checklist item:** "Week 4: Add candidate generation only if time remains."
- Generator (`src/models/sentence_simplifier.py`) currently produces **one** output (greedy or sampling); no `num_return_sequences`, no beam search, no reranking, no MBR.

## Proposed concrete improvement (single bet)

**Candidate generation + reference-free reranking on top of the frozen QLoRA adapter.**

Rationale: the QLoRA model already produces genuinely simplified text but is evaluated with a single greedy decode. Sentence simplification literature (CLEF 2025 systems, MBR decoding for text simplification/generation) repeatedly shows that sampling N candidates and selecting with a quality/consensus criterion beats single greedy decoding, with no additional training. This directly attacks the "identity-copy ≥ QLoRA on SARI" gap by letting us select candidates that are both faithful and adequately edited — and it is the one remaining unfinished item in the existing plan, so it fits scope and timeline.

Two selection strategies to evaluate (both reference-free, both legal for a real submission):
1. **MBR / self-consensus**: generate K samples, pick the candidate with highest mean similarity (e.g., BERTScore-F1 or token-F1) to the other candidates — robust, no reference needed.
2. **Heuristic readability+fidelity reranker**: score each candidate by a combination of (a) source-fidelity (BERTScore/embedding sim to input) and (b) simplicity proxy (FKGL drop or length ratio band), reject empty/degenerate outputs. Mirrors the FKGL-based selection that helped MedSimplify and UM-FHS at CLEF 2025.

Tune K (e.g., 4, 8), temperature, and the selection rule on **val only**; run the winner once on **test**. Decision gate: keep only if it improves SARI on val without raising empty-output rate or harming BERTScore.

## Key questions

1. Does candidate-generation + reranking on the frozen QLoRA adapter beat 47.38 SARI on test (and ideally close the gap to the 47.88 identity artifact) while keeping BLEU/BERTScore?
2. Which selection rule wins on val: MBR self-consensus vs heuristic readability+fidelity?
3. What K / temperature gives the best val SARI without empty/degenerate outputs?
4. Is there a cheaper sub-bet (4B QLoRA, or beam search vs sampling) that should be tested first or instead?
5. What does recent literature (CLEF 2025 SimpleText, MBR for simplification/generation) say about expected gains and pitfalls?

## Evidence needed

- Repo: confirm generator lacks multi-candidate/rerank; confirm metric definitions (SARI in `sari_metric.py` / `src/evaluation/metrics.py`); confirm val split availability and size.
- Literature: CLEF 2025 SimpleText Task 1 overview (candidate selection / FKGL reranking), MBR decoding gains for text generation/simplification, whether sampling+rerank reliably beats greedy for small instruct LLMs.
- Quantify realistic expected SARI delta and risks (e.g., reranking can overfit to a proxy metric, identity-copy artifact may cap SARI gains).

## Scale decision

**Direct search, lead-owned.** 3–6 tool calls (repo reads already done + 3 distinct web/paper queries: CLEF 2025 candidate selection; MBR decoding for simplification; reranking/FKGL selection for LLM simplification). No subagents; self-cite; self-review. Verifier/reviewer subagents NOT used (direct mode).

## Task ledger

| # | Task | Owner | Status |
|---|------|-------|--------|
| T0 | Read CLAUDE.md + IMPROVEMENT_PLAN.md, inspect generator/metrics | lead | done |
| T1 | Write plan + ask confirmation | lead | done (awaiting approval) |
| T2 | 3 distinct searches: CLEF2025 selection / MBR simplification / rerank-FKGL | lead | pending |
| T3 | Write research notes → `outputs/.drafts/...-research-direct.md` | lead | pending |
| T4 | Draft → `...-draft.md` | lead | pending |
| T5 | Self-cite → `...-cited.md` | lead | pending |
| T6 | Self-review → `...-verification.md`; fix FATAL | lead | pending |
| T7 | Deliver `outputs/simpletext-task1-improvement.md` + provenance | lead | pending |

## Verification log

- 2026-06-30: Confirmed via grep that `sentence_simplifier.py` has no `num_return_sequences`/`num_beams`/reranking — single-output only. (supports the "candidate generation is unimplemented" claim)
- 2026-06-30: Confirmed `run_baseline.py` exposes `--sample`, `--temperature`, `--adapter_path`, `--prompt`, `--split` but no candidate/rerank flags.
- (pending) Literature deltas for MBR/rerank vs greedy.

## Decision log

- Chose candidate-generation+reranking as the single bet because: (a) it is the only unfinished checklist item, (b) requires no new training/data (low risk, fits poster timeline), (c) directly targets the SARI-vs-identity gap, (d) RAG/external-data/few-shot bets are already exhausted with negative/parked results.
- Deferred: 4B QLoRA (more compute, separate bet) — will note as alternative, not primary.
