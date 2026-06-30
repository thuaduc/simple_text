# Research Notes (direct mode) — Candidate Generation + Reranking

## Exact search queries used
1. `CLEF 2025 SimpleText Task 1 candidate selection FKGL reranking sentence simplification systems overview`
2. `Minimum Bayes Risk MBR decoding text simplification candidate reranking gains over greedy`
3. `sampling rerank candidate generation small instruct LLM sentence simplification readability fidelity selection`

## Repo facts (verified by grep/read, not external)
- `src/models/sentence_simplifier.py`: single-output generation only; no `num_return_sequences`, no `num_beams`, no reranking. Has `do_sample`, `temperature`, `max_new_tokens`. (grep, 2026-06-30)
- `run_baseline.py`: flags include `--split`, `--prompt`, `--sample`, `--temperature`, `--adapter_path`; no candidate/rerank flags.
- Current best of record: QLoRA `checkpoint-328` = 47.38 SARI / 28.39 BLEU / 0.9248 BERTScore (N=667 test). Identity-copy = 47.88 SARI (artifact). (IMPROVEMENT_PLAN.md)

## Evidence — candidate generation + reranking works for simplification

### EhiMeNLP — TSAR 2025 Shared Task WINNER (most direct analogue)
- Miyata et al., 2025. "Candidate Generation via Iterative Simplification and Reranking by Readability and Semantic Similarity." https://aclanthology.org/2025.tsar-1.18/
- **Won** the TSAR 2025 Readability-Controlled Text Simplification shared task with a **two-step strategy: (1) candidate generation** by combining multiple LLMs with prompts, **(2) reranking** by **readability-based filtering** + **ranking on semantic similarity to the original**.
- This is almost exactly the proposed Option B (fidelity + simplicity reranker). Strong existence proof that the approach can be a winning system. (Note: full per-metric deltas are in the PDF, not extracted here — PDF full-text parsing not performed per workflow; abstract-level claim only.)

### TSAR 2025 (Sharoff/Saggion team) — candidate selection by judges + CEFR filtering
- https://aclanthology.org/2025.tsar-1.16v1.pdf
- Uses "candidate selection guided by automatic and LLM-based judges"; **CEFR filtering** first (retain candidates matching target readability), then select. Confirms readability-first filtering then similarity/quality selection is an established, competitive recipe.

### Medical Text Simplification — reranked beam search by readability
- "Medical Text Simplification: Optimizing for Readability with Unlikelihood Training and Reranked Beam Search Decoding", arXiv:2310.11191. https://ar5iv.labs.arxiv.org/html/2310.11191
- A **modified beam search that reranks intermediate candidates by readability** improved readability up to **2.43 FKGL points** AND human eval, **while maintaining** meaning preservation. Domain = medical simplification (directly relevant). Shows decode-time reranking by readability is effective and meaning-safe.

### MBR decoding (Option A — self-consensus)
- "It's MBR All the Way Down", ACL 2023 BigPicture. https://aclanthology.org/2023.bigpicture-1.9.pdf — Abstract: MBR "provides **reliable several-point improvements across metrics for a wide variety of tasks without any additional data or training**", at "an additional cost at inference time." Directly supports: no training/data needed, gains are real but require K× inference.
- "Follow the Wisdom of the Crowd: Effective Text Generation via MBR", Findings ACL 2023. https://aclanthology.org/2023.findings-acl.262.pdf — greedy/beam suffer degeneration; temperature/top-k/nucleus give diverse but lower-quality outputs; MBR over a sampled pool addresses the diversity-vs-quality tradeoff. Justifies "sample K then select by consensus."
- "Better Instruction-Following Through MBR", ICLR 2025 Spotlight. https://openreview.net/forum?id=7xCSK9BLPy — MBR as test-time compute for instruction-following LLMs; LLM-judge utility. Confirms relevance to instruct models like Qwen3.5.
- "Improving MBR Decoding with Multi-Prompt", EMNLP 2024. https://aclanthology.org/2024.emnlp-main.1255.pdf / arXiv:2407.15343 — explicitly benchmarks on a **text simplification task** (LENS metric); multi-prompt candidate banks + MBR improve quality. Supports generating candidates from several prompt variants (you already have `default_zero_shot`, `nih_k8`, `few_shot`, `definition_augmented`).

### CLEF 2025 SimpleText context (project's own benchmark)
- Task 1 overview: https://ceur-ws.org/Vol-4038/paper_344.pdf ; Track overview: https://link.springer.com/chapter/10.1007/978-3-032-04354-2_23
- UvA CLEF 2025: plan-guided + Cochrane-auto BART systems. https://ceur-ws.org/Vol-4038/paper_359.pdf
- MedSimplify (glossary RAG, 5th in Task 1.2): https://hal.science/hal-05296850v1/document — consistent with project's existing RAG finding; FKGL-based output selection helped slightly (already in IMPROVEMENT_PLAN).

## Synthesis / takeaways
- The proposed bet is well-supported: a *winning* 2025 simplification system (EhiMeNLP) is candidate-gen + readability/similarity rerank; MBR literature promises "several-point" metric gains with **no training/data**; medical reranked decoding improves readability without hurting fidelity.
- Two viable, complementary selectors confirmed: (A) MBR self-consensus, (B) readability-filter-then-similarity (EhiMeNLP/TSAR style). Best practice = readability filter FIRST, then similarity/consensus among survivors.
- Caveats grounded in sources: gains cost K× inference (MBR papers); pure sampling without selection lowers quality (Wisdom-of-Crowd); selecting on a proxy can game that proxy → must validate on `val` with held-out SARI/BLEU/BERTScore and watch empty-output rate.
- SARI-specific risk is project-internal (identity-copy artifact), not from external sources — keep as inference, not claimed external fact.

## Provenance status
- All external claims are abstract/HTML-level. Full-text PDF parsing of TSAR papers NOT performed (workflow avoids PDF extraction); per-metric deltas for EhiMeNLP marked as not-extracted.
