# Provenance: SimpleText Task 1.1 improvement plan (candidate generation + reranking)

- **Date:** 2026-06-30
- **Rounds:** 1 research round (3 distinct web_search queries) + 2 targeted fetches
- **Sources consulted:** 8 primary (EhiMeNLP TSAR2025, TSAR2025 CEFR/judge system, MedText reranked beam search arXiv:2310.11191, MBR BigPicture ACL2023, MBR Wisdom-of-Crowd ACL2023, MBR Multi-Prompt EMNLP2024, MBR Instruction-Following ICLR2025, project IMPROVEMENT_PLAN.md) + 4 CLEF context links
- **Sources accepted:** all 8 primary; [1] and [4] directly fetched and reachable; repo facts verified by grep/read
- **Sources rejected:** none removed; [2],[3],[5],[6],[7] retained from search metadata (not individually re-fetched — noted in verification)
- **Verification:** PASS WITH NOTES — no FATAL issues. Notes: EhiMeNLP per-metric deltas not extracted (PDF parsing intentionally skipped per workflow); "2.43 FKGL" figure from search snippet not independently re-read; several PDF/OpenReview URLs reachability assumed from search provider.
- **Plan:** outputs/.plans/simpletext-task1-improvement.md
- **Research files:** outputs/.drafts/simpletext-task1-improvement-research-direct.md, outputs/.drafts/simpletext-task1-improvement-draft.md, outputs/.drafts/simpletext-task1-improvement-cited.md, outputs/.drafts/simpletext-task1-improvement-verification.md
- **Repo evidence:** grep of src/models/sentence_simplifier.py (single-output, no rerank), run_baseline.py flags, IMPROVEMENT_PLAN.md (47.38 SARI best of record; 47.88 identity-copy artifact)
