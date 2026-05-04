# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

This repository implements solutions for [CLEF 2026 SimpleText Task 1 – Text Simplification](https://simpletext-project.com/2026/), focusing on simplifying complex scientific texts using NLP/LLM techniques.

## Task 1 – Text Simplification

Two subtasks, both using the **Cochrane-auto corpus** (biomedical abstracts and lay summaries from Cochrane systematic reviews):

- **1.1** – Simplify individual sentences extracted from Cochrane-auto.
- **1.2** – Simplify whole documents extracted from Cochrane-auto.

**Evaluation metrics**: SARI, BLEU, LENS, BERTScore (automatic) + human assessment by translation students and professionals.

## Data

Data is distributed to registered participants via the [SimpleText mailing list](https://groups.google.com/g/simpletext). Store it under `papers/` (already gitignored).

### Cochrane-auto corpus

- Parallel data from the same authors (technical abstract + lay summary), aligned at paragraph, sentence, and document levels.
- Incorporates sentence merging, reordering, and discourse-structure alignment.
- **2026 test data**: newly crawled Cochrane systematic reviews from the past year, not part of the training corpus.
