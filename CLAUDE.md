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

## Models (≤ 7B parameters)

Use only models with **≤ 7B parameters**. Larger models (e.g. Llama 3.1/3.3 8B+, Gemma 2 9B, LLaMA 70B) are out of scope.

### Recommended by use case

#### Fine-tuning on Cochrane-auto (seq2seq — best reproducible baselines)

| Model | Params | HuggingFace ID | SimpleText evidence |
| ----- | ------ | -------------- | ------------------- |
| **BART-large** | 400M | `facebook/bart-large` | **42.31 SARI** — UvA plan-guided BART on Cochrane-auto (CLEF 2025); strongest open fine-tuned approach |
| **BART-base** | 140M | `facebook/bart-base` | **41.28 SARI** — UvA sentence-level BART on Cochrane-auto; lighter GPU footprint |
| **T5-base** | 220M | `t5-base` | Fine-tune on Cochrane-auto; weaker than BART on this task but useful for ablations |
| **SciFive-base** | 220M | `razent/SciFive-base-Pubmed` | T5 pretrained on PubMed; good biomedical domain prior before Cochrane fine-tuning |
| **BART-base (WikiLarge)** | 140M | `eilamc14/bart-base-text-simplification` | Pre-fine-tuned on general simplification (ASSET SARI 36.13); warm-start for Cochrane fine-tuning |

**Avoid for zero-shot**: Flan-T5 underperforms on biomedical SimpleText (DUTH: 19.51 base / 35.35 large zero-shot). Fine-tuning Flan-T5 on Cochrane-auto is still viable.

#### Zero-shot / few-shot prompting (instruction LLMs)

| Model | Params | HuggingFace ID | SimpleText evidence |
| ----- | ------ | -------------- | ------------------- |
| **Mistral 7B Instruct** | 7B | `mistralai/Mistral-7B-Instruct-v0.3` | **43.51 SARI** — LIS at SimpleText 2025 (Paper 358), zero-shot + MedSimplify glossary (doc-level); best open LLM result |
| **Qwen2.5 7B Instruct** | ~6.5B | `Qwen/Qwen2.5-7B-Instruct` | Strong general instruct model; good Mistral alternative at ≤7B |
| **Llama 2 7B Chat** | 7B | `meta-llama/Llama-2-7b-chat` | Strong on PLABA biomedical simplification when LoRA fine-tuned |
| **Llama 3.2 3B Instruct** | 3B | `meta-llama/Llama-3.2-3B-Instruct` | Project default; good quality/efficiency tradeoff (~6 GB FP16, ~2 GB 4-bit) |
| **Llama 3.2 1B Instruct** | 1B | `meta-llama/Llama-3.2-1B-Instruct` | Minimal resources (~2 GB FP16); useful for rapid iteration |
| **Qwen2.5 3B Instruct** | 3B | `Qwen/Qwen2.5-3B-Instruct` | Competitive small instruct model |
| **Phi-3 Mini Instruct** | 3.8B | `microsoft/Phi-3-mini-4k-instruct` | Strong reasoning for its size |
| **Gemma 2 2B IT** | 2.6B | `google/gemma-2-2b-it` | Efficient small model for prompt experiments |

**Prompting tip**: Paper 358's DASP_0 zero-shot prompt with MedSimplify definitions (+0.6 SARI) works best with 7B instruct models; one-shot hurt performance.

#### Auxiliary models (classification, jargon detection)

| Model | Params | HuggingFace ID | Role |
| ----- | ------ | -------------- | ---- |
| **RoBERTa-base** | 125M | `roberta-base` | Operation classifier (rephrase/delete/split/ignore/merge) for plan-guided pipeline |
| **BioBERT-base** | 110M | `dmis-lab/biobert-base-cased-v1.1` | Biomedical NER / jargon term identification |
| **DeBERTa-v3-base** | 184M | `microsoft/deberta-v3-base` | Jargon detection (UvA, THM approaches) |

### Priority order for this project

1. **Fine-tune `facebook/bart-large`** on Cochrane-auto + plan-guided operation prefixes (UvA approach, ~42 SARI, no API)
2. **Zero-shot `mistralai/Mistral-7B-Instruct-v0.3`** with MedSimplify glossary (Paper 358 DASP_0, ~43 SARI doc-level)
3. **Hybrid**: BART beam candidates + LLM candidates → MBR reranking with LENS (Best-of-Labs 2023 pattern)
4. **LoRA fine-tune** a 7B instruct model (`Llama-2-7b-chat`, `Qwen2.5-7B-Instruct`) on Cochrane-auto if GPU allows

Use **4-bit quantization** for 7B models when GPU memory is limited (~4 GB vs ~14 GB FP16).
