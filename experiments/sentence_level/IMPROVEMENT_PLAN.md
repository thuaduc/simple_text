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

- [ ] Week 1: Implement prompt variants and split-aware validation runs.
- [ ] Week 1: Compare prompt-only Qwen3.5 2B/4B systems on `val`.
- [ ] Week 2: Build exact-match biomedical glossary retrieval and definition-augmented prompting.
- [ ] Week 2: Report RAG coverage and compare with SARI, BLEU, and BERTScore.
- [ ] Week 3: Fine-tune Qwen3.5 with QLoRA on the rephrase-only train split.
- [ ] Week 3: Compare fine-tuned models against prompt-only and RAG validation winners.
- [ ] Week 4: Run final selected systems on `test`.
- [ ] Week 4: Add candidate generation only if time remains.

## Success Criteria

- Preserve the current baseline result as a reproducible run.
- Achieve a validation SARI gain without increasing empty outputs, factual-number errors, or unreadable outputs.
- Prefer a final system that is simple to submit: one Qwen3.5 model, one selected prompt/retrieval configuration, deterministic output settings, and full metadata in results.
- Use the Task 1.1 rephrase-only files only unless we explicitly decide to expand scope later.
