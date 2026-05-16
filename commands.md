````
uv run python experiments/sentence_level/simplify_sentence.py --sentence "Given the evidence from this Cochrane review, the avoidance of nitrous oxide may be reasonable in participants with pre-existing poor pulmonary function or at high risk of postoperative nausea and vomiting." --reference "The avoidance of nitrous oxide may be reasonable in participants with pre-existing poor pulmonary function or at high risk of postoperative nausea and vomiting." --prompt_type "paper358_zero_shot"
````

````
uv run python experiments/sentence_level/simplify_sentence.py --sentence "Given the evidence from this Cochrane review, the avoidance of nitrous oxide may be reasonable in participants with pre-existing poor pulmonary function or at high risk of postoperative nausea and vomiting." --reference "The avoidance of nitrous oxide may be reasonable in participants with pre-existing poor pulmonary function or at high risk of postoperative nausea and vomiting." --num-shots 1

~/Downloads/tum/SS26/Praktikum/simple_text main !2 ?7 ❯ uv run python experiments/sentence_level/simplify_sentence.py --sentence "Given the evidence from this Cochrane review, the avoidance of nitrous oxide may be reasonable in participants with pre-existing poor pulmonary function or at high risk of postoperative nausea and vomiting." --reference "The avoidance of nitrous oxide may be reasonable in participants with pre-existing poor pulmonary function or at high risk of postoperative nausea and vomiting." --num_shots 1
================================================================================
TEXT SIMPLIFICATION
================================================================================

Model: meta-llama/Llama-3.2-3B-Instruct
Device: CPU
Prompt type: default
Few-shot examples: 1
Temperature: 0.7

Loading model...
INFO:src.models.llama_simplifier:Loading meta-llama/Llama-3.2-3B-Instruct on cpu...
INFO:httpx:HTTP Request: HEAD https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/config.json "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/tokenizer_config.json "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET https://huggingface.co/api/models/meta-llama/Llama-3.2-3B-Instruct/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
INFO:httpx:HTTP Request: GET https://huggingface.co/api/models/meta-llama/Llama-3.2-3B-Instruct/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: GET https://huggingface.co/api/models/meta-llama/Llama-3.2-3B-Instruct "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: HEAD https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/config.json "HTTP/1.1 200 OK"
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 254/254 [00:00<00:00, 532.18it/s]
INFO:httpx:HTTP Request: HEAD https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/generation_config.json "HTTP/1.1 200 OK"
INFO:src.models.llama_simplifier:Model loaded successfully!
Model loaded successfully!

Using 1 few-shot examples:

  Example 1:
    Complex: Resuscitation with a nasal interface may reduce the rate of intubation...
    Simple:  It may reduce the number of newborn babies who are intubated in the de...

--------------------------------------------------------------------------------
INPUT (Complex):
  Given the evidence from this Cochrane review, the avoidance of nitrous oxide may be reasonable in participants with pre-existing poor pulmonary function or at high risk of postoperative nausea and vomiting.
--------------------------------------------------------------------------------

Simplifying...
[transformers] Ignoring clean_up_tokenization_spaces=True for BPE tokenizer TokenizersBackend. The clean_up_tokenization post-processing step is designed for WordPiece tokenizers and is destructive for BPE (it strips spaces before punctuation). Set clean_up_tokenization_spaces=False to suppress this warning, or set clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True to force cleanup anyway.

--------------------------------------------------------------------------------
OUTPUT (Simplified):
  Based on the review, using nitrous oxide may not be a good idea for people who already have lung problems or are at risk of getting sick to their stomach after surgery.
--------------------------------------------------------------------------------

================================================================================
EVALUATION
================================================================================

REFERENCE (Target simplification):
  The avoidance of nitrous oxide may be reasonable in participants with pre-existing poor pulmonary function or at high risk of postoperative nausea and vomiting.

Computing SARI metric...
INFO:src.evaluation.metrics:Evaluating 1 predictions...
INFO:src.evaluation.metrics:Computing SARI...
INFO:src.evaluation.metrics:SARI: 20.69 (Add: 0.00, Keep: 20.95, Del: 41.12)


============================================================
EVALUATION RESULTS
============================================================

SARI Score:              20.6912

============================================================

Interpretation:
  SARI: 0-100 (higher is better)
    - <25: Poor simplification
    - 25-35: Acceptable baseline
    - 35-42: Good system
    - >42: State-of-the-art

  Reference: LIS at SimpleText 2025 (Paper 358)
    - Best result: 43.51 (5th place at CLEF 2025)
    - Mistral 7B with zero-shot + definitions
============================================================


~/Downloads/tum/SS26/Praktikum/simple_text main !2 ?7 ❯                                                                                                               1m 17s  3.12 
````