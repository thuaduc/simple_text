# Evaluation Metrics

Metrics used to evaluate text simplification quality on the Cochrane-auto corpus.

---

## SARI (System output Against References and Input)

**Purpose**: Primary metric for text simplification

**Formula**: `SARI = (F1_add + F1_keep + Precision_del) / 3`

**What it measures**:
- **Add**: New words added (vs source)
- **Keep**: Words kept from source
- **Delete**: Words removed from source

**Range**: 0-100 (higher is better)

**Interpretation**:
- <25: Poor
- 25-35: Acceptable baseline
- 35-42: Good
- \>42: State-of-the-art

**Reference**: Xu et al. (2016)

---

## BLEU (Bilingual Evaluation Understudy)

**Purpose**: Measure n-gram overlap with references

**Formula**: `BLEU = BP × exp(Σ w_n × log(p_n))`

**What it measures**: N-gram precision (1,2,3,4-grams) with brevity penalty

**Range**: 0-100 (higher is better)

**Interpretation**:
- 10-20: Low overlap (common for simplification)
- 20-30: Moderate
- \>30: High

**Note**: Can be low for good simplifications due to intentional changes

**Reference**: Papineni et al. (2002)

---

## BERTScore

**Purpose**: Semantic similarity using contextual embeddings

**Method**: Token-level cosine similarity with RoBERTa-large embeddings

**Formula**:
```
Precision = mean(max similarity for each predicted token)
Recall = mean(max similarity for each reference token)
F1 = 2 × (P × R) / (P + R)
```

**Range**: 0-1 (report F1, higher is better)

**Interpretation**:
- <0.70: Poor semantic preservation
- 0.70-0.80: Good
- \>0.80: Excellent

**Reference**: Zhang et al. (2020)

---

## LENS (Learnable Evaluation Metric)

**Purpose**: Neural metric trained on human judgments

**Method**: Cross-encoder trained on 12K+ human ratings (SimpEval corpus)

**Input**: (source, prediction, reference) triplet

**Range**: 0-5 (higher is better)

**Interpretation**:
- <2.0: Poor
- 2.0-3.0: Acceptable
- 3.0-4.0: Good
- \>4.0: Excellent

**Note**: Highest correlation with human evaluation (ρ ≈ 0.6-0.7)

**Availability**: Requires separate installation

**Reference**: Maddela et al. (2023)

---

## Computing Metrics

All metrics are computed corpus-level (not averaged sentence-level):

```python
from src.evaluation.metrics import evaluate_simplification

results = evaluate_simplification(
    sources=complex_sentences,
    predictions=generated_simple,
    references=gold_references
)
```

**Output**:
```json
{
  "sari": 32.45,
  "bleu": 22.18,
  "bertscore_f1": 0.7541
}
```

---

## References

1. Xu et al. (2016) - "Optimizing Statistical Machine Translation for Text Simplification"
2. Papineni et al. (2002) - "BLEU: a Method for Automatic Evaluation of Machine Translation"
3. Zhang et al. (2020) - "BERTScore: Evaluating Text Generation with BERT"
4. Maddela et al. (2023) - "LENS: A Learnable Evaluation Metric for Text Simplification"
