# Simplification Methods

This document describes the text simplification approaches evaluated on the Cochrane-auto corpus for CLEF 2026 SimpleText Task 1.

## Dataset

**Cochrane-auto Corpus**:
- **Test set**: 1,544 sentence pairs (complex → simple)
- **Domain**: Biomedical abstracts from systematic reviews
- **Operations**: rephrase (60%), delete (20%), split (15%), merge (5%)

See [Cochrane README](cochrane/README.md) for full dataset details.

---

## Method 1: Few-Shot Prompting (Baseline)

**Approach**: Zero-training with in-context learning

**Model**: Mistral 7B Instruct v0.3 (7B parameters)

**Configuration**:
- Number of shots: 3 examples (default)
- Temperature: 0.7
- Max tokens: 256
- 4-bit quantization: Optional (recommended for GPU memory efficiency)

**Prompt Template**:
```
Simplify the following medical text for a general audience. Use plain language and avoid technical jargon.

[Example pairs of complex → simple]

Complex: {complex_sentence}
Simple: [model generates simplification]
```

**Run**:
```bash
uv run python experiments/sentence_level/run_baseline.py
```

**With 4-bit quantization (recommended)**:
```bash
uv run python experiments/sentence_level/run_baseline.py --load_in_4bit
```

**Expected Performance**:
| Metric | Target |
|--------|--------|
| SARI | 35-44 |

**Note**: Based on Paper 358 (LIS at SimpleText 2025), Mistral 7B achieved SARI 43.51 with zero-shot prompting.

---

## Method 2: [To be implemented]

Add new methods here as you develop them.

---

## Comparison

Results from all methods will be compared here.

| Method | SARI | BLEU | BERTScore | Notes |
|--------|------|------|-----------|-------|
| Method 1 (few-shot) | TBD | TBD | TBD | Zero training |
| Method 2 | - | - | - | |
