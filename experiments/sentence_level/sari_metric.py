"""SARI metric for text simplification evaluation.

Uses the HuggingFace `evaluate` library which implements the tensor2tensor
version of SARI (Xu et al. 2016) with community bugfixes:
  - 0/0 = 1 (not 0)
  - Fixed KEEP recall computation
  - ADD score uses F1 (not precision-only)
  - Proper sacrebleu 13a tokenization (separates punctuation)

This is the de facto standard used by CLEF SimpleText and all major
text simplification benchmarks.
"""

from typing import List
import evaluate

_sari_metric = None


def _get_metric():
    global _sari_metric
    if _sari_metric is None:
        _sari_metric = evaluate.load("sari")
    return _sari_metric


def sari_sentence(source: str, prediction: str, references: List[str]) -> float:
    """Compute SARI for a single sentence (0-100 scale)."""
    if not references or not any(r.strip() for r in references):
        return 0.0
    refs = [r for r in references if r.strip()]
    metric = _get_metric()
    result = metric.compute(sources=[source], predictions=[prediction], references=[refs])
    return result["sari"]


def corpus_sari(sources: List[str], predictions: List[str], references_list: List[List[str]]) -> dict:
    """Compute corpus-level SARI. Returns dict with overall score and count."""
    # Filter to entries with non-empty references
    s_filt, p_filt, r_filt = [], [], []
    for src, pred, refs in zip(sources, predictions, references_list):
        clean_refs = [r for r in refs if r.strip()] if refs else []
        if clean_refs:
            s_filt.append(src)
            p_filt.append(pred)
            r_filt.append(clean_refs)

    if not s_filt:
        return {"sari": 0.0, "n_evaluated": 0}

    metric = _get_metric()
    result = metric.compute(sources=s_filt, predictions=p_filt, references=r_filt)
    return {"sari": result["sari"], "n_evaluated": len(s_filt)}


if __name__ == "__main__":
    # Sanity checks
    s = sari_sentence(
        "We included five trials, in which 1406 infants participated.",
        "We found five studies with 1406 babies.",
        ["We found five studies that involved 1406 babies."]
    )
    print(f"Test SARI: {s:.2f}  (expect ~83)")

    # Copy baseline should be ~50+
    s2 = sari_sentence(
        "We included five trials, in which 1406 infants participated.",
        "We included five trials, in which 1406 infants participated.",
        ["We found five studies that involved 1406 babies."]
    )
    print(f"Copy SARI: {s2:.2f}  (expect ~38)")
