"""Evaluation metrics for text simplification.

Uses the HuggingFace `evaluate` library (tensor2tensor SARI with bugfixes).
This is the de facto standard used by CLEF SimpleText.
"""

from typing import List, Dict
import logging
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_sari_metric = None


def _get_metric():
    global _sari_metric
    if _sari_metric is None:
        _sari_metric = evaluate.load("sari")
    return _sari_metric


def compute_sari(
    sources: List[str],
    predictions: List[str],
    references: List[List[str]]
) -> float:
    """
    Compute SARI (System output Against References and Input).
    
    Args:
        sources: List of complex input sentences
        predictions: List of predicted simplified sentences
        references: List of lists of reference simplifications
    
    Returns:
        SARI score (0-100, higher is better)
    """
    logger.info("Computing SARI...")
    
    try:
        metric = _get_metric()
        result = metric.compute(sources=sources, predictions=predictions, references=references)
        sari = result["sari"]
        logger.info(f"SARI: {sari:.2f}")
        return sari
    except Exception as e:
        logger.error(f"Error computing SARI: {e}")
        return 0.0


def evaluate_simplification(
    sources: List[str],
    predictions: List[str],
    references: List[List[str]]
) -> Dict[str, float]:
    """
    Compute SARI evaluation metric for text simplification.
    
    Args:
        sources: List of complex input sentences
        predictions: List of predicted simplified sentences
        references: List of lists of reference simplifications
    
    Returns:
        Dictionary with SARI score
    """
    logger.info(f"Evaluating {len(predictions)} predictions...")
    
    # Filter out empty predictions or references
    valid_sources = []
    valid_predictions = []
    valid_references = []
    
    for i in range(len(predictions)):
        refs = references[i] if i < len(references) else []
        clean_refs = [r for r in refs if r.strip()] if refs else []
        
        if predictions[i].strip() and clean_refs:
            valid_sources.append(sources[i])
            valid_predictions.append(predictions[i])
            valid_references.append(clean_refs)
    
    if len(valid_sources) < len(predictions):
        logger.warning(f"Filtered out {len(predictions) - len(valid_sources)} empty predictions/references")
    
    results = {}
    
    if valid_sources:
        sari = compute_sari(valid_sources, valid_predictions, valid_references)
        results['sari'] = sari
    else:
        results['sari'] = 0.0
    
    results['n_evaluated'] = len(valid_sources)
    
    return results


def print_results(results: Dict[str, float]):
    """Print evaluation results in a formatted table."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nSARI Score:              {results.get('sari', 0):.4f}")
    print(f"Sentences evaluated:     {results.get('n_evaluated', 0)}")
    
    print("\n" + "="*60)
    print("\nInterpretation:")
    print("  SARI: 0-100 (higher is better)")
    print("    - <30: Poor simplification")
    print("    - 30-40: Acceptable baseline")
    print("    - 40-43: Good system (CLEF 2025 top range)")
    print("    - >43: State-of-the-art")
    print("\n  CLEF 2025 Task 1.1 top results:")
    print("    - UM-FHS (GPT-4.1-mini): 43.34")
    print("    - DS@GT (plan-guided LLaMA 70B): 42.33")
    print("    - UvA (plan-guided BART): 42.31")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Testing evaluation metrics...")
    
    sources = [
        "We included five trials, in which 1406 infants participated.",
        "The evidence is very uncertain."
    ]
    
    predictions = [
        "We found five studies with 1406 babies.",
        "We are not sure about the results."
    ]
    
    references = [
        ["We found five studies that involved 1406 babies."],
        ["We are uncertain about the evidence.", "The results are unclear."]
    ]
    
    results = evaluate_simplification(sources, predictions, references)
    print_results(results)
