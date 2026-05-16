"""Evaluation metrics for text simplification."""

from typing import List, Dict
import logging

# Import metrics
from easse.sari import get_corpus_sari_operation_scores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_sari(
    sources: List[str],
    predictions: List[str],
    references: List[List[str]]
) -> float:
    """
    Compute SARI (System output Against References and Input).
    
    SARI measures the quality of simplification by comparing:
    - Additions: n-grams added in prediction (vs source)
    - Deletions: n-grams deleted from source
    - Keeps: n-grams kept from source
    
    Args:
        sources: List of complex input sentences
        predictions: List of predicted simplified sentences
        references: List of lists of reference simplifications
    
    Returns:
        SARI score (0-100, higher is better)
    """
    logger.info("Computing SARI...")
    
    try:
        # EASSE API: orig_sents, sys_sents, refs_sents
        add, keep, delete_score = get_corpus_sari_operation_scores(
            orig_sents=sources,
            sys_sents=predictions,
            refs_sents=references
        )
        sari = (add + keep + delete_score) / 3
        logger.info(f"SARI: {sari:.2f} (Add: {add:.2f}, Keep: {keep:.2f}, Del: {delete_score:.2f})")
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
    
    Following the evaluation approach from LIS at SimpleText 2025 (Paper 358),
    which uses only SARI as the primary evaluation metric.
    
    Args:
        sources: List of complex input sentences
        predictions: List of predicted simplified sentences
        references: List of lists of reference simplifications
    
    Returns:
        Dictionary with SARI score and component scores
    """
    logger.info(f"Evaluating {len(predictions)} predictions...")
    
    # Filter out empty predictions or references
    valid_indices = []
    for i in range(len(predictions)):
        if predictions[i].strip() and references[i] and any(ref.strip() for ref in references[i]):
            valid_indices.append(i)
    
    if len(valid_indices) < len(predictions):
        logger.warning(f"Filtered out {len(predictions) - len(valid_indices)} empty predictions/references")
    
    # Filter data
    sources_filtered = [sources[i] for i in valid_indices]
    predictions_filtered = [predictions[i] for i in valid_indices]
    references_filtered = [references[i] for i in valid_indices]
    
    results = {}
    
    # Compute SARI
    sari = compute_sari(sources_filtered, predictions_filtered, references_filtered)
    results['sari'] = sari
    
    return results


def print_results(results: Dict[str, float]):
    """
    Print evaluation results in a formatted table.
    
    Following the evaluation approach from LIS at SimpleText 2025 (Paper 358).
    
    Args:
        results: Dictionary of metric scores
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nSARI Score:              {results.get('sari', 0):.4f}")
    
    print("\n" + "="*60)
    print("\nInterpretation:")
    print("  SARI: 0-100 (higher is better)")
    print("    - <25: Poor simplification")
    print("    - 25-35: Acceptable baseline")
    print("    - 35-42: Good system")
    print("    - >42: State-of-the-art")
    print("\n  Reference: LIS at SimpleText 2025 (Paper 358)")
    print("    - Best result: 43.51 (5th place at CLEF 2025)")
    print("    - Mistral 7B with zero-shot + definitions")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test the metrics
    print("Testing evaluation metrics...")
    
    # Sample data
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
    
    results = evaluate_simplification(
        sources,
        predictions,
        references
    )
    
    print_results(results)
