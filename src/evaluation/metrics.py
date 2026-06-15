"""Evaluation metrics for text simplification.

Uses the HuggingFace `evaluate` library (tensor2tensor SARI with bugfixes).
This is the de facto standard used by CLEF SimpleText.
"""

from typing import List, Dict
import logging
import evaluate
import sacrebleu
import torch
from bert_score import score as bert_score

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

    if not sources:
        return 0.0
    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError("SARI requires aligned sources, predictions, and references")

    metric = _get_metric()
    result = metric.compute(sources=sources, predictions=predictions, references=references)
    sari = result["sari"]
    logger.info(f"SARI: {sari:.2f}")
    return sari


def compute_bleu(
    predictions: List[str],
    references: List[List[str]]
) -> float:
    """
    Compute corpus-level BLEU score.
    
    Args:
        predictions: List of predicted simplified sentences
        references: List of lists of reference simplifications
    
    Returns:
        BLEU score (0-100, higher is better)
    """
    logger.info("Computing BLEU...")

    if not predictions:
        return 0.0
    if len(predictions) != len(references):
        raise ValueError("BLEU requires aligned predictions and references")
    if any(not refs for refs in references):
        raise ValueError("BLEU requires at least one reference for every prediction")

    # SacreBLEU expects reference streams; use None for missing references so
    # variable-reference corpora do not treat empty strings as valid references.
    max_refs = max(len(refs) for refs in references)
    ref_streams = [
        [refs[ref_idx] if ref_idx < len(refs) else None for refs in references]
        for ref_idx in range(max_refs)
    ]

    bleu = sacrebleu.corpus_bleu(predictions, ref_streams)
    logger.info(f"BLEU: {bleu.score:.2f}")
    return bleu.score


def compute_bertscore(
    predictions: List[str],
    references: List[List[str]]
) -> float:
    """
    Compute corpus-level BERTScore F1 with max-over-references strategy.
    
    Args:
        predictions: List of predicted simplified sentences
        references: List of lists of reference simplifications
    
    Returns:
        BERTScore F1 (0-1, higher is better)
    """
    logger.info("Computing BERTScore...")

    if not predictions:
        return 0.0
    if len(predictions) != len(references):
        raise ValueError("BERTScore requires aligned predictions and references")
    if any(not refs for refs in references):
        raise ValueError("BERTScore requires at least one reference for every prediction")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # For each prediction, compute F1 against all references and take max.
    # Empty predictions are scored as zero instead of being dropped.
    all_f1_scores = []

    for pred, refs in zip(predictions, references):
        if not pred.strip():
            all_f1_scores.append(0.0)
            continue

        _, _, F1 = bert_score(
            [pred] * len(refs),
            refs,
            lang="en",
            model_type="roberta-large",
            device=device,
            verbose=False,
            batch_size=32
        )

        all_f1_scores.append(F1.max().item())

    corpus_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
    logger.info(f"BERTScore F1: {corpus_f1:.4f}")
    return corpus_f1


def evaluate_simplification(
    sources: List[str],
    predictions: List[str],
    references: List[List[str]]
) -> Dict[str, float]:
    """
    Compute evaluation metrics for text simplification.
    
    Args:
        sources: List of complex input sentences
        predictions: List of predicted simplified sentences
        references: List of lists of reference simplifications
    
    Returns:
        Dictionary with SARI, BLEU, and BERTScore F1 scores
    """
    logger.info(f"Evaluating {len(predictions)} predictions...")
    
    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError("Evaluation requires aligned sources, predictions, and references")

    # Filter only examples that cannot be scored because they have no reference.
    # Empty predictions remain in the evaluation and are penalized by the metrics.
    valid_sources = []
    valid_predictions = []
    valid_references = []
    empty_predictions = 0
    
    for source, prediction, refs in zip(sources, predictions, references):
        clean_refs = [r for r in refs if r.strip()] if refs else []

        if clean_refs:
            if not prediction.strip():
                empty_predictions += 1
            valid_sources.append(source)
            valid_predictions.append(prediction.strip())
            valid_references.append(clean_refs)
    
    if len(valid_sources) < len(predictions):
        logger.warning(f"Filtered out {len(predictions) - len(valid_sources)} examples without references")
    if empty_predictions:
        logger.warning(f"Scoring {empty_predictions} empty predictions")
    
    results = {}
    
    if valid_sources:
        sari = compute_sari(valid_sources, valid_predictions, valid_references)
        results['sari'] = sari
        
        bleu = compute_bleu(valid_predictions, valid_references)
        results['bleu'] = bleu
        
        bertscore_f1 = compute_bertscore(valid_predictions, valid_references)
        results['bertscore_f1'] = bertscore_f1
    else:
        results['sari'] = 0.0
        results['bleu'] = 0.0
        results['bertscore_f1'] = 0.0
    
    results['n_evaluated'] = len(valid_sources)
    
    return results


def print_results(results: Dict[str, float]):
    """Print evaluation results in a formatted table."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nSARI Score:              {results.get('sari', 0):.4f}")
    print(f"BLEU Score:              {results.get('bleu', 0):.4f}")
    print(f"BERTScore F1:            {results.get('bertscore_f1', 0):.4f}")
    print(f"Sentences evaluated:     {results.get('n_evaluated', 0)}")
    
    print("\n" + "="*60)
    print("\nInterpretation:")
    print("  SARI: 0-100 (higher is better)")
    print("    - <30: Poor simplification")
    print("    - 30-40: Acceptable baseline")
    print("    - 40-43: Good system (CLEF 2025 top range)")
    print("    - >43: State-of-the-art")
    print("\n  BLEU: 0-100 (higher is better)")
    print("    - 10-20: Low overlap (common for simplification)")
    print("    - 20-30: Moderate")
    print("    - >30: High")
    print("    Note: Can be low for good simplifications due to intentional changes")
    print("\n  BERTScore F1: 0-1 (higher is better)")
    print("    - Raw roberta-large max-over-reference F1")
    print("    - Best used as a relative semantic-preservation signal")
    print("\n  CLEF 2025 Task 1.1 top results:")
    print("    - UM-FHS (GPT-4.1-mini): 43.34 SARI")
    print("    - DS@GT (plan-guided LLaMA 70B): 42.33 SARI")
    print("    - UvA (plan-guided BART): 42.31 SARI")
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
