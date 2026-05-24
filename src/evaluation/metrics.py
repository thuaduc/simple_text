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
    
    try:
        metric = _get_metric()
        result = metric.compute(sources=sources, predictions=predictions, references=references)
        sari = result["sari"]
        logger.info(f"SARI: {sari:.2f}")
        return sari
    except Exception as e:
        logger.error(f"Error computing SARI: {e}")
        return 0.0


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
    
    try:
        # Handle empty references
        if not references:
            return 0.0
        
        # Transpose references from per-sentence to per-reference-variant format
        # Input: [[r1, r2], [r1], [r1, r2, r3], ...]
        # Output: [[r1_sent0, r1_sent1, ...], [r2_sent0, r2_sent1, ...], ...]
        
        # Find max number of references per sentence
        max_refs = max(len(refs) for refs in references)
        
        # Pad shorter reference lists with empty strings
        padded_references = []
        for refs in references:
            padded = list(refs) + [''] * (max_refs - len(refs))
            padded_references.append(padded)
        
        # Transpose to get reference streams
        ref_streams = [list(stream) for stream in zip(*padded_references)]
        
        # Compute BLEU
        bleu = sacrebleu.corpus_bleu(predictions, ref_streams)
        logger.info(f"BLEU: {bleu.score:.2f}")
        return bleu.score
    except Exception as e:
        logger.error(f"Error computing BLEU: {e}")
        return 0.0


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
    
    try:
        # Handle empty references
        if not references or not predictions:
            return 0.0
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # For each prediction, compute F1 against all references and take max
        all_f1_scores = []
        
        for pred, refs in zip(predictions, references):
            if not refs:
                continue
            
            # Compute BERTScore against all references for this prediction
            P, R, F1 = bert_score(
                [pred] * len(refs),
                refs,
                lang="en",
                model_type="roberta-large",
                device=device,
                verbose=False,
                batch_size=32
            )
            
            # Take max F1 across all references
            max_f1 = F1.max().item()
            all_f1_scores.append(max_f1)
        
        # Return corpus-level mean
        corpus_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
        logger.info(f"BERTScore F1: {corpus_f1:.4f}")
        return corpus_f1
    except Exception as e:
        logger.error(f"Error computing BERTScore: {e}")
        return 0.0


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
    print("    - <0.70: Poor semantic preservation")
    print("    - 0.70-0.80: Good")
    print("    - >0.80: Excellent")
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
