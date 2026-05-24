"""Main script to run baseline text simplification evaluation."""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import MODEL_NAME, DATA_DIR, BATCH_SIZE, RANDOM_SEED
from src.utils.data_loader import load_cochrane_sentences, get_few_shot_examples
from src.prompts.few_shot_examples import get_curated_examples
from src.models.llama_simplifier import LlamaSimplifier
from src.evaluation.metrics import evaluate_simplification, print_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline text simplification evaluation"
    )
    
    parser.add_argument(
        '--num_shots',
        type=int,
        default=3,
        help='Number of few-shot examples (default: 3)'
    )
    
    parser.add_argument(
        '--test_size',
        type=int,
        default=None,
        help='Subset of test data to use (default: all)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size for generation (default: {BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/sentence_level/results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=DATA_DIR,
        help=f'Directory containing Cochrane data (default: {DATA_DIR})'
    )
    
    parser.add_argument(
        '--load_in_4bit',
        action='store_true',
        help='Use 4-bit quantization (requires CUDA)'
    )
    
    parser.add_argument(
        '--use_curated_examples',
        action='store_true',
        help='Use manually curated examples instead of sampling from training data'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {RANDOM_SEED})'
    )

    parser.add_argument(
        '--all_labels',
        action='store_true',
        help='Use full dataset with all operation labels (default: rephrase only)'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation pipeline."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    print("="*80)
    print("TEXT SIMPLIFICATION BASELINE EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Few-shot examples: {args.num_shots}")
    print(f"  Test size: {args.test_size or 'all'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  4-bit quantization: {args.load_in_4bit}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Random seed: {args.seed}")
    print(f"  Dataset: {'all labels' if args.all_labels else 'rephrase only'}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    rephrase_only = not args.all_labels

    # Load test data
    print("Loading test data...")
    complex_sentences, simple_references, labels, pair_ids = load_cochrane_sentences(
        split='test',
        data_dir=args.data_dir,
        rephrase_only=rephrase_only,
    )
    
    # Optionally limit test size
    if args.test_size is not None:
        print(f"Limiting to first {args.test_size} examples...")
        complex_sentences = complex_sentences[:args.test_size]
        simple_references = simple_references[:args.test_size]
        labels = labels[:args.test_size]
        pair_ids = pair_ids[:args.test_size]
    
    print(f"Total test examples: {len(complex_sentences)}")
    
    # Get few-shot examples
    print(f"\nPreparing {args.num_shots}-shot examples...")
    if args.use_curated_examples:
        few_shot_examples = get_curated_examples(num_examples=args.num_shots)
        print("Using manually curated examples")
    else:
        few_shot_examples = get_few_shot_examples(
            num_shots=args.num_shots,
            data_dir=args.data_dir,
            seed=args.seed,
            rephrase_only=rephrase_only,
        )
        print("Using examples sampled from training data")
    
    # Display examples
    print("\nFew-shot examples:")
    for i, ex in enumerate(few_shot_examples, 1):
        print(f"\n  Example {i}:")
        print(f"    Complex: {ex['complex'][:80]}...")
        print(f"    Simple:  {ex['simple'][:80]}...")
    
    # Initialize model
    print(f"\nInitializing {MODEL_NAME}...")
    simplifier = LlamaSimplifier(
        load_in_4bit=args.load_in_4bit and torch.cuda.is_available()
    )
    
    # Generate simplifications
    print("\nGenerating simplifications...")
    print(f"Processing {len(complex_sentences)} sentences...")
    
    predictions = simplifier.simplify_batch(
        complex_sentences,
        few_shot_examples=few_shot_examples,
        batch_size=args.batch_size
    )
    
    print(f"Generated {len(predictions)} simplifications")
    
    # Save predictions
    predictions_file = os.path.join(args.output_dir, 'predictions.txt')
    print(f"\nSaving predictions to {predictions_file}...")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')
    
    # Also save input-output pairs for inspection
    pairs_file = os.path.join(args.output_dir, 'input_output_pairs.txt')
    print(f"Saving input-output pairs to {pairs_file}...")
    with open(pairs_file, 'w', encoding='utf-8') as f:
        for i, (inp, pred, refs) in enumerate(zip(complex_sentences, predictions, simple_references)):
            f.write(f"Example {i+1}\n")
            f.write(f"Input:  {inp}\n")
            f.write(f"Output: {pred}\n")
            f.write(f"References: {refs}\n")
            f.write("-" * 80 + "\n\n")
    
    # Evaluate
    print("\nEvaluating predictions...")
    
    results = evaluate_simplification(
        sources=complex_sentences,
        predictions=predictions,
        references=simple_references
    )
    
    # Print results
    print_results(results)
    
    # Save results
    results_file = os.path.join(args.output_dir, 'metrics.json')
    print(f"Saving metrics to {results_file}...")
    
    # Add metadata
    results_with_metadata = {
        'metrics': results,
        'metadata': {
            'model': MODEL_NAME,
            'num_shots': args.num_shots,
            'test_size': len(complex_sentences),
            'batch_size': args.batch_size,
            'load_in_4bit': args.load_in_4bit,
            'use_curated_examples': args.use_curated_examples,
            'rephrase_only': rephrase_only,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat(),
            'evaluation': 'SARI, BLEU, BERTScore (automatic metrics)'
        }
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_with_metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - predictions.txt: Generated simplifications")
    print(f"  - input_output_pairs.txt: Side-by-side comparison")
    print(f"  - metrics.json: All evaluation metrics")
    print()


if __name__ == "__main__":
    main()
