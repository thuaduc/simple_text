"""Main script to run baseline text simplification evaluation."""

import argparse
import json
import sys
from pathlib import Path
import torch
from datetime import datetime

# Add src to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "results"
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from src.config import MODEL_NAME, DATA_DIR, BATCH_SIZE, RANDOM_SEED
from src.utils.data_loader import load_cochrane_sentences
from src.evaluation.metrics import evaluate_simplification, print_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline text simplification evaluation"
    )
    
    parser.add_argument(
        '--test_size',
        type=int,
        default=None,
        help='Subset of test data to use (default: all)'
    )

    parser.add_argument(
        '--example',
        action='store_true',
        help='Run on 10 sentences only (quick smoke test; overrides --test_size)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size for generation (default: {BATCH_SIZE})'
    )

    parser.add_argument(
        '--baseline',
        type=str,
        choices=['model', 'identity', 'reference'],
        default='model',
        help='Baseline to evaluate: model generation, identity copy, or reference copy (default: model)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help='Base directory to save results (default: experiments/sentence_level/results)'
    )

    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Subfolder name for this run under output_dir (e.g. qwen35-4b-v2); avoids overwriting previous results'
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

    parser.add_argument(
        '--skip_bertscore',
        action='store_true',
        help='Skip BERTScore evaluation (useful for fast CPU baseline runs)'
    )
    
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> Path:
    """Resolve and create the output directory if needed."""
    resolved = output_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_output_dir(output_dir: Path, run_name: str | None) -> Path:
    """Resolve the final output directory, optionally scoped to a named run."""
    resolved = ensure_output_dir(output_dir)
    if run_name:
        resolved = ensure_output_dir(resolved / run_name)
    return resolved


def main():
    """Main evaluation pipeline."""
    args = parse_args()

    if args.example:
        args.test_size = 10
    
    output_dir = resolve_output_dir(args.output_dir, args.run_name)
    baseline_model_name = {
        'model': MODEL_NAME,
        'identity': 'identity_copy',
        'reference': 'reference_copy',
    }[args.baseline]
    prompt_name = 'default_zero_shot' if args.baseline == 'model' else None
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    print("="*80)
    print("TEXT SIMPLIFICATION BASELINE EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Baseline: {args.baseline}")
    print(f"  Model: {baseline_model_name}")
    print(f"  Prompt: {prompt_name or 'none'}")
    print(f"  Test size: {args.test_size or 'all'}{' (example mode)' if args.example else ''}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  4-bit quantization: {args.load_in_4bit}")
    print(f"  Output directory: {output_dir}")
    if args.run_name:
        print(f"  Run name: {args.run_name}")
    print(f"  Random seed: {args.seed}")
    print(f"  Dataset: {'all labels' if args.all_labels else 'rephrase only'}")
    print(f"  BERTScore: {'skipped' if args.skip_bertscore else 'enabled'}")
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
    
    if args.baseline == 'identity':
        print("\nUsing identity baseline: output is exactly the input")
        predictions = list(complex_sentences)
        print(f"Copied {len(predictions)} inputs as predictions")
    elif args.baseline == 'reference':
        print("\nUsing reference baseline: output is exactly the first reference")
        predictions = [refs[0] if refs else "" for refs in simple_references]
        print(f"Copied {len(predictions)} references as predictions")
    else:
        print("\nUsing default zero-shot prompt")

        # Initialize model only for the model baseline so identity runs stay lightweight.
        from src.models.sentence_simplifier import SentenceSimplifier

        print(f"\nInitializing {MODEL_NAME}...")
        simplifier = SentenceSimplifier(
            load_in_4bit=args.load_in_4bit and torch.cuda.is_available()
        )

        # Generate simplifications
        print("\nGenerating simplifications...")
        print(f"Processing {len(complex_sentences)} sentences...")

        predictions = simplifier.simplify_batch(
            complex_sentences,
            batch_size=args.batch_size
        )

        print(f"Generated {len(predictions)} simplifications")
    
    # Evaluate
    print("\nEvaluating predictions...")
    
    results = evaluate_simplification(
        sources=complex_sentences,
        predictions=predictions,
        references=simple_references,
        include_bertscore=not args.skip_bertscore,
    )
    
    # Print results before saving files
    print_results(results)
    
    # Save predictions
    ensure_output_dir(output_dir)
    predictions_file = output_dir / 'predictions.txt'
    print(f"\nSaving predictions to {predictions_file}...")
    with predictions_file.open('w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')
    
    # Also save input-output pairs for inspection
    pairs_file = output_dir / 'input_output_pairs.txt'
    print(f"Saving input-output pairs to {pairs_file}...")
    with pairs_file.open('w', encoding='utf-8') as f:
        for i, (inp, pred, refs) in enumerate(zip(complex_sentences, predictions, simple_references)):
            f.write(f"Example {i+1}\n")
            f.write(f"Input:  {inp}\n")
            f.write(f"Output: {pred}\n")
            f.write(f"References: {refs}\n")
            f.write("-" * 80 + "\n\n")
    
    # Save results
    results_file = output_dir / 'metrics.json'
    print(f"Saving metrics to {results_file}...")
    
    # Add metadata
    results_with_metadata = {
        'metrics': results,
        'metadata': {
            'baseline': args.baseline,
            'model': baseline_model_name,
            'prompt': prompt_name,
            'test_size': len(complex_sentences),
            'example_mode': args.example,
            'batch_size': args.batch_size,
            'load_in_4bit': args.load_in_4bit,
            'skip_bertscore': args.skip_bertscore,
            'rephrase_only': rephrase_only,
            'seed': args.seed,
            'run_name': args.run_name,
            'output_dir': str(output_dir),
            'timestamp': datetime.now().isoformat(),
            'evaluation': 'SARI, BLEU' + ('' if args.skip_bertscore else ', BERTScore') + ' (automatic metrics)'
        }
    }
    
    with results_file.open('w', encoding='utf-8') as f:
        json.dump(results_with_metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - predictions.txt: Generated simplifications")
    print(f"  - input_output_pairs.txt: Side-by-side comparison")
    print(f"  - metrics.json: All evaluation metrics")
    print()


if __name__ == "__main__":
    main()
