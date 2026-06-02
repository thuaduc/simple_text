"""Simple script to simplify a single sentence using the text simplification model."""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import MODEL_NAME
from src.models.llama_simplifier import LlamaSimplifier
from src.prompts.few_shot_examples import get_curated_examples
from src.evaluation.metrics import evaluate_simplification, print_results


def main():
    """Simplify a single sentence."""
    parser = argparse.ArgumentParser(
        description="Simplify a single sentence using the text simplification model"
    )
    
    parser.add_argument(
        '--sentence',
        type=str,
        nargs='?',
        help='The sentence to simplify (if not provided, will use an example)'
    )
    
    parser.add_argument(
        '--reference',
        type=str,
        help='Reference simplified sentence for evaluation (optional)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=MODEL_NAME,
        help=f'Model to use (default: {MODEL_NAME})'
    )
    
    parser.add_argument(
        '--num_shots',
        type=int,
        default=0,
        help='Number of few-shot examples (default: 0, zero-shot)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=1024,
        help='Maximum tokens to generate (default: 256)'
    )
    
    parser.add_argument(
        '--prompt_type',
        type=str,
        default='default',
        choices=['default', 'paper358_zero_shot', 'paper358_one_shot'],
        help='Prompt strategy to use (default: default)'
    )
    
    parser.add_argument(
        '--load_in_4bit',
        action='store_true',
        help='Use 4-bit quantization (requires CUDA)'
    )
    
    args = parser.parse_args()
    
    # Use example sentence if none provided
    if args.sentence is None:
        args.sentence = (
            "Resuscitation with a nasal interface may reduce the rate of "
            "intubation in the DR, but the evidence is very uncertain."
        )
        if args.reference is None:
            args.reference = (
                "Using a nose mask to help babies breathe may reduce the need for "
                "a breathing tube, but we're not sure."
            )
        print("No sentence provided, using example sentence.")
        print()
    
    print("="*80)
    print("TEXT SIMPLIFICATION")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Few-shot examples: {args.num_shots}")
    print(f"Temperature: {args.temperature}")
    print()
    
    # Load model
    print("Loading model...")
    simplifier = LlamaSimplifier(
        model_name=args.model,
        load_in_4bit=args.load_in_4bit and torch.cuda.is_available(),
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=True
    )
    print("Model loaded successfully!")
    print()
    
    # Get few-shot examples if requested
    few_shot_examples = None
    if args.num_shots > 0:
        few_shot_examples = get_curated_examples(num_examples=args.num_shots)
        print(f"Using {len(few_shot_examples)} few-shot examples:")
        for i, ex in enumerate(few_shot_examples, 1):
            print(f"\n  Example {i}:")
            print(f"    Complex: {ex['complex'][:70]}...")
            print(f"    Simple:  {ex['simple'][:70]}...")
        print()
    
    # Display input
    print("-"*80)
    print("INPUT (Complex):")
    print(f"  {args.sentence}")
    print("-"*80)
    print()
    
    # Simplify
    print("Simplifying...")
    simplified = simplifier.simplify(args.sentence, few_shot_examples=few_shot_examples, prompt_type=args.prompt_type)
    print()
    
    # Display output
    print("-"*80)
    print("OUTPUT (Simplified):")
    print(f"  {simplified}")
    print("-"*80)
    print()
    
    # Evaluate if reference is provided
    if args.reference:
        print("="*80)
        print("EVALUATION")
        print("="*80)
        print()
        print("REFERENCE (Target simplification):")
        print(f"  {args.reference}")
        print()
        
        print("Computing SARI metric...")
        
        results = evaluate_simplification(
            sources=[args.sentence],
            predictions=[simplified],
            references=[[args.reference]]
        )
        
        print()
        print_results(results)
        print()
    else:
        print("Tip: Use --reference 'your reference text' to evaluate the output")
        print()


if __name__ == "__main__":
    main()
