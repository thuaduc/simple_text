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

from src.config import MODEL_NAME, DATA_DIR, BATCH_SIZE, RANDOM_SEED, TEMPERATURE
from src.utils.data_loader import load_cochrane_sentences
from src.evaluation.metrics import evaluate_simplification, print_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline text simplification evaluation"
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to use (default: test)",
    )

    parser.add_argument(
        "--test_size",
        type=int,
        default=None,
        help="Subset of data to use (default: all)",
    )

    parser.add_argument(
        "--example",
        action="store_true",
        help="Run on 10 sentences only (quick smoke test; overrides --test_size)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for generation (default: {BATCH_SIZE})",
    )

    parser.add_argument(
        "--baseline",
        type=str,
        choices=["model", "identity", "reference"],
        default="model",
        help="Baseline to evaluate: model generation, identity copy, or reference copy (default: model)",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        choices=["default_zero_shot", "few_shot"],
        default="default_zero_shot",
        help="Prompt variant to use (default: default_zero_shot)",
    )

    parser.add_argument(
        "--num_shots",
        type=int,
        default=3,
        help="Number of few-shot examples to retrieve (default: 3, only for few_shot prompt)",
    )

    parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable retrieval-augmented generation using a glossary "
        "(only applies to --baseline model). See --rag_mode for timing.",
    )

    parser.add_argument(
        "--rag_mode",
        type=str,
        choices=["before", "after"],
        default="before",
        help="When to apply RAG relative to the model: 'before' injects retrieved "
        "definitions for the input into the prompt; 'after' post-edits the model "
        "output by retrieving definitions for jargon still present (default: before)",
    )

    parser.add_argument(
        "--glossary_path",
        type=str,
        default=None,
        help="Path to glossary CSV used by --rag (default: cochrane/data/MedSimplify.csv)",
    )

    parser.add_argument(
        "--max_definitions",
        type=int,
        default=10,
        help="Maximum definitions to include per sentence (default: 10)",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Base directory to save results (default: experiments/sentence_level/results)",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Subfolder name for this run under output_dir (e.g. qwen35-4b-v2); avoids overwriting previous results",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR,
        help=f"Directory containing Cochrane data (default: {DATA_DIR})",
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to a trained LoRA adapter to load on top of the base model",
    )

    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sampling during generation (default: greedy decoding for reproducible evaluation)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Sampling temperature when --sample is enabled (default: {TEMPERATURE})",
    )

    parser.add_argument(
        "--num_candidates",
        type=int,
        default=1,
        help="Number of sampled candidates per sentence for candidate-generation "
        "+ reranking (default: 1 = no reranking; uses normal greedy/sample path)",
    )

    parser.add_argument(
        "--rerank",
        type=str,
        choices=["mbr", "readability", "oracle"],
        default="mbr",
        help="Reranking method when --num_candidates > 1: 'mbr' (self-consensus), "
        "'readability' (FKGL filter + fidelity), or 'oracle' (max-SARI ceiling, "
        "uses gold references; not deployable). Default: mbr",
    )

    parser.add_argument(
        "--rerank_temperature",
        type=float,
        default=None,
        help="Sampling temperature for candidate generation (default: --temperature)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})",
    )

    parser.add_argument(
        "--all_labels",
        action="store_true",
        help="Use full dataset with all operation labels (default: rephrase only)",
    )

    parser.add_argument(
        "--skip_bertscore",
        action="store_true",
        help="Skip BERTScore evaluation (useful for fast CPU baseline runs)",
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

    # Resolve glossary path for RAG (applies to both before/after modes)
    if args.rag:
        if args.baseline != "model":
            print(
                f"Warning: --rag only applies to --baseline model; "
                f"ignoring it for baseline '{args.baseline}'."
            )
            args.rag = False
        else:
            if args.glossary_path is None:
                args.glossary_path = str(Path(args.data_dir) / "MedSimplify.csv")
            if not Path(args.glossary_path).exists():
                print(
                    f"Error: --rag requires a glossary. "
                    f"File not found: {args.glossary_path}"
                )
                print("Please download MedSimplify.csv or specify --glossary_path")
                sys.exit(1)

    output_dir = resolve_output_dir(args.output_dir, args.run_name)
    baseline_model_name = {
        "model": MODEL_NAME,
        "identity": "identity_copy",
        "reference": "reference_copy",
    }[args.baseline]

    # Set random seed
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("TEXT SIMPLIFICATION BASELINE EVALUATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Split: {args.split}")
    print(f"  Baseline: {args.baseline}")
    print(f"  Model: {baseline_model_name}")
    print(f"  Prompt: {args.prompt}")
    if args.prompt == "few_shot":
        print(f"  Number of shots: {args.num_shots}")
    if args.rag:
        print(f"  RAG: enabled (mode: {args.rag_mode})")
        print(f"  Glossary: {args.glossary_path}")
        print(f"  Max definitions per sentence: {args.max_definitions}")
    print(
        f"  Data size: {args.test_size or 'all'}{' (example mode)' if args.example else ''}"
    )
    print(f"  Batch size: {args.batch_size}")
    if args.num_candidates > 1:
        print(
            f"  Generation: candidate sampling x{args.num_candidates} "
            f"(rerank: {args.rerank})"
        )
        print(
            f"  Candidate temperature: "
            f"{args.rerank_temperature if args.rerank_temperature is not None else args.temperature}"
        )
        if args.rerank == "oracle":
            print(
                "  NOTE: --rerank oracle uses gold references (ceiling only, "
                "NOT a deployable system)."
            )
    else:
        print(f"  Generation: {'sampling' if args.sample else 'greedy'}")
        if args.sample:
            print(f"  Temperature: {args.temperature}")
    print(f"  Output directory: {output_dir}")
    if args.run_name:
        print(f"  Run name: {args.run_name}")
    print(f"  Random seed: {args.seed}")
    print(f"  Dataset: {'all labels' if args.all_labels else 'rephrase only'}")
    print(f"  BERTScore: {'skipped' if args.skip_bertscore else 'enabled'}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    rephrase_only = not args.all_labels

    # Load data
    print(f"Loading {args.split} data...")
    complex_sentences, simple_references, labels, pair_ids = load_cochrane_sentences(
        split=args.split,
        data_dir=args.data_dir,
        rephrase_only=rephrase_only,
    )

    # Optionally limit data size
    if args.test_size is not None:
        print(f"Limiting to first {args.test_size} examples...")
        complex_sentences = complex_sentences[: args.test_size]
        simple_references = simple_references[: args.test_size]
        labels = labels[: args.test_size]
        pair_ids = pair_ids[: args.test_size]

    print(f"Total examples: {len(complex_sentences)}")

    # RAG post-editing tracking (populated only when --rag --rag_mode after)
    rag_postedit_data = []
    rag_postedit_stats = None

    # Initialize glossary retriever if needed
    retriever = None
    definitions_list = None
    glossary_matches_data = []

    if args.rag and args.rag_mode == "before":
        from src.retrieval import GlossaryRetriever

        print(f"\nInitializing glossary retriever from {args.glossary_path}...")
        retriever = GlossaryRetriever(args.glossary_path)

        print("\nRetrieving definitions for all sentences...")
        definitions_list = []
        for i, sentence in enumerate(complex_sentences):
            matches = retriever.retrieve(sentence, max_definitions=args.max_definitions)
            definitions_list.append(matches)

            # Store for later saving
            glossary_matches_data.append(
                {
                    "index": i,
                    "sentence": sentence,
                    "matched_terms": [term for term, _ in matches],
                    "num_matches": len(matches),
                }
            )

        # Compute and display coverage stats
        coverage = retriever.get_coverage_stats(
            complex_sentences, max_definitions=args.max_definitions
        )
        print(f"\nGlossary coverage:")
        print(
            f"  Sentences with matches: {coverage['sentences_with_matches']}/{coverage['total_sentences']} ({coverage['coverage_rate']*100:.1f}%)"
        )
        print(f"  Avg matches per sentence: {coverage['avg_matches_per_sentence']:.2f}")
        print(
            f"  Avg matches per matched sentence: {coverage['avg_matches_per_matched_sentence']:.2f}"
        )
        print(f"  Total unique matched terms: {coverage['unique_matched_terms']}")
        print(f"  Top 10 matched terms: {coverage['top_10_terms'][:10]}")

    # Initialize few-shot retriever if needed
    few_shot_retriever = None
    examples_list = None

    if args.prompt == "few_shot":
        from src.retrieval import FewShotRetriever

        print(f"\nInitializing few-shot retriever...")
        # Load training data for retrieval
        train_complex, train_simple, train_labels, train_ids = load_cochrane_sentences(
            split="train",
            data_dir=args.data_dir,
            rephrase_only=rephrase_only,
        )
        print(f"Loaded {len(train_complex)} training examples for few-shot retrieval")

        few_shot_retriever = FewShotRetriever(
            train_complex, train_simple, method="lexical"
        )

        print(f"\nRetrieving {args.num_shots} examples per sentence...")
        examples_list = []
        for sentence in complex_sentences:
            examples = few_shot_retriever.retrieve(sentence, k=args.num_shots)
            examples_list.append(examples)

    if args.baseline == "identity":
        print("\nUsing identity baseline: output is exactly the input")
        predictions = list(complex_sentences)
        print(f"Copied {len(predictions)} inputs as predictions")
    elif args.baseline == "reference":
        print("\nUsing reference baseline: output is exactly the first reference")
        predictions = [refs[0] if refs else "" for refs in simple_references]
        print(f"Copied {len(predictions)} references as predictions")
    else:
        print(f"\nUsing {args.prompt} prompt")

        # Initialize model only for the model baseline so identity runs stay lightweight.
        from src.models.sentence_simplifier import SentenceSimplifier

        print(f"\nInitializing {MODEL_NAME}...")
        simplifier = SentenceSimplifier(
            prompt_name=args.prompt,
            adapter_path=args.adapter_path,
            do_sample=args.sample,
            temperature=args.temperature,
        )

        # Generate simplifications
        print("\nGenerating simplifications...")
        print(f"Processing {len(complex_sentences)} sentences...")

        if args.num_candidates > 1:
            print(
                f"\nCandidate-generation + reranking: sampling "
                f"{args.num_candidates} candidates/sentence, rerank='{args.rerank}'"
            )
            candidates_list = simplifier.simplify_candidates_batch(
                complex_sentences,
                num_candidates=args.num_candidates,
                batch_size=args.batch_size,
                temperature=args.rerank_temperature,
                definitions_list=definitions_list,
                examples_list=examples_list,
            )

            from src.rerank import rerank_candidates

            references_list = simple_references if args.rerank == "oracle" else None
            predictions = rerank_candidates(
                sources=complex_sentences,
                candidates_list=candidates_list,
                method=args.rerank,
                references_list=references_list,
            )

            # Persist the full candidate pool + selection for inspection.
            candidates_file = output_dir / "candidates.jsonl"
            print(f"Saving candidate pools to {candidates_file}...")
            with candidates_file.open("w", encoding="utf-8") as f:
                for i, (src, cands, sel) in enumerate(
                    zip(complex_sentences, candidates_list, predictions)
                ):
                    f.write(
                        json.dumps(
                            {
                                "index": i,
                                "source": src,
                                "candidates": cands,
                                "selected": sel,
                            }
                        )
                        + "\n"
                    )
            print(f"Generated {len(predictions)} simplifications (reranked)")
        else:
            predictions = simplifier.simplify_batch(
                complex_sentences,
                batch_size=args.batch_size,
                definitions_list=definitions_list,
                examples_list=examples_list,
            )

            print(f"Generated {len(predictions)} simplifications")

        # RAG (after mode): retrieve definitions for jargon still in each
        # prediction, then regenerate a revised sentence.
        if args.rag and args.rag_mode == "after":
            from src.retrieval import GlossaryRetriever

            print(
                f"\n[RAG post-edit] Initializing glossary retriever from "
                f"{args.glossary_path}..."
            )
            postedit_retriever = GlossaryRetriever(args.glossary_path)

            print("[RAG post-edit] Retrieving definitions from model outputs...")
            postedit_definitions = []
            for i, pred in enumerate(predictions):
                matches = postedit_retriever.retrieve(
                    pred, max_definitions=args.max_definitions
                )
                postedit_definitions.append(matches)
                rag_postedit_data.append(
                    {
                        "index": i,
                        "draft": pred,
                        "matched_terms": [term for term, _ in matches],
                        "num_matches": len(matches),
                    }
                )

            num_to_edit = sum(1 for d in postedit_definitions if d)
            print(
                f"[RAG post-edit] {num_to_edit}/{len(predictions)} outputs contain "
                f"glossary terms; running second pass on those..."
            )

            drafts_before_postedit = list(predictions)
            predictions = simplifier.postedit_batch(
                original_sentences=complex_sentences,
                draft_simplifications=drafts_before_postedit,
                definitions_list=postedit_definitions,
                batch_size=args.batch_size,
            )

            num_changed = sum(
                1
                for before, after in zip(drafts_before_postedit, predictions)
                if before != after
            )
            print(
                f"[RAG post-edit] Revised {num_changed}/{len(predictions)} predictions"
            )
            for entry, before, after in zip(
                rag_postedit_data, drafts_before_postedit, predictions
            ):
                entry["revised"] = after
                entry["changed"] = before != after

            rag_postedit_stats = {
                "glossary_path": args.glossary_path,
                "max_definitions": args.max_definitions,
                "total_glossary_terms": len(postedit_retriever.glossary),
                "outputs_with_terms": num_to_edit,
                "predictions_changed": num_changed,
            }

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
    predictions_file = output_dir / "predictions.txt"
    print(f"\nSaving predictions to {predictions_file}...")
    with predictions_file.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")

    # Also save input-output pairs for inspection
    pairs_file = output_dir / "input_output_pairs.txt"
    print(f"Saving input-output pairs to {pairs_file}...")
    with pairs_file.open("w", encoding="utf-8") as f:
        for i, (inp, pred, refs) in enumerate(
            zip(complex_sentences, predictions, simple_references)
        ):
            f.write(f"Example {i+1}\n")
            f.write(f"Input:  {inp}\n")
            f.write(f"Output: {pred}\n")
            f.write(f"References: {refs}\n")
            f.write("-" * 80 + "\n\n")

    # Save glossary matches if using RAG (before mode)
    if args.rag and args.rag_mode == "before" and glossary_matches_data:
        matches_file = output_dir / "glossary_matches.jsonl"
        print(f"Saving glossary matches to {matches_file}...")
        with matches_file.open("w", encoding="utf-8") as f:
            for match_data in glossary_matches_data:
                f.write(json.dumps(match_data) + "\n")

    # Save RAG post-edit traces if used (after mode)
    if args.rag and args.rag_mode == "after" and rag_postedit_data:
        postedit_file = output_dir / "rag_postedit.jsonl"
        print(f"Saving RAG post-edit traces to {postedit_file}...")
        with postedit_file.open("w", encoding="utf-8") as f:
            for entry in rag_postedit_data:
                f.write(json.dumps(entry) + "\n")

    # Save results
    results_file = output_dir / "metrics.json"
    print(f"Saving metrics to {results_file}...")

    # Add metadata
    results_with_metadata = {
        "metrics": results,
        "metadata": {
            "split": args.split,
            "baseline": args.baseline,
            "model": baseline_model_name,
            "adapter_path": args.adapter_path,
            "prompt": args.prompt,
            "data_size": len(complex_sentences),
            "example_mode": args.example,
            "batch_size": args.batch_size,
            "do_sample": args.sample or args.num_candidates > 1,
            "temperature": args.temperature if args.sample else None,
            "num_candidates": args.num_candidates,
            "rerank": args.rerank if args.num_candidates > 1 else None,
            "rerank_temperature": (
                args.rerank_temperature
                if args.num_candidates > 1
                else None
            ),
            "rag": args.rag,
            "rag_mode": args.rag_mode if args.rag else None,
            "glossary_path": args.glossary_path if args.rag else None,
            "skip_bertscore": args.skip_bertscore,
            "rephrase_only": rephrase_only,
            "seed": args.seed,
            "run_name": args.run_name,
            "output_dir": str(output_dir),
            "timestamp": datetime.now().isoformat(),
            "evaluation": "SARI, BLEU"
            + ("" if args.skip_bertscore else ", BERTScore")
            + " (automatic metrics)",
        },
    }

    # Add glossary metadata if using RAG (before mode)
    if args.rag and args.rag_mode == "before" and retriever:
        coverage = retriever.get_coverage_stats(
            complex_sentences, max_definitions=args.max_definitions
        )
        results_with_metadata["glossary_coverage"] = {
            "glossary_path": args.glossary_path,
            "max_definitions": args.max_definitions,
            "total_glossary_terms": len(retriever.glossary),
            "sentences_with_matches": coverage["sentences_with_matches"],
            "coverage_rate": coverage["coverage_rate"],
            "avg_matches_per_sentence": coverage["avg_matches_per_sentence"],
            "avg_matches_per_matched_sentence": coverage[
                "avg_matches_per_matched_sentence"
            ],
            "unique_matched_terms": coverage["unique_matched_terms"],
            "top_10_terms": coverage["top_10_terms"],
        }

    # Add RAG post-edit stats if used
    if rag_postedit_stats:
        results_with_metadata["rag_postedit"] = rag_postedit_stats

    with results_file.open("w", encoding="utf-8") as f:
        json.dump(results_with_metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - predictions.txt: Generated simplifications")
    print(f"  - input_output_pairs.txt: Side-by-side comparison")
    print(f"  - metrics.json: All evaluation metrics")
    if args.rag and args.rag_mode == "before":
        print(f"  - glossary_matches.jsonl: Matched terms per sentence")
    if args.rag and args.rag_mode == "after":
        print(f"  - rag_postedit.jsonl: Draft, retrieved terms, and revised output")
    print()


if __name__ == "__main__":
    main()
