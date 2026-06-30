"""Phase 1 of the learned reranker: build labeled candidate data.

For each sentence in a split, generate K candidates from the frozen QLoRA
adapter (sampling), compute each candidate's TRUE SARI vs its gold reference
(label), and extract reference-free features. Writes one JSONL row per
(sentence, candidate).

Gold references are used ONLY to compute the SARI label, never as a feature, so
a model trained on these rows is deployable.

Usage:
  python experiments/sentence_level/build_reranker_data.py \
    --split train --num_candidates 8 --temperature 0.7 \
    --adapter_path experiments/sentence_level/lora_adapter/qwen35-2b-best/checkpoint-328 \
    --output cochrane/data/reranker_train.jsonl

Run once for --split train and once for --split val. Re-use a fixed --seed so
train/val feature distributions match the eventual test run.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from src.config import DATA_DIR, RANDOM_SEED
from src.utils.data_loader import load_cochrane_sentences
from src.rerank.features import extract_features
from src.rerank.reranker import quiet_sari


def parse_args():
    p = argparse.ArgumentParser(description="Build labeled reranker training data")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--num_candidates", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--adapter_path", type=str, required=True)
    p.add_argument("--prompt", type=str, default="default_zero_shot")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--test_size", type=int, default=None, help="limit #sentences")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--data_dir", type=str, default=DATA_DIR)
    p.add_argument("--no_scores", action="store_true",
                   help="skip model log-prob feature (faster, no transition scores)")
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Loading {args.split} (rephrase-only)...")
    sources, references, labels, pair_ids = load_cochrane_sentences(
        split=args.split, data_dir=args.data_dir, rephrase_only=True
    )
    if args.test_size is not None:
        sources = sources[: args.test_size]
        references = references[: args.test_size]
        pair_ids = pair_ids[: args.test_size]
    print(f"{len(sources)} sentences; generating {args.num_candidates} candidates each")

    from src.models.sentence_simplifier import SentenceSimplifier

    simplifier = SentenceSimplifier(
        prompt_name=args.prompt, adapter_path=args.adapter_path,
        do_sample=True, temperature=args.temperature,
    )

    out = simplifier.simplify_candidates_batch(
        sources, num_candidates=args.num_candidates,
        batch_size=args.batch_size, temperature=args.temperature,
        return_scores=not args.no_scores,
    )
    if args.no_scores:
        candidates_list, scores_list = out, [[None] * args.num_candidates for _ in out]
    else:
        candidates_list, scores_list = out

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_groups_with_ref = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for gi, (src, cands, scores, refs, pid) in enumerate(
            zip(sources, candidates_list, scores_list, references, pair_ids)
        ):
            clean_refs = [r for r in refs if r and r.strip()]
            if not clean_refs:
                continue  # cannot label without a reference
            n_groups_with_ref += 1

            # Per-candidate true SARI label.
            saris = [
                quiet_sari([src], [c if c.strip() else ""], [clean_refs])
                for c in cands
            ]
            best_idx = max(range(len(cands)), key=lambda i: saris[i])

            for ci, cand in enumerate(cands):
                feats = extract_features(src, cand, cands, logprob=scores[ci])
                row = {
                    "group_id": gi,
                    "pair_id": pid,
                    "cand_idx": ci,
                    "source": src,
                    "candidate": cand,
                    "sari": saris[ci],
                    "is_oracle_best": int(ci == best_idx),
                    "features": feats,
                }
                f.write(json.dumps(row) + "\n")
                n_rows += 1

    print(f"\nWrote {n_rows} rows from {n_groups_with_ref} labeled groups -> {args.output}")
    print("Next: experiments/sentence_level/train_reranker.py (Phase 3)")


if __name__ == "__main__":
    main()
