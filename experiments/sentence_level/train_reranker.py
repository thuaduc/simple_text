"""Phase 3 of the learned reranker: train + tune the candidate scorer.

Reads the labeled JSONL produced by build_reranker_data.py, trains a pointwise
regressor on reference-free features to predict per-candidate SARI, and selects
hyperparameters by the DOWNSTREAM objective: corpus-style mean oracle-gap
recovered on the val groups (i.e. mean SARI of the candidate the model would
pick). Saves {'model', 'feature_names'} to a pickle loadable by
src.rerank.features.load_reranker_scorer.

Leakage controls:
  - Train ONLY on the train JSONL; tune/report on the val JSONL.
  - Features are reference-free (gold refs were used only to make the SARI label).

Dependency: scikit-learn (pip install scikit-learn). Kept out of pyproject until
the approach proves out on val.

Usage:
  python experiments/sentence_level/train_reranker.py \
    --train cochrane/data/reranker_train.jsonl \
    --val   cochrane/data/reranker_val.jsonl \
    --out   experiments/sentence_level/reranker.pkl
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from src.rerank.features import FEATURE_NAMES, features_to_vector


def load_rows(path):
    groups = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            groups[r["group_id"]].append(r)
    return groups


def to_xy(groups):
    X, y = [], []
    for rows in groups.values():
        for r in rows:
            X.append(features_to_vector(r["features"]))
            y.append(r["sari"])
    return X, y


def selected_mean_sari(model, groups):
    """Mean true-SARI of the candidate the model would pick per group (the
    deployable objective)."""
    total, n = 0.0, 0
    for rows in groups.values():
        if not rows:
            continue
        vecs = [features_to_vector(r["features"]) for r in rows]
        scores = model.predict(vecs)
        best = max(range(len(rows)), key=lambda i: scores[i])
        total += rows[best]["sari"]
        n += 1
    return total / max(1, n)


def oracle_and_baseline(groups):
    """Reference points on the val groups: oracle (max SARI) and the model's
    own cand_idx==0 sample (a greedy-ish proxy)."""
    orc, base, n = 0.0, 0.0, 0
    for rows in groups.values():
        if not rows:
            continue
        orc += max(r["sari"] for r in rows)
        base += next((r["sari"] for r in rows if r["cand_idx"] == 0), rows[0]["sari"])
        n += 1
    return orc / max(1, n), base / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        print("ERROR: scikit-learn required. pip install scikit-learn")
        sys.exit(1)

    train_groups = load_rows(args.train)
    val_groups = load_rows(args.val)
    Xtr, ytr = to_xy(train_groups)
    print(f"Train: {len(train_groups)} groups, {len(Xtr)} candidates")
    print(f"Val:   {len(val_groups)} groups")

    val_oracle, val_base = oracle_and_baseline(val_groups)
    print(f"Val reference points (per-group mean SARI): "
          f"first-sample={val_base:.2f}  oracle={val_oracle:.2f}")

    # Small hyperparameter sweep; select by downstream val selected-SARI.
    best_model, best_score, best_cfg = None, float("-inf"), None
    for n_est in (100, 300):
        for depth in (2, 3):
            for lr in (0.05, 0.1):
                m = GradientBoostingRegressor(
                    n_estimators=n_est, max_depth=depth,
                    learning_rate=lr, random_state=args.seed,
                )
                m.fit(Xtr, ytr)
                s = selected_mean_sari(m, val_groups)
                print(f"  n_est={n_est} depth={depth} lr={lr}: val selected-SARI={s:.2f}")
                if s > best_score:
                    best_model, best_score, best_cfg = m, s, (n_est, depth, lr)

    gap = val_oracle - val_base
    recovered = (best_score - val_base) / gap * 100 if gap > 0 else 0.0
    print(f"\nBest cfg {best_cfg}: val selected-SARI={best_score:.2f} "
          f"(first-sample {val_base:.2f}, oracle {val_oracle:.2f}, "
          f"recovered {recovered:.0f}% of the gap)")
    print("DECISION GATE: promote to a test run only if this clears greedy-val (47.42).")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"model": best_model, "feature_names": FEATURE_NAMES}, f)
    print(f"Saved reranker -> {args.out}")


if __name__ == "__main__":
    main()
