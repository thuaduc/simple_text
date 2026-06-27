"""Fetch and filter external biomedical sentence-simplification data into a
rephrase-only CSV that load_cochrane_sentences()/load_rephrase_csv() can read.

Sources:
  - Med-EASi  (HuggingFace: cbasu/Med-EASi)  Expert  -> Simple
  - PLABA     (OSF project rnpmf, data.json) source sentence -> adaptation sentence

Only 1->1 reword-like pairs are kept (identity copies, splits, deletions and
near-total rewrites are dropped) so the output matches a "rephrase" operator.

Output schema (same columns the Cochrane loader expects, plus `source`):
  pair_id, complex, label, simple, source
where `simple` is the string repr of a one-element list and label == 'rephrase'.

Example:
  python experiments/sentence_level/build_external_rephrase.py \
      --output cochrane/data/external_rephrase_train.csv
"""

import argparse
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from Levenshtein import ratio as lev_ratio

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from src.config import DATA_DIR

# OSF direct-download links for the PLABA archive (project rnpmf).
PLABA_DATA_JSON_URL = "https://osf.io/download/4kp7v/"

# Splits an adapted text into sentences to detect one-to-many "split" operations.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, default=DATA_DIR,
                   help="Where to cache PLABA data.json and write the output CSV by default")
    p.add_argument("--output", type=Path, default=None,
                   help="Output CSV path (default: <data_dir>/external_rephrase_train.csv)")
    p.add_argument("--sources", nargs="+", default=["plaba", "medeasi"],
                   choices=["plaba", "medeasi"], help="Which sources to include")
    p.add_argument("--medeasi_splits", nargs="+", default=["train", "validation", "test"],
                   help="Med-EASi splits to pull (none overlap the Cochrane eval sets)")
    # 1->1 rephrase filters
    p.add_argument("--min_sim", type=float, default=0.30,
                   help="Min char Levenshtein ratio (drop near-total rewrites below this)")
    p.add_argument("--max_sim", type=float, default=0.95,
                   help="Max char Levenshtein ratio (drop near-identical copies above this)")
    p.add_argument("--min_words", type=int, default=4, help="Min words in the complex sentence")
    p.add_argument("--max_words", type=int, default=80, help="Max words in the complex sentence")
    p.add_argument("--min_tgt_words", type=int, default=3, help="Min words in the simplified sentence")
    p.add_argument("--keep_splits", action="store_true",
                   help="Keep 1->many (split) targets instead of dropping them")
    return p.parse_args()


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)


def n_sentences(text: str) -> int:
    parts = [p for p in _SENT_SPLIT.split(text.strip()) if p.strip()]
    return len(parts)


def keep_pair(complex_s: str, simple_s: str, args) -> tuple[bool, str]:
    """Return (keep, reason). reason names the filter that dropped the pair."""
    if not isinstance(complex_s, str) or not isinstance(simple_s, str):
        return False, "empty"
    c = complex_s.strip()
    s = simple_s.strip()
    if not c or not s:
        return False, "empty"
    if c == s:
        return False, "identical"
    cw = len(c.split())
    sw = len(s.split())
    if cw < args.min_words or cw > args.max_words:
        return False, "src_len"
    if sw < args.min_tgt_words:
        return False, "tgt_len"
    if not args.keep_splits and n_sentences(s) > 1:
        return False, "split"
    r = lev_ratio(c, s)
    if r > args.max_sim:
        return False, "too_similar"
    if r < args.min_sim:
        return False, "too_different"
    return True, "ok"


def load_plaba_pairs(data_dir: Path) -> list[tuple[str, str, str]]:
    """Sentence-aligned (pair_id, complex, simple) triples from PLABA data.json."""
    cache = data_dir / "plaba_data.json"
    if not cache.exists():
        _download(PLABA_DATA_JSON_URL, cache)
    data = json.loads(cache.read_text())

    pairs = []
    for qid, node in data.items():
        if not isinstance(node, dict):
            continue
        for pmid, pnode in node.items():
            if pmid in ("question", "question_type") or not isinstance(pnode, dict):
                continue
            abstract = pnode.get("abstract", {})
            adaptations = pnode.get("adaptations", {})
            if not isinstance(abstract, dict) or not isinstance(adaptations, dict):
                continue
            for adapt_name, adapt in adaptations.items():
                if not isinstance(adapt, dict):
                    continue
                # Sentence ids are shared between source and adaptation; a missing
                # id (or empty string) is a deletion, which we skip for rephrase.
                for sid, src in abstract.items():
                    tgt = adapt.get(sid)
                    if not isinstance(src, str) or not isinstance(tgt, str):
                        continue
                    pair_id = f"plaba_{qid}_{pmid}_{adapt_name}_{sid}"
                    pairs.append((pair_id, src, tgt))
    print(f"PLABA: {len(pairs)} raw 1->1 sentence pairs")
    return pairs


def load_medeasi_pairs(splits) -> list[tuple[str, str, str]]:
    """(pair_id, complex, simple) triples from Med-EASi (Expert -> Simple)."""
    from datasets import load_dataset

    ds = load_dataset("cbasu/Med-EASi")
    pairs = []
    for split in splits:
        if split not in ds:
            print(f"  Med-EASi: split '{split}' not found, skipping")
            continue
        for i, row in enumerate(ds[split]):
            idx = row.get("idx", i)
            pair_id = f"medeasi_{split}_{idx}"
            pairs.append((pair_id, row["Expert"], row["Simple"]))
    print(f"Med-EASi: {len(pairs)} raw pairs from splits {splits}")
    return pairs


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output = args.output or (data_dir / "external_rephrase_train.csv")

    raw_pairs: list[tuple[str, str, str, str]] = []  # (pair_id, complex, simple, source)
    if "plaba" in args.sources:
        raw_pairs += [(pid, c, s, "plaba") for pid, c, s in load_plaba_pairs(data_dir)]
    if "medeasi" in args.sources:
        raw_pairs += [(pid, c, s, "medeasi") for pid, c, s in load_medeasi_pairs(args.medeasi_splits)]

    kept = []
    drop_reasons: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    n_dup = 0
    for pair_id, c, s, source in raw_pairs:
        ok, reason = keep_pair(c, s, args)
        if not ok:
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            continue
        key = (c.strip(), s.strip())
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        kept.append({
            "pair_id": pair_id,
            "complex": c.strip(),
            "label": "rephrase",
            "simple": repr([s.strip()]),
            "source": source,
        })

    df = pd.DataFrame(kept, columns=["pair_id", "complex", "label", "simple", "source"])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print("\n=== Summary ===")
    print(f"Raw pairs:     {len(raw_pairs)}")
    print(f"Dropped (dup): {n_dup}")
    for reason, n in sorted(drop_reasons.items(), key=lambda x: -x[1]):
        print(f"Dropped ({reason}): {n}")
    print(f"Kept:          {len(df)}")
    if len(df):
        print("By source:     " + ", ".join(f"{k}={v}" for k, v in df["source"].value_counts().items()))
    print(f"Wrote -> {output}")


if __name__ == "__main__":
    main()
