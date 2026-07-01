"""Reference-free features for the learned reranker.

Every feature here is computable at inference time WITHOUT gold references, so a
model trained on these can be deployed. Gold references are used only to *label*
training rows (true SARI), never as a feature.

`extract_features(source, candidate, pool, logprob=None)` -> dict
`FEATURE_NAMES` -> ordered feature list
`features_to_vector(feats)` -> list[float] in FEATURE_NAMES order
`load_reranker_scorer(path)` -> callable(source, candidate, pool) -> float
"""

from __future__ import annotations

import difflib
import math
import re
from typing import Dict, List, Optional, Sequence

from src.rerank.reranker import fkgl, token_f1, _count_syllables, _WORD_RE

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _bigrams(toks: Sequence[str]):
    return set(zip(toks, toks[1:]))


def _levenshtein_ratio(a: str, b: str) -> float:
    """difflib similarity ratio in [0,1] (stdlib, no new dependency)."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def _numeric_consistency(source: str, candidate: str) -> float:
    """Fraction of source numbers preserved in the candidate (1.0 if source has
    no numbers). Guards against fluent-but-wrong candidates that drop/alter
    clinical figures."""
    src_nums = set(_NUM_RE.findall(source))
    if not src_nums:
        return 1.0
    cand_nums = set(_NUM_RE.findall(candidate))
    return len(src_nums & cand_nums) / len(src_nums)


# Ordered feature schema (keep stable; the trained model depends on this order).
FEATURE_NAMES: List[str] = [
    "token_f1_src",        # token-F1 candidate vs source (fidelity)
    "lev_ratio_src",       # char-level similarity vs source
    "len_ratio",           # cand words / source words
    "len_diff_abs",        # |cand words - source words|
    "frac_added",          # added tokens / source tokens (vs source set)
    "frac_deleted",        # deleted tokens / source tokens
    "frac_kept",           # kept tokens / source tokens
    "bigram_novelty",      # cand bigrams not in source / cand bigrams
    "cand_fkgl",           # candidate Flesch-Kincaid grade
    "fkgl_drop",           # source FKGL - candidate FKGL (positive = simpler)
    "cand_wordcount",      # candidate length in words
    "mean_syll",           # mean syllables/word (candidate)
    "complex_frac",        # fraction of words with >=3 syllables
    "numeric_consistency", # source numbers preserved in candidate
    "pool_consensus",      # mean token-F1 to the other candidates
    "len_rank",            # normalized rank of candidate length within pool
    "logprob",             # length-normalized model log-prob (0.0 if unavailable)
]


def extract_features(
    source: str,
    candidate: str,
    pool: Sequence[str],
    logprob: Optional[float] = None,
) -> Dict[str, float]:
    """Compute the reference-free feature dict for one candidate."""
    s_tok = _tokens(source)
    c_tok = _tokens(candidate)
    s_set, c_set = set(s_tok), set(c_tok)
    n_src = max(1, len(s_tok))

    kept = len(s_set & c_set)
    added = len(c_set - s_set)
    deleted = len(s_set - c_set)

    c_bg = _bigrams(c_tok)
    bigram_novelty = (
        len(c_bg - _bigrams(s_tok)) / len(c_bg) if c_bg else 0.0
    )

    syll = [_count_syllables(w) for w in c_tok] or [0]
    mean_syll = sum(syll) / len(syll)
    complex_frac = (sum(1 for x in syll if x >= 3) / len(syll)) if c_tok else 0.0

    src_fkgl = fkgl(source)
    cand_fkgl = fkgl(candidate)

    # Pool-relative features.
    others = [p for p in pool if p is not candidate]
    if others:
        pool_consensus = sum(token_f1(candidate, p) for p in others) / len(others)
        lengths = sorted(len(_tokens(p)) for p in pool)
        rank = sum(1 for L in lengths if L < len(c_tok))
        len_rank = rank / max(1, len(pool) - 1)
    else:
        pool_consensus = 1.0
        len_rank = 0.0

    return {
        "token_f1_src": token_f1(candidate, source),
        "lev_ratio_src": _levenshtein_ratio(source, candidate),
        "len_ratio": len(c_tok) / n_src,
        "len_diff_abs": abs(len(c_tok) - len(s_tok)),
        "frac_added": added / n_src,
        "frac_deleted": deleted / n_src,
        "frac_kept": kept / n_src,
        "bigram_novelty": bigram_novelty,
        "cand_fkgl": cand_fkgl,
        "fkgl_drop": src_fkgl - cand_fkgl,
        "cand_wordcount": float(len(c_tok)),
        "mean_syll": mean_syll,
        "complex_frac": complex_frac,
        "numeric_consistency": _numeric_consistency(source, candidate),
        "pool_consensus": pool_consensus,
        "len_rank": len_rank,
        "logprob": float(logprob) if logprob is not None else 0.0,
    }


def _finite(x: float) -> float:
    """Coerce NaN/inf to 0.0 (the same default used for missing features).
    Guards the trained estimator, which rejects non-finite inputs."""
    x = float(x)
    return x if math.isfinite(x) else 0.0


def features_to_vector(feats: Dict[str, float]) -> List[float]:
    """Flatten a feature dict to a vector in FEATURE_NAMES order."""
    return [_finite(feats.get(name, 0.0)) for name in FEATURE_NAMES]


def load_reranker_scorer(path: str):
    """Load a trained reranker and return a scorer(source, candidate, pool)->float.

    The model file is a joblib/pickle dump of {'model': estimator,
    'feature_names': [...]}. Estimator must expose predict() (regression score)
    or decision_function(); higher = better candidate. The returned scorer takes
    an optional ``logprob`` so the inference path can supply the same model
    log-prob feature used at training time (run_baseline passes it when
    --rerank learned).
    """
    import pickle

    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    names = bundle.get("feature_names", FEATURE_NAMES)

    predict = getattr(model, "predict", None) or getattr(model, "decision_function")

    def scorer(source, candidate, pool, logprob=None) -> float:
        feats = extract_features(source, candidate, pool, logprob=logprob)
        vec = [_finite(feats.get(n, 0.0)) for n in names]
        return float(predict([vec])[0])

    return scorer
