"""Candidate selection for sentence simplification.

History / scope:
  The reference-free selectors (token-F1 MBR, readability filter, SARI-as-utility
  MBR) were all evaluated on val and **all lost to greedy decoding** (see
  `outputs/simpletext-task1-rerank-experiment.md`), so they were removed. What
  remains is:

  - `select_oracle`  — max-SARI candidate per sentence using gold references.
    NOT deployable; used to (a) measure the achievable ceiling and (b) produce
    training labels for the learned reranker.
  - `rerank_candidates` — dispatch supporting 'oracle' and 'learned'. The
    'learned' path takes a trained scorer (see `experiments/sentence_level/
    train_reranker.py`) that ranks candidates from reference-free features.

  The scoring primitives `fkgl`, `token_f1`, and `_quiet_sari` are retained
  because they are reused as features (`src/rerank/features.py`) and for label
  computation, not as standalone selectors.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENT_RE = re.compile(r"[.!?]+")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")


# --------------------------------------------------------------------------- #
# Readability: Flesch-Kincaid Grade Level (lightweight, dependency-free)
# --------------------------------------------------------------------------- #
def _count_syllables(word: str) -> int:
    """Heuristic syllable count for a single word (>=1)."""
    word = word.lower()
    if not word:
        return 0
    groups = _VOWEL_GROUP_RE.findall(word)
    count = len(groups)
    # Common silent trailing 'e' (e.g. "make" -> 1 not 2), but keep words like "the".
    if word.endswith("e") and not word.endswith(("le", "ee", "ie")) and count > 1:
        count -= 1
    return max(1, count)


def fkgl(text: str) -> float:
    """Flesch-Kincaid Grade Level. Higher = harder. Returns 0.0 for empty text.

    FKGL = 0.39*(words/sentences) + 11.8*(syllables/words) - 15.59
    """
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_sentences = max(1, len([s for s in _SENT_RE.split(text) if s.strip()]))
    n_syllables = sum(_count_syllables(w) for w in words)
    return (
        0.39 * (n_words / n_sentences)
        + 11.8 * (n_syllables / n_words)
        - 15.59
    )


# --------------------------------------------------------------------------- #
# Token-overlap utility (reused as a feature)
# --------------------------------------------------------------------------- #
def token_f1(a: str, b: str) -> float:
    """Token-overlap F1 between two strings (case-insensitive). Reference-free."""
    ta = _WORD_RE.findall(a.lower())
    tb = _WORD_RE.findall(b.lower())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    from collections import Counter

    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(tb)
    recall = overlap / len(ta)
    return 2 * precision * recall / (precision + recall)


def _non_degenerate(candidates: Sequence[str]) -> List[int]:
    """Indices of candidates that are non-empty after stripping."""
    return [i for i, c in enumerate(candidates) if c and c.strip()]


# --------------------------------------------------------------------------- #
# SARI helper (labels + oracle)
# --------------------------------------------------------------------------- #
_QUIET_SARI_METRIC = None


def quiet_sari(sources, predictions, references) -> float:
    """Compute SARI via `evaluate` without the per-call INFO logging in
    src.evaluation.metrics (called K times per sentence for labels/oracle)."""
    global _QUIET_SARI_METRIC
    if _QUIET_SARI_METRIC is None:
        import evaluate

        _QUIET_SARI_METRIC = evaluate.load("sari")
    return _QUIET_SARI_METRIC.compute(
        sources=sources, predictions=predictions, references=references
    )["sari"]


# --------------------------------------------------------------------------- #
# Oracle (ceiling + label generation; NOT deployable)
# --------------------------------------------------------------------------- #
def select_oracle(
    source: str,
    candidates: Sequence[str],
    references: Sequence[str],
    sari_fn: Optional[Callable[[List[str], List[str], List[List[str]]], float]] = None,
) -> int:
    """Index of the candidate with the highest per-sentence SARI vs gold refs.

    Requires gold `references` (uses the answer key) -> ceiling only, also used
    to label training data for the learned reranker. `sari_fn` defaults to a
    quiet `evaluate` SARI.
    """
    valid = _non_degenerate(candidates)
    if not valid:
        return 0
    if sari_fn is None:
        sari_fn = quiet_sari

    refs = [r for r in references if r and r.strip()]
    if not refs:
        return valid[0]

    best_idx = valid[0]
    best_score = float("-inf")
    for i in valid:
        score = sari_fn([source], [candidates[i]], [refs])
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def rerank_candidates(
    sources: Sequence[str],
    candidates_list: Sequence[Sequence[str]],
    method: str = "oracle",
    references_list: Optional[Sequence[Sequence[str]]] = None,
    scorer: Optional[Callable[..., float]] = None,
    scores_list: Optional[Sequence[Sequence[float]]] = None,
) -> List[str]:
    """Select one prediction per sentence from a candidate pool.

    Args:
        sources: complex input sentences (n,)
        candidates_list: per-sentence candidate lists (n, k)
        method: 'oracle' (gold-ref ceiling) | 'learned' (trained scorer)
        references_list: gold references per sentence (required for 'oracle')
        scorer: for method='learned', a fn (source, candidate, pool) -> score;
            the argmax-scoring candidate per sentence is selected.

    Returns:
        list of selected predictions (n,)
    """
    if len(sources) != len(candidates_list):
        raise ValueError(
            f"sources ({len(sources)}) and candidates_list "
            f"({len(candidates_list)}) must align"
        )
    if method == "oracle" and references_list is None:
        raise ValueError("method='oracle' requires references_list")
    if method == "learned" and scorer is None:
        raise ValueError("method='learned' requires a scorer callable")

    selected: List[str] = []
    for idx, (src, cands) in enumerate(zip(sources, candidates_list)):
        cands = list(cands)
        if not cands:
            selected.append("")
            continue
        if method == "oracle":
            j = select_oracle(src, cands, references_list[idx])
        elif method == "learned":
            valid = _non_degenerate(cands)
            pool = valid if valid else list(range(len(cands)))
            lp = scores_list[idx] if scores_list is not None else None
            j = max(
                pool,
                key=lambda i: scorer(
                    src, cands[i], cands, lp[i] if lp is not None else None
                ),
            )
        else:
            raise ValueError(f"unknown rerank method: {method}")
        selected.append(cands[j])
    return selected
