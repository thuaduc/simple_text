"""Reference-free candidate reranking for sentence simplification.

Implements the "candidate generation + reranking" plan
(outputs/simpletext-task1-improvement.md):

  - MBR self-consensus  (`select_mbr`): pick the candidate most similar, on
    average, to the other candidates. Reference-free; uses token-F1 utility by
    default (no model load) or any pluggable utility function.
  - Readability-filter-then-similarity  (`select_readability`, EhiMeNLP / TSAR
    2025 style): drop empty/degenerate candidates, keep those that get simpler
    (FKGL drop vs the source within a band), then rank survivors by similarity
    to the *source* to preserve meaning.
  - Oracle  (`select_oracle`): pick the max-SARI candidate per sentence using
    the gold references. NOT a deployable system; used only to measure the
    achievable ceiling on `val` before investing further (decision gate in the
    plan).

Design notes:
  - No new dependencies. FKGL uses a lightweight syllable heuristic; token-F1
    is pure-Python. BERTScore can be plugged in as a utility/similarity fn when
    a GPU run wants semantic scoring.
  - All selectors return an index into the candidate list and never raise on
    empty pools (they fall back to the first candidate / empty string).
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
# Similarity utilities
# --------------------------------------------------------------------------- #
def token_f1(a: str, b: str) -> float:
    """Token-overlap F1 between two strings (case-insensitive). Reference-free."""
    ta = _WORD_RE.findall(a.lower())
    tb = _WORD_RE.findall(b.lower())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    # Multiset overlap.
    from collections import Counter

    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(tb)
    recall = overlap / len(ta)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Selectors (return an index into `candidates`)
# --------------------------------------------------------------------------- #
def _non_degenerate(candidates: Sequence[str]) -> List[int]:
    """Indices of candidates that are non-empty after stripping."""
    return [i for i, c in enumerate(candidates) if c and c.strip()]


def select_mbr(
    candidates: Sequence[str],
    utility: Callable[[str, str], float] = token_f1,
) -> int:
    """MBR self-consensus: index of the candidate with the highest mean
    utility to the *other* candidates. Ties break toward the earlier index.
    """
    valid = _non_degenerate(candidates)
    if not valid:
        return 0
    if len(valid) == 1:
        return valid[0]

    best_idx = valid[0]
    best_score = float("-inf")
    for i in valid:
        others = [j for j in valid if j != i]
        score = sum(utility(candidates[i], candidates[j]) for j in others) / len(others)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def select_readability(
    source: str,
    candidates: Sequence[str],
    similarity: Callable[[str, str], float] = token_f1,
    min_fidelity: float = 0.30,
    max_fidelity: float = 0.97,
) -> int:
    """Readability-filter-then-similarity (EhiMeNLP / TSAR 2025 style).

    1. Drop empty/degenerate candidates.
    2. Keep candidates that are simpler than the source (FKGL drop). If none
       qualify, fall back to all non-degenerate candidates.
    3. Among survivors, prefer fidelity to the source within
       [min_fidelity, max_fidelity]: this rewards meaning preservation while
       rejecting near-identical copies (the SARI identity-copy trap) and total
       rewrites. Rank by similarity-to-source, clipped to the band.
    """
    valid = _non_degenerate(candidates)
    if not valid:
        return 0
    if len(valid) == 1:
        return valid[0]

    src_fkgl = fkgl(source)
    simpler = [i for i in valid if fkgl(candidates[i]) < src_fkgl]
    pool = simpler if simpler else valid

    def banded_score(i: int) -> float:
        sim = similarity(source, candidates[i])
        # Penalize candidates outside the fidelity band (too-identical / too-divergent).
        if sim > max_fidelity:
            return max_fidelity - (sim - max_fidelity)  # push down near-copies
        if sim < min_fidelity:
            return sim  # already low; keep raw (worst)
        return sim

    best_idx = pool[0]
    best_score = float("-inf")
    for i in pool:
        score = banded_score(i)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def select_oracle(
    source: str,
    candidates: Sequence[str],
    references: Sequence[str],
    sari_fn: Optional[Callable[[List[str], List[str], List[List[str]]], float]] = None,
) -> int:
    """Oracle: index of the candidate with the highest per-sentence SARI.

    Requires gold `references`. Used ONLY to measure the achievable ceiling
    (decision gate). `sari_fn` defaults to src.evaluation.metrics.compute_sari.
    """
    valid = _non_degenerate(candidates)
    if not valid:
        return 0
    if sari_fn is None:
        from src.evaluation.metrics import compute_sari as sari_fn  # lazy import

    refs = [r for r in references if r and r.strip()]
    if not refs:
        # No references to score against; fall back to MBR consensus.
        return select_mbr(candidates)

    best_idx = valid[0]
    best_score = float("-inf")
    for i in valid:
        score = sari_fn([source], [candidates[i]], [refs])
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def rerank_candidates(
    sources: Sequence[str],
    candidates_list: Sequence[Sequence[str]],
    method: str = "mbr",
    references_list: Optional[Sequence[Sequence[str]]] = None,
    similarity: Callable[[str, str], float] = token_f1,
) -> List[str]:
    """Select one prediction per sentence from a pool of candidates.

    Args:
        sources: complex input sentences (n,)
        candidates_list: per-sentence candidate lists (n, k)
        method: 'mbr' | 'readability' | 'oracle'
        references_list: gold references per sentence (required for 'oracle')
        similarity: utility/similarity function (token_f1 default; pass a
            BERTScore-backed fn for semantic scoring on GPU runs)

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

    selected: List[str] = []
    for idx, (src, cands) in enumerate(zip(sources, candidates_list)):
        cands = list(cands)
        if not cands:
            selected.append("")
            continue
        if method == "mbr":
            j = select_mbr(cands, utility=similarity)
        elif method == "readability":
            j = select_readability(src, cands, similarity=similarity)
        elif method == "oracle":
            j = select_oracle(src, cands, references_list[idx])
        else:
            raise ValueError(f"unknown rerank method: {method}")
        selected.append(cands[j])
    return selected
