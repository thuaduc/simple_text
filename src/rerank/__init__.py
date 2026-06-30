"""Candidate selection + learned reranking for sentence simplification.

The reference-free selectors (mbr, readability, sari_mbr) were removed after all
lost to greedy on val. What remains: the oracle (ceiling + label generation),
scoring primitives reused as features, and the learned-reranker dispatch.
See `src/rerank/reranker.py` and `src/rerank/features.py`.
"""

from src.rerank.reranker import (
    fkgl,
    token_f1,
    quiet_sari,
    select_oracle,
    rerank_candidates,
)

__all__ = [
    "fkgl",
    "token_f1",
    "quiet_sari",
    "select_oracle",
    "rerank_candidates",
]
