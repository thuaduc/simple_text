"""Offline unit tests for src/rerank/reranker.py (no model/GPU/evaluate needed).

Run: python experiments/sentence_level/test_reranker.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rerank.reranker import (
    fkgl,
    token_f1,
    select_mbr,
    select_readability,
    select_oracle,
    rerank_candidates,
)

passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


print("== fkgl ==")
check("empty -> 0.0", fkgl("") == 0.0)
# A long, multisyllabic sentence should score higher than a short simple one.
hard = "The pharmacological intervention demonstrated statistically significant heterogeneity."
easy = "The drug helped a lot."
check("harder text has higher FKGL", fkgl(hard) > fkgl(easy))

print("== token_f1 ==")
check("identical -> 1.0", abs(token_f1("a b c", "a b c") - 1.0) < 1e-9)
check("disjoint -> 0.0", token_f1("a b", "x y") == 0.0)
check("both empty -> 1.0", token_f1("", "") == 1.0)
check("one empty -> 0.0", token_f1("a", "") == 0.0)
check("symmetric", abs(token_f1("a b c", "a b") - token_f1("a b", "a b c")) < 1e-9)

print("== select_mbr (self-consensus) ==")
# Three agree, one outlier -> pick one of the consensus group (not the outlier).
cands = [
    "the dog ran fast",
    "the dog ran fast",
    "the dog ran fast",
    "completely unrelated text here",
]
check("picks consensus, not outlier", select_mbr(cands) != 3)
check("all-empty -> index 0", select_mbr(["", "  ", ""]) == 0)
check("single valid candidate", select_mbr(["", "only one", ""]) == 1)

print("== select_readability ==")
src = "The pharmacological intervention demonstrated statistically significant heterogeneity."
cands = [
    src,  # identical copy -> should be penalized (too-high fidelity / not simpler)
    "The drug worked well in the studies.",  # simpler + faithful-ish
    "",  # degenerate
]
sel = select_readability(src, cands)
check("does not pick the identical copy", sel != 0)
check("does not pick empty", sel != 2)
check("readability all-empty -> 0", select_readability("x", ["", ""]) == 0)

print("== select_oracle (injected fake SARI) ==")
# Fake SARI: reward candidate that matches the reference token set.
def fake_sari(sources, preds, refs):
    return token_f1(preds[0], refs[0][0]) * 100.0

cands = ["wrong answer", "the correct simple form", "another wrong"]
refs = ["the correct simple form"]
idx = select_oracle("src", cands, refs, sari_fn=fake_sari)
check("oracle picks max-SARI candidate", idx == 1)
check(
    "oracle no-refs falls back to mbr",
    select_oracle("s", ["a a", "a a", "zzz"], [], sari_fn=fake_sari) != 2,
)

print("== rerank_candidates (batch) ==")
sources = ["s1", "s2"]
pool = [["good good", "good good", "bad"], ["x y", "x y", "q"]]
out_mbr = rerank_candidates(sources, pool, method="mbr")
check("batch mbr returns one per sentence", len(out_mbr) == 2)
check("batch mbr avoids outliers", out_mbr[0] != "bad" and out_mbr[1] != "q")
# Batch oracle uses the real compute_sari (needs `evaluate`); skip if unavailable.
try:
    out_or = rerank_candidates(
        ["s"], [["no", "the correct simple form", "nope"]],
        method="oracle", references_list=[["the correct simple form"]],
    )
    check("batch oracle returns one prediction", len(out_or) == 1)
except ImportError:
    print("  SKIP  batch oracle (evaluate not installed)")

try:
    rerank_candidates(["s"], [["a"]], method="bogus")
    check("unknown method raises", False)
except ValueError:
    check("unknown method raises", True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
