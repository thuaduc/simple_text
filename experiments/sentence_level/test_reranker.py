"""Offline unit tests for src/rerank (no model/GPU/evaluate needed).

Run: python experiments/sentence_level/test_reranker.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rerank.reranker import (
    fkgl,
    token_f1,
    select_oracle,
    rerank_candidates,
)
from src.rerank.features import (
    FEATURE_NAMES,
    extract_features,
    features_to_vector,
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
hard = "The pharmacological intervention demonstrated statistically significant heterogeneity."
easy = "The drug helped a lot."
check("harder text has higher FKGL", fkgl(hard) > fkgl(easy))

print("== token_f1 ==")
check("identical -> 1.0", abs(token_f1("a b c", "a b c") - 1.0) < 1e-9)
check("disjoint -> 0.0", token_f1("a b", "x y") == 0.0)
check("both empty -> 1.0", token_f1("", "") == 1.0)
check("symmetric", abs(token_f1("a b c", "a b") - token_f1("a b", "a b c")) < 1e-9)

print("== select_oracle (injected fake SARI) ==")
def fake_sari(sources, preds, refs):
    return token_f1(preds[0], refs[0][0]) * 100.0

cands = ["wrong answer", "the correct simple form", "another wrong"]
refs = ["the correct simple form"]
check("oracle picks max-SARI candidate",
      select_oracle("src", cands, refs, sari_fn=fake_sari) == 1)
check("oracle no-refs -> first valid",
      select_oracle("s", ["", "a a", "zzz"], [], sari_fn=fake_sari) == 1)
check("oracle all-empty -> 0",
      select_oracle("s", ["", "  "], ["r"], sari_fn=fake_sari) == 0)

print("== rerank_candidates dispatch ==")
try:
    out = rerank_candidates(["s"], [["a a", "the correct simple form", "no"]],
                            method="oracle", references_list=[["the correct simple form"]])
    check("oracle batch returns one", len(out) == 1 and out[0] != "")
except ImportError:
    print("  SKIP  oracle batch (evaluate not installed)")

# learned dispatch with a toy scorer (prefer longer candidates)
scorer = lambda src, cand, pool, logprob=None: len(cand)
out_l = rerank_candidates(["s", "s"], [["short", "much longer candidate"], ["a", "bb"]],
                          method="learned", scorer=scorer)
check("learned picks argmax scorer", out_l == ["much longer candidate", "bb"])
# logprob threading: scorer that just returns the per-candidate logprob
lp_scorer = lambda src, cand, pool, logprob=None: (logprob or 0.0)
out_lp = rerank_candidates(["s"], [["a", "b", "c"]], method="learned",
                           scorer=lp_scorer, scores_list=[[-2.0, -0.5, -3.0]])
check("learned uses threaded logprob", out_lp == ["b"])

try:
    rerank_candidates(["s"], [["a"]], method="learned")  # missing scorer
    check("learned requires scorer", False)
except ValueError:
    check("learned requires scorer", True)
try:
    rerank_candidates(["s"], [["a"]], method="mbr")  # removed selector
    check("removed selector raises", False)
except ValueError:
    check("removed selector raises", True)

print("== features ==")
src = "We included 16 RCTs (2232 couples) over 5 years."
cand = "We found 16 studies with 2232 couples."
feats = extract_features(src, cand, [cand, "We found 16 studies.", src])
check("feature dict has all names", all(n in feats for n in FEATURE_NAMES))
check("vector length matches schema", len(features_to_vector(feats)) == len(FEATURE_NAMES))
full = extract_features(src, "We found 16 studies, 2232 couples, 5 years.", [cand])
check("all numbers preserved -> consistency 1.0", full["numeric_consistency"] == 1.0)
check("partial numbers -> consistency 2/3", abs(feats["numeric_consistency"] - 2/3) < 1e-9)
bad = extract_features(src, "We found some studies with some couples.", [cand])
check("dropped numbers -> consistency < 1.0", bad["numeric_consistency"] < 1.0)
check("identical-to-source has high token_f1", extract_features(src, src, [src])["token_f1_src"] > 0.99)
check("logprob defaults to 0.0 when absent", feats["logprob"] == 0.0)
check("logprob passes through", extract_features(src, cand, [cand], logprob=-1.5)["logprob"] == -1.5)
check("simpler candidate -> positive fkgl_drop", feats["fkgl_drop"] > 0)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
