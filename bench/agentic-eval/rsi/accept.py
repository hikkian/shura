"""RSIL0: acceptance of a harness edit by OUR measurements, not by an editor's opinion (RSIL1 card, step 3).

Paired comparison on the dev part: for every (task, repetition) the score and the resolved flag of the candidate minus the base.
The uncertainty comes from resampling FAMILIES (a family is the unit of the split), so the interval is honest about the few clusters.

THRESHOLDS below are stubs: reports/rsit1/DATASET.md does not exist yet (checked in RSIL0). They are marked as such in every verdict,
and the driver refuses to accept on them unless `allow_stub=True` is given explicitly.
"""
import random
from collections import defaultdict
from pathlib import Path

DATASET = Path("/home/alisher/Desktop/AI/reports/rsit1/DATASET.md")


def load_thresholds(path=DATASET):
    """The acceptance rule of reports/rsit1/DATASET.md §5, as the numbers it states: the paired score difference must be above zero with
    its 95 % interval, the resolved difference must be above -0.03 with its interval, and every task needs at least two repeats (the
    pilot checks the repeats). The cost limit (1.15) is from the RSIL1 card: §5 does not state it. Missing file: FileNotFoundError."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: the acceptance thresholds are not known (only --fake may run on stubs)")
    text = p.read_text(encoding="utf-8")
    if "## 5. Правило принятия" not in text or "0.03" not in text:
        raise ValueError(f"{p}: the §5 acceptance rule is not in the form this driver reads; check it before a real run")
    return {"score_ci_low_min": 0.0, "resolved_ci_low_min": -0.03, "cost_growth_max": 1.15, "time_growth_max_significant": 1.35,
            "repeats_min": 2, "source": f"{p} §5"}


STUB = {
    "score_ci_low_min": 0.0,       # candidate - base mean score: the 95 % interval must be above this
    "resolved_ci_low_min": -0.03,  # resolved must not be worse (lower bound above this)
    "cost_growth_max": 1.15,       # time and tokens per task may grow at most by this factor
    "time_growth_max_significant": 1.35,   # time may grow to this only when the score is significantly better and resolved is not worse
    "source": "STUB (reports/rsit1/DATASET.md not found)",
}


def paired_bootstrap(pairs, reps=2000, seed=20261021, alpha=0.05):
    """pairs: list of (family, diff). Returns (mean, low, high) with the family-cluster bootstrap at 1 - alpha (95 % by default;
    RSI2D's decide2 passes a smaller alpha for its Bonferroni-corrected one-sided bound)."""
    by_family = defaultdict(list)
    for fam, d in pairs:
        by_family[fam].append(d)
    fams = sorted(by_family)
    if not fams:
        return 0.0, 0.0, 0.0
    means = [sum(by_family[f]) / len(by_family[f]) for f in fams]
    mean = sum(means) / len(means)
    rng = random.Random(seed)
    boots = []
    for _ in range(reps):
        sample = [means[rng.randrange(len(fams))] for _ in fams]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * (reps - 1))]
    hi = boots[int((1 - alpha / 2) * (reps - 1))]
    return round(mean, 4), round(lo, 4), round(hi, 4)


def _key(r):
    return (r["task"], r.get("rep", 1))


def decide(candidate, base, thresholds=None, allow_stub=False):
    """candidate and base: lists of result dicts {task, family, score, resolved, seconds, tokens}. Returns a verdict dict."""
    th = thresholds or STUB
    if th is STUB and not allow_stub:
        raise RuntimeError("acceptance uses STUB thresholds: pass allow_stub=True to decide, and say so in the report")
    b = {_key(r): r for r in base}
    pairs_score, pairs_res, time_ratio, tok_ratio = [], [], [], []
    per_family = defaultdict(list)
    for c in candidate:
        r = b.get(_key(c))
        if r is None:
            continue
        pairs_score.append((c["family"], c["score"] - r["score"]))
        pairs_res.append((c["family"], float(bool(c["resolved"])) - float(bool(r["resolved"]))))
        per_family[c["family"]].append(c["score"] - r["score"])
        if r.get("seconds"):
            time_ratio.append(c["seconds"] / r["seconds"])
        if r.get("tokens"):
            tok_ratio.append(c.get("tokens", 0) / r["tokens"])
    if not pairs_score:
        return {"accept": False, "reason": "no paired tasks", "stub": th is STUB}
    s_mean, s_lo, s_hi = paired_bootstrap(pairs_score)
    r_mean, r_lo, r_hi = paired_bootstrap(pairs_res)
    t = sum(time_ratio) / len(time_ratio) if time_ratio else 1.0
    k = sum(tok_ratio) / len(tok_ratio) if tok_ratio else 1.0
    regressions = {f: round(sum(v) / len(v), 4) for f, v in per_family.items() if sum(v) / len(v) < -0.1}
    ok_score = s_lo > th["score_ci_low_min"]
    ok_res = r_lo > th["resolved_ci_low_min"]
    # the time limit is wider (1.35) only when the score is significantly better and resolved is not worse (orchestrator's decision,
    # RSIL0d); the token limit stays at 1.15
    time_max = th.get("time_growth_max_significant", th["cost_growth_max"]) if (ok_score and ok_res) else th["cost_growth_max"]
    ok_cost = t <= time_max and k <= th["cost_growth_max"]
    verdict = {
        "accept": bool(ok_score and ok_res and ok_cost and not regressions),
        "score": {"mean": s_mean, "ci95": [s_lo, s_hi], "ok": ok_score},
        "resolved": {"mean": r_mean, "ci95": [r_lo, r_hi], "ok": ok_res},
        "time_ratio": round(t, 3), "tokens_ratio": round(k, 3), "time_limit": time_max, "cost_ok": ok_cost,
        "family_regressions": regressions, "pairs": len(pairs_score), "stub": th is STUB, "thresholds": th.get("source", "given"),
    }
    return verdict
