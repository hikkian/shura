"""GEVAL1: the acceptance rule with the general-v1 gate (reports/geval1/DATASET-GENERAL.md §5).

A harness/config edit for a CODE model (Tiel, Sage) is accepted when the code dev verdict of rsi.accept.decide accepts AND the
general dev score does not fall beyond noise: the family-cluster 95 % interval of the paired general score difference
(candidate − base) has its lower bound above −NOISE. For a GENERAL model (Gemma) the roles are swapped: general must improve
(interval above 0, resolved not worse), code must not regress beyond the same noise.

NOISE = 0.05 on the mean score (the repeat-to-repeat spread of general-v1 is unknown until the first GPU run: the value is a
pre-registered stub and is reported as such in every verdict; the first run's two repeats replace it with the measured spread).
"""
from collections import defaultdict

from rsi.accept import decide, paired_bootstrap

NOISE = 0.05
NOISE_SOURCE = "STUB (pre-registered 0.05 until the first general-v1 GPU run measures the repeat spread)"


def _pairs(candidate, base):
    b = {(r["task"], r.get("rep", 1)): r for r in base}
    score, res = [], []
    for c in candidate:
        r = b.get((c["task"], c.get("rep", 1)))
        if r is None:
            continue
        fam = c.get("family") or c["task"].rsplit("-general-", 1)[0]
        score.append((fam, float(c.get("score") or 0.0) - float(r.get("score") or 0.0)))
        res.append((fam, float(bool(c.get("resolved"))) - float(bool(r.get("resolved")))))
    return score, res


def general_gate(candidate, base, noise=NOISE, improve=False):
    """candidate/base: general-v1 result dicts (judge scores already attached). improve=True demands a gain, else only no-regression."""
    score, res = _pairs(candidate, base)
    if not score:
        return {"ok": False, "reason": "no paired general tasks", "pairs": 0, "noise": noise, "noise_source": NOISE_SOURCE}
    s_mean, s_lo, s_hi = paired_bootstrap(score)
    r_mean, r_lo, r_hi = paired_bootstrap(res)
    ok = (s_lo > 0 and r_lo > -0.03) if improve else (s_lo > -noise)
    return {"ok": bool(ok), "score": {"mean": s_mean, "ci95": [s_lo, s_hi]}, "resolved": {"mean": r_mean, "ci95": [r_lo, r_hi]},
            "pairs": len(score), "families": len({f for f, _ in score}), "mode": "improve" if improve else "no-regression",
            "noise": noise, "noise_source": NOISE_SOURCE}


def decide_with_general(code_candidate, code_base, general_candidate, general_base, *, model_kind="code", thresholds=None,
                        allow_stub=False, noise=NOISE):
    """model_kind: "code" (Tiel, Sage: code decides, general must not regress) or "general" (Gemma: general decides, code must not regress)."""
    if model_kind == "code":
        primary = decide(code_candidate, code_base, thresholds, allow_stub)
        guard = general_gate(general_candidate, general_base, noise, improve=False)
    else:
        primary = general_gate(general_candidate, general_base, noise, improve=True)
        guard = decide(code_candidate, code_base, thresholds, allow_stub)
        # for a general model the code verdict is only a no-regression guard: its score interval must not be below -noise
        guard = dict(guard, ok=bool(guard.get("score", {}).get("ci95", [-1])[0] > -noise), role="guard (no code regression)")
    accept = bool(primary.get("accept", primary.get("ok")) and guard.get("ok"))
    return {"accept": accept, "model_kind": model_kind, "primary": primary, "guard": guard,
            "rule": ("code dev accepted by rsi.accept.decide AND general dev score CI low > -noise" if model_kind == "code"
                     else "general dev score CI low > 0 and resolved CI low > -0.03 AND code dev score CI low > -noise")}
