"""Recalculate detectable paired score gaps with the unchanged v1 power model."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import power  # canonical simulation; intentionally not edited


def _delta_for_points(target, shape):
    lo, hi = 0.0, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if power.avg_points(mid, shape, n=20000) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _mixed_power(delta_logit, trials=200, flips=1200, seed=7):
    """Use the power.py design with 22 two-variant v1 and 20 four-variant v2 clusters."""
    rng = power.random.Random(seed)
    hit = 0
    for _ in range(trials):
        base = [min(0.97, max(0.03, rng.betavariate(0.7, 0.7))) for _ in range(42)]
        runs = []
        for arm, shift in (("incumbent", 0.0), ("challenger", delta_logit)):
            for t, p in enumerate(base):
                q = power.sigmoid(power.logit(p) + shift)
                reps = 2 if t < 22 else 4
                for rep in range(reps):
                    runs.append({"arm": arm, "task": f"family-{t}", "rep": rep,
                                 "resolved": rng.random() < q, "tokens": 1, "seconds": 1})
        scores = power.analyze.per_task_scores(runs)
        diffs = power.analyze.paired_diffs(scores, "challenger", "incumbent")
        p_value = power.analyze.sign_flip_p(diffs, flips, power.random.Random(rng.random()))
        if p_value * 3 < power.analyze.ALPHA and sum(diffs) > 0:
            hit += 1
    return hit / trials


def report(trials=200, flips=2000):
    specs = {
        "U-shaped task difficulty": (0.7, 0.7),
        "pilot-filtered difficulty": (2.0, 2.0),
    }
    # Variants within one family are repeated measurements, not new independent units.
    # V1: 22 family clusters x 2 variants; v2 total: 42 clusters x 4 variants.
    for label, shape in specs.items():
        print(f"\n{label}")
        for clusters, reps, cases in ((22, 2, 44),):
            for target in (5, 10):
                delta = _delta_for_points(target, shape)
                p = power.power(delta, clusters, reps=reps, shape=shape, contrasts=3,
                                trials=trials, flips=flips, boot=100)
                print(f"clusters={clusters:2d} ({cases} cases), gap={target:2d} pp: power={p:.3f}")
        for target in (5, 10):
            delta = _delta_for_points(target, shape)
            p = _mixed_power(delta, trials=trials, flips=flips)
            print(f"clusters=42 (124 cases; 22x2 + 20x4), gap={target:2d} pp: power={p:.3f}")
        # A coarse empirical 80%-power detectable-gap search, in score points.
        lo, hi = 0.0, 50.0
        for _ in range(7):
            mid = (lo + hi) / 2
            delta = _delta_for_points(mid, shape)
            p = _mixed_power(delta, trials=trials, flips=flips)
            if p < 0.8:
                lo = mid
            else:
                hi = mid
        print(f"42 clusters: empirical ~80%-power gap ≈ {hi:.1f} pp")


if __name__ == "__main__":
    report()
