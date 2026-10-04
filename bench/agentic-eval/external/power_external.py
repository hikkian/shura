"""Paired sign-flip power for public task sets, using the existing power.py model."""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import power

SETS = {
    "Aider Polyglot Python": 34,
    "ShuraCode subset 40": 40,
    "ShuraCode subset 50": 50,
    "Aider Polyglot validated subset": 92,
    "Aider Polyglot JavaScript": 49,
    "Aider Polyglot Python+JavaScript": 83,
    "Aider Polyglot full": 225,
    "BFCL simple_python": 400,
    "BFCL multi_turn_base": 200,
}


def delta_for_points(points: float, shape: tuple[float, float]) -> float:
    lo, hi = 0.0, 6.0
    for _ in range(44):
        mid = (lo + hi) / 2
        if power.avg_points(mid, shape, n=20000) < points:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _mde(n: int, shape: tuple[float, float], trials: int, flips: int, contrasts: int) -> float:
    lo, hi = 0.0, 45.0
    for _ in range(6):
        mid = (lo + hi) / 2
        p = power.power(delta_for_points(mid, shape), n, reps=1, shape=shape,
                        contrasts=contrasts, trials=trials, flips=flips, boot=100)
        if p < 0.8:
            lo = mid
        else:
            hi = mid
    return hi


def report(trials: int = 80, flips: int = 1000, contrasts: int = 4) -> list[dict]:
    shapes = {
        "broad difficulty mix": (0.7, 0.7),
        "middle difficulty mix": (2.0, 2.0),
    }
    rows = []
    for benchmark, n in SETS.items():
        for shape_name, shape in shapes.items():
            p5 = power.power(delta_for_points(5, shape), n, reps=1, shape=shape,
                             contrasts=contrasts, trials=trials, flips=flips, boot=100)
            p10 = power.power(delta_for_points(10, shape), n, reps=1, shape=shape,
                              contrasts=contrasts, trials=trials, flips=flips, boot=100)
            mde = _mde(n, shape, trials, flips, contrasts) if shape_name == "broad difficulty mix" else None
            row = {"benchmark": benchmark, "independent_tasks": n, "shape": shape_name,
                   "power_5pp": round(p5, 3), "power_10pp": round(p10, 3),
                   "mde_80pp": round(mde, 1) if mde is not None else None}
            rows.append(row)
            mde_text = f" 80%MDE≈{mde:.1f}pp" if mde is not None else ""
            print(f"{benchmark:35s} n={n:3d} {shape_name:24s} 5pp={p5:.3f} 10pp={p10:.3f}{mde_text}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--flips", type=int, default=1000)
    parser.add_argument("--contrasts", type=int, default=4)
    args = parser.parse_args()
    report(args.trials, args.flips, args.contrasts)
