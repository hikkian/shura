"""Power of the candidate designs (how often a real difference of `delta` points is detected), by simulation.
    python3 bench/agentic-eval/power.py
A design is: how many tasks, how many repetitions, how spread out the task difficulty is, and how many comparisons are
corrected for. Planned contrasts against ONE incumbent (3 tests, Holm over 3) are far more powerful than 'beat every other
arm' (6 tests and three conditions at once)."""
import math
import random

import analyze


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def logit(p):
    return math.log(p / (1 - p))


def power(delta_logit, tasks, reps, shape, contrasts, trials=300, seed=7, flips=2000, boot=400):
    rng = random.Random(seed)
    hit = 0
    for _ in range(trials):
        base = [min(0.97, max(0.03, rng.betavariate(*shape))) for _ in range(tasks)]
        runs = []
        for arm, shift in (("incumbent", 0.0), ("challenger", delta_logit)):
            for t, p in enumerate(base):
                q = sigmoid(logit(p) + shift)
                for r in range(reps):
                    runs.append({"arm": arm, "task": f"t{t}", "rep": r, "resolved": rng.random() < q, "tokens": 1, "seconds": 1})
        scores = analyze.per_task_scores(runs)
        d = analyze.paired_diffs(scores, "challenger", "incumbent")
        p = analyze.sign_flip_p(d, flips, random.Random(rng.random()))
        if p * contrasts < analyze.ALPHA and sum(d) > 0:       # Holm for the smallest p is p * number_of_contrasts
            hit += 1
    return hit / trials


def avg_points(delta_logit, shape, n=20000):
    rng = random.Random(1)
    tot = 0.0
    for _ in range(n):
        p = min(0.97, max(0.03, rng.betavariate(*shape)))
        tot += sigmoid(logit(p) + delta_logit) - p
    return 100 * tot / n


if __name__ == "__main__":
    shapes = {"U-shaped (as benchmarks are)": (0.7, 0.7), "informative tasks (pilot-filtered)": (2.0, 2.0)}
    for name, shape in shapes.items():
        print(f"\n{name}")
        for delta in (0.25, 0.5, 0.8):
            pts = avg_points(delta, shape)
            row = []
            for tasks, reps in ((30, 2), (60, 2), (60, 3), (120, 2), (200, 2)):
                row.append(f"{tasks}x{reps}: {100 * power(delta, tasks, reps, shape, contrasts=3, trials=200):3.0f}%")
            print(f"  true gap ~{pts:4.1f} pts   " + "   ".join(row))
