"""Does the decision rule in analyze.py behave? Simulate whole experiments with KNOWN truth and count how often it
declares a (wrong or right) winner. Run before trusting any real result:

    python3 bench/agentic-eval/simulate.py            # ~ a few minutes on one core

Task difficulty is drawn from a U-shaped Beta (many tasks are easy or hard for everybody, a minority is in between), the
way real benchmark tasks behave. An arm's advantage is a shift on the logit scale, converted to roughly `delta` points
of average success rate.
"""
import math
import random
import sys

import analyze

TASKS, REPS, ARMS = 60, 2, 4


def logit(p):
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def experiment(rng, shifts, tasks=TASKS, reps=REPS):
    base = [min(0.98, max(0.02, rng.betavariate(0.7, 0.7))) for _ in range(tasks)]
    runs = []
    for arm, shift in shifts.items():
        for t, p in enumerate(base):
            q = sigmoid(logit(p) + shift)
            for r in range(reps):
                runs.append({"arm": arm, "task": f"t{t:02d}", "rep": r, "resolved": rng.random() < q,
                             "tokens": rng.randint(8_000, 40_000), "seconds": 1.0})
    return runs


def rate(shifts, trials, flips=3000, boot=600, seed=1):
    rng = random.Random(seed)
    wins = {a: 0 for a in shifts}
    ties = 0
    for _ in range(trials):
        runs = experiment(rng, shifts)
        res = analyze.compare(runs, flips=flips, boot=boot)
        v = analyze.verdict(res)
        if v["kind"] == "winner":
            wins[v["arm"]] += 1
        else:
            ties += 1
    return wins, ties


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    arms = ["A", "B", "C", "D"]
    null = {a: 0.0 for a in arms}
    wins, ties = rate(null, trials)
    print(f"all arms equal ({trials} experiments): winner declared {sum(wins.values())} times "
          f"({100 * sum(wins.values()) / trials:.1f}%), ties {ties}")
    for label, shift in (("about +5 points", 0.25), ("about +10 points", 0.5), ("about +15 points", 0.8)):
        shifts = {**null, "A": shift}
        wins, ties = rate(shifts, trials, seed=2)
        print(f"A better by {label}: A declared winner {100 * wins['A'] / trials:.1f}%, wrong winner "
              f"{100 * sum(v for k, v in wins.items() if k != 'A') / trials:.1f}%, tie {100 * ties / trials:.1f}%")


if __name__ == "__main__":
    main()
