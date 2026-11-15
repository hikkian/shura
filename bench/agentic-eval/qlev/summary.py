"""Paired QLEV1 summaries and DATASET §5 gates; standard library only."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from qlev.arms import QUALITY_ARMS, SPEED_ARMS

METRICS = ("resolved", "score", "seconds", "reasoning_tokens", "timed_out")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _number(row, key):
    value = row.get(key)
    if value is None:
        return None
    if key in ("resolved", "timed_out"):
        # a flag of one run is True/False; a row that stands for several runs (a baseline mean such as 0.5) is a number and stays one
        # (VRAMDIP1: float(bool(0.5)) made it 1.0 and moved the baseline and the CI)
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return float(value.strip().lower() == "true")
        if isinstance(value, (int, float)):
            return float(value)
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values):
    return sum(values) / len(values) if values else None


def _percentile(sorted_values, q):
    if not sorted_values:
        return None
    ix = (len(sorted_values) - 1) * q
    low, high = int(ix), min(int(ix) + 1, len(sorted_values) - 1)
    frac = ix - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def bootstrap_ci(values, seed=20261021, samples=10000, confidence=0.95):
    """Percentile CI for a mean; resamples the passed independent units with replacement."""
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(_mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(samples))
    alpha = (1 - confidence) / 2
    return [_percentile(means, alpha), _percentile(means, 1 - alpha)]


def paired_task_differences(records, base_arm, candidate_arm):
    by_key = {}
    for row in records:
        if row.get("arm") not in (base_arm, candidate_arm):
            continue
        key = (str(row.get("task")), str(row.get("rep")))
        by_key.setdefault(key, {})[row["arm"]] = row
    by_task = {}
    pair_counts = defaultdict(int)
    for (task, rep), pair in by_key.items():
        if base_arm not in pair or candidate_arm not in pair:
            continue
        base, cand = pair[base_arm], pair[candidate_arm]
        family = cand.get("family") or base.get("family") or "unknown"
        task_row = by_task.setdefault(task, {"family": family, "reps": [], "metrics": defaultdict(list)})
        task_row["reps"].append(rep)
        pair_counts[task] += 1
        for metric in METRICS:
            b, c = _number(base, metric), _number(cand, metric)
            if b is not None and c is not None:
                task_row["metrics"][metric].append(c - b)
    results = {}
    for task, row in by_task.items():
        results[task] = {"family": row["family"], "n_pairs": len(row["reps"]),
                         "metrics": {key: _mean(values) for key, values in row["metrics"].items()}}
    return results


def summarize_pair(records, base_arm, candidate_arm, bootstrap_samples=10000, seed=20261021, minimum_reps=2):
    paired = paired_task_differences(records, base_arm, candidate_arm)
    common = sorted(paired)
    if not common:
        return {"base_arm": base_arm, "arm": candidate_arm, "paired_tasks": 0, "gate": "no paired base results", "metrics": {}}
    family_groups = defaultdict(list)
    for task in common:
        family_groups[paired[task]["family"]].append(task)
    metrics = {}
    for index, metric in enumerate(METRICS):
        task_values = [paired[task]["metrics"][metric] for task in common if metric in paired[task]["metrics"]]
        family_values = [_mean([paired[task]["metrics"][metric] for task in tasks if metric in paired[task]["metrics"]])
                         for tasks in family_groups.values()]
        family_values = [v for v in family_values if v is not None]
        if task_values:
            metrics[metric] = {"mean_delta": _mean(task_values),
                               "task_bootstrap_95": bootstrap_ci(task_values, seed + index, bootstrap_samples),
                               "family_bootstrap_95": bootstrap_ci(family_values, seed + 100 + index, bootstrap_samples)}
    min_pairs = min(paired[t]["n_pairs"] for t in common)
    group = "quality" if candidate_arm in QUALITY_ARMS else "speed" if candidate_arm in SPEED_ARMS else "unclassified"
    if min_pairs < minimum_reps:
        gate = f"insufficient repeats: min {min_pairs}, need {minimum_reps}"
    elif group == "quality":
        resolved_ci = metrics.get("resolved", {}).get("task_bootstrap_95")
        score_ci = metrics.get("score", {}).get("task_bootstrap_95")
        gate = "pass" if (resolved_ci and score_ci and resolved_ci[0] > -0.03 and score_ci[0] > 0) else "fail"
    elif group == "speed":
        resolved_ci = metrics.get("resolved", {}).get("task_bootstrap_95")
        score_ci = metrics.get("score", {}).get("task_bootstrap_95")
        seconds = metrics.get("seconds", {}).get("mean_delta")
        gate = "pass" if (resolved_ci and score_ci and resolved_ci[0] > -0.03 and score_ci[0] >= 0 and seconds is not None and seconds < 0) else "fail"
    else:
        gate = "unclassified arm"
    baseline = {}
    candidate = {}
    for metric in METRICS:
        bvals, cvals = [], []
        for row in records:
            value = _number(row, metric)
            if value is None or str(row.get("task")) not in paired:
                continue
            if row.get("arm") == base_arm:
                bvals.append(value)
            elif row.get("arm") == candidate_arm:
                cvals.append(value)
        baseline[metric] = _mean(bvals)
        candidate[metric] = _mean(cvals)
    return {"base_arm": base_arm, "arm": candidate_arm, "group": group, "gate": gate,
            "paired_tasks": len(common), "families": len(family_groups), "minimum_pairs_per_task": min_pairs,
            "baseline_means": baseline, "candidate_means": candidate, "metrics": metrics,
            "task_differences": paired}


def summarize(records, base_arm="base", arms=None, bootstrap_samples=10000, seed=20261021, minimum_reps=2):
    arms = arms or sorted({r.get("arm") for r in records if r.get("arm") != base_arm})
    return {"base_arm": base_arm,
            "comparisons": [summarize_pair(records, base_arm, arm, bootstrap_samples, seed + i * 1000, minimum_reps)
                            for i, arm in enumerate(arms)]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="QLEV1 results.jsonl")
    parser.add_argument("--base", default="base")
    parser.add_argument("--arms", nargs="*")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--minimum-reps", type=int, default=2)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    result = summarize(read_jsonl(args.results), args.base, args.arms, args.bootstrap, args.seed, args.minimum_reps)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
