"""Statistics for the agentic model comparison. Pure standard library, deterministic (seeded).

Input: a list of run records, one per (arm, task, repetition):
    {"arm": str, "task": str, "rep": int, "resolved": bool, "tokens": int, "seconds": float, ...}

Design (fixed before any run, see PROTOCOL.md):
  * the unit of analysis is the TASK, not the run: repetitions of one task are averaged first, because they are not
    independent (the same task is easy or hard for every arm);
  * arms are compared in pairs, task by task (paired design), which removes the task difficulty from the noise;
  * p-values come from an exact-style SIGN-FLIP randomisation test of the mean paired difference (Monte Carlo, 100k flips),
    confidence intervals from a cluster bootstrap over tasks (20k resamples);
  * the family of all pairwise comparisons is corrected with Holm's method;
  * a WINNER is declared only if one arm is significantly better than EVERY other arm (Holm-adjusted p < 0.05 and a
    positive difference). Otherwise the result is a TIE among the arms that are not significantly worse than the best one,
    and the tie is broken by the pre-registered secondary metric (tokens per solved task, paired the same way).
"""
import itertools
import json
import random
import statistics
import sys

FLIPS = 100_000
BOOT = 20_000
ALPHA = 0.05


def per_task_scores(runs):
    """{arm: {task: mean resolved over repetitions}}. Tasks missing for an arm are an error: the design is complete."""
    cell = {}
    for r in runs:
        cell.setdefault((r["arm"], r["task"]), []).append(1.0 if r["resolved"] else 0.0)
    arms = sorted({a for a, _ in cell})
    tasks = sorted({t for _, t in cell})
    out = {a: {} for a in arms}
    for a in arms:
        for t in tasks:
            if (a, t) not in cell:
                raise ValueError(f"arm {a!r} has no run for task {t!r}")
            out[a][t] = statistics.fmean(cell[(a, t)])
    return out


def paired_diffs(scores, a, b):
    tasks = sorted(scores[a])
    return [scores[a][t] - scores[b][t] for t in tasks]


def sign_flip_p(diffs, flips=FLIPS, rng=None):
    """Two-sided p-value of H0 'no difference' for the mean of paired differences (zeros carry no information)."""
    rng = rng or random.Random(20261001)
    nz = [d for d in diffs if d != 0.0]
    if not nz:
        return 1.0
    observed = abs(sum(nz))
    n = len(nz)
    hits = 0
    for _ in range(flips):
        s = 0.0
        bits = rng.getrandbits(n)
        for i, d in enumerate(nz):
            s += d if (bits >> i) & 1 else -d
        if abs(s) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (flips + 1)


def bootstrap_ci(diffs, boot=BOOT, rng=None, level=0.95):
    rng = rng or random.Random(20261002)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(boot))
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return means[int(lo * boot)], means[min(boot - 1, int(hi * boot))]


def holm(pvals):
    """Holm step-down adjusted p-values for {name: p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def compare(runs, flips=FLIPS, boot=BOOT):
    scores = per_task_scores(runs)
    arms = sorted(scores)
    pairs = list(itertools.combinations(arms, 2))
    raw, rows = {}, {}
    for a, b in pairs:
        d = paired_diffs(scores, a, b)
        mean = statistics.fmean(d)
        lo, hi = bootstrap_ci(d, boot)
        raw[(a, b)] = sign_flip_p(d, flips)
        rows[(a, b)] = {"mean_diff": mean, "ci": (lo, hi), "n_tasks": len(d)}
    adj = holm(raw)
    for k in rows:
        rows[k]["p"], rows[k]["p_holm"] = raw[k], adj[k]
    means = {a: statistics.fmean(scores[a].values()) for a in arms}
    return {"arms": arms, "means": means, "pairs": rows, "scores": scores}


def verdict(result):
    """Winner, or the tie group. Pre-registered rule, see the module docstring."""
    arms, rows = result["arms"], result["pairs"]

    def diff(a, b):                     # significant advantage of a over b?  (mean diff, adjusted p)
        if (a, b) in rows:
            r = rows[(a, b)]
            return r["mean_diff"], r["p_holm"]
        r = rows[(b, a)]
        return -r["mean_diff"], r["p_holm"]

    best = max(arms, key=lambda a: result["means"][a])
    not_worse = [a for a in arms if a == best or not (diff(best, a)[0] > 0 and diff(best, a)[1] < ALPHA)]
    if len(not_worse) == 1:
        return {"kind": "winner", "arm": best, "group": not_worse}
    return {"kind": "tie", "arm": None, "group": not_worse, "best_by_mean": best}


def tokens_per_solved(runs, arm):
    """Median over tasks of the mean tokens spent per solved repetition; tasks never solved by the arm are skipped."""
    by_task = {}
    for r in runs:
        if r["arm"] == arm and r["resolved"]:
            by_task.setdefault(r["task"], []).append(r["tokens"])
    return {t: statistics.fmean(v) for t, v in by_task.items()}


def tie_break(runs, group):
    """Cheapest arm of the tie group: median of the tokens-per-solved over the tasks ALL group arms solved at least once."""
    per = {a: tokens_per_solved(runs, a) for a in group}
    common = set.intersection(*(set(v) for v in per.values())) if per else set()
    if not common:
        return None, {}
    med = {a: statistics.median(per[a][t] for t in common) for a in group}
    return min(med, key=med.get), {"tasks_in_common": len(common), "median_tokens_per_solved": med}


def report(runs, flips=FLIPS, boot=BOOT):
    res = compare(runs, flips, boot)
    ver = verdict(res)
    lines = ["Arm results (mean over tasks of the per-task success rate):"]
    for a in sorted(res["arms"], key=lambda x: -res["means"][x]):
        lines.append(f"  {a:28} {100 * res['means'][a]:5.1f}%")
    lines.append("")
    lines.append("Paired differences (first minus second), 95% bootstrap CI, Holm-adjusted p:")
    for (a, b), r in sorted(res["pairs"].items()):
        lo, hi = r["ci"]
        lines.append(f"  {a:>22} - {b:<22} {100 * r['mean_diff']:+6.1f} pts  [{100 * lo:+6.1f}, {100 * hi:+6.1f}]  p={r['p_holm']:.4f}")
    lines.append("")
    if ver["kind"] == "winner":
        lines.append(f"WINNER: {ver['arm']} (significantly better than every other arm)")
    else:
        cheap, info = tie_break(runs, ver["group"])
        lines.append("NO STATISTICAL WINNER on quality. Arms not significantly worse than the best "
                     f"({ver['best_by_mean']}): {', '.join(ver['group'])}")
        if cheap:
            lines.append(f"Tie-break (fewest tokens per solved task over {info['tasks_in_common']} common tasks): {cheap}")
            for a, m in info["median_tokens_per_solved"].items():
                lines.append(f"  {a:28} median {m:,.0f} tokens per solved task")
    return "\n".join(lines), res, ver


def main(argv):
    runs = [json.loads(line) for line in open(argv[1]) if line.strip()]
    print(report(runs)[0])


if __name__ == "__main__":
    main(sys.argv)
