"""GEVAL1: summary of a general-v1 run — per category and overall, with a family-cluster bootstrap 95 % CI, and the paired base/candidate
differences the acceptance rule needs (rsi/accept.py decide_with_general). Standard library only, deterministic.

    python3 general_summary.py results.jsonl [--judged judged.jsonl] [--base base --candidate cand] [--split dev]

Scores: programmatic tasks use the record's `score` (0..1, from hidden/check.py); writing tasks (`judge_pending` in the test tail)
take the judge's score from judged.jsonl by the run tag `<task>-<arm>-<rep>-…` (what check.py copied to $GEVAL_ANSWER_DIR);
a judge task without a judgement counts as 0 and is listed. Headline = mean over categories of the category mean (equal weights
unless WEIGHTS says otherwise); web tasks are reported but outside the headline unless they were run.
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHTS = {"questions": 15, "reasoning": 15, "instructions": 20, "writing": 15, "files": 10, "review_code": 6, "review_text": 4, "long_context": 10, "web": 5,
           # GEVAL1b mirror categories
           "design": 8, "media": 8, "document": 5, "posts": 6, "diagnosis": 8, "research": 5, "orchestration": 8}


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def category_of(task_id, tasks_root=HERE / "tasks_general"):
    fam, kind = task_id.rsplit("-general-", 1)
    tj = tasks_root / fam / f"general-{kind}" / "task.json"
    return json.loads(tj.read_text(encoding="utf-8"))["category"] if tj.is_file() else "?"


def attach_judgements(records, judged):
    """Judge scores by (task, arm, rep): the judged tag starts with `<task>-<arm>-<rep>-`."""
    by = {}
    for j in judged:
        tag = j.get("tag", "")
        for r in records:
            prefix = f"{r['task']}-{r['arm']}-{r['rep']}-"
            if tag.startswith(prefix):
                by[(r["task"], r["arm"], r["rep"])] = j.get("score")
    out, missing = [], []
    for r in records:
        r = dict(r)
        if "judge_pending" in (r.get("test_tail") or ""):
            s = by.get((r["task"], r["arm"], r["rep"]))
            if s is None:
                missing.append(r["task"])
                s = 0.0
            r["score"] = float(s)
            r["resolved"] = s >= 0.75
            r["judged"] = True
        out.append(r)
    return out, missing


def family_bootstrap(values_by_family, reps=4000, seed=20261029):
    fams = sorted(values_by_family)
    if not fams:
        return None, None, None
    means = [statistics.fmean(values_by_family[f]) for f in fams]
    mean = statistics.fmean(means)
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(means[rng.randrange(len(means))] for _ in means) for _ in range(reps))
    return round(mean, 4), round(boots[int(0.025 * (reps - 1))], 4), round(boots[int(0.975 * (reps - 1))], 4)


def per_task(records, arm):
    """{task: mean score over reps} for one arm."""
    acc = defaultdict(list)
    for r in records:
        if r["arm"] == arm:
            acc[r["task"]].append(float(r.get("score") or 0.0))
    return {t: statistics.fmean(v) for t, v in acc.items()}


def summarize_arm(records, arm):
    scores = per_task(records, arm)
    cats = defaultdict(lambda: defaultdict(list))
    for t, s in scores.items():
        cats[category_of(t)][t.rsplit("-general-", 1)[0]].append(s)
    rows = {}
    for c, fams in sorted(cats.items()):
        mean, lo, hi = family_bootstrap(fams)
        rows[c] = {"tasks": sum(len(v) for v in fams.values()), "families": len(fams), "mean": mean, "ci": [lo, hi]}
    present = [c for c in rows if c != "web" or rows["web"]["tasks"]]
    w = {c: WEIGHTS.get(c, 1) for c in present}
    headline = round(sum(rows[c]["mean"] * w[c] for c in present) / sum(w.values()), 4) if present else None
    return {"arm": arm, "tasks": len(scores), "categories": rows, "headline": headline, "weights": w}


def paired(records, base, cand):
    a, b = per_task(records, base), per_task(records, cand)
    common = sorted(set(a) & set(b))
    diffs = defaultdict(list)
    for t in common:
        diffs[t.rsplit("-general-", 1)[0]].append(b[t] - a[t])
    mean, lo, hi = family_bootstrap(diffs)
    return {"base": base, "candidate": cand, "tasks": len(common), "families": len(diffs), "mean_diff": mean, "ci": [lo, hi]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--judged")
    ap.add_argument("--base")
    ap.add_argument("--candidate")
    ap.add_argument("--split")
    a = ap.parse_args(argv)
    records = read_jsonl(a.results)
    if a.split:
        fams = json.loads((HERE / "splits.json").read_text())["families"]
        records = [r for r in records if fams.get(r.get("family") or r["task"].rsplit("-general-", 1)[0]) == a.split]
    missing = []
    if a.judged:
        records, missing = attach_judgements(records, read_jsonl(a.judged))
    out = {"arms": [summarize_arm(records, arm) for arm in sorted({r["arm"] for r in records})], "judge_missing": sorted(set(missing))}
    if a.base and a.candidate:
        out["paired"] = paired(records, a.base, a.candidate)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    main()
