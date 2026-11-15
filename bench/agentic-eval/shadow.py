#!/usr/bin/env python3
"""SHADOW1 night layer: paired night runs of a grey candidate against the stack, until a green or red zone (MASTER §5а).

A candidate in the shadow never acts for the person. At night the same whole tasks run on two configurations, `stack` and
`stack+candidate`; the paired results pile up in the shadow store and after every night gate/gate.py decides on the combined
data. The rules are written into the candidate's preregistration before the first run and are frozen from then on:
target row and "not worse" rows, δ (0.03), the number of looks (at most three, the significance level is split between them),
the life time (at most four weeks; then the candidate is removed as "not proven"). Night pairs are whole tasks: cutting them to
the tasks where the switch fired is not allowed (MASTER §5а; that needs replay from the firing point, FORK1).

    shadow.py init NAME --switch "..." --adds '{"max_tokens": 48}' [--target-row] [--delta 0.03] [--looks 3] [--life-days 28]
    shadow.py plan NAME [NAME...] --tasks-per-night 10 --server server.json --out RUN_DIR [--stack-extra '{...}'] [--night N]
    shadow.py ingest RUN_DIR NAME [NAME...]          # after run_experiment.py: append the pairs to each candidate's store
    shadow.py verdict NAME                           # gate.py on the combined data; one line per row
    shadow.py summary                                # the morning line

Store: reports/shadow1/<candidate>/{prereg.json, results.jsonl, nights.json, verdict-look<k>.json}. Results use the record format
of gate/example-results ({"row", "arm", "task", "family", "rep", "seed", "score"}) plus a few extra fields gate ignores.
`proxy.py`, `harness.py` and `run_experiment.py` are not changed: plan writes their inputs, ingest reads their outputs.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from gate import gate as G  # noqa: E402

STORE = Path(os.environ.get("SHADOW_STORE", str(Path.home() / "Desktop/AI/reports/shadow1")))
MODEL = "/home/alisher/Desktop/AI/models/Tiel-Coder-35B-A3B.APEX-I-MiniPlus-V2.1.q8mtp.gguf"
MAX_LOOKS = 3
MAX_LIFE_DAYS = 28
DEFAULT_DELTA = 0.03


def _now():
    return time.time()


def _iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat(timespec="seconds")


def cand_dir(name, store=None):
    if not name or any(c in name for c in "/\\") or name in (".", "..", "stack"):
        raise ValueError(f"bad candidate name {name!r}")
    return Path(store or STORE) / name


def _read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _prereg_hash(pre):
    frozen = {k: v for k, v in pre.items() if k not in ("look", "n")}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# --------------------------------------------------------------------------------------------- preregistration (before any run)

def init(name, switch, adds=None, profile=None, target_row=False, delta=DEFAULT_DELTA, looks=MAX_LOOKS, life_days=MAX_LIFE_DAYS,
         alpha=0.05, tasks_per_night=10, seed=20261113, rows=None, store=None, now=None):
    """Write the candidate's preregistration. Refused once results exist (the rules cannot move after the first run)."""
    if int(looks) < 1 or int(looks) > MAX_LOOKS:
        raise ValueError(f"looks must be 1..{MAX_LOOKS}")
    if float(life_days) <= 0 or float(life_days) > MAX_LIFE_DAYS:
        raise ValueError(f"the shadow life is at most {MAX_LIFE_DAYS} days")
    if adds is None and profile is None:
        raise ValueError("the candidate must say what it adds to a request (adds) or which server profile it needs (profile)")
    d = cand_dir(name, store)
    if _read_jsonl(d / "results.jsonl"):
        raise ValueError(f"{name}: results exist; the preregistration is frozen")
    t = now if now is not None else _now()
    rows = rows or [{"name": "agent-dev", "kind": "agent", "target": bool(target_row), "delta": float(delta)}]
    for r in rows:
        r.setdefault("kind", "agent")
        r.setdefault("delta", float(delta))
        r.setdefault("target", False)
    pre = {"candidate": name, "control": "stack", "metric": "score", "k": 1, "seed": int(seed), "looks": int(looks),
           "alpha": float(alpha), "bootstrap_reps": 10000, "rows": rows,
           "shadow": {"switch": switch, "adds": adds or {}, "profile": profile, "created": _iso(t), "created_ts": t,
                      "expires_ts": t + float(life_days) * 86400, "life_days": float(life_days), "tasks_per_night": int(tasks_per_night),
                      "task_source": "agentic-eval:dev (search part)", "whole_tasks_only": True}}
    G.validate_prereg(pre)
    d.mkdir(parents=True, exist_ok=True)
    pre["shadow"]["prereg_sha256"] = _prereg_hash(pre)
    (d / "prereg.json").write_text(json.dumps(pre, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return pre


def load_prereg(name, store=None):
    d = cand_dir(name, store)
    pre = json.loads((d / "prereg.json").read_text(encoding="utf-8"))
    want = pre["shadow"].pop("prereg_sha256", None)
    if want and _prereg_hash(pre) != want:
        raise ValueError(f"{name}: the preregistration was changed after it was written")
    pre["shadow"]["prereg_sha256"] = want
    return pre


# --------------------------------------------------------------------------------------------- task sources

class OpenTaskSource:
    """The open agentic-eval tasks of the search part (dev). Personal tasks are never opened here."""

    def __init__(self, rows=None):
        self._rows = rows

    def tasks(self):
        if self._rows is None:
            sys.path.insert(0, str(HERE))
            from qlev import selection
            self._rows = [dict(r, source="open", dir=str(HERE / "tasks_v3" / r["family"] / r["kind"]))
                          for r in selection.dev_code_tasks()]
        return list(self._rows)


class JournalTaskSource:
    """New real tasks from the experience ledger: only with the `exam` consent, only blocks drawn to the exam split (never train,
    never sealed), only records that carry both tests and a repository snapshot, never `personal` ones. No live data exists yet."""

    def __init__(self, ledger):
        self.ledger = ledger

    def tasks(self):
        if not self.ledger.consent("exam"):
            return []
        out = []
        for rec in self.ledger.list():
            if rec.get("kind") != "task" or rec.get("privacy") == "personal":
                continue
            p = rec["payload"]
            if p.get("split") != "exam" or not p.get("tests") or not p.get("repo_snapshot"):
                continue
            out.append({"id": f"journal-{rec['id']}", "family": str(p.get("repo") or p.get("block") or f"journal-{rec['id']}"),
                        "kind": "journal", "source": "journal", "event_id": rec["id"], "prompt": p["request"],
                        "repo_snapshot": p["repo_snapshot"], "tests": p["tests"]})
        return out


def pick_tasks(source, n, seed, used=()):
    """n fresh tasks, deterministic for the seed; never ones this candidate already ran (a re-check uses new tasks)."""
    rows = sorted((t for t in source.tasks() if t["id"] not in set(used)), key=lambda t: t["id"])
    random.Random(seed).shuffle(rows)
    return rows[:n]


# --------------------------------------------------------------------------------------------- the night plan

def used_tasks(name, store=None):
    return sorted({r["task_id"] for r in _read_jsonl(cand_dir(name, store) / "results.jsonl")})


def plan(names, server, out, tasks_per_night=None, stack_extra=None, night=None, source=None, store=None, model=MODEL,
         tasks_root=None, parallel=2, max_hours=3.0):
    """arms.json for run_experiment.py: `stack` once, `stack+<candidate>` for each candidate, same tasks and seeds.
    Returns {"arms": path, "tasks": [...], "command": [...]}."""
    pres = [load_prereg(n, store) for n in names]
    t = _now()
    for pre in pres:
        if final_verdict(pre["candidate"], store):
            raise ValueError(f"{pre['candidate']}: already decided ({final_verdict(pre['candidate'], store)['status']}); no more nights")
        if t > pre["shadow"]["expires_ts"]:
            raise ValueError(f"{pre['candidate']}: the shadow life is over; it is removed as not proven")
        if pre["shadow"]["profile"]:
            raise ValueError(f"{pre['candidate']}: a server-profile candidate needs its own run (two servers); plan it alone")
    n = int(tasks_per_night or max(p["shadow"]["tasks_per_night"] for p in pres))
    used = set().union(*(set(used_tasks(p["candidate"], store)) for p in pres))
    nights = [len(_nights(p["candidate"], store)) for p in pres]
    night = int(night if night is not None else max(nights) + 1)
    seed = pres[0]["seed"] + night
    tasks = pick_tasks(source or OpenTaskSource(), n, seed, used)
    if len(tasks) < n:
        raise ValueError(f"only {len(tasks)} fresh tasks left for these candidates")
    missing = [x["id"] for x in tasks if not x.get("dir")]
    if missing:
        raise ValueError(f"tasks without a task folder (journal tasks must be materialized first): {missing[:3]}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(tasks_root or (Path("/dev/shm/SHADOW1") / f"tasks-night{night}"))
    for task in tasks:
        link = root / task["family"] / task["kind"]
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(task["dir"], target_is_directory=True)
    base = dict(stack_extra or {})
    arms = [{"name": "stack", "model": model, "extra": base}]
    for pre in pres:
        arms.append({"name": pre["candidate"], "model": model, "extra": {**base, **pre["shadow"]["adds"]}})
    cfg = {"server": server, "arms": arms,
           "shadow": {"night": night, "seed": seed, "tasks": [x["id"] for x in tasks], "candidates": names}}
    path = out / "arms.json"
    path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    cmd = [sys.executable, str(HERE / "run_experiment.py"), str(path), "--out", str(out), "--tasks-root", str(root),
           "--reps", "1", "--round-size", str(n), "--parallel", str(parallel), "--max-hours", str(max_hours),
           "--seed", str(seed), "--judge", "off", "--port", "18196"]
    return {"arms": str(path), "tasks": [x["id"] for x in tasks], "night": night, "seed": seed, "command": cmd}


# --------------------------------------------------------------------------------------------- after the night

def final_verdict(name, store=None):
    """The decision once a candidate left the shadow (red, green, or removed); None while it is in the shadow."""
    p = cand_dir(name, store) / "final.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _nights(name, store=None):
    p = cand_dir(name, store) / "nights.json"
    return json.loads(p.read_text()) if p.exists() else []


def _row_of(task_row, task):
    want = task_row.get("tasks") or {}
    return all(task.get(k) == v for k, v in want.items())


def ingest(run_dir, names, store=None):
    """Append the complete pairs (stack and candidate on the same task and rep) of one night to each candidate's store."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "arms.json").read_text())
    night, seed = cfg["shadow"]["night"], cfg["shadow"]["seed"]
    runs = _read_jsonl(run_dir / "results.jsonl")
    added = {}
    for name in names:
        pre = load_prereg(name, store)
        d = cand_dir(name, store)
        if final_verdict(name, store):
            raise ValueError(f"{name}: already decided; its store takes no more nights")
        if any(n["night"] == night and n["run"] == str(run_dir) for n in _nights(name, store)):
            added[name] = 0
            continue
        cells = {}
        for r in runs:
            if r["arm"] in ("stack", name):
                cells.setdefault((r["task"], r["rep"]), {})[r["arm"]] = r
        recs = []
        for (task, rep), arms in sorted(cells.items()):
            if set(arms) != {"stack", name}:
                continue                                   # an unpaired run (crash, time budget) is left out, never imputed
            if any("score" not in r for r in arms.values()):
                raise ValueError(f"{name}: a result without a score for {task}/{rep} in {run_dir}; fix the run before ingesting")
            for arm, r in arms.items():
                for row in pre["rows"]:
                    if not _row_of(row, r):
                        continue
                    recs.append({"row": row["name"], "arm": arm, "task": f"{task}@n{night}", "task_id": task,
                                 "family": r.get("family"), "rep": rep, "seed": seed, "score": float(r.get("score", 0.0)),
                                 "night": night, "seconds": r.get("seconds"), "reasoning_tokens": r.get("reasoning_tokens")})
        if not recs:
            added[name] = 0                                # a night without pairs spends no look
            continue
        with open(d / "results.jsonl", "a", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        nights = _nights(name, store) + [{"night": night, "run": str(run_dir), "pairs": len(recs) // 2 // max(1, len(pre["rows"])),
                                          "at": _iso(_now())}]
        (d / "nights.json").write_text(json.dumps(nights, indent=1) + "\n")
        added[name] = len(recs)
    return added


def verdict(name, store=None, now=None):
    """gate.py on the combined data of all nights so far; look = number of nights (at most `looks`)."""
    final = final_verdict(name, store)
    if final:
        return final                                       # a decided candidate stays decided (no re-analysis of grown data)
    pre = load_prereg(name, store)
    d = cand_dir(name, store)
    recs = _read_jsonl(d / "results.jsonl")
    nights = _nights(name, store)
    t = now if now is not None else _now()
    look = min(len(nights), pre["looks"])
    if not recs or look == 0:
        if t > pre["shadow"]["expires_ts"]:
            v = {"candidate": name, "overall": "grey", "status": "removed: not proven", "look": 0, "rows": [], "expired": True}
            (d / "final.json").write_text(json.dumps(v, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            return v
        return {"candidate": name, "overall": "grey", "status": "waiting", "look": 0, "rows": []}
    run_pre = {k: v for k, v in pre.items() if k != "shadow"}
    run_pre["look"] = look
    rows = []
    for row in pre["rows"]:
        tasks = {r["task"] for r in recs if r["row"] == row["name"]}
        rows.append(G.analyze_row(dict(row, n=len(tasks), k=1), recs, run_pre))
    v = G.verdict(run_pre, rows)
    expired = t > pre["shadow"]["expires_ts"]
    if v["overall"] == "red":
        status = "removed: worse"
    elif v["overall"] == "green":
        status = "proven: next train"
    elif look >= pre["looks"] or expired:
        status = "removed: not proven"
    else:
        status = "in shadow"
    v.update({"status": status, "look": look, "looks": pre["looks"], "expired": expired, "nights": len(nights)})
    (d / f"verdict-look{look}.json").write_text(json.dumps(v, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if status != "in shadow":
        (d / "final.json").write_text(json.dumps(v, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return v


def row_line(v):
    out = []
    for r in v["rows"]:
        if r.get("lower_bound") is None:
            out.append(f"{r['name']}: нет пар")
            continue
        out.append(f"{r['name']}: {r['zone']}, разница {r['mean_difference']:+.3f} [{r['lower_bound']:+.3f}; {r['upper_bound']:+.3f}], "
                   f"задач {r['n']}, δ {r['delta']}")
    return "; ".join(out)


def summary(store=None, now=None):
    """The morning line: 'на проверке в тени: N задач, пока ничья / лучше / хуже', and the outcome when a candidate leaves the grey zone."""
    root = Path(store or STORE)
    active, done = [], []
    for pre_path in sorted(root.glob("*/prereg.json")):
        name = pre_path.parent.name
        v = verdict(name, root, now)
        if v["status"] == "waiting":
            continue
        if v["status"] == "in shadow":
            active.append((name, v))
        else:
            done.append((name, v))
    parts = []
    if active:
        n = sum(r.get("n", 0) for _, v in active for r in v["rows"][:1])
        trend = []
        for name, v in active:
            r = v["rows"][0] if v["rows"] else {}
            if r.get("lower_bound") is None:
                trend.append(f"{name} — пар пока нет")
                continue
            word = "лучше" if r["lower_bound"] > 0 else "хуже" if r["upper_bound"] < 0 else "ничья"
            trend.append(f"{name} — пока {word}")
        parts.append(f"на проверке в тени: {n} задач ({', '.join(trend)})")
    for name, v in done:
        word = {"removed: worse": "хуже стека — снят", "proven: next train": "не хуже — идёт в поезд",
                "removed: not proven": "серая зона до конца срока — снят как не доказанный"}[v["status"]]
        parts.append(f"тень: {name} — {word} ({row_line(v)})")
    return "; ".join(parts) if parts else "в тени никого нет"


# --------------------------------------------------------------------------------------------- simulation of the night layer (no GPU)

def _fast_bootstrap(diffs, families=None, reps=10000, seed=1, level=.95):
    """gate.paired_bootstrap with numpy: the same resampling (tasks, or families then tasks) and the same percentile rule."""
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = np.asarray(diffs, dtype=float)
    n = len(arr)
    groups = {}
    for i, f in enumerate(families if families else ["__all__"] * n):
        groups.setdefault(f, []).append(i)
    if not families or all(len(v) == 1 for v in groups.values()):
        means = arr[rng.integers(0, n, size=(reps, n))].mean(axis=1)
    else:
        names = sorted(groups)
        means = np.empty(reps)
        for b in range(reps):
            picked = [groups[names[j]] for j in rng.integers(0, len(names), size=len(names))]
            vals = np.concatenate([arr[np.asarray(g)[rng.integers(0, len(g), size=len(g))]] for g in picked])
            means[b] = vals.mean()
    tail = (1 - level) / 2
    lo, hi = np.percentile(means, [100 * tail, 100 * (1 - tail)])
    return float(lo), float(hi)


def _fast_harm_test(diffs, delta, _orig=None):
    """gate.exact_harm_test: exact for <= 22 non-zero pairs (the original), else 100 000 random sign flips with numpy."""
    import numpy as np
    centered = np.asarray([d + delta for d in diffs if d + delta], dtype=float)
    n = len(centered)
    if n <= 22:
        return _orig(diffs, delta)
    observed = centered.mean()
    rng = np.random.default_rng(20261113)
    hits = 0
    for _ in range(5):
        signs = rng.integers(0, 2, size=(20000, n)) * 2 - 1
        hits += int(((signs * centered).mean(axis=1) <= observed + 1e-15).sum())
    return (hits + 1) / 100001


class FastGate:
    """Context: gate.py with the numpy statistics above (zones are decided by gate.analyze_row itself)."""

    def __enter__(self):
        self.saved = (G.paired_bootstrap, G.exact_harm_test)
        orig = G.exact_harm_test
        G.paired_bootstrap = _fast_bootstrap
        G.exact_harm_test = lambda d, delta: _fast_harm_test(d, delta, orig)
        return self

    def __exit__(self, *a):
        G.paired_bootstrap, G.exact_harm_test = self.saved


def _synthetic_records(effect, pairs, sigma, rng, start, row="agent-dev", cand="cand"):
    recs = []
    for i in range(start, start + pairs):
        base = rng.gauss(0.6, 0.15)
        recs.append({"row": row, "arm": "stack", "task": f"t{i}", "family": f"t{i}", "rep": 0, "seed": 1, "score": base})
        recs.append({"row": row, "arm": cand, "task": f"t{i}", "family": f"t{i}", "rep": 0, "seed": 1,
                     "score": base + effect + rng.gauss(0, sigma)})
    return recs


def simulate(effect, reps=1000, pairs_per_look=30, looks=3, sigma=0.22, delta=DEFAULT_DELTA, target=True, seed=20261113,
             bootstrap_reps=10000):
    """Sequential shadow with gate.py zones: up to `looks` looks on the combined data (30, 60, 90 pairs), stop at the first green
    or red. Pair noise: the candidate-minus-stack difference of one task has SD `sigma` around `effect`.
    Returns shares of repeats that ended green / red / grey, and the share per look."""
    rng = random.Random(seed)
    pre = {"candidate": "cand", "control": "stack", "metric": "score", "k": 1, "looks": looks, "alpha": 0.05,
           "bootstrap_reps": bootstrap_reps, "seed": 1,
           "rows": [{"name": "agent-dev", "kind": "agent", "target": target, "delta": delta}]}
    ends = {"green": 0, "red": 0, "grey": 0}
    by_look = {z: [0] * looks for z in ("green", "red")}
    with FastGate():
        for r in range(reps):
            recs = []
            zone = "grey"
            for look in range(1, looks + 1):
                recs += _synthetic_records(effect, pairs_per_look, sigma, rng, (look - 1) * pairs_per_look)
                run = dict(pre, look=look, seed=seed + r)
                res = G.analyze_row(dict(pre["rows"][0], n=look * pairs_per_look, k=1, seed=seed + r), recs, run)
                zone = res["zone"]
                if zone != "grey":
                    by_look[zone][look - 1] += 1
                    break
            ends[zone] += 1
    return {"effect": effect, "reps": reps, "pairs_per_look": pairs_per_look, "looks": looks, "sigma": sigma, "delta": delta,
            "target_row": target, "share": {z: ends[z] / reps for z in ends},
            "by_look": {z: [x / reps for x in v] for z, v in by_look.items()}}


def check_fast_gate(cases=30, pairs=30, seed=7):
    """Zones of the numpy statistics against the original gate.py on random data sets (borderline cases may differ by the
    bootstrap's own randomness)."""
    rng = random.Random(seed)
    agree, rows = 0, []
    for c in range(cases):
        effect = rng.choice([0.0, -0.05, -0.15, 0.05])
        recs = _synthetic_records(effect, pairs, 0.22, rng, 0)
        pre = {"candidate": "cand", "control": "stack", "metric": "score", "k": 1, "looks": 3, "alpha": 0.05, "look": 1,
               "bootstrap_reps": 10000, "seed": 1, "rows": [{"name": "agent-dev", "kind": "agent", "target": True, "delta": 0.03}]}
        row = dict(pre["rows"][0], n=pairs, k=1)
        slow = G.analyze_row(row, recs, pre)
        with FastGate():
            fast = G.analyze_row(row, recs, pre)
        agree += slow["zone"] == fast["zone"]
        rows.append({"effect": effect, "slow": slow["zone"], "fast": fast["zone"],
                     "lo": [round(slow["lower_bound"], 4), round(fast["lower_bound"], 4)],
                     "hi": [round(slow["upper_bound"], 4), round(fast["upper_bound"], 4)],
                     "p_harm": [round(slow["exact_p_harm"], 4), round(fast["exact_p_harm"], 4)]})
    return {"cases": cases, "agree": agree, "rows": rows}

# --------------------------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init")
    a.add_argument("name")
    a.add_argument("--switch", required=True)
    a.add_argument("--adds", default=None, help="JSON merged into every request of the candidate arm")
    a.add_argument("--profile", default=None)
    a.add_argument("--target-row", action="store_true")
    a.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    a.add_argument("--looks", type=int, default=MAX_LOOKS)
    a.add_argument("--life-days", type=float, default=MAX_LIFE_DAYS)
    a.add_argument("--tasks-per-night", type=int, default=10)
    a.add_argument("--rows", default=None, help='JSON list of rows, e.g. [{"name": "own", "target": true, "tasks": {"kind": "bugfix"}}, '
                                                '{"name": "rest", "tasks": {}}] (default: one not-worse row over all tasks)')
    p = sub.add_parser("plan")
    p.add_argument("names", nargs="+")
    p.add_argument("--server", required=True, help="server.json: {exe, args, prefix} as in run_experiment's arms.json")
    p.add_argument("--out", required=True)
    p.add_argument("--tasks-per-night", type=int, default=None)
    p.add_argument("--stack-extra", default=None)
    p.add_argument("--night", type=int, default=None)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--max-hours", type=float, default=3.0)
    i = sub.add_parser("ingest")
    i.add_argument("run_dir")
    i.add_argument("names", nargs="+")
    v = sub.add_parser("verdict")
    v.add_argument("name")
    sub.add_parser("summary")
    m = sub.add_parser("simulate")
    m.add_argument("--reps", type=int, default=1000)
    m.add_argument("--effects", default="0,-0.15")
    m.add_argument("--no-target", action="store_true")
    m.add_argument("--check", type=int, default=0, help="first compare the numpy statistics with gate.py on this many data sets")
    o = ap.parse_args(argv)
    if o.cmd == "init":
        pre = init(o.name, o.switch, json.loads(o.adds) if o.adds else None, o.profile, o.target_row, o.delta, o.looks,
                   o.life_days, tasks_per_night=o.tasks_per_night, rows=json.loads(o.rows) if o.rows else None, store=o.store)
        print(json.dumps(pre, indent=1, ensure_ascii=False))
    elif o.cmd == "plan":
        res = plan(o.names, json.loads(Path(o.server).read_text()), o.out, o.tasks_per_night,
                   json.loads(o.stack_extra) if o.stack_extra else None, o.night, store=o.store, parallel=o.parallel,
                   max_hours=o.max_hours)
        print(json.dumps(res, indent=1, ensure_ascii=False))
    elif o.cmd == "ingest":
        print(json.dumps(ingest(o.run_dir, o.names, o.store)))
    elif o.cmd == "verdict":
        v = verdict(o.name, o.store)
        print(f"{o.name}: {v['status']} (заход {v['look']}); {row_line(v)}")
    elif o.cmd == "summary":
        print(summary(o.store))
    elif o.cmd == "simulate":
        if o.check:
            print(json.dumps(check_fast_gate(o.check), ensure_ascii=False), flush=True)
        for e in (float(x) for x in o.effects.split(",")):
            print(json.dumps(simulate(e, o.reps, target=not o.no_target), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
