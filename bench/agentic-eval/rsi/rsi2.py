"""RSI2 (RSI2D): the cause-driven harness cycle. The old task-by-task pilot stays as `pilot`; this is the `cycle2` step.

What changed against RSI1 (reports/REPORT-RSI1.md, REPORT-RSI2D.md):
- candidates are born from FREQUENT CAUSES over the whole set of base runs (mine_causes), one cause -> one short edit, not from one task;
- the measurement set is the GROWTH set: tasks with room (base mean <= 1 - min_room), chosen from train/dev base runs only, never test;
  a short SAFETY set of ceiling tasks checks that nothing broke;
- the number of repeats comes from the power table (simulate), the decision is a pre-registered rule (decide2): paired family-cluster
  bootstrap, one-sided lower bound at alpha / K (Bonferroni over the K candidates of the night) above zero, resolved not worse, cost
  within the limits. No editor can override it; the composer only MERGES the accepted edits, and the merge is measured again;
- base runs are reused across the candidates of one night (and from an earlier night of the same harness fingerprint, with a drift
  check), the final test runs once at the end.
"""
import collections
import hashlib
import json
import os
import random
import re
import statistics
import time
from pathlib import Path

from . import accept, driver, executor, pilot, trajectory

MIN_ROOM = 0.1            # a growth task has at least this much score to gain (base mean <= 1 - MIN_ROOM)
SAFETY_N = 8              # ceiling tasks in the safety set (one repeat each: at the ceiling the noise is ~0)
SAFETY_MAX_DROP = 0.05    # the combined edit may not lower the safety set's mean score by more than this
ALPHA = 0.05
DRIFT_TASKS = 4           # reused base runs are trusted when this many fresh runs agree with them (drift check)
PRIOR_LOG_ROOT = Path("/home/alisher/Desktop/AI/reports/rsil1/runs")

# ---- trajectory features and causes -------------------------------------------------------------------------------------------
NUDGE = "Continue if you have next steps"
COMPACTION = "Let me analyze the conversation"


def log_features(log_path):
    """What the agent log shows about HOW the run went (no verdict): reasoning overflow, compactions, nudges, refused (mistyped) paths,
    output tokens, when the first write happened."""
    f = dict(max_reason=0, compactions=0, nudges=0, refusals=0, misspelled=0, length_steps=0, steps=0, first_write_step=None,
             out_tokens=0, test_runs=0, edits=0)
    p = Path(log_path)
    if not p.exists():
        return f
    work = None
    with open(p, errors="replace") as fh:
        lines = fh.readlines()
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        t, part = e.get("type"), e.get("part") or {}
        if t == "user" and NUDGE in (e.get("text") or ""):
            f["nudges"] += 1
        elif t == "reasoning":
            text = part.get("text") or ""
            f["max_reason"] = max(f["max_reason"], len(text))
            if text.startswith(COMPACTION):
                f["compactions"] += 1
        elif t == "step_finish":
            f["steps"] += 1
            f["out_tokens"] += (part.get("tokens") or {}).get("output") or 0
            if part.get("reason") == "length":
                f["length_steps"] += 1
        elif t == "tool_use":
            st = part.get("state") or {}
            inp = st.get("input") or {}
            path = str(inp.get("filePath") or "")
            if work is None and st.get("status") == "completed" and "/work" in path:
                work = path.split("/work")[0] + "/work"
            err = str(st.get("error") or "") + (str(st.get("output") or "") if st.get("status") == "error" else "")
            if "specified a rule" in err:
                f["refusals"] += 1
                if work and not path.startswith(work) or (not work and path.startswith("/")):
                    f["misspelled"] += 1
            if part.get("tool") in ("write", "edit"):
                f["edits"] += 1
                if f["first_write_step"] is None:
                    f["first_write_step"] = f["steps"]
            if part.get("tool") == "bash" and re.search(r"\b(pytest|unittest|node --test|npm test|\.tmp/|test_)", str(inp.get("command") or "")):
                f["test_runs"] += 1
    return f


# score causes: the name, the test for it, whether a harness edit can plausibly address it (the report's column), the editor brief
CAUSES = collections.OrderedDict([
    ("error_contract", dict(fixable=True, brief="the task names error cases (wrong type, out of range, invalid input) and the hidden "
                                                "tests check that the stated exception is raised; the run implemented the happy path and "
                                                "skipped some error clauses")),
    ("deliverables", dict(fixable=True, brief="the task asks for a project-level deliverable besides the code (a package layout, a "
                                              "README or design note to complete, a file count); the run left it out although the code passed")),
    ("timeout_reasoning", dict(fixable=True, brief="the run timed out: long reasoning (tens of thousands of characters per step), context "
                                                   "compactions that lost the drafted code, the first write came late")),
    ("logic", dict(fixable=False, brief="a wrong result on normal inputs (the model's own mistake, no harness pattern)")),
])
COST_CAUSES = ("path_misspell", "compaction")


def tag_run(rec, feats):
    """The causes of one run's loss (score < 1 or timeout), from the hidden-test tail and the log features. Cost tags do not lose score."""
    tags = []
    tail = (rec.get("test_tail") or "")
    score = rec.get("score") or 0.0
    lost = score < 1 or rec.get("timed_out")
    if lost:
        if rec.get("timed_out") and (feats["max_reason"] >= 20000 or feats["compactions"] >= 2):
            tags.append("timeout_reasoning")
        elif re.search(r"not raised|operator: 'throws'|assertRaises|assert\.throws|Error not raised", tail):
            tags.append("error_contract")
        elif re.search(r"existsSync|Starter project|supporting|package is split|README|test_layout|test_project_docs|generatedMessage: true", tail):
            tags.append("deliverables")
        elif rec.get("timed_out"):
            tags.append("timeout_reasoning")
        else:
            tags.append("logic")
    if feats["misspelled"]:
        tags.append("path_misspell")
    if feats["compactions"]:
        tags.append("compaction")
    return tags


def mine_causes(records):
    """records: [(rec, log_path)] of BASE runs (train and dev, this night and earlier nights). Returns the cause table: runs, tasks,
    lost score, share of all lost score, fixable, example runs (for the editor's pool), ranked by lost score among the fixable ones."""
    table = collections.OrderedDict((c, dict(cause=c, runs=0, tasks=set(), lost=0.0, examples=[], fixable=v["fixable"], brief=v["brief"]))
                                    for c, v in CAUSES.items())
    cost = collections.Counter()
    lost_total = 0.0
    for rec, log in records:
        feats = log_features(log)
        lost_total += 1 - (rec.get("score") or 0.0)
        for tag in tag_run(rec, feats):
            if tag in table:
                row = table[tag]
                row["runs"] += 1
                row["tasks"].add(rec["task"])
                row["lost"] += 1 - (rec.get("score") or 0.0)
                row["examples"].append((rec, log))
            else:
                cost[tag] += 1
    rows = []
    for row in table.values():
        rows.append(dict(row, tasks=sorted(row["tasks"]), lost=round(row["lost"], 3),
                         share=round(row["lost"] / lost_total, 3) if lost_total else 0.0))
    rows.sort(key=lambda r: (-r["fixable"], -r["lost"]))
    return {"causes": rows, "lost_total": round(lost_total, 3), "cost_tags": dict(cost), "runs": len(records)}


def prior_records(run_name, root=PRIOR_LOG_ROOT):
    """The base runs (train base and dev) of an earlier run folder with their agent logs, for cause mining without the GPU."""
    state = driver.load_state(run_name) if (driver.RUNS / run_name / "state.json").exists() else json.loads((root / run_name / "state.json").read_text())
    base = root / run_name
    out = []
    for key, step in state.get("steps", {}).items():
        rec = step.get("value") if isinstance(step, dict) else None
        if not isinstance(rec, dict) or "score" not in rec:
            continue
        arm = rec.get("arm") or ""
        task = rec["task"]
        if arm == "base":
            tdir = key.split(".base.")[0]
            log = base / "runs" / tdir / f"{task}__base__{rec['rep']}.agent.log"
        elif arm.startswith("dev-"):
            log = base / "runs" / "dev" / arm[4:] / f"{task}__{arm}__{rec['rep']}.agent.log"
        else:
            continue
        out.append((rec, log))
    return out


# ---- the sets ---------------------------------------------------------------------------------------------------------------------
TASKS_ROLES = Path(__file__).resolve().parent.parent / "tasks_roles"


def ru_safety_tasks(n, root=TASKS_ROLES, splits=None, seed="rsi2-ru"):
    """RSI2E: n Russian role tasks from the dev part (never test) for the safety set, chosen by hash so the choice is stable."""
    if not n:
        return []
    splits = splits or json.loads((Path(__file__).resolve().parent.parent / "splits.json").read_text())["families"]
    tasks = [t for t in executor.all_tasks(root) if t.get("prompt_lang") == "ru" and splits.get(t["family"]) == "dev"]
    tasks.sort(key=lambda t: hashlib.sha256(f"{seed}:{t['id']}".encode()).hexdigest())
    return tasks[:n]

def task_means(records):
    by = collections.defaultdict(list)
    for rec, _ in records:
        by[rec["task"]].append(rec.get("score") or 0.0)
    return {t: statistics.mean(v) for t, v in by.items()}


def growth_set(tasks, records, min_room=MIN_ROOM, max_tasks=None):
    """The tasks with room: base mean <= 1 - min_room, most room first; `tasks` are the task dicts allowed (train and dev, never test)."""
    means = task_means(records)
    allowed = {t["id"]: t for t in tasks}
    chosen = [(means[t], t) for t in means if t in allowed and means[t] <= 1 - min_room]
    chosen.sort(key=lambda x: (x[0], x[1]))
    out = [allowed[t] for _, t in chosen]
    return out[:max_tasks] if max_tasks else out


def safety_set(tasks, records, n=SAFETY_N, seed="rsi2"):
    """Ceiling tasks (base mean 1.0), n of them by a hash so the choice does not depend on the night."""
    means = task_means(records)
    ceiling = [t for t in tasks if means.get(t["id"]) == 1.0]
    ceiling.sort(key=lambda t: hashlib.sha256(f"{seed}:{t['id']}".encode()).hexdigest())
    return ceiling[:n]


# ---- the pre-registered rule ---------------------------------------------------------------------------------------------------
def t_quantile(p, df):
    """The one-sided t quantile from the normal one (Cornish-Fisher expansion; within ~0.01 for df >= 5, exact enough for the rule)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(p)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


def family_lower(pairs, alpha):
    """One-sided lower bound of the mean paired difference with FAMILIES as the units: mean of family means minus t(1 - alpha, n - 1) times
    their standard error. The percentile bootstrap undercovers with 7-18 clusters (simulate_rules: 6-10 % false acceptance per cycle
    against 2.5-4.5 % for this bound), so the rule uses this; the bootstrap numbers are reported beside it."""
    fam = collections.defaultdict(list)
    for f, d in pairs:
        fam[f].append(d)
    fm = [statistics.mean(v) for v in fam.values()]
    n = len(fm)
    if n < 2:
        return (statistics.mean(fm) if fm else 0.0), -1.0, n
    mean = statistics.mean(fm)
    se = statistics.stdev(fm) / (n ** 0.5)
    return mean, mean - t_quantile(1 - alpha, n - 1) * se, n


def decide2(candidate, base, k=1, alpha=ALPHA, thresholds=None, allow_stub=False, cost_decides=False):
    """Paired (task, rep) differences candidate - base, families as the units. Accept when the one-sided t lower bound at alpha / k of the
    mean score difference is above zero (family_lower), the resolved lower bound is above -0.03 and the cost ratios are within the limits
    (accept.decide's). k = the candidates measured against the same base this night (Bonferroni). The family bootstrap interval is
    reported for comparison, it does not decide. RSI2E: the cost ratios are informational (cost_decides=False, the candidate measurement:
    the A/A recount of RSI1 showed 53 % false cost refusals at few pairs) and decide only on the confirmation of the combined version
    (cost_decides=True)."""
    th = thresholds or accept.STUB
    if th is accept.STUB and not allow_stub:
        raise RuntimeError("decide2 on STUB thresholds: pass allow_stub=True (and say so in the report)")
    b = {(r["task"], r.get("rep", 1)): r for r in base}
    pairs_s, pairs_r, time_ratio, tok_ratio = [], [], [], []
    per_family = collections.defaultdict(list)
    for c in candidate:
        r = b.get((c["task"], c.get("rep", 1)))
        if r is None:
            continue
        d = c["score"] - r["score"]
        pairs_s.append((c["family"], d))
        pairs_r.append((c["family"], float(bool(c["resolved"])) - float(bool(r["resolved"]))))
        per_family[c["family"]].append(d)
        if r.get("seconds"):
            time_ratio.append(c["seconds"] / r["seconds"])
        if r.get("tokens"):
            tok_ratio.append(c.get("tokens", 0) / r["tokens"])
    if not pairs_s:
        return {"accept": False, "reason": "no paired runs", "k": k}
    a = alpha / max(1, k)
    s_mean, s_lo, n_fam = family_lower(pairs_s, a)
    r_mean, r_lo, _ = family_lower(pairs_r, a)
    b_mean, b_lo, _ = accept.paired_bootstrap(pairs_s, alpha=2 * a)      # reported only: the two-sided interval at 2a, lower bound at a
    t = statistics.mean(time_ratio) if time_ratio else 1.0
    kk = statistics.mean(tok_ratio) if tok_ratio else 1.0
    ok_score, ok_res = s_lo > th["score_ci_low_min"], r_lo > th["resolved_ci_low_min"]
    time_max = th.get("time_growth_max_significant", th["cost_growth_max"]) if (ok_score and ok_res) else th["cost_growth_max"]
    ok_cost = t <= time_max and kk <= th["cost_growth_max"]
    regressions = {f: round(statistics.mean(v), 4) for f, v in per_family.items() if statistics.mean(v) < -0.1}
    ok_all = bool(ok_score and ok_res and not regressions and (ok_cost or not cost_decides))
    return {"accept": ok_all, "cost_decides": cost_decides, "k": k, "alpha_each": round(a, 4), "families": n_fam,
            "score": {"mean": round(s_mean, 4), "lower": round(s_lo, 4), "ok": ok_score, "bootstrap_lower": b_lo},
            "resolved": {"mean": round(r_mean, 4), "lower": round(r_lo, 4), "ok": ok_res},
            "time_ratio": round(t, 3), "tokens_ratio": round(kk, 3), "time_limit": time_max, "cost_ok": ok_cost,
            "family_regressions": regressions, "pairs": len(pairs_s), "stub": th is accept.STUB, "rule": "RSI2D decide2 (pre-registered)"}


def safety_ok(candidate, base, max_drop=SAFETY_MAX_DROP):
    """The safety set: the paired mean score difference may not fall below -max_drop, and no task may go from resolved to unresolved in
    every repeat."""
    b = {(r["task"], r.get("rep", 1)): r for r in base}
    diffs, broke = [], []
    by_task = collections.defaultdict(list)
    for c in candidate:
        r = b.get((c["task"], c.get("rep", 1)))
        if r is None:
            continue
        diffs.append(c["score"] - r["score"])
        by_task[c["task"]].append((bool(c["resolved"]), bool(r["resolved"])))
    for task, v in by_task.items():
        if all((not cr) and br for cr, br in v):
            broke.append(task)
    mean = statistics.mean(diffs) if diffs else 0.0
    return {"ok": bool(diffs) and mean > -max_drop and not broke, "mean": round(mean, 4), "broke": broke, "pairs": len(diffs)}


def drift_ok(fresh, reused, sigma=0.213, z=2.0):
    """Reused base runs (an earlier night, same harness fingerprint) are trusted when the fresh runs of the same tasks do not differ in
    mean by more than z * sigma / sqrt(n)."""
    b = collections.defaultdict(list)
    for r in reused:
        b[r["task"]].append(r["score"])
    diffs = [r["score"] - statistics.mean(b[r["task"]]) for r in fresh if r["task"] in b]
    if not diffs:
        return {"ok": True, "n": 0}
    mean = statistics.mean(diffs)
    bound = z * sigma / (len(diffs) ** 0.5)
    return {"ok": abs(mean) <= bound, "mean": round(mean, 4), "bound": round(bound, 4), "n": len(diffs)}


# ---- the power simulation (the report's table; the tests run it small) ---------------------------------------------------------
def simulate(levels, residuals, n_tasks, r, k, deltas, cycles=200, alpha=ALPHA, seed=1, boot_reps=300):
    """levels: base means of headroom tasks; residuals: empirical within-task run residuals. One cycle = fresh base runs (r per task) and k
    candidates with the given deltas (an edit's effect applies to headroom tasks only, scores are clipped to [0, 1]). Returns the
    acceptance share per delta position and the per-cycle share with any null candidate accepted."""
    rng = random.Random(seed)
    acc = [0] * len(deltas)
    any_null = 0

    def lower(pairs, a):
        return family_lower(pairs, a)[1]

    for _ in range(cycles):
        tasks = [rng.randrange(len(levels)) for _ in range(n_tasks)]
        fams = [i // 2 for i in range(n_tasks)]           # families of two (the py_/js_ pair of one idea)
        base = [[min(1.0, max(0.0, levels[t] + rng.choice(residuals))) for _ in range(r)] for t in tasks]
        null_hit = False
        for j, delta in enumerate(deltas):
            cand = [[min(1.0, max(0.0, levels[t] + delta + rng.choice(residuals))) for _ in range(r)] for t in tasks]
            pairs = [(fams[i], cand[i][x] - base[i][x]) for i in range(n_tasks) for x in range(r)]
            res = statistics.mean((cand[i][x] >= 1) - (base[i][x] >= 1) for i in range(n_tasks) for x in range(r))
            ok = lower(pairs, alpha / max(1, k)) > 0 and res > -0.03
            acc[j] += ok
            if ok and delta <= 0:
                null_hit = True
        any_null += null_hit
    return {"accept": [a / cycles for a in acc], "false_accept_cycle": any_null / cycles, "runs": n_tasks * r * (k + 1)}


def noise_from_rows(rows):
    """From RSI1-style run rows [(key, arm, task, rep, score, ...)] of base and dev runs: the headroom tasks' levels and residuals."""
    by = collections.defaultdict(list)
    for r in rows:
        if r[1] == "base" or str(r[1]).startswith("dev-"):
            by[r[2]].append(r[4])
    levels, resid = [], []
    for t, v in by.items():
        if min(v) < 1 and len(v) >= 2:
            m = statistics.mean(v)
            levels.append(m)
            resid.extend((s - m) * (2 ** 0.5) for s in v)
    return levels, resid


# ---- LHSET1: the long-horizon final block ----------------------------------------------------------------------------------------
TASKS_LH = Path(__file__).resolve().parent.parent / "tasks_lh"


def lh_tasks(part="dev", root=TASKS_LH):
    """The long-horizon tasks of one part (dev or test) by tasks_lh/splits_lh.json; start/ is built from the pinned repositories first."""
    splits = root / "splits_lh.json"
    if not splits.exists():
        return []
    fams = json.loads(splits.read_text(encoding="utf-8"))["families"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_tasks_lh", root / "build_tasks_lh.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build()
    out = []
    for tj in sorted(root.glob("*/*/task.json")):
        t = json.loads(tj.read_text(encoding="utf-8"))
        t["dir"] = tj.parent
        if fams.get(t["family"]) == part:
            out.append(t)
    return out


def lh_metrics(records, out_dir_of):
    """The LHSET1 metrics (lh/metrics_lh.py) of every record: from its agent log and the grade's JSON parts in its test tail."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lh import metrics_lh
    rows = []
    for r in records:
        log = out_dir_of(r)
        m = metrics_lh.compute(log) if log and Path(log).exists() else {}
        m.update(metrics_lh.from_grade(metrics_lh.grade_parts(r.get("test_tail"))))
        rows.append({"task": r["task"], "rep": r.get("rep"), "score": r.get("score"), "resolved": r.get("resolved"), "timed_out": r.get("timed_out"),
                     **{k: v for k, v in m.items() if not isinstance(v, list) or k in ("regressions", "exact_contract_failed")}})
    return rows


def lh_calibration(name, model, runner, part="dev", n=8, repeats=2, harness_root=None):
    """LHCAL1: the long-horizon tasks of one part under one harness (the product's config by default), each run its own saved step, the
    script metrics per run and a summary; no decision. Writes RUNS/<name>/state.json like a night and returns the summary."""
    state = driver.load_state(name)
    steps = state.setdefault("steps", {})
    meta = state.setdefault("meta", {})
    meta.update({"scheme": "lhcal", "model": model, "part": part, "repeats": repeats, "started": meta.get("started") or time.time(), "status": "running"})
    root = Path(harness_root) if harness_root else pilot.HARNESS_CFG
    fp = executor.fingerprint(root / model) if (root / model).exists() else "none"
    tasks = lh_tasks(part)[:n]
    out = driver.RUNS / name / "runs" / "lh" / fp
    recs = []
    for t in tasks:
        for rep in range(1, repeats + 1):
            key = f"lh.{fp}.{t['id']}.{rep}"
            if key not in steps:
                rec = runner.run(t, root, model, f"lh-{fp}", rep, out)
                steps[key] = {"value": rec, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                driver.save_state(name, state)
            recs.append(steps[key]["value"])
    rows = lh_metrics(recs, lambda r: out / f"{r['task']}__lh-{fp}__{r['rep']}.agent.log")
    keys = ("score", "hallucinated_paths", "repeated_calls", "compactions", "recovery_calls_total", "unverified_done", "start_rule_violation")
    def mean(k):
        vals = [float(r[k]) for r in rows if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        return round(statistics.mean(vals), 3) if vals else None
    summary = {"tasks": [t["id"] for t in tasks], "repeats": repeats, "harness": fp, "resolved": sum(1 for r in recs if r.get("resolved")), "of": len(recs),
               "timed_out": sum(1 for r in recs if r.get("timed_out")), "seconds_mean": round(statistics.mean(r.get("seconds") or 0 for r in recs), 1) if recs else None,
               "means": {k: mean(k) for k in keys}, "rows": rows}
    steps["lh.summary"] = {"value": summary, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    meta.update(status="done", ended=time.time())
    driver.save_state(name, state)
    return summary


# ---- the cycle -------------------------------------------------------------------------------------------------------------------
class Cycle2(pilot.Pilot):
    """The RSI2 night. Steps (each saved, a rerun skips the finished ones):
    base runs on the candidate pool of tasks -> growth and safety sets -> causes -> one proposal per cause -> measurement of every
    candidate on the growth set -> the pre-registered decision (k = number of candidates) -> merge of the accepted -> the merge measured
    on growth + safety -> version -> final test once."""

    def __init__(self, *args, max_candidates=3, min_room=MIN_ROOM, prior_runs=(), max_growth=None, cause_editor="opus", mechanisms=(),
                 safety_ru=(), screen_reps=0, lh_final=0, **kw):
        super().__init__(*args, **kw)
        self.lh_final = lh_final          # LHSET1: run this many LH dev tasks (original vs accepted) once at the end, reported only
        self.max_candidates, self.min_room, self.prior_runs, self.max_growth = max_candidates, min_room, tuple(prior_runs), max_growth
        self.cause_editor = cause_editor
        # RSI2E: mechanism plugins as candidates (one per name, measured by the same rule as a text edit); Russian role tasks for the
        # safety set; a one-repeat screen of the pool before the base runs, so the growth set is chosen cheaply
        self.mechanisms, self.safety_ru, self.screen_reps = tuple(mechanisms), list(safety_ru), screen_reps

    def _records(self, label, root, fp, tasks, reps):
        out = self.dir / "runs" / label / fp
        return [self._run_step(f"{label}.{fp}.{t['id']}.{rep}", t, root, f"{label}-{fp}", rep, out) for t in tasks for rep in range(1, reps + 1)]

    def _with_logs(self, label, fp, recs):
        return [(r, self.dir / "runs" / label / fp / f"{r['task']}__{label}-{fp}__{r['rep']}.agent.log") for r in recs]

    def run_cycle(self):
        self.meta.setdefault("started", time.time())
        self.meta.update({"model": self.model, "fake": self.fake, "repeats": self.repeats, "limits": self.limits, "timebox_min": self.timebox_min,
                          "thresholds": self.thresholds, "allow_stub": self.allow_stub, "status": "running", "scheme": "rsi2",
                          "max_candidates": self.max_candidates, "min_room": self.min_room, "prior_runs": list(self.prior_runs),
                          "train": [t["id"] for t in self.tasks["train"]], "dev": [t["id"] for t in self.tasks["dev"]],
                          "test": [t["id"] for t in self.tasks.get("test", [])]})
        self._save()
        try:
            status = self._night()
        except pilot.Stop as stop:
            status = f"stopped: {stop}"
        self.meta.update(status=status, ended=time.time())
        self._save()
        return status

    def _night(self):
        pool = self.tasks["train"] + self.tasks["dev"]
        fa = executor.fingerprint(self.acc_dir)
        prior = []
        for name in self.prior_runs:
            prior.extend(prior_records(name))
        if self.screen_reps:
            # 1a. RSI2E: a cheap screen of the pool (screen_reps per task) picks the growth and safety sets; the full base (repeats) runs
            # only on those sets. The screen runs count as base records for the mining and the choice, not for the paired decision.
            screen = self._records("s2", self.acc_root, fa, pool, self.screen_reps)
            records = prior + self._with_logs("s2", fa, screen)
            sets = self.step("sets", lambda: self._sets(pool, records))
            by_id = {t["id"]: t for t in pool + self.safety_ru}
            growth = [by_id[i] for i in sets["growth"]]
            safety = [by_id[i] for i in sets["safety"]]
            if not growth:
                return "done: no task with room (RSI2D step 2: the model is at the ceiling of this set; no harness cycle needed)"
            base_growth = self._records("b2", self.acc_root, fa, growth, self.repeats)
            base_safety = self._records("b2", self.acc_root, fa, safety, self.repeats)
            records = records + self._with_logs("b2", fa, base_growth)
        else:
            # 1b. the base on the whole pool (RSI2D)
            base = self._records("b2", self.acc_root, fa, pool, self.repeats)
            records = prior + self._with_logs("b2", fa, base)
            sets = self.step("sets", lambda: self._sets(pool, records))
            by_id = {t["id"]: t for t in pool + self.safety_ru}
            growth = [by_id[i] for i in sets["growth"]]
            safety = [by_id[i] for i in sets["safety"]]
            if not growth:
                return "done: no task with room (RSI2D step 2: the model is at the ceiling of this set; no harness cycle needed)"
            base_growth = [r for r in base if r["task"] in sets["growth"]]
            ru = [t for t in safety if t["id"] not in {x["id"] for x in pool}]
            base_safety = [r for r in base if r["task"] in sets["safety"]] + (self._records("b2", self.acc_root, fa, ru, self.repeats) if ru else [])
        # 2. the causes, from every base record (this night and the prior ones)
        mined = self.step("causes", lambda: self._causes(records))
        # --fake (the stand's plumbing check): the fake runs lose score for no harness reason, so every cause with runs is taken
        fixable = [c for c in mined["causes"] if (c["fixable"] or self.fake) and c["runs"] > 0][: self.max_candidates]
        if not fixable and not self.mechanisms:
            return "done: no repeated fixable cause (RSI2D step 2: a legitimate outcome)"
        # 3. one proposal per cause
        proposals = {}
        for c in fixable:
            proposals[c["cause"]] = self.step(f"propose.{c['cause']}", lambda c=c: self._propose_cause(c, records))
        diffs = {c: p["diff"] for c, p in proposals.items() if p.get("diff")}
        for name in self.mechanisms:          # RSI2E: a mechanism plugin is a candidate too: a folder diff that adds plugins.json
            diffs[f"plugin-{name}"] = self._mechanism_diff(name)
        if not diffs:
            return "done: no candidate (every editor kept the base or was refused)"
        k = len(diffs)
        self.meta["k"] = k
        # 4. every candidate on the growth set; the rule decides, nobody else
        verdicts = {}
        for cause, diff in diffs.items():
            verdicts[cause] = self.step(f"measure.{cause}", lambda cause=cause, diff=diff: self._measure(cause, diff, growth, base_growth, k))
        accepted = [c for c, v in verdicts.items() if v.get("applies") and v["verdict"]["accept"]]
        self.step("decision", lambda: {"k": k, "accepted": accepted,
                                       "verdicts": {c: {kk: vv for kk, vv in v.items() if kk != "runs"} for c, v in verdicts.items()}})
        if not accepted:
            return "done: KEEP_BASE (no candidate passed the rule)"
        # 5. the merge (composer merges only) and its own measurement on growth + safety
        combined = self.step("combine", lambda: self._combine(accepted, diffs, verdicts))
        confirm = self.step("confirm", lambda: self._confirm(combined["diff"], growth, safety, base_growth, base_safety))
        final_diff = combined["diff"]
        if not confirm["ok"] and len(accepted) > 1:
            best = max(accepted, key=lambda c: verdicts[c]["verdict"]["score"]["mean"])
            final_diff = diffs[best]
            confirm = self.step("confirm.single", lambda: self._confirm(final_diff, growth, safety, base_growth, base_safety, label="single"))
        if not confirm["ok"]:
            return "done: KEEP_BASE (the merge did not pass the confirmation)"
        self.step("version", lambda: self._adopt(final_diff))
        if self.tasks.get("test"):
            self.step("final.test", self._final_test)
        if self.lh_final:
            self.step("lh.final", self._lh_final)
        return "done: version adopted"

    def _sets(self, pool, records):
        safety = [t["id"] for t in safety_set(pool, records)] + [t["id"] for t in self.safety_ru]
        return {"growth": [t["id"] for t in growth_set(pool, records, self.min_room, self.max_growth)], "safety": safety,
                "safety_ru": [t["id"] for t in self.safety_ru], "means": {k: round(v, 3) for k, v in task_means(records).items()}}

    def _mechanism_diff(self, name):
        """A candidate that switches one mechanism plugin on: a diff adding <model>/plugins.json (the only difference to the base)."""
        if name not in executor.RSI2_PLUGINS:
            raise pilot.Stop(f"unknown mechanism {name!r}; known: {sorted(executor.RSI2_PLUGINS)}")
        body = json.dumps({"plugins": [{"name": name, "options": {executor.RSI2_PLUGINS[name]: True}}]}, indent=1) + "\n"
        lines = body.count("\n")
        return "--- /dev/null\n+++ b/plugins.json\n@@ -0,0 +1," + str(lines) + " @@\n" + "".join("+" + l + "\n" for l in body.splitlines())

    def _lh_final(self):
        """LHSET1: the long-horizon dev tasks under the original and the accepted harness, once, with the script metrics; reported only
        (no decision): resolved/score, hallucinated paths and symbols, start-rule violations, repeats, regressions, unverified done,
        compactions and the recovery cost after them."""
        tasks = lh_tasks("dev")[: self.lh_final]
        if not tasks:
            return {"note": "no LH dev tasks"}
        fo, fa = executor.fingerprint(self.orig_dir), executor.fingerprint(self.acc_dir)
        base = self._records("lh-orig", self.orig_root, fo, tasks, self.repeats)
        cand = self._records("lh-acc", self.acc_root, fa, tasks, self.repeats)
        rows_b = lh_metrics(base, lambda r: self.dir / "runs" / "lh-orig" / fo / f"{r['task']}__lh-orig-{fo}__{r['rep']}.agent.log")
        rows_c = lh_metrics(cand, lambda r: self.dir / "runs" / "lh-acc" / fa / f"{r['task']}__lh-acc-{fa}__{r['rep']}.agent.log")
        def mean(rows, k):
            vals = [float(r[k]) for r in rows if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
            return round(statistics.mean(vals), 3) if vals else None
        keys = ("score", "hallucinated_paths", "repeated_calls", "compactions", "recovery_calls_total", "unverified_done", "start_rule_violation")
        return {"tasks": [t["id"] for t in tasks], "repeats": self.repeats, "original": fo, "accepted": fa,
                "summary": {k: {"original": mean(rows_b, k), "accepted": mean(rows_c, k)} for k in keys},
                "resolved": {"original": sum(1 for r in base if r.get("resolved")), "accepted": sum(1 for r in cand if r.get("resolved")), "of": len(base)},
                "rows": {"original": rows_b, "accepted": rows_c}, "note": "reported only: LH tasks decide nothing (LHSET1)"}

    def _causes(self, records):
        mined = mine_causes(records)
        for row in mined["causes"]:
            row["examples"] = [(r["task"], r.get("rep"), str(log)) for r, log in row["examples"]][:12]
        return mined

    def _propose_cause(self, cause_row, records):
        """The editor sees the cause brief first, then the trajectories of the runs that show it (several tasks), and proposes ONE short
        edit. The same surface rules as the pilot's proposals (surface_ok, size, growth)."""
        self._budget(self.cause_editor)
        tid = f"cause-{cause_row['cause']}"
        pool_dir = self._pool_dir(tid)
        entries = []
        by_task = {r["task"]: r for r, _ in records}
        for task, rep, log in cause_row["examples"]:
            rec = next((r for r, l in records if r["task"] == task and r.get("rep") == rep), by_task.get(task))
            if rec and Path(log).exists():
                entries.append(trajectory.entry(rec, {"lang": rec.get("lang", "?")}, log, repeats=self.repeats))
        sel = trajectory.select(entries)
        trajectory.write_pool(pool_dir, sel["chosen"])
        brief = (f"# CAUSE: {cause_row['cause']}\n{cause_row['brief']}.\nSeen in {cause_row['runs']} base runs of {len(cause_row['tasks'])} tasks "
                 f"({', '.join(cause_row['tasks'][:8])}), losing {cause_row['lost']} score points ({round(100 * cause_row['share'])} % of all lost score).\n"
                 f"Propose ONE short edit that addresses THIS cause across these tasks. Nothing else.\n")
        (pool_dir / "00-CAUSE.txt").write_text(brief, encoding="utf-8")
        os.environ["RSI_FAKE_BASE"] = str(self.h_file)
        cand_dir = self.dir / "candidates" / tid
        try:
            diff = driver.propose(self.name, "cause", self.h_dir, None, usage_log=self.usage, editor=self.cause_editor, model=self.model,
                                  library_dir=self._library_view(), cand_dir=cand_dir, trajectories=pool_dir)
        except Exception as err:           # a leak or a refused surface is a refusal, not a crash (as in the pilot)
            return {"editor": self.cause_editor, "diff": "", "refused": "editor", "reason": str(err)[:200]}
        if not diff or not pilot.surface_ok(diff):
            return {"editor": self.cause_editor, "diff": "", "refused": "surface" if diff else None}
        size = pilot.surface_size(diff)
        root = self._trial(tid, diff)
        if not size["ok"] or root is None:
            return {"editor": self.cause_editor, "diff": "", "refused": "size" if not size["ok"] else "does not apply", "size": size}
        growth_info = self._growth_of(root)
        if not growth_info["ok"]:
            return {"editor": self.cause_editor, "diff": "", "refused": "budget", "growth": growth_info}
        return {"editor": self.cause_editor, "diff": diff, "refused": None, "size": size, "growth": growth_info, "cause": cause_row["cause"]}

    def _measure(self, cause, diff, growth, base_growth, k):
        root = self._trial(f"cause-{cause}", diff)
        if root is None:
            return {"applies": False}
        fp = executor.fingerprint(root / self.model)
        cand = self._records(f"c2.{cause}", root, fp, growth, self.repeats)
        verdict = decide2(cand, base_growth, k=k, thresholds=self.thresholds, allow_stub=self.allow_stub, cost_decides=False)
        fired = {}
        for r in cand:                      # RSI2E: a mechanism candidate must have fired; a plugin that never acted is not a measurement of it
            for name, n in (r.get("mechanism_events") or {}).items():
                fired[name] = fired.get(name, 0) + n
        if cause.startswith("plugin-"):
            name = cause[len("plugin-"):]
            if not fired.get(name) and name != "compact-preserve":   # compaction may legitimately not happen in a short run
                verdict = dict(verdict, accept=False, mechanism_never_fired=name)
        return {"applies": True, "verdict": verdict, "runs": pilot.paired_runs(cand, base_growth), "families": pilot.family_means(cand, base_growth),
                "mechanism_events": fired}

    def _combine(self, accepted, diffs, verdicts):
        """One accepted edit is the combination. Several: the composer merges them (it may not drop one; a merge that drops an accepted edit
        is replaced by the plain concatenation applied in order)."""
        if len(accepted) == 1:
            return {"diff": diffs[accepted[0]], "how": "single"}
        self._budget("fable")
        cand_dir = self.dir / "candidates" / "combine"
        cand_dir.mkdir(parents=True, exist_ok=True)
        for c in accepted:          # the composer reads <role>.diff files: the accepted edits under their cause names
            (cand_dir / f"{c}.diff").write_text(diffs[c], encoding="utf-8")
        evidence = {c: {k: v for k, v in verdicts[c]["verdict"].items()} for c in accepted}
        evidence["instruction"] = "MERGE ONLY: every listed edit passed the pre-registered rule; combine them into one diff and resolve overlaps; do not drop or weaken an edit."
        os.environ["RSI_FAKE_BASE"] = str(self.h_file)
        try:
            merged = driver.compose(self.name, self.h_dir, evidence, usage_log=self.usage, model=self.model, library_dir=self._library_view(), cand_dir=cand_dir)
        except Exception:
            merged = ""
        if merged and pilot.surface_ok(merged) and self._trial("combine", merged) is not None:
            return {"diff": merged, "how": "composer"}
        # the fallback: apply the accepted diffs one after another
        root = self.dir / "trials" / "combine-seq"
        if root.exists():
            pilot.shutil.rmtree(root)
        (root / self.model).mkdir(parents=True)
        pilot.copy_tree(self.h_dir, root / self.model)
        for c in accepted:
            if not driver.apply_diff(root / self.model, diffs[c]):
                return {"diff": diffs[accepted[0]], "how": "first accepted (the others did not apply on top)"}
        seq = driver.diff_folders(self.h_dir, root / self.model) if hasattr(driver, "diff_folders") else None
        if seq is None:
            return {"diff": diffs[accepted[0]], "how": "first accepted (no folder diff)"}
        return {"diff": seq, "how": "sequential"}

    def _confirm(self, diff, growth, safety, base_growth, base_safety, label="combined"):
        root = self._trial(f"confirm-{label}", diff)
        if root is None:
            return {"ok": False, "reason": "does not apply"}
        fp = executor.fingerprint(root / self.model)
        cand_g = self._records(f"confirm-{label}", root, fp, growth, self.repeats)
        v = decide2(cand_g, base_growth, k=1, thresholds=self.thresholds, allow_stub=self.allow_stub, cost_decides=True)
        s = {"ok": True, "pairs": 0}
        if safety:
            cand_s = self._records(f"safety-{label}", root, fp, safety, self.repeats)
            s = safety_ok(cand_s, base_safety)
        out = {"ok": bool(v["accept"] and s["ok"]), "growth": v, "safety": s, "fingerprint": fp}
        if self.general is not None and self.general.get("fast") and out["ok"]:
            screen = self._general_screen(f"confirm-{label}", diff)
            out["general"] = screen["gate"]
            out["ok"] = bool(screen["ok"])
        return out

    def _adopt(self, diff):
        root = self._trial("adopt", diff)
        if root is None:
            return {"applied": False}
        pilot.copy_tree(root / self.model, self.h_dir)
        pilot.copy_tree(self.h_dir, self.acc_dir)
        n = sum(1 for p in (self.library / "harnesses").glob("cp*")) + 1 if (self.library / "harnesses").exists() else 1
        pilot.copy_tree(self.acc_dir, self.library / "harnesses" / f"cp{n:03d}")
        self._index()
        self.library.mkdir(parents=True, exist_ok=True)
        with open(self.library / "EXPERIENCE.md", "a", encoding="utf-8") as f:
            f.write(f"\n## rsi2 {self.name} — version adopted (cp{n:03d})\n{diff[:2000]}\n")
        return {"applied": True, "fingerprint": executor.fingerprint(self.acc_dir), "checkpoint": f"cp{n:03d}"}
