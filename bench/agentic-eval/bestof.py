"""QTTS1: «лучшая из N по тестам» — best-of-N for one task, sequentially on one slot, each attempt in its own copy of the repo
with its own sampling seed, and the final state picked WITHOUT the hidden tests:

  (a) the task's visible tests (bugfix: public-repro.test.js / public_repro.py) or, for implement and refactor, the registry check
      written from the prompt (visible_v3/, QTTS1q), run from the ORIGINAL files in a copy of the attempt's tree, so an attempt
      that edited the repro cannot flatter itself;
  (b) on a tie, the agent's own checks: its test files in the tree (node --test discovery, unittest discover) and, after that,
      whether it ran the project's test command and how many throwaway checks it ran (harness.project_checks);
  (c) on a tie, the first attempt.

The hidden tests are run only AFTER the choice (`select` never sees them): the chosen attempt's result is the official one; the
other attempts' hidden results are kept for diagnostics only (first-attempt and oracle rates in the report).

N = 1 is harness.run_one itself (the same code path, so the result record is byte for byte the old one). N > 1 adds the field
"best_of" to the record and sums the attempts' time and tokens into "seconds" / "tokens" (the price of the mode), keeping the
chosen attempt's own numbers in best_of.attempts[].

    python3 bestof.py summary results.jsonl        # per arm: solved with a CI, time and tokens ×N, how often not the first attempt
"""
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import proxy as proxy_mod  # noqa: E402
import visible  # noqa: E402

VISIBLE_TIMEOUT_S = 120
OWN_RUNNERS = {"python": ["python3", "-B", "-m", "unittest", "discover", "-q", "-s", ".", "-p", "test*.py"],
               "javascript": ["node", "--test"]}
GRADER_DIRS = ("hidden_tests", "visible_tests", "__pycache__", "node_modules", ".tmp")


# ---- what the attempt may be judged by -------------------------------------------------------------------------------------
def visible_files(task):
    """The names the visible check puts into the tree (the task's own visible files, or the registry's check: visible.py)."""
    return visible.names(task)


def _tree_copy(work):
    """A throwaway copy of the attempt's tree without the graders' folders: the checks run there, the tree itself is not touched."""
    tmp = Path(tempfile.mkdtemp(prefix="bestof-check-", dir=str(Path(work).parent)))
    dst = tmp / "tree"
    shutil.copytree(work, dst, ignore=shutil.ignore_patterns(*GRADER_DIRS))
    return tmp, dst


def _run(cmd, cwd, timeout):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "NO_COLOR": "1"})
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except OSError as err:
        return 127, str(err)


def visible_check(task, work):
    """(a): visible.run_in_copy — the task's visible tests or the registry's check (QTTS1q), from the original files, on a copy."""
    res = visible.run_in_copy(task, work)
    return {k: v for k, v in res.items() if k != "output"}


def own_check(task, work, log_path):
    """(b): the agent's own tests in its tree (discovery by the language's runner; the visible files are removed first, they are (a)'s),
    plus what the log says it ran. {"has": bool, "ok": bool, "total": int, "tests_run_project": bool, "adhoc_checks": int}."""
    runner = OWN_RUNNERS.get(task.get("lang"))
    log_facts = harness.project_checks(log_path) if Path(log_path).exists() else {"tests_run_project": False, "adhoc_checks": 0}
    if not runner:
        return {"has": False, **log_facts}
    tmp, tree = _tree_copy(work)
    try:
        for f in visible_files(task):
            (tree / f).unlink(missing_ok=True)
        code, out = _run(runner, tree, VISIBLE_TIMEOUT_S)
        soft = harness.soft_score(runner, code, out)
        total = soft.get("tests_total") or 0
        if task.get("lang") == "python" and total == 0:
            m = re.search(r"^Ran (\d+) tests?", out, re.MULTILINE)
            total = int(m.group(1)) if m else 0
        return {"has": total > 0, "ok": total > 0 and code == 0, "total": total, **log_facts}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def rank_key(attempt):
    """The order of the rule: (a) visible score, then (b) own tests pass, their count, the project's tests run, throwaway checks,
    then (c) the earlier attempt. Attempts without a signal rank below attempts with one at every step."""
    v, o = attempt["visible"], attempt["own"]
    return ((v.get("score", 1.0 if v.get("ok") else 0.0) if v.get("has") else -1.0),
            (1 if o.get("ok") else 0) if o.get("has") else -1,
            o.get("total", 0) if o.get("has") else -1,
            1 if o.get("tests_run_project") else 0,
            int(o.get("adhoc_checks") or 0),
            -attempt["index"])


def select(attempts):
    """The chosen attempt's index and the step of the rule that decided it. Pure: reads only visible / own / index."""
    keys = [rank_key(a) for a in attempts]
    best = max(range(len(attempts)), key=lambda i: keys[i])
    others = [k for i, k in enumerate(keys) if i != best]
    if not others:
        return best, "only"
    top = keys[best]
    for step, name in enumerate(("visible", "own_pass", "own_count", "project_tests", "adhoc_checks")):
        if any(k[step] != top[step] for k in others):
            return best, name
    return best, "first"


# ---- the attempts ----------------------------------------------------------------------------------------------------------
def attempt_seed(base, task_id, rep, index):
    """A seed per attempt that a rerun reproduces: the experiment's seed, the task, the repetition and the attempt number."""
    import zlib
    return (int(base) * 1_000_003 + zlib.crc32(f"{task_id}:{rep}:{index}".encode())) % 2_147_483_647


class _SeedProxy:
    """A measuring proxy in front of the arm's proxy for ONE attempt: it adds the attempt's `seed` to every request (llama-server's
    sampling seed) and keeps the attempt's own request log, so time and tokens are known per attempt."""

    def __init__(self, upstream_port, seed, log_path):
        self.log = Path(log_path)
        self.srv = proxy_mod.serve(0, f"http://127.0.0.1:{upstream_port}", str(self.log), {"seed": int(seed)})
        self.port = self.srv.server_port
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def run_best_of(task, arm, rep, port, proxy_log, out_dir, n=1, seed=0, scratch_root="/dev/shm/agentic-eval", context=32768,
                runner=None, keep_hidden_of_others=True):
    """Best-of-`n` for one cell. n == 1: harness.run_one unchanged. Otherwise n sequential attempts, the rule above, the hidden
    tests after the choice. The record is run_one's for the chosen attempt with the price summed over the attempts and the field
    "best_of" (n, chosen, reason, seeds, per-attempt facts, first-attempt and oracle hidden results)."""
    if int(n) <= 1:
        return harness.run_one(task, arm, rep, port, proxy_log, out_dir, scratch_root=scratch_root, context=context, runner=runner)
    from qlev.arms import is_arm
    qlev_arm = arm if is_arm(arm) else None
    runner = runner or harness.engine_runner()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(proxy_log).unlink(missing_ok=True)
    attempts, scratches = [], []
    try:
        for i in range(int(n)):
            scratch = Path(tempfile.mkdtemp(prefix=f"{task['id']}-{arm}-{rep}-a{i}-", dir=harness._ensure(scratch_root)))
            scratches.append(scratch)
            work = scratch / "work"
            shutil.copytree(task["dir"] / "start", work) if (task["dir"] / "start").is_dir() else work.mkdir(parents=True)
            harness.write_project_rules(work)
            s = attempt_seed(seed, task["id"], rep, i)
            sp = _SeedProxy(port, s, scratch / "attempt-proxy.jsonl")
            try:
                cfg = harness.write_config(scratch / "cfg", sp.port, context=context, arm=qlev_arm, workdir=work, task=task)
                log_path = out_dir / f"{task['id']}__{arm}__{rep}__a{i}.agent.log"
                with harness.question_policy(task, out_dir) as policy:             # SIMUSER1
                    code, timed_out, secs = runner(work, task["prompt"], cfg, scratch / "data", task.get("timeout_s", 900), log_path)
            finally:
                sp.close()
            tot = proxy_mod.totals(sp.log)
            attempts.append({"index": i, "seed": s, "work": work, "log": log_path, "agent_exit": code, "timed_out": timed_out,
                             "seconds": round(secs, 1), "tokens": tot["completion_tokens"], "prompt_tokens": tot["prompt_tokens"],
                             "requests": tot["requests"], "request_errors": tot["errors"],
                             "max_prompt_tokens": harness.request_prompt_peak(sp.log),
                             "visible": visible_check(task, work), "own": own_check(task, work, log_path), "question_stats": policy.stats()})
        chosen, reason = select(attempts)                      # the hidden tests have not run yet: they cannot take part
        for a in attempts:
            if a["index"] == chosen or keep_hidden_of_others:
                passed, tail = harness.run_tests(a["work"], task["dir"] / "hidden", task["test_cmd"])
                soft = harness.LAST_TEST_RUN.get("soft") or {}
                a["hidden"] = {"resolved": bool(passed), "score": 1.0 if passed else float(soft.get("score", 0.0)),
                               "tests_total": soft.get("tests_total"), "tests_passed": soft.get("tests_passed"), "tail": tail[-400:]}
        c = attempts[chosen]
        hidden = c["hidden"]
        rec = {"arm": arm, "task": task["id"], "family": task["family"], "kind": task["kind"], "rep": rep,
               "resolved": hidden["resolved"], "score": hidden["score"], "tests_total": hidden["tests_total"],
               "tests_passed": hidden["tests_passed"], "timed_out": any(a["timed_out"] for a in attempts),
               "agent_exit": c["agent_exit"],
               "seconds": round(sum(a["seconds"] for a in attempts), 1), "tokens": sum(a["tokens"] for a in attempts),
               "prompt_tokens": sum(a["prompt_tokens"] for a in attempts), "requests": sum(a["requests"] for a in attempts),
               "request_errors": sum(a["request_errors"] for a in attempts),
               "max_prompt_tokens": max(a["max_prompt_tokens"] for a in attempts),
               **harness.count_events(c["log"]), **harness.project_checks(c["log"]), **c["question_stats"]}
        if not hidden["resolved"]:
            rec["test_tail"] = hidden["tail"]
        rec["best_of"] = {
            "n": int(n), "chosen": chosen, "reason": reason, "seeds": [a["seed"] for a in attempts],
            "attempts": [{k: a[k] for k in ("index", "seed", "seconds", "tokens", "prompt_tokens", "requests", "agent_exit",
                                            "timed_out", "visible", "own")}
                         | {"hidden": {k: v for k, v in (a.get("hidden") or {}).items() if k != "tail"}}
                         for a in attempts],
            "first_resolved": (attempts[0].get("hidden") or {}).get("resolved"),
            "oracle_resolved": any((a.get("hidden") or {}).get("resolved") for a in attempts) if keep_hidden_of_others else None,
        }
        return rec
    finally:
        for s in scratches:
            shutil.rmtree(s, ignore_errors=True)


# ---- the morning numbers ---------------------------------------------------------------------------------------------------
def _ci(values, boot=4000, seed=1):
    import random
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(boot))
    return (means[int(0.025 * boot)], means[int(0.975 * boot) - 1])


def summary(runs, control=None):
    """Per arm: tasks, solved % with a 95 % cluster-bootstrap CI over tasks, mean seconds and tokens per task (and the ratio to the
    control arm = the N=1 arm, or the arm with the smallest N), and for best-of arms: how often the choice was not the first
    attempt, by which step of the rule, the first-attempt and oracle rates."""
    by_arm = {}
    for r in runs:
        by_arm.setdefault(r["arm"], []).append(r)
    ns = {a: max((r.get("best_of") or {}).get("n", 1) for r in rs) for a, rs in by_arm.items()}
    control = control or min(ns, key=lambda a: (ns[a], a))
    out = {}
    for arm, rs in by_arm.items():
        per_task = {}
        for r in rs:
            per_task.setdefault(r["task"], []).append(1.0 if r["resolved"] else 0.0)
        means = [statistics.fmean(v) for v in per_task.values()]
        lo, hi = _ci(means)
        row = {"n": ns[arm], "tasks": len(per_task), "runs": len(rs), "solved": statistics.fmean(means) if means else 0.0,
               "ci95": (lo, hi), "seconds_per_task": statistics.fmean(r["seconds"] for r in rs),
               "tokens_per_task": statistics.fmean(r["tokens"] for r in rs)}
        bo = [r["best_of"] for r in rs if r.get("best_of")]
        if bo:
            row["not_first"] = sum(1 for b in bo if b["chosen"] != 0) / len(bo)
            row["by_reason"] = {k: sum(1 for b in bo if b["reason"] == k) for k in sorted({b["reason"] for b in bo})}
            firsts = [b["first_resolved"] for b in bo if b.get("first_resolved") is not None]
            oracles = [b["oracle_resolved"] for b in bo if b.get("oracle_resolved") is not None]
            row["first_attempt_solved"] = statistics.fmean(1.0 if f else 0.0 for f in firsts) if firsts else None
            row["oracle_solved"] = statistics.fmean(1.0 if o else 0.0 for o in oracles) if oracles else None
        out[arm] = row
    base = out.get(control)
    for arm, row in out.items():
        if base and base["seconds_per_task"] and base["tokens_per_task"]:
            row["seconds_x"] = row["seconds_per_task"] / base["seconds_per_task"]
            row["tokens_x"] = row["tokens_per_task"] / base["tokens_per_task"]
    return {"control": control, "arms": out}


def summary_text(res):
    lines = [f"control arm: {res['control']}", "",
             "| arm | N | tasks | solved | 95% CI | s/task (×) | tokens/task (×) | not first | first att. | oracle |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, r in sorted(res["arms"].items(), key=lambda kv: (kv[1]["n"], kv[0])):
        lo, hi = r["ci95"]
        nf = f"{100 * r['not_first']:.0f}% ({', '.join(f'{k} {v}' for k, v in r['by_reason'].items())})" if "not_first" in r else "—"
        fa = f"{100 * r['first_attempt_solved']:.1f}%" if r.get("first_attempt_solved") is not None else "—"
        orc = f"{100 * r['oracle_solved']:.1f}%" if r.get("oracle_solved") is not None else "—"
        lines.append(f"| {arm} | {r['n']} | {r['tasks']} | {100 * r['solved']:.1f}% | [{100 * lo:.1f}, {100 * hi:.1f}] | "
                     f"{r['seconds_per_task']:.0f} (×{r.get('seconds_x', 1):.2f}) | "
                     f"{r['tokens_per_task']:,.0f} (×{r.get('tokens_x', 1):.2f}) | {nf} | {fa} | {orc} |")
    return "\n".join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["summary"])
    ap.add_argument("results")
    ap.add_argument("--control", default=None)
    a = ap.parse_args(argv)
    runs = [json.loads(line) for line in Path(a.results).read_text().splitlines() if line.strip()]
    print(summary_text(summary(runs, a.control)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
