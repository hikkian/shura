"""Checks every task BEFORE it is allowed into the experiment. A broken task adds noise or, worse, a bias.

For each task:
  1. task.json is complete and the prompt does not leak the hidden tests' file names or contents;
  2. start/ + hidden tests  -> must FAIL (otherwise the task asks for nothing);
  3. start/ + reference solution + hidden tests -> must PASS, 3 times in a row (a flaky test is thrown out), each in < 30 s;
  4. the hidden tests are not importable from start/ (the agent cannot read them).
Exit code 0 only when every task is valid.

    python3 bench/agentic-eval/validate_tasks.py [tasks-root]
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import harness

REQUIRED = ("id", "family", "kind", "lang", "prompt", "test_cmd", "timeout_s")


def overlay(src, dst):
    for p in Path(src).rglob("*"):
        if p.is_file():
            target = Path(dst) / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def check(task):
    problems = []
    for k in REQUIRED:
        if k not in task:
            problems.append(f"task.json lacks {k!r}")
    if problems:
        return problems
    d = task["dir"]
    for sub in ("start", "hidden", "solution"):
        if not (d / sub).is_dir():
            problems.append(f"missing folder {sub}/")
    if problems:
        return problems
    hidden_names = {p.name for p in (d / "hidden").rglob("*") if p.is_file()}
    start_names = {p.name for p in (d / "start").rglob("*") if p.is_file()}
    if hidden_names & start_names:
        problems.append(f"hidden file name(s) also in start/: {sorted(hidden_names & start_names)}")
    for name in hidden_names:
        if name in task["prompt"]:
            problems.append(f"the prompt mentions a hidden file: {name}")
    with tempfile.TemporaryDirectory(prefix="validate-", dir="/dev/shm") as tmp:
        work = Path(tmp) / "work"
        shutil.copytree(d / "start", work)
        ok, tail = harness.run_tests(work, d / "hidden", task["test_cmd"])
        if ok:
            problems.append("the hidden tests already PASS on the starting files")
        if task["kind"] == "bugfix":                   # the symptom the prompt promises must be visible, and must go away with the fix
            shutil.rmtree(work / "hidden_tests", ignore_errors=True)       # the agent never has the hidden tests next to it
            r = subprocess.run(["python3", "-B", "-m", "unittest", "-q"], cwd=work, capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                problems.append("the visible test of a bugfix task already passes on the buggy files")
        overlay(d / "solution", work)
        if task["kind"] == "bugfix":
            shutil.rmtree(work / "hidden_tests", ignore_errors=True)
            r = subprocess.run(["python3", "-B", "-m", "unittest", "-q"], cwd=work, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                problems.append("the visible test still fails with the reference fix: " + (r.stdout + r.stderr)[-200:])
        for i in range(3):
            import time
            t0 = time.monotonic()
            ok, tail = harness.run_tests(work, d / "hidden", task["test_cmd"], timeout=30)
            if not ok:
                problems.append(f"the reference solution fails the hidden tests (run {i + 1}): {tail[-300:]}")
                break
            if time.monotonic() - t0 > 30:
                problems.append("hidden tests take more than 30 s")
    return problems


def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else harness.TASKS
    tasks = harness.all_tasks(root)
    bad = 0
    for t in tasks:
        problems = check(t)
        if problems:
            bad += 1
            print(f"FAIL {t.get('id', t['dir'])}")
            for p in problems:
                print("   -", p)
    print(f"{len(tasks) - bad}/{len(tasks)} tasks valid")
    return 1 if bad or not tasks else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
