"""Offline, network-isolated validator for the separate tasks_v2 tree."""
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import build_tasks_v2

HERE = Path(__file__).resolve().parent
ROOT = HERE / "tasks_v2"
REQUIRED = ("id", "family", "kind", "lang", "difficulty", "prompt", "test_cmd", "timeout_s", "solution_remove")


def _copy(src, dst):
    for p in Path(src).rglob("*"):
        if p.is_file():
            target = Path(dst) / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def _overlay(src, dst):
    for p in Path(src).rglob("*"):
        if p.is_file():
            target = Path(dst) / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def _run(cmd, cwd, timeout=30):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "NO_PROXY": "*", "no_proxy": "*", "HOME": "/dev/null"}
    if shutil.which("bwrap"):
        work = str(Path(cwd).resolve())
        argv = ["bwrap", "--die-with-parent", "--unshare-net", "--ro-bind", "/", "/", "--tmpfs", "/tmp",
                "--dir", "/tmp/work", "--bind", work, "/tmp/work", "--proc", "/proc", "--dev", "/dev",
                "--tmpfs", "/dev/shm", "--chdir", "/tmp/work",
                "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin", "--setenv", "HOME", "/dev/null",
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "NO_PROXY", "*", "--setenv", "no_proxy", "*", "--"] + list(cmd)
    else:
        raise RuntimeError("bwrap is required to enforce the no-network validation boundary")
    t0 = time.monotonic()
    try:
        result = subprocess.run(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=timeout, check=False)
        return result.returncode == 0, result.stdout[-1200:], time.monotonic() - t0
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return False, "timeout: " + output[-1000:], time.monotonic() - t0


def _run_hidden(task_dir, work, cmd):
    hidden_dir = Path(work) / "hidden_tests"
    if hidden_dir.exists():
        shutil.rmtree(hidden_dir)
    shutil.copytree(Path(task_dir) / "hidden", hidden_dir)
    return _run(cmd, work)


def check(task_dir):
    problems = []
    task_dir = Path(task_dir)
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    for key in REQUIRED:
        if key not in task:
            problems.append(f"task.json lacks {key}")
    if problems:
        return task, problems
    for folder in ("start", "hidden", "solution"):
        if not (task_dir / folder).is_dir():
            problems.append(f"missing {folder}/")
    if problems:
        return task, problems
    hidden = task_dir / "hidden"
    start = task_dir / "start"
    hidden_names = {p.name for p in hidden.rglob("*") if p.is_file()}
    start_names = {p.name for p in start.rglob("*") if p.is_file()}
    leaks = hidden_names & start_names
    if leaks:
        problems.append(f"hidden filename also in start/: {sorted(leaks)}")
    prompt = task["prompt"]
    for name in hidden_names:
        if name in prompt:
            problems.append(f"prompt leaks hidden filename: {name}")
    if any(p.suffix in (".pyc", ".log") for p in task_dir.rglob("*")):
        problems.append("generated cache/log file found in task")

    with tempfile.TemporaryDirectory(prefix="eval-v2-", dir="/dev/shm") as tmp:
        work = Path(tmp) / "work"
        shutil.copytree(start, work)
        ok, tail, elapsed = _run_hidden(task_dir, work, task["test_cmd"])
        if ok:
            problems.append("initial files already pass hidden tests")
        if elapsed >= 30:
            problems.append(f"initial test phase exceeded 30 s ({elapsed:.2f})")

        visible_cmd = task.get("visible_cmd")
        if visible_cmd:
            ok, tail, _ = _run(visible_cmd, work)
            if task["kind"] == "bugfix" and ok:
                problems.append("bugfix reproduction passes before the fix")

        for rel in task.get("solution_remove", []):
            target = work / rel
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
        _overlay(task_dir / "solution", work)

        if visible_cmd:
            ok, tail, elapsed = _run(visible_cmd, work)
            if not ok:
                problems.append("visible reproduction fails with reference: " + tail[-500:])
        for run_number in range(1, 4):
            ok, tail, elapsed = _run_hidden(task_dir, work, task["test_cmd"])
            if not ok:
                problems.append(f"reference hidden test run {run_number}/3 failed: {tail[-500:]}")
                break
            if elapsed >= 30:
                problems.append(f"reference test run {run_number}/3 took {elapsed:.2f} s")
        # Hidden graders and implementation tasks must not need a network escape hatch.
        for sub in ("start", "solution", "hidden"):
            for file in (task_dir / sub).rglob("*"):
                if file.is_file() and file.suffix in (".py", ".js", ".cjs"):
                    content = file.read_text(errors="replace")
                    if "socket.create_connection" in content or "fetch('http" in content or "fetch(\"http" in content:
                        problems.append(f"network access token found in {file.relative_to(task_dir)}")
    return task, problems


def validate(root=ROOT):
    dirs = sorted(Path(root).glob("*/*"))
    task_dirs = [d for d in dirs if (d / "task.json").is_file()]
    bad = 0
    for task_dir in task_dirs:
        task, problems = check(task_dir)
        if problems:
            bad += 1
            print(f"FAIL {task.get('id', task_dir)}")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"PASS {task['id']}")
    print(f"{len(task_dirs)-bad}/{len(task_dirs)} v2 tasks valid")
    return 1 if bad or not task_dirs else 0


if __name__ == "__main__":
    raise SystemExit(validate())
