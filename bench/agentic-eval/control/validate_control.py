"""LL3a: validate the control-set candidates (DESIGN-LL0 section 6).

    python3 control/validate_control.py public  [--root DIR] [--only FAMILY] [--jobs N]
    python3 control/validate_control.py private [--root DIR] [--only FAMILY] [--jobs N]

public: tasks_control_public/<family>/<kind>/ in the tasks_v3 format. Checks (as RSIT1 FORMAT.md section 7): the reference passes the hidden
tests three times in a row, each run within 30 s; the start does not pass; a bugfix reproduction fails on the start and passes on the
reference; every mutant (at least three, each a plausible wrong or naive solution) fails the hidden tests; the soft metric parses with at
least 6 units; no hidden file name in start/ or in the prompt; no network tokens. Every run is in bwrap without network.

private: tasks built from git history by build_from_git.py (task.json holds repo, parent commit, commit; hidden/ holds the tests of the
commit). The start is `git archive <parent>`, the reference is the start with the non-test files of the commit applied. Same pass/fail rules.
Nothing from the private directory is printed except ids and counts.
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EVAL / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v2 = _load("validate_tasks_v2")
harness = _load("harness")
PUBLIC_ROOT = EVAL / "tasks_control_public"
PRIVATE_ROOT = Path.home() / ".local/share/shura/private/control"
MIN_UNITS = 6


def _overlay(src, dst):
    v2._overlay(src, dst)


def _put_hidden(task_dir, work):
    hidden_dir = Path(work) / "hidden_tests"
    if hidden_dir.exists():
        shutil.rmtree(hidden_dir)
    shutil.copytree(Path(task_dir) / "hidden", hidden_dir)


def _run_hidden(task_dir, work, cmd, timeout=60):
    _put_hidden(task_dir, work)
    return v2._run(cmd, work, timeout)


def sandbox_run(cmd, cwd, timeout=60):
    """Like validate_tasks_v2._run (bwrap, no network, a throw-away HOME) but returns the whole output: (returncode, text, seconds)."""
    work = str(Path(cwd).resolve())
    argv = ["bwrap", "--die-with-parent", "--unshare-net", "--ro-bind", "/", "/", "--tmpfs", "/tmp", "--dir", "/tmp/work",
            "--bind", work, "/tmp/work", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/dev/shm", "--chdir", "/tmp/work",
            "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin", "--setenv", "HOME", "/dev/null", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--setenv", "NO_PROXY", "*", "--setenv", "no_proxy", "*", "--"] + list(cmd)
    t0 = time.monotonic()
    try:
        r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        return r.returncode, r.stdout.decode("utf-8", "replace"), time.monotonic() - t0
    except subprocess.TimeoutExpired as exc:
        return 124, "timeout: " + (exc.stdout or b"").decode("utf-8", "replace"), time.monotonic() - t0


def _soft(task, work, task_dir):
    """The soft score of one sandboxed run of the hidden tests, as the stand computes it (harness.soft_score)."""
    _put_hidden(task_dir, work)
    code, text, _ = sandbox_run(task["test_cmd"], work, 90)
    soft = harness.soft_score(task["test_cmd"], code, text)
    soft["skipped"] = _skipped(text)
    soft["text"] = text
    return soft


_SKIPPED_PY = re.compile(r"skipped=(\d+)")
_SKIPPED_NODE = re.compile(r"^(?:#|\u2139)\s*(?:skipped|todo) (\d+)", re.M)


def _skipped(text):
    """Tests that did not run: unittest `OK (skipped=N)` and node:test `# skipped N` / `# todo N`."""
    return sum(int(m.group(1)) for m in _SKIPPED_PY.finditer(text)) + sum(int(m.group(1)) for m in _SKIPPED_NODE.finditer(text))


def _mutants(task, task_dir, work, problems):
    kind = task.get("kind")
    mutants = sorted((task_dir / "mutants").glob("m*.json")) if (task_dir / "mutants").is_dir() else []
    if kind not in ("implement", "bugfix", "refactor", "feature", "security"):
        return
    if len(mutants) < 3:
        problems.append("fewer than 3 mutants")
    for m in mutants:
        cfg = json.loads(m.read_text())
        # after a refactor the mutated code may live in another file of the package: look for the anchor by file name
        candidates = [work / cfg["file"]] + [c for c in work.rglob(Path(cfg["file"]).name) if "hidden_tests" not in c.parts]
        target = next((c for c in candidates if c.is_file() and c.read_text().count(cfg["old"]) == 1), None)
        if target is None:
            problems.append(f"{m.name}: anchor not found exactly once in {cfg['file']}")
            continue
        original = target.read_text()
        target.write_text(original.replace(cfg["old"], cfg["new"], 1))
        ok, tail, _ = _run_hidden(task_dir, work, task["test_cmd"])
        target.write_text(original)
        if ok:
            problems.append(f"{m.name}: mutant passes the hidden tests (tests too weak)")


def check_public(task_dir):
    task_dir = Path(task_dir)
    task, problems = v2.check(task_dir)  # reference 3x, start fails, leaks, network, 30 s
    if "test_cmd" not in task:
        return task, problems
    with tempfile.TemporaryDirectory(prefix="control-pub-", dir="/dev/shm") as tmp:
        work = Path(tmp) / "work"
        shutil.copytree(task_dir / "start", work)
        for rel in task.get("solution_remove", []):
            target = work / rel
            shutil.rmtree(target) if target.is_dir() else target.unlink(missing_ok=True)
        _overlay(task_dir / "solution", work)
        soft = _soft(task, work, task_dir)
        if soft["score"] != 1.0:
            problems.append(f"soft metric on the reference is {soft}")
        if task.get("soft", "counts") == "counts" and (soft.get("tests_total") or 0) < MIN_UNITS:
            problems.append(f"fewer than {MIN_UNITS} hidden units: {soft.get('tests_total')}")
        _mutants(task, task_dir, work, problems)
    return task, problems


def _git_tree(repo, rev, dst, only=None):
    """`git archive rev` piped into tar (modes and symlinks kept, nothing read into memory), only the given paths if `only`."""
    cmd = ["git", "--no-optional-locks", "--literal-pathspecs", "-C", repo, "archive", rev] + (["--"] + list(only) if only else [])
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dst)], stdin=p.stdout, check=True)
    p.stdout.close()
    if p.wait() != 0:
        raise subprocess.CalledProcessError(p.returncode, "git archive")


def materialize_private(task, work):
    """start = the parent commit."""
    Path(work).mkdir(parents=True, exist_ok=True)
    _git_tree(os.path.expanduser(task["repo"]), task["parent"], work)


def apply_reference(task, work):
    """The reference = the start with the non-test files of the commit (git archive of those paths: modes and symlinks come along)."""
    for rel in task.get("solution_remove", []):
        target = Path(work) / rel
        shutil.rmtree(target) if target.is_dir() else target.unlink(missing_ok=True)
    if task["source_files"]:
        _git_tree(os.path.expanduser(task["repo"]), task["commit"], work, only=task["source_files"])


_ERR_AT_START = re.compile(r"(ModuleNotFoundError|ImportError|AttributeError|SyntaxError|NameError|Cannot find module|ERR_MODULE_NOT_FOUND)")


def check_private(task_dir):
    task_dir = Path(task_dir)
    problems = []
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    for key in ("id", "family", "kind", "repo", "parent", "commit", "test_cmd", "source_files", "test_files"):
        if key not in task:
            problems.append(f"task.json lacks {key}")
    if problems:
        return task, problems
    if not (task_dir / "hidden").is_dir():
        return task, ["missing hidden/"]
    info = {"commit": task["commit"]}
    # leak check: the prompt must not name a hidden test file or a test function
    prompt = task.get("prompt", "")
    for rel in task["test_files"]:
        if Path(rel).name in prompt or Path(rel).stem in prompt:
            problems.append("the prompt names a hidden test file")
            break
    hidden_text = " ".join((task_dir / "hidden" / "files" / rel).read_text(errors="replace") for rel in task["test_files"]
                           if (task_dir / "hidden" / "files" / rel).is_file())
    names = set(re.findall(r"def (test_\w+)", hidden_text)) | set(re.findall(r"\b(?:test|it)\(['\"]([^'\"]+)", hidden_text))
    if any(n in prompt for n in names if len(n) > 8):
        problems.append("the prompt contains the name of a hidden test")
    timeout = task.get("check_timeout_s", 90)
    with tempfile.TemporaryDirectory(prefix="control-priv-", dir="/dev/shm") as tmp:
        # the start: a fresh tree
        start = Path(tmp) / "start"
        try:
            materialize_private(task, start)
        except subprocess.CalledProcessError:
            return task, problems + ["git archive of the parent failed"]
        soft0 = _soft(task, start, task_dir)
        if soft0["text"].count("\n") and soft0.get("score") == 1.0 and not soft0["skipped"]:
            problems.append("the start already passes the hidden tests")
        info["start_total"] = soft0.get("tests_total")
        info["start_passed"] = soft0.get("tests_passed")
        info["start_score"] = soft0.get("score")
        # the start must fail for the reason of the task: a test that ran and failed by assertion, not only an import error of a new module
        info["start_import_error"] = bool(_ERR_AT_START.search(soft0["text"]))
        info["start_assertion_failure"] = "AssertionError" in soft0["text"] or "failureType" in soft0["text"] or "not ok" in soft0["text"]
        shutil.rmtree(start, ignore_errors=True)
        # the reference: a fresh tree for every run (nothing a previous run created can help the next one)
        times, last = [], None
        for run_number in range(1, 4):
            ref = Path(tmp) / f"ref{run_number}"
            try:
                materialize_private(task, ref)
                apply_reference(task, ref)
            except subprocess.CalledProcessError:
                problems.append("git archive of the commit failed")
                break
            _put_hidden(task_dir, ref)
            code, text, elapsed = sandbox_run(task["test_cmd"], ref, timeout)
            times.append(elapsed)
            last = harness.soft_score(task["test_cmd"], code, text)
            last["skipped"] = _skipped(text)
            shutil.rmtree(ref, ignore_errors=True)
            if code != 0:
                problems.append(f"reference run {run_number}/3 failed")
                break
            info["ref_skipped"] = last["skipped"]  # skips (a UI test without a browser) are normal; only the tests that ran are units
        if last is not None and not problems:
            units = (last.get("tests_total") or 0) - last["skipped"]
            if last.get("score") != 1.0:
                problems.append("soft metric on the reference is not 1.0")
            if units < 2:
                problems.append(f"fewer than 2 hidden units that ran: {units}")
            info["units"] = units
        info["ref_s"] = [round(t, 1) for t in times]
        # the part of the score a do-nothing agent already gets (the stand can normalise by it)
        if info.get("start_total"):
            info["fail_to_pass"] = info["start_total"] - (info["start_passed"] or 0)
            if info["fail_to_pass"] < 1:
                problems.append("no test fails at the start")
        else:
            problems.append("the start run reports no test count (it did not reach the tests)")
    task["_validation"] = info
    return task, problems


def checked(checker, d):
    try:
        return checker(d)
    except Exception as exc:  # one broken task must not abort the whole run
        try:
            task = json.loads((Path(d) / "task.json").read_text())
        except Exception:
            task = {"id": Path(d).name}
        return task, [f"validator error: {type(exc).__name__}"]


def run(mode, root, only, jobs):
    checker = check_public if mode == "public" else check_private
    if mode == "private" and subprocess.run(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"], capture_output=True).returncode == 0:
        raise SystemExit("the private root must be outside every git working tree (validation.json is written into it)")
    dirs = sorted(Path(root).glob("*/*"))
    task_dirs = [d for d in dirs if (d / "task.json").is_file() and (only is None or d.parent.name == only)]
    if "--skip-valid" in sys.argv and mode == "private":
        def done(d):
            v = d / "validation.json"
            return v.is_file() and json.loads(v.read_text()).get("ok") and json.loads(v.read_text()).get("commit") == json.loads((d / "task.json").read_text())["commit"]
        task_dirs = [d for d in task_dirs if not done(d)]
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for d, (task, problems) in zip(task_dirs, ex.map(lambda d: checked(checker, d), task_dirs)):
            results.append((d, task, problems))
    bad = 0
    for d, task, problems in results:
        name = task.get("id", d.name)
        if problems:
            bad += 1
            print(f"FAIL {name}")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"PASS {name}")
        if mode == "private":
            (d / "validation.json").write_text(json.dumps({"ok": not problems, "problems": problems, **task.get("_validation", {})}))
    print(f"{len(results) - bad}/{len(results)} {mode} control tasks valid")
    return 1 if bad or not results else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in ("public", "private"):
        raise SystemExit(__doc__)
    mode = args[0]

    def opt(name, default):
        return args[args.index(name) + 1] if name in args else default

    root = Path(opt("--root", PUBLIC_ROOT if mode == "public" else PRIVATE_ROOT))
    raise SystemExit(run(mode, root, opt("--only", None), int(opt("--jobs", 3))))
