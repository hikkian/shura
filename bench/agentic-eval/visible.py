"""QTTS1q: the VISIBLE check of a task — the independent signal best-of-N may pick by (bestof.py, step (a)). Never the hidden tests.

Two sources, in this order:
  1. the task's own `visible_cmd` (bugfix: public-repro.test.js / public_repro.py in start/);
  2. the registry visible_v3/<family>/: a check written by hand from the task PROMPT and the start folder's module name, for the
     implement and refactor tasks that have no visible_cmd. provenance.json says who wrote it, from which sentences of the prompt,
     and which kinds it applies to. The hidden tests were not opened to write them.

The check always runs from the ORIGINAL files (start/ or the registry) copied over a throwaway copy of the attempt's tree, so an
attempt that edited or deleted them cannot flatter itself.

Honesty guards (`python3 visible.py guard`), run after the checks were written and never used to write them: (1) an assertion line a
check shares with the hidden tests is allowed only when every literal in it occurs in the prompt (an example the prompt gives may
coincide with a hidden test; anything else shared is a leak and fails the guard); (2) string literals of 8+ characters that are not
made of the prompt's and start folder's words are listed as "invented inputs" for review (allowed: an input the author made up is
not a leak, a copied one is caught by (1)).  `python3 visible.py verify` runs each registry
check against its reference tree (the refactor task's start/, which is the working module, else the implement task's solution/):
a check that fails there misread the prompt.  `python3 visible.py coverage [--ids FILE]` counts the tasks with a signal.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402

REGISTRY = HERE / "visible_v3"
ROOTS = (HERE / "tasks_v3", HERE / "tasks_v2", HERE / "tasks")
TIMEOUT_S = 120
GRADER_DIRS = ("hidden_tests", "visible_tests", "__pycache__", "node_modules", ".tmp")
MIN_LITERAL = 8


def provenance(family):
    p = REGISTRY / family / "provenance.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def spec(task):
    """{"cmd", "files": {name in the tree: source path}, "source": "task" | "registry"} or None when the task has no visible signal."""
    cmd = task.get("visible_cmd") or []
    start = Path(task["dir"]) / "start"
    if cmd:
        files = {}
        for arg in cmd:
            if (start / arg).is_file():
                files[arg] = start / arg
            elif (start / f"{arg}.py").is_file():              # python3 -m unittest public_repro
                files[f"{arg}.py"] = start / f"{arg}.py"
        return {"cmd": list(cmd), "files": files, "source": "task"}
    if task["kind"] == "bugfix" and (start / "test_repro.py").is_file():     # the first task set: the failing repro the prompt names
        return {"cmd": ["python3", "-B", "-m", "unittest", "-q", "test_repro"], "files": {"test_repro.py": start / "test_repro.py"},
                "source": "task"}
    prov = provenance(task["family"])
    if prov and task["kind"] in prov.get("applies_to", []):
        return {"cmd": list(prov["cmd"]), "files": {prov["file"]: REGISTRY / task["family"] / prov["file"]}, "source": "registry"}
    return None


def names(task):
    """The file names the visible check puts into the tree: the agent's own checks (bestof step (b)) must not count them."""
    s = spec(task)
    return list(s["files"]) if s else []


def _run(cmd, cwd, timeout=TIMEOUT_S):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False,
                           env={"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "NO_COLOR": "1", "LANG": "C.UTF-8"})
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except OSError as err:
        return 127, str(err)


def run_in_copy(task, tree):
    """The visible check on a copy of `tree` (the graders' folders left out, the original visible files copied over).
    {"has": True, "ok", "score", "passed", "total", "source", "output"} or {"has": False}."""
    s = spec(task)
    if not s:
        return {"has": False}
    tmp = Path(tempfile.mkdtemp(prefix="visible-", dir=str(Path(tree).parent)))
    try:
        dst = tmp / "tree"
        shutil.copytree(tree, dst, ignore=shutil.ignore_patterns(*GRADER_DIRS))
        for name, src in s["files"].items():
            (dst / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / name)
        code, out = _run(s["cmd"], dst)
        soft = harness.soft_score(s["cmd"], code, out)
        return {"has": True, "ok": code == 0, "score": soft.get("score", 1.0 if code == 0 else 0.0), "passed": soft.get("tests_passed"),
                "total": soft.get("tests_total"), "source": s["source"], "output": out[-600:]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- the registry's honesty ---------------------------------------------------------------------------------------------------
def task_dirs(family):
    out = {}
    for root in ROOTS:
        for kind in ("implement", "refactor"):
            d = root / family / kind
            if (d / "task.json").is_file() and kind not in out:
                out[kind] = d
    return out


def _text_of(folder):
    parts = []
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


_LITERAL = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"")


FRAMEWORK = ("__main__", "node:test", "node:assert/strict", "node:crypto")
_WORD = re.compile(r"[A-Za-z0-9]+")


def literals(text, minimum=MIN_LITERAL):
    """The string literals of a check, without the test framework's own and the test titles (the first argument of test(...))."""
    titles = set(re.findall(r"^\s*test\(\s*(['\"])(.*?)\1", text, re.MULTILINE))
    titles = {t[1] for t in titles}
    out = []
    for m in _LITERAL.finditer(text):
        s = m.group(1) if m.group(1) is not None else m.group(2)
        if len(s) >= minimum and s not in FRAMEWORK and s not in titles and not s.startswith("./"):
            out.append(s)
    return out


def _backed(literal, corpus):
    """A literal is backed by the corpus when it occurs there, or every word of it (letters and digits) does."""
    plain = literal.encode("utf-8").decode("unicode_escape", errors="ignore")
    if literal in corpus or plain in corpus:
        return True
    words = _WORD.findall(plain)
    return bool(words) and all(w in corpus for w in words)


def guard(family):
    """{"family", "invented": [literal...], "coincidences": [(kind, line)...], "leaks": [(kind, line)...], "ok"}. A line shared with the
    hidden tests is a leak unless every literal in it is in the PROMPT; invented inputs are reported, not forbidden."""
    prov = provenance(family)
    check = (REGISTRY / family / prov["file"]).read_text(encoding="utf-8")
    dirs = task_dirs(family)
    prompt, corpus = "", ""
    for kind in prov["applies_to"]:
        if kind in dirs:
            task = json.loads((dirs[kind] / "task.json").read_text(encoding="utf-8"))
            prompt += task["prompt"] + "\n"
            corpus += task["prompt"] + "\n" + _text_of(dirs[kind] / "start")
    invented = sorted({lit for lit in literals(check) if not _backed(lit, corpus)})
    norm = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    bare = re.compile(r"^\s*with self\.assertRaises\(\w+\)( as \w+)?:\s*$")     # no literal, no call: says nothing on its own
    mine = {norm(line): line for line in check.splitlines()
            if len(norm(line)) >= 25 and ("assert" in line or "throws" in line) and not bare.match(line)}
    coincidences, leaks = [], []
    for kind, d in dirs.items():
        if kind not in prov["applies_to"]:
            continue
        theirs = {norm(line) for line in _text_of(d / "hidden").splitlines()}
        for key in sorted(mine.keys() & theirs):
            line = mine[key]
            coincidences.append((kind, line.strip()))
            if not all(lit in prompt for lit in literals(line, minimum=2)):
                leaks.append((kind, line.strip()))
    return {"family": family, "invented": invented, "coincidences": coincidences, "leaks": leaks, "ok": not leaks}


def verify(family):
    """The check against its reference tree: refactor start/ (the working module) when the check applies to refactor, else the implement
    task's solution/. {"family", "reference", "ok", "output"}."""
    prov = provenance(family)
    dirs = task_dirs(family)
    if "refactor" in prov["applies_to"] and "refactor" in dirs:
        ref, label = dirs["refactor"] / "start", "refactor/start"
    elif "implement" in dirs:
        ref, label = dirs["implement"] / "solution", "implement/solution"
    else:
        return {"family": family, "reference": None, "ok": False, "output": "no task folder"}
    kind = "refactor" if label.startswith("refactor") else "implement"
    task = harness.load_task(dirs[kind])
    with tempfile.TemporaryDirectory(prefix="visible-verify-") as tmp:
        tree = Path(tmp) / "tree"
        shutil.copytree(ref, tree)
        if label.startswith("implement"):          # solution/ holds the changed files only: the rest comes from start/
            shutil.copytree(dirs["implement"] / "start", tree, dirs_exist_ok=True)
            shutil.copytree(ref, tree, dirs_exist_ok=True)
        res = run_in_copy(task, tree)
    return {"family": family, "reference": label, "ok": bool(res.get("ok")), "output": res.get("output", "")}


def coverage(tasks, ids=None):
    """Per kind: tasks, with a visible signal, by source."""
    out = {}
    for t in tasks:
        if ids is not None and t["id"] not in ids:
            continue
        row = out.setdefault(t["kind"], {"tasks": 0, "with_signal": 0, "task": 0, "registry": 0, "without": []})
        row["tasks"] += 1
        s = spec(t)
        if s:
            row["with_signal"] += 1
            row[s["source"]] += 1
        else:
            row["without"].append(t["id"])
    return out


def all_tasks():
    seen, out = set(), []
    for root in ROOTS:
        for t in harness.all_tasks(root) if root.is_dir() else []:
            if t["id"] not in seen:
                seen.add(t["id"])
                out.append(t)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["coverage", "guard", "verify"])
    ap.add_argument("--ids", help="a file with task ids (comma or newline separated), e.g. reports/qual0b/tasks60.txt")
    a = ap.parse_args(argv)
    fams = sorted(p.name for p in REGISTRY.iterdir() if (p / "provenance.json").is_file())
    if a.step == "coverage":
        ids = None
        if a.ids:
            ids = {x.strip() for x in re.split(r"[,\s]+", Path(a.ids).read_text()) if x.strip()}
        cov = coverage(all_tasks(), ids)
        for kind, r in sorted(cov.items()):
            pct = 100 * r["with_signal"] / r["tasks"] if r["tasks"] else 0
            print(f"{kind:12s} {r['with_signal']:3d} of {r['tasks']:3d} ({pct:.0f} %)  task {r['task']}, registry {r['registry']}"
                  + (f"; without: {', '.join(r['without'])}" if r["without"] and len(r["without"]) <= 12 else ""))
        return 0
    bad = 0
    for fam in fams:
        res = guard(fam) if a.step == "guard" else verify(fam)
        bad += 0 if res["ok"] else 1
        if a.step == "guard":
            print(f"{fam:20s} {'ok ' if res['ok'] else 'LEAK'} shared with hidden: {len(res['coincidences'])} "
                  f"(prompt-backed: {len(res['coincidences']) - len(res['leaks'])}); invented inputs: {res['invented'] or '-'}")
            for kind, line in res["coincidences"]:
                print(f"    shared [{kind}]: {line}")
        else:
            print(f"{fam:20s} {'ok ' if res['ok'] else 'FAIL'} against {res['reference']}" + ("" if res["ok"] else f"\n{res['output']}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
