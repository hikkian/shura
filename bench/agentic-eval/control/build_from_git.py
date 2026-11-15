"""LL3a: build candidate control tasks from the git history of the person's own repositories (DESIGN-LL0 section 6.2, items 1 and 3).

    python3 control/build_from_git.py scan  [--repo NAME] [--max-src-lines 300]
    python3 control/build_from_git.py build [--repo NAME] [--commits sha,sha] [--limit N] [--force]
    python3 control/build_from_git.py materialize TASK_ID DEST   # the start tree of a private task (the stand calls this: `git archive <parent>`)
    python3 control/build_from_git.py list

A commit that adds or changes a test together with the code it tests becomes a task (the SWE-bench / LHSET1 method): the start is the
parent commit, the hidden tests are the tests of the commit, the reference is the non-test files of the commit. Only repositories named
in REPOS are read, and only their git objects (no checkout, no fetch, no working tree changes). The output goes to the private directory
(outside every git repository), never to a repository: task ids, prompts and the candidate list contain commit messages.

    ~/.local/share/shura/private/control/<family>/<kind>/{task.json, hidden/}    tasks (start is `git archive <parent>` at run time)
    ~/.local/share/shura/private/control/candidates/<repo>.json                 scan results
    ~/.local/share/shura/private/control/CANDIDATES.md                          the list for the person (yes/no)

The prompt of a built task is a DRAFT made from the commit message ("prompt_draft": true). The person or the local model rewrites it
before the task enters the gates (DESIGN-LL0 6.2: "the local model writes, the person edits").
Work repositories and customer code are not listed here and must not be added: only the person's own repositories.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
PRIVATE = Path.home() / ".local/share/shura/private/control"

# The person's own repositories (the list from the LL3 card). Nothing else is opened.
REPOS = {
    "shura-integration": {"path": "~/Projects/shura-integration", "env": {}},
    "shuracode-integration": {"path": "~/Projects/shuracode-integration", "env": {}},
    "barid": {"path": "~/Projects/barid", "env": {"BARID_SKIP_UI": "1"}},
    "hikkian.github.io": {"path": "~/Projects/hikkian.github.io", "env": {}},
    "arabic-academy": {"path": "~/Desktop/arabic-academy", "env": {}},
    "open-go-academy": {"path": "~/Desktop/open-go-academy", "env": {}},
}

# test files: any file under a tests/ folder, or by name
TEST_RE = re.compile(r"(^|/)(tests?|__tests__)/|(^|/)test_[^/]*\.py$|_test\.(py|go)$|\.(test|spec)\.(js|ts|mjs|cjs|tsx)$")
RUNNABLE_PY = re.compile(r"(^|/)test_[^/]*\.py$")
RUNNABLE_JS = re.compile(r"\.test\.(js|mjs|cjs)$")
# never part of a task: documents and pictures, and generated task sets and reports of the stand (they contain other tasks' tests).
# SKIP_RE is applied to every changed file; DOC_RE only to files that are not tests (a fixture under tests/ is part of the hidden tests).
SKIP_RE = re.compile(r"(^|/)(reports|laptop|vendor_js|node_modules)/|"
                     r"bench/agentic-eval/(tasks[^/]*|visible_v3|families[^/]*|patches_v2|tests_v2)/")
DOC_RE = re.compile(r"\.(md|txt|lock|png|svg|jpg|gif|jpeg|ico)$|(^|/)docs?/|(^|/)LICENSE|\.gitignore$")
LANG = {".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".ts": "typescript", ".tsx": "typescript",
        ".go": "go", ".sh": "bash", ".sql": "sql"}


def git(repo, *args, check=True):
    return subprocess.run(["git", "--no-optional-locks", "--literal-pathspecs", "-c", "core.quotepath=false", "-C", os.path.expanduser(repo), *args], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=check).stdout


def excluded_commits():
    """Commits that already are the source of a task in tasks_lh (tasks_v3 and the public control tasks are written from scratch and have no git source)."""
    found = set()
    for p in (EVAL / "tasks_lh").glob("*/*/task.json"):
        try:
            src = json.loads(p.read_text()).get("source") or {}
        except Exception:
            continue
        if isinstance(src, dict) and src.get("commit"):
            found.add(src["commit"][:7])
    return found


def scan_repo(name, max_src_lines=300, min_src_lines=6):
    repo = REPOS[name]["path"]
    out = git(repo, "log", "--no-merges", "--no-renames", "--format=@@%H\t%P\t%ad\t%s", "--date=short", "--numstat").decode("utf-8", "replace")
    commits, cur = [], None
    for line in out.splitlines():
        if line.startswith("@@"):
            h, parents, d, s = line[2:].split("\t", 3)
            cur = {"commit": h, "parent": parents.split()[0] if parents else None, "date": d, "subject": s, "tests": [], "src": [],
                   "src_lines": 0, "tests_lines": 0}
            commits.append(cur)
        elif line.strip() and cur is not None:
            a, b, f = line.split("\t", 2)
            if SKIP_RE.search(f):
                continue
            if TEST_RE.search(f):
                cur["tests"].append(f)  # binary fixtures show "-" for the counts
                cur["tests_lines"] += int(a) if a != "-" else 0
            elif not DOC_RE.search(f) and a != "-":
                cur["src"].append(f)
                cur["src_lines"] += int(a) + int(b)
    skip = excluded_commits()
    out = []
    for c in commits:
        runnable = [f for f in c["tests"] if RUNNABLE_PY.search(f) or RUNNABLE_JS.search(f)]
        if not (c["parent"] and runnable and any(Path(f).suffix in LANG for f in c["src"]) and min_src_lines <= c["src_lines"] <= max_src_lines):
            continue
        # one language per task: the stand's soft metric reads one kind of runner output
        if any(RUNNABLE_PY.search(f) for f in runnable) and any(RUNNABLE_JS.search(f) for f in runnable):
            continue
        if c["commit"][:7] in skip:
            continue
        # a test-only fix of a sibling commit is not a task: the change must have code that the tests exercise
        c["runnable_tests"] = runnable
        out.append(c)
    return out


def exists_at(repo, commit, rel):
    return subprocess.run(["git", "-C", os.path.expanduser(repo), "cat-file", "-e", f"{commit}:{rel}"], capture_output=True).returncode == 0


def family_of(name, c):
    # the family is the repository plus the top-level directory of the first changed code file: one area is one split bucket
    code = next((f for f in c["src"] if Path(f).suffix in LANG), c["src"][0])
    parts = Path(code).parts
    area = parts[0] if len(parts) > 1 else "root"
    return f"{name}__{re.sub(r'[^A-Za-z0-9_]+', '_', area)}"


def difficulty_of(c, test_adds):
    score = 0
    score += c["src_lines"] >= 80
    score += len(c["src"]) >= 3
    score += test_adds >= 6
    return "hard" if score >= 2 else ("medium" if score == 1 else "easy")


def kind_of(subject):
    s = subject.lower()
    if s.startswith(("fix", "bug", "hotfix")) or " fix" in s[:30]:
        return "bugfix"
    if s.startswith(("refactor", "clean", "simplif")):
        return "refactor"
    return "feature"


def build_cmds(name, c):
    """The commands that run the tests of the commit, from the repository root of the work directory."""
    cmds = []
    # only the test files that exist at the commit (a deleted or renamed test file would be a ModuleNotFoundError)
    py = [f for f in c["tests"] if RUNNABLE_PY.search(f) and exists_at(REPOS[name]["path"], c["commit"], f)]
    js = [f for f in c["tests"] if RUNNABLE_JS.search(f) and exists_at(REPOS[name]["path"], c["commit"], f)]
    dotted = [f[:-3].replace("/", ".") for f in py if all(part.isidentifier() for part in f[:-3].split("/"))]
    if len(dotted) == len(py) and py:
        cmds.append(["python3", "-B", "-m", "unittest", "-q", *dotted])
    for f in py if len(dotted) != len(py) else []:  # a hyphen in a directory name: discover by file
        cmds.append(["python3", "-B", "-m", "unittest", "discover", "-q", "-s", str(Path(f).parent), "-p", Path(f).name])
    if js:
        cmds.append(["node", "--test", *js])
    return cmds


RUNNER = '''"""Generated by build_from_git.py: put the tests of the commit in place and run them (output and exit code as the stand expects)."""
import json, os, shutil, stat, subprocess, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(here, "_cfg.json")))
for rel in cfg["test_files"]:
    src = os.path.join(here, "files", rel)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(rel) or ".", exist_ok=True)
        shutil.copy2(src, rel)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **cfg["env"])
# tests that import their helpers as siblings (`from test_rb import ...`) need their own folder on the path, as `unittest discover -s tests` gives
dirs = sorted({os.path.dirname(rel) or "." for rel in cfg["test_files"] if rel.endswith(".py")})
env["PYTHONPATH"] = os.pathsep.join([os.path.abspath(d) for d in dirs] + [os.getcwd()] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
quiet = tempfile.mkdtemp(prefix="control-quiet-")  # a silent notify-send first on PATH: no desktop notification from a test
p = os.path.join(quiet, "notify-send")
open(p, "w").write("#!/bin/sh\\nexit 0\\n")
os.chmod(p, stat.S_IRWXU)
env["PATH"] = quiet + os.pathsep + env.get("PATH", "")
code = 0
try:
    for cmd in cfg["cmds"]:
        r = subprocess.run(cmd, env=env)
        code = code or r.returncode
finally:
    shutil.rmtree(quiet, ignore_errors=True)
sys.exit(code)
'''


def test_additions(repo, c):
    out = git(repo, "show", "--format=", "--no-renames", c["commit"], "--", *c["tests"]).decode("utf-8", "replace")
    return len(re.findall(r"^\+\s*(?:def test_|async def test_|test\(|it\()", out, flags=re.M))


def build_task(name, c, force=False):
    repo = REPOS[name]["path"]
    family = family_of(name, c)
    kind = "c" + c["commit"][:8]
    tdir = PRIVATE / family / kind
    if (tdir / "task.json").is_file() and not force:
        return None  # an edited prompt must not be overwritten by the draft; --force rebuilds
    (tdir / "validation.json").unlink(missing_ok=True)
    (tdir / "hidden" / "files").mkdir(parents=True, exist_ok=True)
    # the tests as they are at the commit (and the fixtures under tests/): hidden/files/<path>
    for rel in c["tests"]:
        data = git(repo, "show", f"{c['commit']}:{rel}", check=False)
        if not data and not exists_at(repo, c["commit"], rel):
            continue  # the commit deleted this test file
        dst = tdir / "hidden" / "files" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    cmds = build_cmds(name, c)
    (tdir / "hidden" / "_cfg.json").write_text(json.dumps({"test_files": c["tests"], "cmds": cmds, "env": REPOS[name]["env"]}))
    (tdir / "hidden" / "_run.py").write_text(RUNNER)
    body = "\n".join(l for l in git(repo, "show", "-s", "--format=%b", c["commit"]).decode("utf-8", "replace").splitlines()
                     if not re.match(r"\s*(co-authored-by|signed-off-by|tests?\b|test plan)", l, re.I) and not re.search(r"\btest_\w+|tests?/", l)).strip()
    adds = test_additions(repo, c)
    deleted = [f for f in c["src"] if not exists_at(repo, c["commit"], f)]
    task = {
        "id": f"{family}-{kind}", "family": family, "kind": kind, "task_kind": kind_of(c["subject"]),
        "lang": LANG.get(Path(c["src"][0]).suffix, "text"), "difficulty": difficulty_of(c, adds),
        "prompt": f"In this repository: {c['subject']}\n\n{body}".strip(), "prompt_draft": True, "prompt_lang": "en",
        # the words after _run.py are ignored by the runner; they make the stand's soft metric (harness.soft_score) read the relayed
        # output like unittest or node:test output
        "test_cmd": ["python3", "-B", "hidden_tests/_run.py"] + (["unittest"] if any(RUNNABLE_PY.search(f) for f in c["tests"]) else ["node", "--test"]), "timeout_s": 1800, "check_timeout_s": 120,
        "solution_remove": deleted, "visible_cmd": None, "set": "control-private", "soft": "counts",
        "tags": ["git", name], "source": "own-git", "repo": REPOS[name]["path"], "parent": c["parent"], "commit": c["commit"],
        "source_files": [f for f in c["src"] if f not in deleted], "test_files": c["tests"],
        "src_lines": c["src_lines"], "tests_added": adds, "date": c["date"],
    }
    (tdir / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=1))
    return task


def cmd_scan(args):
    names = [args[args.index("--repo") + 1]] if "--repo" in args else list(REPOS)
    mx = int(args[args.index("--max-src-lines") + 1]) if "--max-src-lines" in args else 300
    (PRIVATE / "candidates").mkdir(parents=True, exist_ok=True)
    for n in names:
        if n not in REPOS:
            raise SystemExit(f"{n} is not in REPOS (only the person's own repositories are read)")
        found = scan_repo(n, mx)
        (PRIVATE / "candidates" / f"{n}.json").write_text(json.dumps(found, ensure_ascii=False, indent=1))
        print(f"{n}: {len(found)} candidates")


def cmd_build(args):
    names = [args[args.index("--repo") + 1]] if "--repo" in args else list(REPOS)
    wanted = set(args[args.index("--commits") + 1].split(",")) if "--commits" in args else None
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    n_built = 0
    for n in names:
        f = PRIVATE / "candidates" / f"{n}.json"
        if not f.is_file():
            continue
        for c in json.loads(f.read_text()):
            if wanted and not any(c["commit"].startswith(w) for w in wanted):
                continue
            if limit is not None and n_built >= limit:
                break
            if build_task(n, c, force="--force" in args) is not None:
                n_built += 1
    print(f"built {n_built} tasks into the private directory")


def cmd_materialize(args):
    """The start tree of a private task: `git archive <parent>` into DEST (the stand calls this instead of copying start/)."""
    task_id, dest = args
    for p in PRIVATE.glob("*/*/task.json"):
        t = json.loads(p.read_text())
        if t["id"] == task_id:
            Path(dest).mkdir(parents=True, exist_ok=True)
            archive = subprocess.Popen(["git", "--no-optional-locks", "-C", os.path.expanduser(t["repo"]), "archive", t["parent"]], stdout=subprocess.PIPE)
            subprocess.run(["tar", "-x", "-C", dest], stdin=archive.stdout, check=True)
            archive.stdout.close()
            if archive.wait() != 0:
                raise SystemExit("git archive failed")
            return
    raise SystemExit("no such task")


def cmd_list(args):
    rows = []
    for p in sorted(PRIVATE.glob("*/*/task.json")):
        t = json.loads(p.read_text())
        v = p.parent / "validation.json"
        ok = json.loads(v.read_text()).get("ok") if v.is_file() else None
        rows.append((t["id"], t["repo"].rsplit("/", 1)[-1], t["task_kind"], t["lang"], t["difficulty"], t["src_lines"], t["tests_added"], ok))
    for r in rows:
        print(*r, sep="\t")
    print(len(rows), "tasks", file=sys.stderr)


if __name__ == "__main__":
    cmds = {"scan": cmd_scan, "build": cmd_build, "list": cmd_list, "materialize": cmd_materialize}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    cmds[sys.argv[1]](sys.argv[2:])
