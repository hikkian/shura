#!/usr/bin/env python3
"""EXAM200b: micro tasks (П1 стадия) from PUBLIC commit history, validated in a no-network bwrap sandbox.

Method (CTRL1, public sources only): a commit of a public repository that changes non-test Python code and adds or
changes its unittest files. The task starts from the parent state; the hidden tests are the test files of the commit;
the reference is the commit state. Only the origin/main history is used (commits that are not pushed are not read).

Per candidate, the checks (acceptance rule, written before the build):
  1. start (parent state + hidden tests) must FAIL;
  2. reference (commit state + hidden tests) must PASS three times in a row;
  3. up to six mutants of the reference's changed lines are generated; the first three that compile are kept;
     at least two of them must be caught by the hidden tests;
  4. the empty solution equals the start (check 1).

Usage:
  build_micro.py list   [--repo barid|shura] [--limit N]         candidate commits (no test run)
  build_micro.py check  --repo R --commit SHA [--out DIR]        build and validate one candidate
  build_micro.py manifest --out DIR                              validate and collect all checked candidates
Scratch: /dev/shm/EXAM200b/. Nothing is written into the repositories.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/dev/shm/EXAM200b")
SOURCES = {
    "barid": {"path": Path.home() / "Projects/barid", "ref": "origin/main", "url": "https://github.com/hikkian/barid",
              "license": "MIT"},
    "shura": {"path": Path.home() / "Projects/shura", "ref": "origin/main", "url": "https://github.com/hikkian/shura",
              "license": "MIT"},
}
TEST_RE = re.compile(r"(^|/)tests/test_[^/]*\.py$")
TIMEOUT_S = 120
MUT_NEEDED, MUT_KEEP = 2, 3
MUT_TRY = 6
SAFE_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}


def git(repo: Path, *args: str, text: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text, timeout=300, check=True)
    return r.stdout


def export_tree(repo: Path, rev: str, dest: Path) -> Path:
    """Materialise a commit's tree into dest (regular files only, no git metadata)."""
    blob = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", rev], capture_output=True,
                          timeout=300, check=True).stdout
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = Path(m.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe member {m.name}")
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tf.extractfile(m).read())
    return dest


def commits(repo_key: str):
    """Yield (sha, parent, subject, [(status, path, added, deleted)]) for non-merge commits on the public ref."""
    src = SOURCES[repo_key]
    repo = src["path"]
    log = git(repo, "log", src["ref"], "--no-merges", "--format=@@%H %P%x09%s", "--numstat")
    cur = None
    for line in log.splitlines():
        if line.startswith("@@"):
            if cur:
                yield cur
            head, _, subject = line[2:].partition("\t")
            parts = head.split()
            cur = {"sha": parts[0], "parent": parts[1] if len(parts) > 1 else "", "subject": subject, "files": []}
        elif line.strip() and cur is not None:
            a, d, p = (line.split("\t") + ["", "", ""])[:3]
            cur["files"].append((a, d, p))
    if cur:
        yield cur


def candidate_ok(c: dict) -> dict | None:
    """Commit rule: non-test Python changes (2–150 lines, 1–3 files) and a unittest file added or changed."""
    if not c["parent"]:
        return None
    src = [(a, d, p) for a, d, p in c["files"] if p.endswith(".py") and not TEST_RE.search(p) and not p.startswith("tests/")]
    tests = [(a, d, p) for a, d, p in c["files"] if TEST_RE.search(p)]
    if not src or not tests or len(src) > 3:
        return None
    lines = 0
    for a, d, _ in src:
        if not (a.isdigit() and d.isdigit()):
            return None           # binary change
        lines += int(a) + int(d)
    if not 2 <= lines <= 150:
        return None
    return {"sha": c["sha"], "parent": c["parent"], "subject": c["subject"],
            "src": [p for _, _, p in src], "tests": [p for _, _, p in tests]}


def run_bwrap(work: Path, cmd: list[str]) -> tuple[int, str]:
    # memory cap for every run (rule of 14 Nov: a runaway test must not take the desktop down with it)
    argv = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=4G", "-p", "MemorySwapMax=0",
            "nice", "-n", "19", "ionice", "-c3", "bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/tmp", "--tmpfs", str(Path.home()), "--bind", str(work), str(work),
            "--unshare-net", "--chdir", str(work), "--clearenv"]
    for k, v in SAFE_ENV.items():
        argv += ["--setenv", k, v]
    argv += cmd
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_S)
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return 124, out + "\n[timeout]"


_RAN = re.compile(r"^Ran (\d+) tests?", re.M)
_FAILED = re.compile(r"^FAILED \((.*)\)", re.M)


def counts(rc: int, out: str) -> dict:
    """unittest counts: passed = ran - failures - errors (as in harness.soft_score)."""
    m = _RAN.search(out)
    total = int(m.group(1)) if m else None
    if total is None:
        return {"rc": rc, "total": None, "passed": 0, "import_error": "ModuleNotFoundError" in out or "ImportError" in out}
    failed = 0
    f = _FAILED.search(out)
    if f:
        for part in f.group(1).split(","):
            k, _, v = part.strip().partition("=")
            if k.strip() in ("failures", "errors") and v.strip().isdigit():
                failed += int(v)
    return {"rc": rc, "total": total, "passed": 0 if rc != 0 and total is None else max(0, total - failed),
            "import_error": "ImportError" in out or "ModuleNotFoundError" in out}


def stage_hidden(work: Path, hidden_dir: Path) -> None:
    """Copy the hidden test files over the work folder (they replace the parent versions or add new files)."""
    for f in hidden_dir.rglob("*"):
        if f.is_file():
            rel = f.relative_to(hidden_dir)
            (work / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, work / rel)


def run_hidden(work: Path, hidden_dir: Path, test_files: list[str]) -> dict:
    stage_hidden(work, hidden_dir)
    rc, out = run_bwrap(work, ["python3", "-m", "unittest", *test_files])
    return {**counts(rc, out), "tail": out[-800:]}


MUTATION_PAIRS = [("==", "!="), ("!=", "=="), ("<=", "<"), (">=", ">"), (" < ", " <= "), (" > ", " >= "),
                  (" and ", " or "), (" or ", " and "), ("True", "False"), ("False", "True"),
                  (" not ", " ")]


def mutants(reference: Path, changed: dict[str, list[int]]) -> list[tuple[str, str, str]]:
    """Deterministic one-line mutants on lines added by the commit. Yields (file, description, new_text)."""
    out = []
    for rel, lines in sorted(changed.items()):
        path = reference / rel
        text = path.read_text(encoding="utf-8").splitlines(keepends=True)
        for ln in lines:
            if not 1 <= ln <= len(text):
                continue
            line = text[ln - 1]
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", '"""', "'''", "def ", "class ", "@", "import ", "from ")):
                continue
            for old, new in MUTATION_PAIRS:
                if old in line:
                    mutated = line.replace(old, new, 1)
                    out.append((rel, f"L{ln}: {old.strip()!r}->{new.strip()!r}", ln, mutated))
                    break
            else:
                if re.search(r"\b\d+\b", line) and not stripped.startswith(("return ", "raise ")):
                    m = re.search(r"\b(\d+)\b", line)
                    mutated = line[:m.start()] + str(int(m.group(1)) + 1) + line[m.end():]
                    out.append((rel, f"L{ln}: number +1", ln, mutated))
                elif stripped.startswith("return "):
                    out.append((rel, f"L{ln}: return None", ln, line[:line.index("return")] + "return None\n"))
    return out


def compiles(path: Path) -> bool:
    r = subprocess.run([sys.executable, "-c", "import ast,sys;ast.parse(open(sys.argv[1],encoding='utf-8').read())",
                        str(path)], capture_output=True, timeout=60)
    return r.returncode == 0


def changed_lines(repo: Path, parent: str, sha: str, rel: str) -> list[int]:
    """Line numbers (new file) of lines added by the commit in rel."""
    diff = git(repo, "diff", "-U0", parent, sha, "--", rel)
    lines = []
    for m in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", diff, re.M):
        start, n = int(m.group(1)), int(m.group(2) or 1)
        lines.extend(range(start, start + n))
    return lines


def check(repo_key: str, sha: str, out: Path | None) -> dict:
    src = SOURCES[repo_key]
    repo = src["path"]
    meta = next((c for c in commits(repo_key) if c["sha"] == sha), None)
    if meta is None:
        raise SystemExit(f"commit {sha} is not on {src['ref']}")
    cand = candidate_ok(meta)
    if cand is None:
        raise SystemExit(f"commit {sha} does not meet the candidate rule")
    work_root = SCRATCH / "work" / f"{repo_key}-{sha[:10]}"
    parent_dir = export_tree(repo, cand["parent"], work_root / "parent")
    ref_dir = export_tree(repo, sha, work_root / "reference")
    # reference = commit tree; hidden = commit versions of the test files
    hidden = work_root / "hidden"
    shutil.rmtree(hidden, ignore_errors=True)
    for rel in cand["tests"]:
        (hidden / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ref_dir / rel, hidden / rel)
    # the reference solution keeps the parent's test files (the hidden ones are copied in at test time)
    for rel in cand["tests"]:
        if (parent_dir / rel).exists():
            shutil.copyfile(parent_dir / rel, ref_dir / rel)
        else:
            (ref_dir / rel).unlink()
    test_files = cand["tests"]
    result: dict = {"repo": repo_key, "commit": sha, "parent": cand["parent"], "subject": cand["subject"],
                    "src": cand["src"], "tests": cand["tests"], "test_cmd": ["python3", "-m", "unittest", *test_files],
                    "timeout_s": TIMEOUT_S}

    def fresh(name: str, tree: Path) -> Path:
        w = work_root / name
        shutil.rmtree(w, ignore_errors=True)
        shutil.copytree(tree, w)
        return w

    start_run = run_hidden(fresh("w-start", parent_dir), hidden, test_files)
    result["start"] = {k: start_run[k] for k in ("rc", "total", "passed", "import_error")}
    refs = [run_hidden(fresh(f"w-ref{i}", ref_dir), hidden, test_files) for i in range(3)]
    result["reference"] = [{k: r[k] for k in ("rc", "total", "passed")} for r in refs]
    changed = {rel: changed_lines(repo, cand["parent"], sha, rel) for rel in cand["src"]}
    muts = []
    for rel, desc, ln, text in mutants(ref_dir, changed)[:MUT_TRY * 4]:
        if len(muts) >= MUT_TRY:
            break
        w = fresh("w-mut", ref_dir)
        target = w / rel
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[ln - 1] = text
        target.write_text("".join(lines), encoding="utf-8")
        if not compiles(target):
            continue
        muts.append((rel, desc, w))
        if len(muts) == MUT_KEEP:
            break
    caught = []
    mut_rows = []
    for rel, desc, w in muts:
        r = run_hidden(w, hidden, test_files)
        ok = r["rc"] != 0 or (r["total"] is not None and r["passed"] < r["total"])
        caught.append(ok)
        mut_rows.append({"file": rel, "change": desc, "caught": ok, "passed": r["passed"], "total": r["total"]})
        shutil.rmtree(w, ignore_errors=True)
    result["mutants"] = mut_rows
    ref_ok = all(r["rc"] == 0 and r["total"] and r["passed"] == r["total"] for r in refs)
    start_fails = start_run["rc"] != 0 and not (start_run["total"] and start_run["passed"] == start_run["total"])
    result["checks"] = {
        "start_fails": start_fails,
        "reference_3x_pass": ref_ok,
        "mutants_kept": len(mut_rows),
        "mutants_caught": sum(caught),
        "mutants_ok": len(mut_rows) == MUT_KEEP and sum(caught) >= MUT_NEEDED,
        "start_only_import_error": bool(start_run["import_error"]) and not (start_run["total"] and start_run["passed"] < start_run["total"]),
    }
    result["accepted"] = bool(start_fails and ref_ok and result["checks"]["mutants_ok"])
    result["unit"] = f"{repo_key}:{cand['src'][0]}"
    result["id"] = f"{repo_key}-{sha[:10]}"
    result["hidden_sha256"] = hashlib.sha256(
        b"".join(f.read_bytes() for f in sorted(hidden.rglob("*")) if f.is_file())).hexdigest()[:16]
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{result['id']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(work_root, ignore_errors=True)
    return result


def cmd_list(args) -> int:
    repos = [args.repo] if args.repo else list(SOURCES)
    total = 0
    for key in repos:
        n = 0
        for c in commits(key):
            cand = candidate_ok(c)
            if cand:
                n += 1
                if args.limit and n > args.limit:
                    break
                print(f"{key}\t{cand['sha']}\t{','.join(cand['src'])}\t{cand['subject'][:70]}")
        total += n
        print(f"# {key}: {n} candidates", file=sys.stderr)
    print(f"# total {total}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list")
    p_list.add_argument("--repo", choices=list(SOURCES))
    p_list.add_argument("--limit", type=int, default=0)
    p_check = sub.add_parser("check")
    p_check.add_argument("--repo", required=True, choices=list(SOURCES))
    p_check.add_argument("--commit", required=True)
    p_check.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        return cmd_list(a)
    res = check(a.repo, a.commit, a.out)
    print(json.dumps({k: res[k] for k in ("id", "accepted", "checks", "start", "reference", "mutants")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
