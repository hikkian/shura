#!/usr/bin/env python3
"""EXAM200: build the corrected task set tasks_v3r1 from tasks_v3 without touching tasks_v3.

Three kinds of defects are fixed (reports/errmap/REPORT-ERRMAP1.md, reports/exam200/REPORT-EXAM200.md):
  layout   the refactor prompt now names the files the hidden test requires;
  docs     the hidden test no longer demands a phrase that exists only in the reference solution; the prompt says what to do;
  inputs   requirements that hidden tests check but the prompt never stated are added to the prompt (exam/prompt-additions.json).
Hidden contract tests, reference solutions and mutants are not changed, except the one docs assertion.

    python3 exam/build_v3r1.py --out exam/tasks_v3r1 [--validate]      # run from bench/agentic-eval
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "tasks_v3"
ADDITIONS = HERE / "prompt-additions.json"
DOC_PHRASE = "Validation is explicit"
DOC_SENTENCE = (" Finish the project notes too: README.md must no longer say 'Starter project', and the 'to be completed' "
                "placeholder in docs/design.md must be replaced with a short description of the implementation.")


def first_paragraph_append(prompt, sentence):
    """Add a sentence to the end of the contract (the first paragraph of the prompt)."""
    head, sep, tail = prompt.partition("\n\n")
    return head.rstrip() + " " + sentence.strip() + sep + tail


def layout_sentence(hidden_text):
    js = re.search(r"existsSync\('([\w-]+)/index\.js'\).*?existsSync\('\1/core\.js'\).*?!fs\.existsSync\('([\w.-]+)'\)", hidden_text, re.S)
    if js:
        return (f" The layout is fixed: `{js.group(1)}/index.js` is the public facade, `{js.group(1)}/core.js` holds the "
                f"implementation, and the old `{js.group(2)}` is removed.")
    py = re.search(r'Path\("([\w-]+)/__init__\.py"\).*?Path\("\1/core\.py"\).*?Path\("([\w.-]+)"\)\.exists\(\)', hidden_text, re.S)
    if py:
        return (f" The layout is fixed: `{py.group(1)}/__init__.py` is the public facade, `{py.group(1)}/core.py` holds the "
                f"implementation, and the old `{py.group(2)}` is removed.")
    return None


def fix_docs_test(path):
    text = path.read_text(encoding="utf-8")
    new = text.replace('self.assertIn("Validation is explicit", design)', 'self.assertNotIn("to be completed", design)')
    new = new.replace("assert.ok(design.includes('Validation is explicit'))", "assert.ok(!design.includes('to be completed'))")
    if new != text:
        path.write_text(new, encoding="utf-8")
        return True
    return False


def build(out):
    additions = json.loads(ADDITIONS.read_text(encoding="utf-8")) if ADDITIONS.exists() else []
    add_by = {}
    for a in additions:
        for kind in a["kinds"]:
            add_by.setdefault((a["family"], kind), []).append(a["add_to_prompt"])
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SRC, out)
    log = []
    for tj_path in sorted(out.glob("*/*/task.json")):
        task_dir = tj_path.parent
        tj = json.loads(tj_path.read_text(encoding="utf-8"))
        fixes = []
        hidden_files = [p for p in (task_dir / "hidden").rglob("*") if p.is_file()]
        hidden_text = "".join(p.read_text(encoding="utf-8", errors="replace") for p in hidden_files)
        prompt = tj["prompt"]
        if tj["kind"] == "refactor" and "public package is split" in hidden_text.replace("_", " "):
            sentence = layout_sentence(hidden_text)
            if sentence is None:
                raise SystemExit(f"{tj['id']}: layout test found but file names not recognised")
            marker = "Keep the change reviewable across 3 files."
            if marker not in prompt:
                raise SystemExit(f"{tj['id']}: refactor prompt has no known anchor sentence")
            prompt = prompt.replace(marker, marker + sentence, 1)
            fixes.append("layout")
        if DOC_PHRASE in hidden_text:
            start_design = (task_dir / "start" / "docs" / "design.md").read_text(encoding="utf-8")
            sol_design = (task_dir / "solution" / "docs" / "design.md").read_text(encoding="utf-8")
            if "to be completed" not in start_design or "to be completed" in sol_design:
                raise SystemExit(f"{tj['id']}: docs placeholder assumption does not hold")
            if not any(fix_docs_test(p) for p in hidden_files):
                raise SystemExit(f"{tj['id']}: docs assertion not found in a known form")
            prompt = prompt.rstrip() + DOC_SENTENCE
            fixes.append("docs")
        for sentence in add_by.get((tj["family"], tj["kind"]), []):
            if sentence.strip() not in prompt:
                prompt = first_paragraph_append(prompt, sentence)
                if "inputs" not in fixes:
                    fixes.append("inputs")
        if fixes:
            tj["prompt"] = prompt
        tj["set"] = "v3r1"
        tj["fixes"] = fixes
        tj_path.write_text(json.dumps(tj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        digest = hashlib.sha256()
        for p in sorted((task_dir / "hidden").rglob("*")):
            if p.is_file():
                digest.update(p.relative_to(task_dir).as_posix().encode() + b"\0" + p.read_bytes())
        log.append({"id": tj["id"], "family": tj["family"], "kind": tj["kind"], "lang": tj.get("lang"), "fixes": fixes,
                    "hidden_sha256": digest.hexdigest(), "prompt_sha256": hashlib.sha256(tj["prompt"].encode()).hexdigest()})
    return log


def run_tests(task_dir, with_solution):
    """Start files (+ the reference solution) + hidden tests in a scratch folder; returns the exit code of the task's test command."""
    tj = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="exam200-") as tmp:
        work = Path(tmp) / "work"
        shutil.copytree(task_dir / "start", work)
        if with_solution:
            for name in tj.get("solution_remove", []):
                target = work / name
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
            shutil.copytree(task_dir / "solution", work, dirs_exist_ok=True)
        shutil.copytree(task_dir / "hidden", work / "hidden_tests")
        argv = ["nice", "-n", "19", "bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
                "--bind", str(work), str(work), "--unshare-net", "--chdir", str(work)] + tj["test_cmd"]
        try:
            return subprocess.run(argv, capture_output=True, text=True, timeout=tj.get("timeout_s", 300)).returncode
        except subprocess.TimeoutExpired:
            return 124


def validate(out, only=None):
    bad = []
    tasks = sorted(out.glob("*/*/task.json"))
    for i, tj_path in enumerate(tasks):
        if only and only not in str(tj_path):
            continue
        task_dir = tj_path.parent
        sol = run_tests(task_dir, True)
        start = run_tests(task_dir, False)
        if sol != 0 or start == 0:
            bad.append({"task": f"{task_dir.parent.name}-{task_dir.name}", "solution_rc": sol, "start_rc": start})
    return len(tasks), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "tasks_v3r1"))
    ap.add_argument("--validate", action="store_true", help="reference solution must pass, start files must fail (needs bwrap)")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    log = build(out)
    (HERE / "build-v3r1.log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    counts = {}
    for row in log:
        for fix in row["fixes"]:
            counts[fix] = counts.get(fix, 0) + 1
    print(f"built {len(log)} tasks into {out}; fixes: {counts}; untouched: {sum(1 for r in log if not r['fixes'])}")
    if args.validate:
        n, bad = validate(out, args.only)
        print(f"validated {n} tasks; problems: {len(bad)}")
        for b in bad:
            print("  ", b)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
