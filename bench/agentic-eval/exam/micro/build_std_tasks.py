#!/usr/bin/env python3
"""EXAM200e: build candidate task directories from open benchmarks (source A) and validate them.

Sources (both checked for licence before use, see SOURCES.md):
  humaneval  openai/human-eval, data/HumanEval.jsonl.gz, MIT. 164 problems.
  quixbugs   jkoppel/QuixBugs, python_programs (buggy) and correct_python_programs, json_testcases, MIT.

Rewriting before validation (card step A): the function is renamed to solve_<id> (all word occurrences in the
file, docstring examples included). The start state is the buggy file (QuixBugs) or the prompt with an empty body
(HumanEval). The hidden tests are the benchmark's own tests, rewritten to import `solution`.
Behaviour changes and parameter renames are NOT made in this version (reported as not done).

Usage:
  build_std_tasks.py build --raw DIR --cand DIR            write candidate task dirs
  build_std_tasks.py validate --cand DIR --out DIR         run std_check.check on every candidate
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import std_check  # noqa: E402

HE_LICENCE = {"src": "humaneval", "licence": "MIT", "url": "https://github.com/openai/human-eval"}
QB_LICENCE = {"src": "quixbugs", "licence": "MIT", "url": "https://github.com/jkoppel/QuixBugs"}


def _rename(text: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, text)


def _write_task(cand: Path, tid: str, meta: dict, start: str, ref: str, hidden_files: dict[str, str]) -> None:
    d = cand / tid
    for sub in ("start", "reference", "hidden"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    (d / "start" / "solution.py").write_text(start, encoding="utf-8")
    (d / "reference" / "solution.py").write_text(ref, encoding="utf-8")
    for name, text in hidden_files.items():
        (d / "hidden" / name).write_text(text, encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def build_humaneval(raw: Path, cand: Path) -> int:
    n = 0
    with gzip.open(raw / "HumanEval.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            num = int(r["task_id"].split("/")[1])
            tid = f"he-{num:03d}"
            new = f"solve_he{num:03d}"
            old = r["entry_point"]
            prompt = _rename(r["prompt"], old, new)
            ref = prompt + _rename(r["canonical_solution"], old, new)
            test = r["test"]
            hidden = {
                "test_hidden.py": "import solution as _s\n" + test + f"\n\ncheck(_s.{new})\nprint('ok')\n",
            }
            meta = {"id": tid, "src": "humaneval", "licence": "MIT", "source": r["task_id"],
                    "source_url": HE_LICENCE["url"], "entry": new, "original_entry": old,
                    "language": "python", "behaviour_changed": False}
            _write_task(cand, tid, meta, prompt, ref, hidden)
            n += 1
    return n


def build_quixbugs(raw: Path, cand: Path) -> int:
    qb = raw / "quixbugs"
    n = 0
    for buggy in sorted((qb / "python_programs").glob("*.py")):
        name = buggy.stem
        correct = qb / "correct_python_programs" / buggy.name
        cases = qb / "json_testcases" / f"{name}.json"
        if not correct.exists() or not cases.exists():
            continue
        new = f"solve_{name}"
        start = _rename(buggy.read_text(encoding="utf-8"), name, new)
        ref = _rename(correct.read_text(encoding="utf-8"), name, new)
        hidden = {
            "test_hidden.py": (
                "import json\nimport solution as _s\n"
                f"f = _s.{new}\n"
                "for line in open('cases.json', encoding='utf-8'):\n"
                "    args, expected = json.loads(line)\n"
                "    assert f(*args) == expected, (args, expected)\n"
                "print('ok')\n"),
            "cases.json": cases.read_text(encoding="utf-8"),
        }
        meta = {"id": f"qb-{name}", "src": "quixbugs", "licence": "MIT", "source": f"python_programs/{buggy.name}",
                "source_url": QB_LICENCE["url"], "entry": new, "original_entry": name, "language": "python",
                "behaviour_changed": False, "kind": "bugfix"}
        _write_task(cand, f"qb-{name}", meta, start, ref, hidden)
        n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", type=Path, required=True)
    b.add_argument("--cand", type=Path, required=True)
    v = sub.add_parser("validate")
    v.add_argument("--cand", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        a.cand.mkdir(parents=True, exist_ok=True)
        he = build_humaneval(a.raw, a.cand)
        qb = build_quixbugs(a.raw, a.cand)
        print(json.dumps({"humaneval": he, "quixbugs": qb}))
        return 0
    accepted = 0
    total = 0
    for d in sorted(p for p in a.cand.iterdir() if p.is_dir()):
        res = std_check.check(d, a.out)
        total += 1
        accepted += int(res["accepted"])
        print(d.name, "ACC" if res["accepted"] else "REJ", "; ".join(res["reasons"]), flush=True)
    print(json.dumps({"total": total, "accepted": accepted}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
