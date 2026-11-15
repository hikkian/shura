#!/usr/bin/env python3
"""EXAM200e: validation of single-function micro tasks from open benchmarks (source A).

A task is a directory with:
  start/solution.py       the state the model receives (buggy file, or the stub with docstring)
  reference/solution.py   the reference (correct file, or prompt + canonical solution)
  hidden/test_hidden.py   the hidden test script; it imports `solution` and exits 0 when the candidate passes
  hidden/cases.json       (QuixBugs only) json-lines test cases read by the hidden script
  meta.json               id, src, entry (function name after renaming), lines of the reference body, licence

Checks (same rule as EXAM200b, written before the build):
  1. start must FAIL the hidden tests (rc != 0);
  2. the empty solution is the start (same check);
  3. reference must PASS three times in a row;
  4. up to six mutants of the reference's body lines are made; the first three that compile are kept;
     at least two of them must be caught by the hidden tests.
  5. the reference body is at most 60 lines (one model answer).

Every run happens in a fresh work directory under /dev/shm/EXAM200e/work, inside bwrap (no network,
read-only root, private HOME and /tmp) from build_micro.run_bwrap. Nothing is written into the sources.

Usage:
  std_check.py check TASK_DIR [--out DIR]     validate one task directory, print and save the verdict
  std_check.py run  TASK_DIR                  run the hidden script on TASK_DIR/reference (debug)
"""
from __future__ import annotations

import ast
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_micro as bm  # noqa: E402  (sandbox runner and mutant generator from EXAM200b)

WORK_ROOT = Path("/dev/shm/EXAM200e/work")
MAX_BODY_LINES = 60
bm.TIMEOUT_S = 20  # outer limit; the inner `timeout` below decides first


def _run(work: Path) -> int:
    """Run the hidden script inside the sandbox in `work`. Returns the exit code (124 on timeout)."""
    # inner timeout: a looping program is killed after 10 s and counts as a failure;
    # memory is capped by run_bwrap (systemd scope, MemoryMax=4G)
    rc, _out = bm.run_bwrap(work, ["timeout", "-k", "2", "10", "python3", "test_hidden.py"])
    return rc


def _fresh(name: str, solution_text: str, hidden_dir: Path) -> Path:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=WORK_ROOT))
    (work / "solution.py").write_text(solution_text, encoding="utf-8")
    for f in hidden_dir.iterdir():
        shutil.copy(f, work / f.name)
    return work


def _compiles(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False

# Operator swaps added for source A: a short reference body gives few mutants with the EXAM200b list alone.
_SWAPS = [(" + ", " - "), (" - ", " + "), ("<=", ">="), (">=", "<="), ("==", "!="), ("!=", "=="),
          (" < ", " >= "), (" > ", " <= "), (" and ", " or "), (" or ", " and "), ("not ", "")]


def _extra_mutants(ref_text: str, body: list[int]) -> list[tuple[str, str, int, str]]:
    """Operator-swap mutants on the body lines. Yields (rel, description, line number, new line)."""
    lines = ref_text.splitlines(keepends=True)
    out = []
    for ln in body:
        line = lines[ln - 1]
        for old, new in _SWAPS:
            if old in line:
                out.append(("solution.py", f"L{ln}: {old.strip()!r}->{new.strip()!r} (swap)", ln,
                            line.replace(old, new, 1)))
    return out


def _body_lines(reference_text: str, entry: str) -> list[int]:
    """1-based line numbers of the function body of `entry` in the reference (for mutants)."""
    tree = ast.parse(reference_text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body = body[1:]  # skip the docstring: its lines are not code
            if not body:
                return []
            start = body[0].lineno
            end = node.end_lineno or node.lineno
            return list(range(start, end + 1))
    return []


def check(task_dir: Path, out: Path | None) -> dict:
    meta = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    tid, entry = meta["id"], meta["entry"]
    hidden = task_dir / "hidden"
    start_text = (task_dir / "start" / "solution.py").read_text(encoding="utf-8")
    ref_text = (task_dir / "reference" / "solution.py").read_text(encoding="utf-8")
    res: dict = {"id": tid, "src": meta["src"], "entry": entry, "accepted": False, "reasons": []}

    body = _body_lines(ref_text, entry)
    body_len = len(body)
    res["reference_body_lines"] = body_len
    if body_len > MAX_BODY_LINES:
        res["reasons"].append(f"reference body {body_len} lines > {MAX_BODY_LINES}")

    w = _fresh(tid + "-start", start_text, hidden)
    res["start_rc"] = _run(w)
    shutil.rmtree(w, ignore_errors=True)
    if res["start_rc"] == 0:
        res["reasons"].append("start passes the hidden tests")

    refs = []
    for i in range(3):
        w = _fresh(f"{tid}-ref{i}", ref_text, hidden)
        refs.append(_run(w))
        shutil.rmtree(w, ignore_errors=True)
    res["reference_rc"] = refs
    if any(rc != 0 for rc in refs):
        res["reasons"].append(f"reference not 3 of 3 passing: {refs}")

    mutants_kept, caught = [], 0
    if all(rc == 0 for rc in refs):
        tmp_ref = Path(tempfile.mkdtemp(prefix=f"{tid}-mut-", dir=WORK_ROOT))
        (tmp_ref / "solution.py").write_text(ref_text, encoding="utf-8")
        cands = (bm.mutants(tmp_ref, {"solution.py": body}) if body else []) + _extra_mutants(ref_text, body)
        ref_lines = ref_text.splitlines(keepends=True)
        for rel, desc, ln, mutated_line in cands[:bm.MUT_TRY]:
            if len(mutants_kept) >= bm.MUT_KEEP:
                break
            # bm.mutants returns the changed line only: put it back into the whole reference file
            lines = list(ref_lines)
            lines[ln - 1] = mutated_line if mutated_line.endswith("\n") else mutated_line + "\n"
            mutated = "".join(lines)
            if not _compiles(mutated):
                continue
            mutants_kept.append((desc, mutated))
        for desc, mutated in mutants_kept:
            w = _fresh(f"{tid}-mut", mutated, hidden)
            rc = _run(w)
            shutil.rmtree(w, ignore_errors=True)
            if rc != 0:
                caught += 1
        shutil.rmtree(tmp_ref, ignore_errors=True)
    res["mutants_kept"] = len(mutants_kept)
    res["mutants_caught"] = caught
    res["mutant_descriptions"] = [d for d, _ in mutants_kept]
    if caught < bm.MUT_NEEDED or len(mutants_kept) < bm.MUT_KEEP:
        res["reasons"].append(f"mutants {caught} of {len(mutants_kept)} caught (need 2 of 3)")

    res["accepted"] = not res["reasons"]
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{tid}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("task_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = check(a.task_dir, a.out)
    print(json.dumps({k: res[k] for k in ("id", "accepted", "start_rc", "reference_rc", "mutants_kept",
                                         "mutants_caught", "reasons")}, ensure_ascii=False))
    return 0 if res["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
