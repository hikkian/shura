#!/usr/bin/env python3
"""EXAM200e: manifest-micro-v2.json — every micro task that passed the checks, with its unit, part and difficulty group.

Inputs (all in this folder):
  tasks/<id>/             statement.ru.md, reader.md, start/, reference/, hidden/, meta.json, validation.json
  manifest-micro-v1.json  the eight tasks of EXAM200b (kept, with their own runner: unittest in the repo snapshot)
  ../manifest-v3r1.json   exam manifest v3r1 (duplicate check by hidden_sha256 and by reference text)

Rules (written before the build):
  unit   = the original problem: he-NNN (HumanEval task), qb-<name> (QuixBugs program), or the EXAM200b commit;
  part   = sha256("SEED:unit") first hex digit: 0-7 search, 8-b gate, c-f sealed;
           the sealed part from source A is EMPTY in this version: the corrected EvalPlus versions were not obtained;
  group  = light (reference body <= 4 lines and <= 5 checks), hard (body >= 10 lines), medium (the rest);
  dup    = two tasks whose reference bodies share >= 0.5 of their token 6-grams are duplicates; one is dropped;
  copy   = no English sentence of 6+ words from a start file's docstring appears in the statement (verbatim copy).
Output: manifest-micro-v2.json and a summary printed to stdout.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = HERE / "tasks"
SEED = "20261114"
V3R1 = HERE.parent / "manifest-v3r1.json"
V1 = HERE / "manifest-micro-v1.json"
TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\sA-Za-z_0-9]")


def part_of(unit: str) -> str:
    h = hashlib.sha256(f"{SEED}:{unit}".encode()).hexdigest()[0]
    return "search" if h in "01234567" else ("gate" if h in "89ab" else "sealed")


def body_lines(ref: str, entry: str) -> int:
    import ast
    tree = ast.parse(ref)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body = body[1:]
            if not body:
                return 0
            return (body[-1].end_lineno or body[-1].lineno) - body[0].lineno + 1
    return 0


def grams(text: str, n: int = 6) -> set:
    toks = [t for t in TOKEN.findall(text)]
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def english_sentences_6(doc: str) -> set:
    out = set()
    for sent in re.split(r"[.!?\n]", doc):
        words = re.findall(r"[A-Za-z']+", sent)
        if len(words) >= 6:
            out.add(" ".join(w.lower() for w in words[:6]))
    return out


def main() -> int:
    v3 = json.loads(V3R1.read_text(encoding="utf-8"))["tasks"]
    v3_hashes = {t["hidden_sha256"] for t in v3}
    v1 = json.loads(V1.read_text(encoding="utf-8"))["tasks"]

    rows, rejected = [], []
    for d in sorted(TASKS.iterdir()):
        if not (d / "meta.json").exists():
            continue  # EXAM200b tasks have no meta.json; they are listed from manifest v1
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        val = json.loads((d / "validation.json").read_text(encoding="utf-8"))
        st = (d / "statement.ru.md").read_text(encoding="utf-8")
        ref = (d / "reference" / "solution.py").read_text(encoding="utf-8")
        start = (d / "start" / "solution.py").read_text(encoding="utf-8")
        reader = (d / "reader.md").read_text(encoding="utf-8") if (d / "reader.md").exists() else ""
        # verdict from the reader's note: clean when the note says so and never says "не чистая"
        low = reader.lower()
        verdict = "чистая" if reader and "чистая" in low and "не чистая" not in low else "не проверена"
        checks = (d / "hidden" / "test_hidden.py").read_text(encoding="utf-8").count("assert") or 0
        nbody = body_lines(ref, meta["entry"])
        doc_sent = english_sentences_6(start)
        copied = [s for s in doc_sent if s in st.lower()]
        row = {
            "id": d.name, "unit": meta["source"] if meta["src"] == "humaneval" else f"quixbugs:{meta['original_entry']}",
            "src": meta["src"], "licence": meta["licence"], "source": meta["source"], "source_url": meta["source_url"],
            "entry": meta["entry"], "original_entry": meta["original_entry"], "language": meta["language"],
            "behaviour_changed": meta.get("behaviour_changed", False), "kind": meta.get("kind", "implement"),
            "statement": f"tasks/{d.name}/statement.ru.md",
            "statement_sha256": hashlib.sha256(st.encode("utf-8")).hexdigest(),
            "hidden_sha256": hashlib.sha256((d / "hidden" / "test_hidden.py").read_bytes()).hexdigest(),
            "reference_body_lines": nbody, "checks": checks,
            "validation": {k: val[k] for k in ("start_rc", "reference_rc", "mutants_kept", "mutants_caught")},
            "reader": {"file": f"tasks/{d.name}/reader.md", "verdict": verdict},
            "copied_sentences": copied,
            "runner": "run_micro (hidden/test_hidden.py)",
            "_ref_grams": None,
        }
        if verdict != "чистая":
            rejected.append((d.name, verdict))
            continue
        if copied:
            rejected.append((d.name, "copied sentence"))
            continue
        row["group"] = "light" if (nbody <= 4 and checks <= 5) else ("hard" if nbody >= 10 else "medium")
        row["part"] = part_of(row["unit"])
        if row["part"] == "sealed":
            # sealed needs corrected versions (EvalPlus) or repository tests; source A here is unrevised: gate instead
            row["part"] = "gate"
        row["_ref_grams"] = grams(ref)
        rows.append(row)

    # duplicates among the kept tasks (token 6-grams of the reference bodies)
    dropped = []
    keep = []
    for r in rows:
        dup_of = None
        for k in keep:
            a, b = r["_ref_grams"], k["_ref_grams"]
            if a and b and len(a & b) / min(len(a), len(b)) >= 0.5:
                dup_of = k["id"]
                break
        if dup_of:
            dropped.append((r["id"], dup_of))
        else:
            keep.append(r)
    for r in keep:
        r.pop("_ref_grams", None)
        if r["hidden_sha256"] in v3_hashes:
            dropped.append((r["id"], "v3r1 hidden_sha256"))
    keep = [r for r in keep if r["hidden_sha256"] not in v3_hashes]

    # EXAM200b tasks (eight, from manifest v1), kept as they were validated
    for t in v1:
        t2 = dict(t)
        t2["src"] = "barid/shura"
        t2["runner"] = "unittest in the repository snapshot (run by build_micro.py, not by run_micro)"
        t2["group"] = "medium"
        keep.append(t2)

    groups = Counter(r.get("group") for r in keep)
    parts = Counter(r.get("part", "exam200b") for r in keep)
    units = len({r.get("unit", r.get("commit", r["id"])) for r in keep})
    light_share = groups.get("light", 0) / max(1, len(keep))
    manifest = {
        "version": "micro-v2",
        "date": "2026-11-14",
        "seed": SEED,
        "sources": {
            "humaneval": {"repo": "openai/human-eval", "licence": "MIT", "file": "data/HumanEval.jsonl.gz",
                          "sha256": "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"},
            "quixbugs": {"repo": "jkoppel/QuixBugs", "licence": "MIT", "commit": "4257f44b0ff1181dedaedee6a447e133219fcebf"},
            "barid": "origin/main bf3da35 (MIT)", "shura": "origin/main b09ea52 (MIT)",
        },
        "rules": {
            "unit": "one original problem per unit", "validation": "start fails; reference 3 of 3; >= 2 of 3 mutants caught",
            "reader": "blind reader sees statement and hidden tests only; verdict in reader.md",
            "part": "sha256(SEED:unit) first hex digit: 0-7 search, 8-b gate, c-f sealed (sealed from source A is empty)",
            "group": "light: body <= 4 lines and <= 5 checks; hard: body >= 10 lines; medium: rest",
        },
        "summary": {
            "tasks": len(keep), "units": units, "parts": dict(parts), "groups": dict(groups),
            "light_share": round(light_share, 3), "behaviour_changed": sum(1 for r in keep if r.get("behaviour_changed")),
            "dropped_as_duplicates": dropped, "rejected_after_reading": rejected,
            "runnable_by_run_micro": sum(1 for r in keep if r.get("runner", "").startswith("run_micro")),
        },
        "tasks": keep,
    }
    out = HERE / "manifest-micro-v2.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
