#!/usr/bin/env python3
"""EXAM200b: manifest of the micro tasks that passed every check.

Input: the validation results written by build_micro.py check (OUT), the statements and the
blind-reader notes in tasks/<sha10>/ (statement.ru.md, reader.md, hidden-added.txt).
Output: manifest-micro-v1.json next to this file.

Part by hash: sha256("SEED:id"), first hex digit 0-7 -> search, 8-b -> gate, c-f -> sealed.
Dedup: hidden_sha256 against exam/manifest-v3r1.json, and repo/commit against the same manifest.
"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path("/dev/shm/EXAM200b/out")
TASKS = HERE / "tasks"
V3 = HERE.parent / "manifest-v3r1.json"
SEED = "20261114"

# (repo, full sha) of tasks whose statement was checked by the blind reader and kept
CHOSEN = [
    ("barid", "a0a06f757136ee7f6ac79a3f5f843411643cc482", "server: do not look up the host name at start"),
    ("barid", "bb6a1b5ea8c7557358b86520b16b374abd44c10e", "rb move (task priority)"),
    ("barid", "bfea8ba0958ddf8fbf2399e15c8b1fa23526198d", "handoff between agents in the prompt"),
    ("barid", "cc255315e685c25d6f8d00594775c30d9c0713e0", "board language settable"),
    ("barid", "9bec0b30fa93f66f7073a51736e7a6fbb2159e78", "default board language from the system locale"),
    ("barid", "7b534815573278e5b7f0584c423a6063caa51731", "op_lane_add"),
    ("barid", "fe7a173a661bebf2b7e07d01f00e050e39d21118", "connect text for agent windows"),
    ("shura", "a3ee7697c13a9ea41a1f0039e598d157dc5ceed4", "bandwidth: physical CPU ids from sysfs"),
]


def part_of(task_id: str) -> str:
    h = hashlib.sha256(f"{SEED}:{task_id}".encode()).hexdigest()[0]
    return "search" if h in "01234567" else ("gate" if h in "89ab" else "sealed")


def verdict(path: Path) -> str:
    if not path.exists():
        return "no reader.md"
    m = re.search(r"[Вв]ердикт\W*([^\n]*)", path.read_text(encoding="utf-8"))
    return m.group(1).strip(" .*:") if m else "reader.md without verdict line"


def main() -> int:
    v3 = json.loads(V3.read_text(encoding="utf-8"))["tasks"]
    v3_hashes = {t["hidden_sha256"] for t in v3}
    rows = []
    for repo, sha, title in CHOSEN:
        sha10 = sha[:10]
        res = json.loads((OUT / f"{repo}-{sha10}.json").read_text(encoding="utf-8"))
        assert res["accepted"], f"{sha10} not accepted"
        st = TASKS / sha10 / "statement.ru.md"
        rd = TASKS / sha10 / "reader.md"
        hid = TASKS / sha10 / "hidden-added.txt"
        task_id = f"{repo}-{sha10}"
        assert res["hidden_sha256"] not in v3_hashes, "duplicate of v3r1"
        rows.append({
            "id": task_id,
            "title": title,
            "repo": repo,
            "commit": sha,
            "parent": res["parent"],
            "unit": sha,
            "module": res["src"],
            "tests": res["tests"],
            "test_cmd": res["test_cmd"],
            "timeout_s": res["timeout_s"],
            "statement": str(st.relative_to(HERE)),
            "statement_sha256": hashlib.sha256(st.read_bytes()).hexdigest(),
            "hidden_added": str(hid.relative_to(HERE)),
            "hidden_sha256": res["hidden_sha256"],
            "reader": {"file": str(rd.relative_to(HERE)), "verdict": verdict(rd)},
            "checks": {
                "start_rc": res["start"]["rc"],
                "reference_rc": [r["rc"] for r in res["reference"]] if isinstance(res["reference"], list) else None,
                "mutants_kept": len(res["mutants"]),
                "mutants_caught": sum(1 for m in res["mutants"] if m["caught"]),
                "accepted": True,
            },
            "part": part_of(task_id),
            "dedup_v3r1": "no hidden_sha256 match; other repo",
        })
    manifest = {
        "version": "micro-v1",
        "date": "2026-11-14",
        "seed": SEED,
        "sources": {"barid": "origin/main bf3da35 (MIT, public)", "shura": "origin/main b09ea52 (MIT, public)"},
        "rules": {
            "unit": "one commit = one task; modules are listed separately",
            "validation": "start fails (rc != 0); reference passes 3 of 3; at least 2 of 3 mutants caught",
            "reader": "blind reader sees statement and added test lines only; every check labelled",
            "part": "sha256(SEED:id) first hex digit: 0-7 search, 8-b gate, c-f sealed",
        },
        "summary": {
            "tasks": len(rows),
            "independent_units": len({r["unit"] for r in rows}),
            "modules": sorted({m for r in rows for m in r["module"]}),
        },
        "tasks": rows,
    }
    (HERE / "manifest-micro-v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
