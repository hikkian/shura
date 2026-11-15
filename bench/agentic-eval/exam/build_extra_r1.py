#!/usr/bin/env python3
"""EXAM200: corrected copies of the control set and the long-horizon set (prompts only).

tasks_control_public -> exam/tasks_control_public_r1, tasks_lh -> exam/tasks_lh_r1. The originals are not touched.
Each sentence of exam/prompt-additions-extra.json is appended to the prompt of the tasks it names; hidden tests,
reference solutions and start files are copied unchanged, so earlier validation of those sets still holds.

    python3 exam/build_extra_r1.py        # run from bench/agentic-eval
"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SETS = {"tasks_control_public": "tasks_control_public_r1", "tasks_lh": "tasks_lh_r1"}


def main():
    additions = json.loads((HERE / "prompt-additions-extra.json").read_text(encoding="utf-8"))
    for src, dst in SETS.items():
        out = HERE / dst
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(ROOT / src, out, symlinks=True)
    touched = {}
    for a in additions:
        if a.get("drop"):
            continue
        for original in a["prompt_files"]:
            rel = Path(original).resolve().relative_to(ROOT)
            target = HERE / SETS[rel.parts[0]] / Path(*rel.parts[1:])
            tj = json.loads(target.read_text(encoding="utf-8"))
            sentence = a["add_to_prompt"].strip()
            if sentence not in tj["prompt"]:
                tj["prompt"] = tj["prompt"].rstrip() + " " + sentence
            tj.setdefault("fixes", [])
            if "inputs" not in tj["fixes"]:
                tj["fixes"].append("inputs")
            target.write_text(json.dumps(tj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            touched[str(target.relative_to(HERE))] = hashlib.sha256(tj["prompt"].encode()).hexdigest()
    log = {"touched": touched, "sets": {}}
    for dst in SETS.values():
        log["sets"][dst] = len(list((HERE / dst).glob("*/*/task.json")))
    (HERE / "build-extra-r1.log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"copied sets {log['sets']}; prompts changed: {len(touched)}")


if __name__ == "__main__":
    main()
