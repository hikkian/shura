#!/usr/bin/env python3
"""EXAM200c: Russian statements for the «search» and «gate-1» parts of exam version v3r1.

A statement is a list of paragraphs; every unique English paragraph (exam/ru/paragraphs.json) has one translation
(exam/ru/ru-batch*.json, corrected by exam/ru/ru-fixes.json). Hidden tests, reference solutions and start files are the
same as in exam/tasks_v3r1, so only the statement language differs.

    python3 exam/build_v3r1_ru.py --check          # literals kept? every paragraph translated?
    python3 exam/build_v3r1_ru.py [--validate]     # writes exam/tasks_v3r1_ru and exam/manifest-v3r1-ru.json
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "tasks_v3r1"
OUT = HERE / "tasks_v3r1_ru"
RU = HERE / "ru"
PARTS = ("search", "gate-1")


def load_translations():
    paragraphs = json.loads((RU / "paragraphs.json").read_text(encoding="utf-8"))
    ru = {}
    for path in sorted(RU.glob("ru-batch*.json")):
        ru.update({int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()})
    fixes = RU / "ru-fixes.json"
    if fixes.exists():
        ru.update({int(k): v for k, v in json.loads(fixes.read_text(encoding="utf-8")).items()})
    return paragraphs, ru


def literals(text):
    """Fragments that must survive translation byte for byte: code spans and quoted strings."""
    found = set(re.findall(r"`[^`\n]+`", text))
    # a quoted string does not start or end with a space: that would be the text between two quoted strings
    found |= set(re.findall(r"(?<![A-Za-z0-9])'(?! )[^'\n]{1,80}(?<! )'(?![A-Za-z0-9])", text))
    found |= set(re.findall(r'"[^"\n]{1,80}"', text))
    return found


def check(paragraphs, ru):
    problems = []
    for p in paragraphs:
        text = ru.get(p["id"])
        if not text or not text.strip():
            problems.append((p["id"], "missing translation"))
            continue
        lost = [lit for lit in literals(p["en"]) if lit not in text]
        if lost:
            problems.append((p["id"], "literals changed or lost: " + "; ".join(sorted(lost)[:6])))
        if not re.search(r"[а-яА-Я]", text):
            problems.append((p["id"], "no Russian text"))
        ratio = len(text) / max(1, len(p["en"]))
        if ratio < 0.6 or ratio > 1.9:
            problems.append((p["id"], f"length ratio {ratio:.2f}"))
    return problems


def build(paragraphs, ru):
    by_en = {p["en"]: ru[p["id"]] for p in paragraphs}
    manifest = json.loads((HERE / "manifest-v3r1.json").read_text(encoding="utf-8"))
    if OUT.exists():
        shutil.rmtree(OUT)
    rows = []
    for t in manifest["tasks"]:
        if t["part"] not in PARTS:
            continue
        src = SRC / t["family"] / t["kind"]
        dst = OUT / t["family"] / t["kind"]
        shutil.copytree(src, dst)
        tj = json.loads((dst / "task.json").read_text(encoding="utf-8"))
        tj["prompt_en"] = tj["prompt"]
        tj["prompt"] = "\n\n".join(by_en[p] for p in tj["prompt_en"].split("\n\n"))
        tj["prompt_lang"] = "ru"
        tj["set"] = "v3r1-ru"
        (dst / "task.json").write_text(json.dumps(tj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        row = dict(t)
        row["slices"] = dict(t["slices"], statement_lang="ru")
        row["prompt_sha256"] = hashlib.sha256(tj["prompt"].encode()).hexdigest()
        rows.append(row)
    out_manifest = {"version": "v3r1-ru", "date": "2026-11-13", "tasks_root": "exam/tasks_v3r1_ru", "source": "exam/tasks_v3r1 + exam/ru",
                    "unit_of_independence": manifest["unit_of_independence"], "parts": list(PARTS),
                    "note": "same hidden tests and solutions as v3r1; only the statement language differs", "tasks": rows}
    (HERE / "manifest-v3r1-ru.json").write_text(json.dumps(out_manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    paragraphs, ru = load_translations()
    problems = check(paragraphs, ru)
    print(f"paragraphs {len(paragraphs)}, translated {sum(1 for p in paragraphs if ru.get(p['id']))}, problems {len(problems)}")
    for pid, what in problems:
        print(f"  #{pid}: {what}")
    if args.check:
        sys.exit(1 if problems else 0)
    if any(what == "missing translation" for _, what in problems):
        sys.exit("cannot build: translations are missing")
    rows = build(paragraphs, ru)
    print(f"built {len(rows)} tasks into {OUT}")
    if args.validate:
        sys.path.insert(0, str(HERE))
        import build_v3r1
        n, bad = build_v3r1.validate(OUT)
        print(f"validated {n} tasks; problems: {len(bad)}")
        for b in bad:
            print("  ", b)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
