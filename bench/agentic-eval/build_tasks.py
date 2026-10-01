"""Renders the task families in families/*.py into tasks/<family>/<variant>/ (the layout harness.py expects).

A family is one idea (say "a duration parser") and gives two tasks that share their hidden tests:
    implement - the agent gets a stub and a precise specification and must write the module;
    bugfix    - the agent gets the finished module with one realistic bug injected, a failing visible test that shows a symptom,
                and must find and fix the cause. The hidden tests also cover behaviour the visible test does not show, so a
                special case for the visible example does not pass.
Both variants of a family are one CLUSTER in the statistics (they are not independent of each other).

FAMILY = {
  "name": "f01_x", "lang": "python",
  "implement": {"prompt": str, "start": {path: text}, "solution": {path: text}},
  "hidden": {path: text},                              # hidden_tests/ contents; __init__.py is added
  "bugfix": {"symptom": str, "bug": [(path, old, new)], "repro": {path: text}},
}
"""
import importlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "families"))

TEST_CMD = {"python": ["python3", "-m", "unittest", "discover", "-s", "hidden_tests", "-t", ".", "-q"]}


def write(root, files):
    for rel, text in files.items():
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def render(fam, out):
    name, lang = fam["name"], fam.get("lang", "python")
    base = Path(out) / name
    if base.exists():
        shutil.rmtree(base)
    hidden = {**fam["hidden"], "__init__.py": ""}
    imp = fam["implement"]
    d = base / "implement"
    write(d / "start", imp["start"])
    write(d / "solution", imp["solution"])
    write(d / "hidden", hidden)
    (d / "task.json").write_text(json.dumps({
        "id": f"{name}-implement", "family": name, "kind": "implement", "lang": lang, "prompt": imp["prompt"],
        "test_cmd": TEST_CMD[lang], "timeout_s": 900}, indent=2))
    bug = fam["bugfix"]
    d = base / "bugfix"
    buggy = dict(imp["solution"])
    for rel, old, new in bug["bug"]:
        if buggy[rel].count(old) != 1:
            raise ValueError(f"{name}: bug pattern must match exactly once in {rel}: {old!r}")
        buggy[rel] = buggy[rel].replace(old, new)
    write(d / "start", {**buggy, **bug["repro"]})
    write(d / "solution", imp["solution"])
    write(d / "hidden", hidden)
    prompt = (bug["symptom"].strip() + "\n\nThe visible test shows it: run `python3 -m unittest -q` in the project folder. "
              "Find the cause in the implementation and fix it there; do not edit the visible test.")
    (d / "task.json").write_text(json.dumps({
        "id": f"{name}-bugfix", "family": name, "kind": "bugfix", "lang": lang, "prompt": prompt,
        "test_cmd": TEST_CMD[lang], "timeout_s": 900}, indent=2))


def build(out):
    n = 0
    for p in sorted((HERE / "families").glob("f*.py")):
        render(importlib.import_module(p.stem).FAMILY, out)
        n += 1
    return n


def main():
    out = HERE / "tasks"
    print(f"rendered {build(out)} families into {out}")


if __name__ == "__main__":
    main()
