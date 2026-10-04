"""Run the unchanged v2 validator against imported Polyglot tasks.

Run inside the network-isolated benchmark namespace. The ID manifest is intended
for /dev/shm and contains only tasks that have a checked-in reference and passed
the starter-fails/reference-passes-three-times gates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import harness
from validate_tasks import check


def validate(tasks_root: Path) -> dict:
    valid: list[str] = []
    no_reference: list[str] = []
    unsupported_runtime: dict[str, str] = {}
    failures: dict[str, list[str]] = {}
    tasks = harness.all_tasks(tasks_root)
    for index, task in enumerate(tasks, 1):
        if not task.get("reference_available"):
            no_reference.append(task["id"])
            print(f"SKIP {task['id']}: no checked-in reference", flush=True)
            continue
        runtime_problem = _runtime_problem(task["lang"], task)
        if runtime_problem:
            unsupported_runtime[task["id"]] = runtime_problem
            print(f"SKIP {task['id']}: {runtime_problem}", flush=True)
            continue
        problems = check(task)
        if problems:
            failures[task["id"]] = problems
            print(f"FAIL {task['id']}: {'; '.join(problems)}", flush=True)
        else:
            valid.append(task["id"])
            print(f"PASS {task['id']} ({index}/{len(tasks)})", flush=True)
    return {
        "total_imported": len(tasks),
        "valid": len(valid),
        "no_reference": len(no_reference),
        "unsupported_runtime": len(unsupported_runtime),
        "failed_validation": len(failures),
        "valid_task_ids": valid,
        "no_reference_task_ids": no_reference,
        "unsupported_runtime_tasks": unsupported_runtime,
        "failures": failures,
    }


def _runtime_problem(language: str, task: dict | None = None) -> str | None:
    required = {"go": "go", "cpp": "g++", "rust": "cargo"}
    if language in required and shutil.which(required[language]) is None:
        return f"missing {required[language]} executable; no packages installed"
    if language == "javascript":
        modules = Path(os.environ.get("POLYGLOT_NPM_MODULES", "/npm-install/node_modules"))
        if not (modules / ".bin/jest").exists():
            return "missing preinstalled Jest; no npm packages installed"
    if language == "java":
        if shutil.which("gradle") is None:
            return "missing system Gradle; wrapper downloads are disabled"
    if language == "cpp" and task:
        for root_key in ("start", "solution"):
            root = task["dir"] / root_key
            for source in root.rglob("*"):
                if source.is_file() and source.suffix in {".h", ".hpp", ".cpp", ".cc"}:
                    try:
                        if "#include <boost/" in source.read_text(encoding="utf-8", errors="ignore"):
                            return "missing Boost headers; packages are not installed during corpus validation"
                    except OSError:
                        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="write validated IDs under /dev/shm")
    args = parser.parse_args()
    if not str(args.manifest.resolve()).startswith("/dev/shm/"):
        raise SystemExit("refusing to write validation manifest outside /dev/shm")
    if os.environ.get("SHURA_POLYGLOT_NETNS") != "1":
        raise SystemExit("run validation inside the network-isolated bubblewrap")
    report = validate(args.tasks)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(report["valid_task_ids"]) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("task_ids") and k not in {"failures", "unsupported_runtime_tasks"}},
                     ensure_ascii=False, indent=2))
    if report["unsupported_runtime_tasks"]:
        print(json.dumps(report["unsupported_runtime_tasks"], ensure_ascii=False, indent=2))
    if report["failures"]:
        print(json.dumps(report["failures"], ensure_ascii=False, indent=2))
    return 1 if report["failed_validation"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
