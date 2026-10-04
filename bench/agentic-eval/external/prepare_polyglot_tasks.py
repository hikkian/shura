"""Convert the official Aider Polyglot Exercism tree to independent ShuraCode tasks.

All converted data is written to a caller-selected scratch directory (use /dev/shm).
The source checkout and this project are never modified.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


TEST_COMMANDS = {
    "python": ["pytest", "-q"],
    "javascript": ["/aider/benchmark/npm-test.sh"],
    "go": ["go", "test", "./..."],
    "java": ["./gradlew", "test"],
    "cpp": ["/aider/benchmark/cpp-test.sh"],
    "rust": ["cargo", "test", "--", "--include-ignored"],
}
SUPPORTED = set(TEST_COMMANDS)
BASE_PROMPT = """Complete this Exercism exercise through the ShuraCode agent. Follow the exercise description and preserve the requested API. Edit only the solution files listed below; do not change tests, benchmark metadata, or package-manager lockfiles. Public test files are included in this working directory. Run them during the task, fix failures, and leave every test file unchanged."""


def _copy_tree(source: Path, dest: Path, *, excluded: set[str] = frozenset()) -> None:
    for src in source.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(source)
        if any(part in {".meta", ".docs", ".git", "node_modules", "__pycache__"} for part in rel.parts):
            continue
        if rel.as_posix() in excluded:
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


def render_exercise(exercise: Path, out: Path, language: str) -> str:
    meta = exercise / ".meta" / "config.json"
    if not meta.is_file():
        raise ValueError(f"missing .meta/config.json: {exercise}")
    config = json.loads(meta.read_text(encoding="utf-8"))
    files = config.get("files", {})
    solution = files.get("solution", [])
    tests = files.get("test", [])
    if not solution or not tests:
        raise ValueError(f"exercise requires solution and test paths: {exercise.name}")
    missing = [p for p in [*solution, *tests] if not (exercise / p).is_file()]
    if missing:
        raise ValueError(f"missing configured files in {exercise.name}: {missing}")

    family = f"aider-{language}-{exercise.name}"
    task_dir = out / family / "implement"
    if task_dir.exists():
        shutil.rmtree(task_dir)
    start, hidden = task_dir / "start", task_dir / "hidden"
    solution_dir = task_dir / "solution"
    excluded = set(tests)
    _copy_tree(exercise, start, excluded=excluded)
    # Keep the Exercism tests visible to the agent, while retaining an untouched
    # copy under hidden/ for the post-run scoring pass.
    for rel in tests:
        visible = start / rel
        visible.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exercise / rel, visible)
    solution_dir.mkdir(parents=True, exist_ok=True)
    for rel in tests:
        dest = hidden / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exercise / rel, dest)

    # Polyglot publishes Exercism starter files and hidden tests, but not a complete
    # reference answer. Where Exercism includes its official .meta/example.* files,
    # preserve them as the validator's reference solution. Map by extension because
    # C++ examples use generic names (example.cpp/example.h).
    examples = sorted((exercise / ".meta").glob("example.*"))
    by_suffix: dict[str, list[Path]] = {}
    for example in examples:
        by_suffix.setdefault(example.suffix, []).append(example)
    targets_by_suffix: dict[str, list[str]] = {}
    data_targets: list[str] = []
    source_suffixes = {".py", ".js", ".go", ".java", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".rs"}
    for target in solution:
        suffix = Path(target).suffix
        if suffix in source_suffixes:
            targets_by_suffix.setdefault(suffix, []).append(target)
        else:
            data_targets.append(target)
    reference_available = bool(examples)
    if examples:
        if not set(targets_by_suffix).issubset(by_suffix) or any(
            len(by_suffix[suffix]) != len(targets_by_suffix[suffix]) for suffix in by_suffix
            if suffix in targets_by_suffix
        ):
            reference_available = False
        else:
            for suffix, targets in targets_by_suffix.items():
                for example, target in zip(by_suffix[suffix], sorted(targets)):
                    destination = solution_dir / target
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(example, destination)
            for target in data_targets:
                source = start / target
                if not source.is_file():
                    reference_available = False
                    break
                destination = solution_dir / target
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

    docs = []
    for doc in ("introduction.md", "instructions.md", "instructions.append.md"):
        path = exercise / ".docs" / doc
        if path.is_file():
            docs.append(path.read_text(encoding="utf-8").strip())
    prompt = "\n\n".join(p for p in docs if p)
    test_list = ", ".join(f"`{name}`" for name in tests)
    solution_list = ", ".join(f"`{name}`" for name in solution)
    prompt += "\n\n" + BASE_PROMPT + f"\n\nSolution files: {solution_list}.\nPublic tests in this folder: {test_list}."

    language_command = list(TEST_COMMANDS[language])
    if language == "python":
        language_command += list(tests)
    elif language == "javascript":
        # The exercise's npm script owns the test selection; hidden test files are restored
        # at their original project-relative paths only after the agent exits.
        pass
    else:
        # Native runners discover the restored test files from the project root.
        pass
    # The agent sees public tests in start/. The scoring adapter restores the
    # untouched reference copies from hidden/ after the agent exits.
    command = [sys.executable, str(Path(__file__).with_name("run_aider_tests.py")),
               "--task-dir", str(task_dir)]

    task = {
        "id": family,
        "family": family,
        "kind": "implement",
        "lang": language,
        "prompt": prompt,
        "test_cmd": command,
        "language_test_cmd": language_command,
        "timeout_s": 900,
        "benchmark": "aider-polyglot",
        "benchmark_source_id": exercise.name,
        "solution_files": solution,
        "visible_test_files": tests,
        "reference_available": reference_available,
        "hidden_files": tests,
        "example_files": files.get("example", []),
        "test_mode": "restore-at-original-path-after-agent",
    }
    (task_dir / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return family


def prepare(source: Path, out: Path, languages: list[str]) -> dict[str, list[str]]:
    if not (source / "README.md").is_file():
        raise ValueError(f"not an Aider Polyglot checkout: {source}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    result: dict[str, list[str]] = {}
    for language in languages:
        if language not in SUPPORTED:
            raise ValueError(f"unsupported language: {language}")
        practice = source / language / "exercises" / "practice"
        if not practice.is_dir():
            raise ValueError(f"missing practice directory for {language}")
        ids = []
        for exercise in sorted(p for p in practice.iterdir() if p.is_dir()):
            if (exercise / ".meta" / "config.json").exists():
                ids.append(render_exercise(exercise, out, language))
        result[language] = ids
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Use /dev/shm; this directory is replaced")
    ap.add_argument("--languages", default="python,javascript")
    args = ap.parse_args()
    languages = [x.strip().lower() for x in args.languages.split(",") if x.strip()]
    result = prepare(args.source, args.out, languages)
    print(json.dumps({k: len(v) for k, v in result.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
