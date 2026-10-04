"""Restore Aider tests after the agent turn; caller runs this inside the offline bubblewrap."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import re


def run(task_dir: Path, workdir: Path) -> int:
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    for rel in task.get("hidden_files", []):
        src = workdir / "hidden_tests" / rel
        dest = workdir / rel
        if not src.is_file():
            print(f"missing hidden fixture: {rel}", file=sys.stderr)
            return 2
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    # validate_tasks.py stages fixtures under this directory; after restoring their
    # original Exercism paths, leaving it in the project causes recursive runners
    # (notably `go test ./...`) to compile a second, invalid test package.
    shutil.rmtree(workdir / "hidden_tests", ignore_errors=True)
    tmp_root = Path(os.environ.get("POLYGLOT_TMPDIR", "/dev/shm/shura-quality-20261004/tmp"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "GOPROXY": "off",
        "GOSUMDB": "off",
        "GOTOOLCHAIN": "local",
        "TMPDIR": str(tmp_root),
        "JEST_CACHE_DIRECTORY": str(tmp_root / "jest-cache"),
        "npm_config_cache": str(tmp_root / "npm-cache"),
        "GOCACHE": str(tmp_root / "go-cache"),
        "GOMODCACHE": str(tmp_root / "go-mod-cache"),
        "CARGO_HOME": str(tmp_root / "cargo-home"),
        "GRADLE_USER_HOME": str(tmp_root / "gradle-home"),
    }
    command = list(task.get("language_test_cmd", task["test_cmd"]))
    if task["lang"] == "javascript":
        # The upstream helper hardcodes /npm-install. Use the caller's read-only
        # preinstalled tree and link it from this disposable task directory.
        modules = env.get("POLYGLOT_NPM_MODULES", "/npm-install/node_modules")
        modules_path = Path(modules)
        if not modules_path.is_dir():
            raise SystemExit(f"preinstalled read-only npm modules not found: {modules_path}")
        node_modules = workdir / "node_modules"
        if node_modules.exists() and not node_modules.is_symlink():
            shutil.rmtree(node_modules)
        if not node_modules.exists():
            node_modules.symlink_to(modules_path, target_is_directory=True)
        env["NODE_PATH"] = modules
        env["PATH"] = str(modules_path / ".bin") + os.pathsep + env.get("PATH", "/usr/bin:/bin")
        for spec in workdir.glob("*.spec.js"):
            spec.write_text(spec.read_text(encoding="utf-8").replace("xtest(", "test("), encoding="utf-8")
        command = ["npm", "test", "--", "--runInBand"]
    elif task["lang"] == "cpp":
        # Compile the exact Exercism test target directly. The upstream CMake file
        # derives its project target from cwd basename and normally assumes a
        # pre-installed Catch2 package; the checked-in test/catch.hpp is sufficient.
        solution_cpp = [p for p in task.get("solution_files", []) if p.endswith(".cpp")]
        test_cpp = [p for p in task.get("hidden_files", []) if p.endswith(".cpp")]
        if not solution_cpp or not test_cpp or not (workdir / "test/tests-main.cpp").is_file():
            print("C++ task lacks a local source, hidden test, or Exercism test main", file=sys.stderr)
            return 2
        digest = hashlib.sha256()
        for rel in sorted([*solution_cpp, *test_cpp, "test/tests-main.cpp"]):
            digest.update(rel.encode())
            digest.update((workdir / rel).read_bytes())
        binary = tmp_root / (re.sub(r"[^a-zA-Z0-9_-]", "_", task["id"]) + "-" + digest.hexdigest()[:16] + ".test")
        command = ["g++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                   "-DEXERCISM_RUN_ALL_TESTS", "-I.", *solution_cpp, *test_cpp,
                   "test/tests-main.cpp", "-o", str(binary)]
        if not binary.is_file():
            build = subprocess.run(command, cwd=workdir, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            if build.returncode:
                sys.stdout.write(build.stdout)
                return build.returncode
        try:
            result = subprocess.run([str(binary)], cwd=workdir, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        except subprocess.TimeoutExpired:
            print("C++ test binary timed out", file=sys.stderr)
            return 124
        sys.stdout.write(result.stdout)
        return result.returncode
    try:
        result = subprocess.run(command, cwd=workdir, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"test runner error: {exc}", file=sys.stderr)
        return 124
    sys.stdout.write(result.stdout)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SHURA_POLYGLOT_NETNS") != "1":
        raise SystemExit("tests must run inside the network-isolated benchmark bubblewrap")
    raise SystemExit(run(args.task_dir, Path.cwd()))


if __name__ == "__main__":
    main()
