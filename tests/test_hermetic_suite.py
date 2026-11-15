"""Static checks on the test suite itself. Nothing here runs a test or touches the machine: every file is parsed with `ast`.

1. No test file hardcodes the owner's path (/home/alisher, /mnt/data/alisher). The suite must run from any checkout, by any
   user. Other /home/... strings are fixture data (for example what the scrubbing tests must remove) and are allowed.
2. No test starts a tool that can reach the live gateway, the desktop or the GPU (shura, systemctl, pkill, nvidia-smi,
   the KDE helpers, docker) through `subprocess`, unless the call sits in a function that patches something, or the test
   class sets up tests/support/live_guard.py in its setUp, or the module imports live_guard.

Known limits: `from subprocess import run` and calls made through a helper function are not seen; the runtime guard
(live_guard.hermetic, the gateway harness's stand-ins) is what covers those.
"""
import ast
import os
import time
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
OWNER_PREFIXES = ("/home/alisher", "/mnt/data/alisher")
LIVE_TOOLS = {"shura", "shuracode", "systemctl", "pkill", "killall", "nvidia-smi", "busctl", "kwriteconfig6",
              "kscreen-doctor", "qdbus6", "docker"}
SUBPROCESS_FUNCS = {"run", "Popen", "call", "check_call", "check_output"}
GUARD_NAMES = {"patch", "hermetic"}


def modules():
    for path in sorted(TESTS.rglob("*.py")):
        if path == THIS or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        yield path, text, ast.parse(text, filename=str(path))


def dotted(node):
    """`mock.patch.object` for an attribute chain of plain names, '' for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def live_tool_started(call):
    """The live tool a subprocess call starts (by name or by absolute path), or '' when it starts none."""
    if not call.args:
        return ""
    first = call.args[0]
    if isinstance(first, (ast.List, ast.Tuple)) and first.elts and isinstance(first.elts[0], ast.Constant):
        word = str(first.elts[0].value)
    elif isinstance(first, ast.Constant) and isinstance(first.value, str):
        words = first.value.split()
        word = words[0] if words else ""
    else:
        return ""
    name = os.path.basename(word)
    return name if name in LIVE_TOOLS else ""


def guarded(tree, text, call):
    """True when the call is replaced or guarded: the module uses live_guard, the enclosing function patches or
    guards something, or the enclosing class sets up the guard in its setUp."""
    if "live_guard" in text:
        return True
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= call.lineno <= node.end_lineno:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and GUARD_NAMES & set(dotted(inner.func).split(".")):
                    return True
        if isinstance(node, ast.ClassDef) and node.lineno <= call.lineno <= node.end_lineno:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "setUp" and "hermetic" in ast.dump(item):
                    return True
    return False


def scan():
    home_hits, tool_hits = [], []
    started = time.monotonic()
    for path, text, tree in modules():
        where = path.relative_to(TESTS.parent)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(OWNER_PREFIXES):
                home_hits.append(f"{where}:{node.lineno}  {node.value[:60]!r}")
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in SUBPROCESS_FUNCS
                    and dotted(node.func.value).split(".")[-1] == "subprocess"):
                tool = live_tool_started(node)
                if tool and not guarded(tree, text, node):
                    tool_hits.append(f"{where}:{node.lineno}  subprocess starts {tool!r} without a patch or live_guard")
    return home_hits, tool_hits, time.monotonic() - started


class SuiteIsHermetic(unittest.TestCase):
    def test_no_test_hardcodes_a_home_path(self):
        home_hits, _, _ = scan()
        self.assertEqual(home_hits, [], "a test hardcodes the owner's path; use __file__, Path.home() or a temporary folder:\n  "
                         + "\n  ".join(home_hits))

    def test_live_tools_are_only_started_through_a_patch_or_the_guard(self):
        _, tool_hits, _ = scan()
        self.assertEqual(tool_hits, [], "a test can reach the live machine:\n  " + "\n  ".join(tool_hits))

    def test_the_scan_stays_under_two_seconds(self):
        _, _, elapsed = scan()
        self.assertLess(elapsed, 2.0, f"the static scan took {elapsed:.2f} s")


if __name__ == "__main__":
    unittest.main()
