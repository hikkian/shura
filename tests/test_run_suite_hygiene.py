"""The suite runner must keep the desktop quiet: a test run once sent a real "game over" notification to the user (LL1, 2026-11-08)."""
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_suite  # noqa: E402


class QuietDesktop(unittest.TestCase):
    def test_the_runner_puts_a_silent_notify_send_first_on_path(self):
        env = {"PATH": os.environ.get("PATH", "")}
        folder = run_suite.quiet_path(env)
        self.addCleanup(shutil.rmtree, folder, True)
        self.assertTrue(env["PATH"].startswith(folder + os.pathsep))
        found = shutil.which("notify-send", path=env["PATH"])
        self.assertEqual(found, os.path.join(folder, "notify-send"))
        r = subprocess.run([found, "Shura", "a test notification"], capture_output=True, timeout=10)
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, b"", b""))

    def test_a_child_process_inherits_the_quiet_path(self):
        env = {**os.environ}
        folder = run_suite.quiet_path(env)
        self.addCleanup(shutil.rmtree, folder, True)
        out = subprocess.run([sys.executable, "-c", "import shutil; print(shutil.which('notify-send'))"], env=env, capture_output=True, text=True, timeout=20).stdout.strip()
        self.assertEqual(out, os.path.join(folder, "notify-send"))


if __name__ == "__main__":
    unittest.main()
