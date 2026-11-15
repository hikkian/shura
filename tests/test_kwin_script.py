"""MEMV2: the KWin script (kwin/shura-desk-signal) run under node against a fake `workspace`: what it prints, that it prints only on a
change, that it never writes to a window and never prints a title. Skipped when node is not installed."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "kwin" / "shura-desk-signal" / "contents" / "code" / "main.js"
FAKE = ROOT / "tests" / "support" / "kwin_fake.js"
TAG = "shura-desk-signal "


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class KwinScript(unittest.TestCase):
    def run_script(self, steps):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(steps, f)
        try:
            out = subprocess.run(["node", str(FAKE), str(SCRIPT), f.name], capture_output=True, text=True, timeout=30)
        finally:
            Path(f.name).unlink()
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.splitlines()
        writes = int(lines.pop().split()[1])
        return [json.loads(line[len(TAG):]) for line in lines if line.startswith(TAG)], writes, out.stdout

    def test_initial_line_and_changes_only(self):
        lines, writes, _ = self.run_script([
            {"op": "load", "windows": [{"name": "a", "cls": "brave-browser"}, {"name": "g", "cls": "steam_app_620"}], "active": "a"},
            {"op": "activate", "name": "a"},                      # same state: no new line
            {"op": "activate", "name": "g"},
            {"op": "fullscreen", "name": "g", "value": True}, {"op": "emit", "name": "g", "signal": "fullScreenChanged"},
            {"op": "emit", "name": "g", "signal": "fullScreenChanged"},   # nothing changed again
        ])
        self.assertEqual(writes, 0)
        self.assertEqual([(l["active"], l["fullscreen"], l["any_fullscreen"], l["fullscreen_class"]) for l in lines],
                         [("brave-browser", False, False, ""), ("steam_app_620", False, False, ""), ("steam_app_620", True, True, "steam_app_620")])
        self.assertTrue(all(l["v"] == 1 for l in lines))

    def test_fullscreen_on_another_window_is_reported_and_minimized_does_not_count(self):
        lines, _, _ = self.run_script([
            {"op": "load", "windows": [{"name": "a", "cls": "code"}, {"name": "v", "cls": "vlc", "fullScreen": True}], "active": "a"},
            {"op": "add", "name": "m", "cls": "mpv", "fullScreen": True, "minimized": True},
        ])
        self.assertEqual((lines[0]["fullscreen"], lines[0]["any_fullscreen"], lines[0]["fullscreen_class"]), (False, True, "vlc"))
        self.assertEqual(len(lines), 1)                           # the minimized full-screen window changes nothing

    def test_window_removed(self):
        lines, _, _ = self.run_script([
            {"op": "load", "windows": [{"name": "a", "cls": "code"}, {"name": "v", "cls": "vlc", "fullScreen": True}], "active": "a"},
            {"op": "remove", "name": "v"},
        ])
        self.assertEqual([l["any_fullscreen"] for l in lines], [True, False])

    def test_no_titles_are_ever_printed(self):
        _, _, raw = self.run_script([{"op": "load", "windows": [{"name": "a", "cls": "code", "fullScreen": True}], "active": "a"}])
        self.assertNotIn("SECRET", raw)

    def test_an_error_is_reported_as_a_line_not_thrown(self):
        lines, _, _ = self.run_script([
            {"op": "load", "windows": [{"name": "a", "cls": "code"}], "active": "a"},
            {"op": "break"},
        ])
        self.assertIn("error", lines[-1])


if __name__ == "__main__":
    unittest.main()
