"""The journal never runs a command it read from the file: only a closed list of actions, on names Shura installs."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from installer.universal import journal  # noqa: E402


class JournalActions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.calls = []

    def runner(self, cmd, **kw):
        self.calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def planted(self, entry):
        (self.home / journal.NAME).write_text(json.dumps(entry) + "\n")
        return journal.Journal.load(self.home)

    def test_planted_undo_command_is_not_run(self):
        marker = Path(self.tmp.name) / "pwned"
        jr = self.planted({"op": "service", "kind": "service", "path": "ai-gateway.service",
                           "undo": ["sh", "-c", f"touch {marker}"]})
        res = jr.rollback(runner=self.runner)
        self.assertEqual(self.calls, [])
        self.assertFalse(marker.exists())
        self.assertFalse(res.ok)
        self.assertTrue(any("ready command is not run" in p for p in res.problems), res.problems)
        self.assertEqual(len(journal.Journal.load(self.home).entries), 1)      # the entry stays

    def test_replaced_outside_the_shura_folder_is_not_touched(self):
        user = Path(self.tmp.name) / "userhome"
        (user / "Documents").mkdir(parents=True)
        victim = user / "Documents" / "x.txt"
        victim.write_text("mine")
        for entry in ({"op": "replaced", "kind": "file", "path": str(victim), "old": "ZXZpbA==", "mode": None},
                      {"op": "created", "kind": "file", "path": str(victim)},
                      {"op": "snapshot", "kind": "dir", "path": str(user / "Documents"), "children": []}):
            with self.subTest(op=entry["op"]):
                old_home = os.environ.get("HOME")
                os.environ["HOME"] = str(user)
                try:
                    res = self.planted(entry).rollback(runner=self.runner)
                finally:
                    os.environ["HOME"] = old_home
                self.assertEqual(victim.read_text(), "mine")
                self.assertFalse(res.ok)
                self.assertTrue(any("path outside the Shura folder: not undone" in p for p in res.problems), res.problems)

    def test_unknown_action_is_an_error_without_a_call(self):
        res = self.planted({"op": "service", "action": "rm-rf", "name": "ai-gateway.service"}).rollback(runner=self.runner)
        self.assertEqual(self.calls, [])
        self.assertFalse(res.ok)

    def test_bad_names_are_errors_without_a_call(self):
        for name in ("../x", "shura-a;rm", "shura-evil.service", "dbus.service", None, ["ai-gateway.service"]):
            with self.subTest(name=name):
                res = self.planted({"op": "service", "action": "user-service-disable", "name": name}).rollback(runner=self.runner)
                self.assertEqual(self.calls, [])
                self.assertFalse(res.ok)

    def test_correct_entry_runs_exactly_the_built_command(self):
        res = self.planted({"op": "service", "action": "user-service-disable", "name": "ai-gateway.service"}).rollback(runner=self.runner)
        self.assertTrue(res.ok, res.problems)
        self.assertEqual(self.calls, [["systemctl", "--user", "disable", "--now", "ai-gateway.service"]])

    def test_record_service_writes_action_and_name_not_a_command(self):
        jr = journal.Journal(self.home)
        jr.record_service("ai-gateway.service", "user-service-disable")
        e = json.loads((self.home / journal.NAME).read_text())
        self.assertEqual((e["action"], e["name"]), ("user-service-disable", "ai-gateway.service"))
        self.assertNotIn("undo", e)
        with self.assertRaises(journal.JournalError):
            jr.record_service("ai-gateway.service", "nope")
        with self.assertRaises(journal.JournalError):
            jr.record_service("../x", "user-service-disable")


if __name__ == "__main__":
    unittest.main()
