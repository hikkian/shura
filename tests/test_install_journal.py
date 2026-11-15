"""INSTALL1 step 8: the installation journal. An install that fails anywhere puts the machine back; `shura uninstall` removes
what the journal lists (models only on request). Everything runs in a temporary folder, the engine step is faked."""
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
sys.path.insert(0, str(ROOT))
from universal import cli, engine, install, journal, launch, planner, report  # noqa: E402
from tests._installer_env import patch_free_space  # noqa: E402
from tests.test_install import HW, Out, args, tiny_catalog  # noqa: E402

MODEL = "Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf"


def snapshot(root):
    """{relative path: sha256 of the content, or None for a folder} of everything under root (symlinks as their target)."""
    result = {}
    for p in sorted(Path(root).rglob("*")):
        rel = str(p.relative_to(root))
        if p.is_symlink():
            result[rel] = "link:" + os.readlink(p)
        elif p.is_dir():
            result[rel] = None
        else:
            result[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


class Machine(unittest.TestCase):
    """A throw-away 'machine': root/ with some user files; the Shura home is root/data/shura (made by the install)."""

    def setUp(self):
        patch_free_space(self)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "machine"
        (self.root / "docs").mkdir(parents=True)
        (self.root / "docs" / "notes.txt").write_text("my notes\n")
        (self.root / "settings.toml").write_text("a = 1\n")
        self.home = self.root / "data" / "shura"
        self.faults = []                                     # fault points the fake steps know about

    # --- a fake installation: the engine/model step creates what the real one does, nothing is downloaded or started
    def fake_attempt(self, args_, hw, catalog, out, release, cfg, home, cache, label, only_backend=None):
        plan = planner.plan(hw, catalog, config=install.relaxed_config(cfg, 0, hw), model=args_.model_id, quant=args_.quant,
                            menu=False)
        engines = Path(home) / "engines" / "b0-vulkan"
        engines.mkdir(parents=True, exist_ok=True)
        (engines / "llama-server").write_text("#!/bin/sh\n")
        (engines / "lib.so").write_bytes(b"\0" * 100)
        (Path(home) / "models").mkdir(parents=True, exist_ok=True)
        (Path(home) / "models" / MODEL).write_bytes(b"gguf" * 1000)
        (Path(home) / "models" / engine.SMOKE_MODEL["name"]).write_bytes(b"smoke")
        (Path(home) / "logs").mkdir(parents=True, exist_ok=True)
        (Path(home) / "logs" / "verify.log").write_text("ok\n")
        if "after-files" in self.faults:
            raise RuntimeError("boom after the files are on disk")
        verdict = {"ok": True, "tok_s": 30.0, "load_s": 1.0, "prompt_tok_s": 100.0}
        return {"ok": True, "plan": plan, "argv": ["llama-server"], "verdict": verdict, "calibration": {}, "backend": "vulkan",
                "server": str(engines / "llama-server"), "path": str(Path(home) / "models" / MODEL), "hw": hw, "label": label,
                "first_predicted": 30.0, "first_measured": 30.0, "errors": []}

    def install(self, **kw):
        out = Out()
        a = args(self.home, **kw)
        with mock.patch.object(install, "_attempt", self.fake_attempt):
            code = install.run(a, HW["rx9070xt_16g_32g"], tiny_catalog(0), out)
        return code, out

    def cli(self, *argv):
        o, e = io.StringIO(), io.StringIO()
        with redirect_stdout(o), redirect_stderr(e), mock.patch("sys.stdin", io.StringIO("")):
            code = cli.main(list(argv))
        return code, o.getvalue(), e.getvalue()


class FailAnywhere(Machine):
    def count_flushes(self):
        counter = [0]
        real = journal.Journal._flush

        def counting(j):
            counter[0] += 1
            return real(j)
        with mock.patch.object(journal.Journal, "_flush", counting):
            self.assertEqual(self.install()[0], 0)
        shutil.rmtree(self.root / "data")
        return counter[0]

    def test_a_failure_at_every_journal_write_restores_the_tree(self):
        before = snapshot(self.root)
        total = self.count_flushes()
        self.assertGreater(total, 8)                                      # the sweep must really cover many steps
        self.assertEqual(snapshot(self.root), before)
        for n in range(1, total + 1):
            with self.subTest(failing_write=n):
                counter = [0]
                real = journal.Journal._flush

                def failing(j, n=n, counter=counter, real=real):
                    counter[0] += 1
                    if counter[0] == n:
                        raise OSError(28, "No space left on device")
                    return real(j)
                with mock.patch.object(journal.Journal, "_flush", failing):
                    code, out = self.install()
                self.assertEqual(code, 1)
                self.assertIn("Install failed", out.text)
                self.assertIn("undone", out.text)
                self.assertEqual(snapshot(self.root), before)

    def test_a_failure_after_the_files_are_written_restores_the_tree(self):
        before = snapshot(self.root)
        self.faults = ["after-files"]
        code, out = self.install()
        self.assertEqual(code, 1)
        self.assertIn("boom after the files", out.text)
        self.assertEqual(snapshot(self.root), before)

    def test_a_failure_at_every_later_step_restores_the_tree(self):
        before = snapshot(self.root)
        for name, target in (("save_state", launch), ("make_report", report)):
            with self.subTest(step=name), mock.patch.object(target, name, side_effect=OSError(5, "I/O error")):
                code, out = self.install()
            self.assertEqual(code, 1)
            self.assertIn("I/O error", out.text)
            self.assertEqual(snapshot(self.root), before)

    def test_ctrl_c_restores_the_tree_and_propagates(self):
        before = snapshot(self.root)
        with mock.patch.object(launch, "save_state", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.install()
        self.assertEqual(snapshot(self.root), before)

    def test_existing_home_with_old_install_comes_back_unchanged(self):
        self.home.mkdir(parents=True)
        (self.home / "state.json").write_text('{"old": true}\n')
        (self.home / "server.pid").write_text("12345")
        (self.home / "engines" / "old-build").mkdir(parents=True)
        (self.home / "engines" / "old-build" / "llama-server").write_text("old")
        (self.home / "models").mkdir()
        (self.home / "models" / "mine.gguf").write_bytes(b"x" * 10)
        (self.home / "foreign.txt").write_text("not ours")
        before = snapshot(self.root)
        self.faults = ["after-files"]
        self.assertEqual(self.install()[0], 1)
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual((self.home / "state.json").read_text(), '{"old": true}\n')


class Replace(Machine):
    def test_replaced_settings_file_gets_its_old_content_back(self):
        path = self.root / "settings.toml"
        path.chmod(0o640)
        jr = journal.Journal(self.root)
        jr.track_file(path, kind="setting")
        path.write_text("a = 2\n")
        path.chmod(0o600)
        res = jr.rollback()
        self.assertTrue(res.ok, res.problems)
        self.assertEqual(path.read_text(), "a = 1\n")
        self.assertEqual(path.stat().st_mode & 0o777, 0o640)

    def test_replacement_by_install_is_undone(self):
        self.home.mkdir(parents=True)
        report_path = self.home / "shura-hardware-report.md"
        report_path.write_text("the previous report\n")
        self.faults = ["after-files"]
        self.assertEqual(self.install()[0], 1)
        self.assertEqual(report_path.read_text(), "the previous report\n")

    def test_service_is_undone_by_its_command(self):
        calls = []

        def runner(cmd, **kw):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        self.home.mkdir(parents=True)
        jr = journal.Journal(self.home)
        jr.record_service("ai-gateway.service", "user-service-disable")
        res = jr.rollback(runner=runner)
        self.assertTrue(res.ok)
        self.assertEqual(calls, [["systemctl", "--user", "disable", "--now", "ai-gateway.service"]])


class Atomic(Machine):
    def test_an_interrupted_journal_write_leaves_the_previous_journal_intact(self):
        self.home.mkdir(parents=True)
        jr = journal.Journal(self.home)
        jr.record_created(self.home / "a.txt")
        good = (self.home / journal.NAME).read_bytes()
        real_replace = os.replace
        for broken in ("replace", "write"):
            with self.subTest(broken=broken):
                if broken == "replace":
                    patch = mock.patch("os.replace", side_effect=OSError(5, "power cut"))
                else:
                    patch = mock.patch("os.fsync", side_effect=OSError(5, "power cut"))
                with patch, self.assertRaises(OSError):
                    jr.record_created(self.home / "b.txt")
                jr.entries = jr.entries[:1]
                self.assertEqual((self.home / journal.NAME).read_bytes(), good)
                loaded = journal.Journal.load(self.home)
                self.assertEqual([e["path"] for e in loaded.entries], [str(self.home / "a.txt")])
        self.assertIs(os.replace, real_replace)

    def test_a_torn_last_line_is_ignored_on_load(self):
        self.home.mkdir(parents=True)
        line = json.dumps({"op": "created", "kind": "file", "path": str(self.home / "a.txt")})
        (self.home / journal.NAME).write_text(line + "\n" + line[:15])
        loaded = journal.Journal.load(self.home)
        self.assertEqual(len(loaded.entries), 1)

    def test_no_journal_means_none(self):
        self.assertIsNone(journal.Journal.load(self.root))


class RollbackItself(Machine):
    def test_a_file_removed_by_hand_is_reported_not_raised(self):
        self.home.mkdir(parents=True)
        jr = journal.Journal(self.home)
        gone, other = self.home / "gone.txt", self.home / "other.txt"
        jr.record_created(gone)
        jr.record_created(other)
        gone.write_text("x")
        other.write_text("y")
        gone.unlink()                                                      # the user deleted it before the rollback
        res = jr.rollback()
        self.assertFalse(other.exists())
        self.assertTrue(res.ok)
        self.assertTrue(any("gone.txt" in s and "already gone" in s for s in res.skipped), res.skipped)
        self.assertIn("already gone 1", res.summary())

    def test_a_real_failure_is_listed_with_the_reason_and_retried_later(self):
        self.home.mkdir(parents=True)
        locked = self.home / "locked"
        jr = journal.Journal(self.home)
        jr.record_created(locked, "dir")
        locked.mkdir()
        (locked / "f.txt").write_text("x")
        locked.chmod(0o500)                                                # cannot delete inside
        try:
            if os.access(locked, os.W_OK):
                self.skipTest("running as a user that ignores permissions")
            res = jr.rollback()
            self.assertFalse(res.ok)
            self.assertIn("locked", res.problems[0])
            self.assertIn("PermissionError", res.problems[0])
            self.assertTrue((self.home / journal.NAME).exists())          # kept: a second try is possible
        finally:
            locked.chmod(0o700)
        again = jr.rollback()
        self.assertTrue(again.ok, again.problems)
        self.assertFalse(locked.exists())

    def test_install_reports_a_failed_rollback_without_a_traceback(self):
        self.faults = ["after-files"]
        real = journal.Journal.rollback

        def broken(j, *a, **k):
            (j.home / "engines" / "b0-vulkan").chmod(0o500)
            try:
                return real(j, *a, **k)
            finally:
                (j.home / "engines" / "b0-vulkan").chmod(0o700)
        with mock.patch.object(journal.Journal, "rollback", broken):
            code, out = self.install()
        self.assertEqual(code, 1)
        if os.access("/", os.W_OK) and os.geteuid() == 0:
            self.skipTest("root ignores permissions")
        self.assertIn("could not be undone", out.text)
        self.assertNotIn("Traceback", out.text)

    def test_rollback_is_idempotent(self):
        self.faults = ["after-files"]
        before = snapshot(self.root)
        jr_holder = {}
        real = journal.Journal.rollback

        def twice(j, *a, **k):
            jr_holder["res"] = real(j, *a, **k)
            return real(j, *a, **k)
        with mock.patch.object(journal.Journal, "rollback", twice):
            self.install()
        self.assertTrue(jr_holder["res"].ok)
        self.assertEqual(snapshot(self.root), before)

    def test_refuses_to_delete_the_users_home(self):
        jr = journal.Journal(self.root)
        jr.entries.append({"op": "created", "kind": "dir", "path": str(Path.home())})
        res = jr.rollback()
        self.assertFalse(res.ok)
        self.assertTrue(Path.home().exists())


class Uninstall(Machine):
    def test_uninstall_after_a_full_install_restores_the_tree_except_models(self):
        before = snapshot(self.root)
        self.assertEqual(self.install()[0], 0)
        self.assertTrue((self.home / journal.NAME).exists())
        self.assertTrue(journal.Journal.load(self.home).committed)
        (self.root / "docs" / "later.txt").write_text("made by the user after the install")
        code, out, err = self.cli("uninstall", "--yes", "--dir", str(self.home))
        self.assertEqual(code, 0, err)
        left = snapshot(self.root)
        extra = {k for k in left if k not in before}
        models = {"data", "data/shura", "data/shura/models", f"data/shura/models/{MODEL}"}
        self.assertEqual(extra - models - {"docs/later.txt"}, set())      # the only leftovers are the kept models
        self.assertIn(f"data/shura/models/{MODEL}", extra)
        self.assertNotIn("data/shura/models/" + engine.SMOKE_MODEL["name"] + "", extra)
        self.assertEqual({k: v for k, v in before.items()}, {k: left[k] for k in before})
        self.assertTrue((self.root / "docs" / "later.txt").exists())      # not in the journal: not touched
        self.assertIn("Kept the models", out)

    def test_uninstall_with_models_flag_restores_the_whole_tree(self):
        before = snapshot(self.root)
        self.assertEqual(self.install()[0], 0)
        (self.root / "docs" / "later.txt").write_text("user file")
        code, out, err = self.cli("uninstall", "--yes", "--models", "--dir", str(self.home))
        self.assertEqual(code, 0, err)
        before["docs/later.txt"] = hashlib.sha256(b"user file").hexdigest()
        self.assertEqual(snapshot(self.root), before)

    def test_models_are_not_deleted_without_consent_when_asked_interactively(self):
        self.assertEqual(self.install()[0], 0)
        with mock.patch("builtins.input", side_effect=["y", "n"]), \
                mock.patch("sys.stdin", mock.Mock(isatty=lambda: True)):
            o, e = io.StringIO(), io.StringIO()
            with redirect_stdout(o), redirect_stderr(e):
                code = cli.main(["uninstall", "--dir", str(self.home)])
        self.assertEqual(code, 0)
        self.assertTrue((self.home / "models" / MODEL).exists())

    def test_dry_run_deletes_nothing(self):
        self.assertEqual(self.install()[0], 0)
        before = snapshot(self.root)
        code, out, _ = self.cli("uninstall", "--dry-run", "--dir", str(self.home))
        self.assertEqual(code, 0)
        self.assertEqual(snapshot(self.root), before)

    def test_without_a_journal_and_without_markers_nothing_is_deleted(self):
        self.home.mkdir(parents=True)
        (self.home / "engines").mkdir()
        (self.home / "stuff.txt").write_text("x")
        before = snapshot(self.root)
        code, out, err = self.cli("uninstall", "--yes", "--dir", str(self.home))
        self.assertEqual(code, 1)
        self.assertEqual(snapshot(self.root), before)
        self.assertIn("nothing was deleted", err)

    def test_files_that_are_not_in_the_journal_survive(self):
        self.assertEqual(self.install()[0], 0)
        (self.home / "my-notes.txt").write_text("keep me")
        code, _, err = self.cli("uninstall", "--yes", "--models", "--dir", str(self.home))
        self.assertEqual(code, 0, err)
        self.assertEqual((self.home / "my-notes.txt").read_text(), "keep me")


if __name__ == "__main__":
    unittest.main()
