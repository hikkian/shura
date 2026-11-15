"""INSTALL1c step 4: settings and the user service from the chosen mode. Everything in /dev/shm-style temporary folders; the
runner (systemctl) is fake, no real service is made."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
sys.path.insert(0, str(ROOT))
from universal import install, journal, settings  # noqa: E402

CHOICE = SimpleNamespace(ok=True, model_id="tiel-coder", context=32768)


class FakeRunner:
    def __init__(self, fail_on=None, jr=None):
        self.calls, self.fail_on, self.jr = [], fail_on, jr
        self.journal_at_call = []

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        if self.jr is not None:       # what the journal on disk held when this command ran
            self.journal_at_call.append(self.jr.path.read_text() if self.jr.path.exists() else "")
        rc = 1 if self.fail_on and self.fail_on in cmd else 0
        return SimpleNamespace(returncode=rc, stdout="", stderr="boom" if rc else "")


def tree(root):
    out = {}
    for p in sorted(Path(root).rglob("*")):
        out[str(p.relative_to(root))] = None if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class Base(unittest.TestCase):
    def setUp(self):
        base = "/dev/shm/INSTALL1c" if Path("/dev/shm").is_dir() else None
        if base:
            os.makedirs(base, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root / "shura-home"
        self.outside = self.root / "outside"
        self.outside.mkdir()
        (self.outside / "keep.txt").write_text("x")
        self.home.mkdir()
        self.fake_home = self.root / "userhome"
        self.xdg = self.root / "xdg"
        self.fake_home.mkdir()
        env = mock.patch.dict(os.environ, {"HOME": str(self.fake_home), "XDG_CONFIG_HOME": str(self.xdg)})
        env.start()
        self.addCleanup(env.stop)
        self.unit = self.xdg / "systemd" / "user" / settings.SERVICE

    def apply(self, runner=None, **kw):
        runner = runner or FakeRunner()
        return settings.apply(self.home, CHOICE, model_path="/m/tiel.gguf", exe_path="/rel/llama-server.sh",
                              runner=runner, consent=True, **kw), runner

    def cfg(self, name):
        return json.loads((self.home / "config" / name).read_text())


class Units(Base):
    def test_1_settings_follow_the_mode(self):
        self.apply()
        launch = self.cfg("model-launch.json")
        prof = launch["models"]["tiel-coder"]
        self.assertEqual(prof["engine"], "v1")
        self.assertEqual(prof["exePath"], "/rel/llama-server.sh")
        self.assertEqual(prof["ctxSize"], 32768)
        self.assertEqual(prof["vramChunkMiB"], 52)
        self.assertEqual(self.cfg("guardian.json")["vramPolicy"], "governor")
        self.assertTrue(self.cfg("guardian.json")["vramGuard"])
        for key in settings.LAYOUT_KEYS:
            self.assertNotIn(key, prof)
            self.assertNotIn(key, launch)
        sc = self.cfg("shuracode.json")
        self.assertEqual(sc["model"], "tiel-local/tiel-coder")
        self.assertEqual(sc["small_model"], "tiel-local/tiel-coder")
        self.assertEqual(sc["$schema"], "https://opencode.ai/config.json")
        self.assertIs(sc["autoupdate"], False)
        self.assertEqual(sc["share"], "disabled")
        prov = sc["provider"]["tiel-local"]
        self.assertEqual(set(prov), {"npm", "name", "options", "models"})
        self.assertEqual(prov["npm"], "@ai-sdk/openai-compatible")
        self.assertEqual(prov["options"]["baseURL"], f"http://127.0.0.1:{settings.GATEWAY_PORT}/v1")
        m = prov["models"]["tiel-coder"]
        self.assertEqual(m["limit"], {"context": 32768, "output": 16384})
        self.assertIs(m["attachment"], False)
        self.assertEqual(m["modalities"], {"input": ["text"], "output": ["text"]})
        self.assertEqual(m["settings"], {"reasoningEffort": "medium"})
        self.assertEqual(m["options"], {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
        self.assertEqual(sc["permission"]["bash"]["sudo *"], "ask")
        self.assertEqual(sc["permission"]["bash"]["rm -rf*"], "deny")
        self.assertEqual(sc["permission"]["webfetch"], "deny")
        self.assertNotIn("plugin", sc)
        self.assertNotIn("mcp", sc)
        self.assertNotRegex(json.dumps(sc), r"(?i)api.?key|token|secret|/home/")
        unit = self.unit.read_text()
        self.assertEqual(unit.splitlines()[0], journal.UNIT_MARKER)
        self.assertIn("ExecStart=", unit)

    def test_2_old_files_return_on_rollback(self):
        (self.home / "config").mkdir()
        old = {n: f'{{"old": "{n}"}}'.encode() for n in ("model-launch.json", "guardian.json")}
        for n, b in old.items():
            (self.home / "config" / n).write_bytes(b)
        jr = journal.Journal(self.home)
        runner = FakeRunner()
        settings.apply(self.home, CHOICE, model_path="/m", exe_path="/e", runner=runner, consent=True, jr=jr)
        self.assertNotEqual((self.home / "config/guardian.json").read_bytes(), old["guardian.json"])
        res = jr.rollback(runner=runner)
        self.assertTrue(res.ok, res.summary())
        for n, b in old.items():
            self.assertEqual((self.home / "config" / n).read_bytes(), b)
        self.assertFalse((self.home / "config/shuracode.json").exists())

    def test_3_failure_after_a_file_leaves_no_traces(self):
        before = tree(self.root)
        real = settings.write_file
        n = {"c": 0}

        def flaky(path, data):
            n["c"] += 1
            if n["c"] == 2:
                raise OSError("disk full")
            real(path, data)
        runner = FakeRunner()
        with mock.patch.object(settings, "write_file", flaky), self.assertRaises(OSError):
            self.apply(runner)
        self.assertEqual(tree(self.root), before)
        self.assertEqual(runner.calls, [])

    def test_4_journal_before_enable_start(self):
        jr = journal.Journal(self.home)
        runner = FakeRunner(jr=jr)
        settings.apply(self.home, CHOICE, model_path="/m", exe_path="/e", runner=runner, consent=True, jr=jr)
        names = [c[2] for c in runner.calls]
        self.assertEqual(names, ["daemon-reload", "enable", "start"])
        for text in runner.journal_at_call:
            self.assertIn('"op": "service"', text)
            self.assertIn(settings.SERVICE, text)
        entry = next(e for e in jr.entries if e["op"] == "service")
        self.assertEqual(set(entry) - {"op", "kind", "path"}, {"action", "name"})
        self.assertNotIn("undo", entry)

    def test_5_enable_failure_rolls_back_the_service(self):
        before = tree(self.root)
        runner = FakeRunner(fail_on="enable")
        with self.assertRaises(settings.SettingsError):
            self.apply(runner)
        self.assertIn(journal.ACTIONS["user-service-disable"](settings.SERVICE), runner.calls)
        self.assertEqual(tree(self.root), before)

    def test_6_nothing_outside_home(self):
        before = tree(self.outside)
        self.apply()
        self.assertEqual(tree(self.outside), before)
        self.assertEqual({p.name for p in self.root.iterdir()}, {"shura-home", "outside", "userhome", "xdg"})
        self.assertEqual(tree(self.outside), before)
        with self.assertRaises(TypeError):                    # no unit_dir in the public API any more
            self.apply(unit_dir=self.outside / "units")

    def test_6a_unit_at_xdg_path(self):
        self.apply()
        self.assertTrue(self.unit.is_file())
        self.assertEqual(self.unit, self.xdg / "systemd/user/ai-gateway.service")
        self.assertFalse((self.home / ".config").exists())

    def test_6b_unit_default_path_without_xdg(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("XDG_CONFIG_HOME")
            self.apply()
        self.assertTrue((self.fake_home / ".config/systemd/user/ai-gateway.service").is_file())

    def test_6c_foreign_unit_not_overwritten(self):
        self.unit.parent.mkdir(parents=True)
        self.unit.write_text("[Service]\nExecStart=/bin/true\n")
        before = tree(self.root)
        runner = FakeRunner()
        with self.assertRaises(settings.SettingsError) as cm:
            self.apply(runner)
        self.assertIn("not overwriting", str(cm.exception))
        self.assertEqual(tree(self.root), before)             # no journal, no config files
        self.assertEqual(runner.calls, [])
        self.assertEqual(self.unit.read_text(), "[Service]\nExecStart=/bin/true\n")

    def test_6d_rollback_removes_our_unit_and_keeps_the_dirs_clean(self):
        before = tree(self.root)
        jr = journal.Journal(self.home)
        runner = FakeRunner()
        settings.apply(self.home, CHOICE, model_path="/m", exe_path="/e", runner=runner, consent=True, jr=jr)
        self.assertTrue(self.unit.is_file())
        res = jr.rollback(runner=runner)
        self.assertTrue(res.ok, res.summary())
        self.assertFalse(self.unit.exists())
        self.assertEqual(tree(self.root), before)

    def test_6e_our_older_unit_is_replaced_and_comes_back(self):
        self.unit.parent.mkdir(parents=True)
        old = (journal.UNIT_MARKER + "\n[Service]\nExecStart=/old\n").encode()
        self.unit.write_bytes(old)
        jr = journal.Journal(self.home)
        runner = FakeRunner()
        settings.apply(self.home, CHOICE, model_path="/m", exe_path="/e", runner=runner, consent=True, jr=jr)
        self.assertNotEqual(self.unit.read_bytes(), old)
        self.assertTrue(any(e["op"] == "replaced" and e["path"] == str(self.unit) for e in jr.entries))
        self.assertTrue(jr.rollback(runner=runner).ok)
        self.assertEqual(self.unit.read_bytes(), old)

    def test_6f_symlinked_config_is_refused(self):
        elsewhere = self.root / "ELSEWHERE"
        elsewhere.mkdir()
        link = self.fake_home / ".config"
        link.symlink_to(elsewhere)
        with mock.patch.dict(os.environ):
            os.environ.pop("XDG_CONFIG_HOME")
            before = tree(self.root)
            with self.assertRaises(settings.SettingsError):
                self.apply()
            self.assertEqual(tree(self.root), before)
            self.assertEqual(list(elsewhere.iterdir()), [])

    def test_7_no_consent_no_writes(self):
        before = tree(self.root)
        runner = FakeRunner()
        self.assertEqual(settings.apply(self.home, CHOICE, model_path="/m", exe_path="/e", runner=runner), [])
        self.assertEqual(tree(self.root), before)
        self.assertEqual(runner.calls, [])
        # the install hook: no --yes and no terminal -> apply is not called
        args = SimpleNamespace(install_settings=True, yes=False)
        out = install.Out()
        out.ask = lambda q, default=True: False
        with mock.patch.object(settings, "apply") as ap:
            self.assertEqual(install._maybe_apply_settings(args, out, None, self.home, {}, "s", "m"), [])
            ap.assert_not_called()
        with mock.patch.object(settings, "apply") as ap:
            install._maybe_apply_settings(SimpleNamespace(install_settings=False, yes=True), out, None, self.home, {}, "s", "m")
            ap.assert_not_called()

    def test_8_memory_low_untouched(self):
        src = (ROOT / "installer/universal/settings.py").read_text()
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        self.assertNotIn("/sys/fs/cgroup", code.split('"""', 2)[2])
        self.assertNotIn("memory.low", code.split('"""', 2)[2])


if __name__ == "__main__":
    unittest.main()
