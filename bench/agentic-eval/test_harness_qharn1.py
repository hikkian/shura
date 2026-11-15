"""QHARN1: the eval harness runs the product's way (serve mode, plugins, project rules, project-check detector). Offline: a fake
engine stands in for ShuraCode (an HTTP server that answers the same routes), so this checks the plumbing, not a model.
Run: python3 -m unittest discover -s bench/agentic-eval -p 'test_harness_qharn1.py'"""
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402

FAKE_ENGINE = r'''#!/usr/bin/env python3
"""A stand-in for `shuracode serve`: the routes the harness uses, with a scripted conversation (one plugin message included)."""
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
SESSION = "S1"
MESSAGES = []
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        if self.path == "/session":
            return self._send(200, [])
        if self.path == f"/session/{SESSION}/message":
            return self._send(200, MESSAGES)
        self._send(404, {})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/session":
            return self._send(200, {"id": SESSION})
        if self.path == f"/session/{SESSION}/message":
            text = body["parts"][0]["text"]
            MESSAGES.append({"info": {"role": "user", "time": {"created": 1}}, "parts": [{"type": "text", "text": text}]})
            MESSAGES.append({"info": {"role": "user", "time": {"created": 2}}, "parts": [{"type": "text", "text": "[completion gate] run the project tests"}]})
            MESSAGES.append({"info": {"role": "assistant", "time": {"created": 3}}, "parts": [
                {"type": "text", "text": "done", "time": {"start": 3}},
                {"type": "step-finish", "time": {"start": 4}}]})
            return self._send(200, {})
        self._send(404, {})
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def write_fake_engine(folder):
    path = Path(folder) / "shuracode"
    path.write_text(FAKE_ENGINE)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class ServeRunner(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.scratch = root / "scratch"
        self.work = self.scratch / "work"
        self.work.mkdir(parents=True)
        self.log = root / "out" / "run.agent.log"
        self.log.parent.mkdir(parents=True)
        self.engine = write_fake_engine(root)

    def test_serve_mode_is_the_default_and_run_is_the_old_path(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENTIC_EVAL_ENGINE_MODE", None)
            self.assertIs(harness.engine_runner(), harness.run_agent_serve)
            os.environ["AGENTIC_EVAL_ENGINE_MODE"] = "run"
            self.assertIs(harness.engine_runner(), harness.run_agent)

    def test_plugin_messages_arrive_as_user_events_in_the_log(self):
        with mock.patch.dict(os.environ, {"AGENTIC_EVAL_NO_SANDBOX": "1"}), mock.patch.object(harness, "ENGINE", str(self.engine)):
            code, timed_out, secs = harness.run_agent_serve(self.work, "fix the bug", self.scratch / "cfg", self.scratch / "data",
                                                            60, self.log, settle_s=0.5)
        self.assertEqual(code, 0)
        self.assertFalse(timed_out)
        events = [json.loads(line) for line in self.log.read_text().splitlines()]
        users = [e["text"] for e in events if e["type"] == "user"]
        self.assertIn("fix the bug", users)
        self.assertIn("[completion gate] run the project tests", users)
        self.assertTrue(any(e["type"] == "text" and e["part"]["text"] == "done" for e in events))


class ConfigAndRules(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.plugin_src = root / "src" / "plugin"
        (self.plugin_src / "reliability").mkdir(parents=True)
        for name in harness.PLUGIN_MODULES:
            (self.plugin_src / name).write_text("// fake\n")
        for name in harness.RELIABILITY:
            (self.plugin_src / "reliability" / f"{name}.ts").write_text("// fake\n")
        self.nm = root / "src" / "node_modules"
        (self.nm / "@opencode-ai" / "plugin").mkdir(parents=True)
        (self.nm / "@opencode-ai" / "plugin" / "package.json").write_text("{}")
        self.patches = mock.patch.multiple(harness, PLUGIN_SRC=self.plugin_src, NODE_MODULES_SRC=self.nm)
        self.patches.start()
        self.addCleanup(self.patches.stop)

    def test_the_config_has_the_product_plugins_and_no_instruction_file_outside_the_project(self):
        cfg = Path(self._tmp.name) / "cfg"
        with mock.patch.dict(os.environ, {"AGENTIC_EVAL_RELIABILITY": "task-card,completion-gate"}):
            harness.write_config(cfg, port=1234)
        conf = json.loads((cfg / "shuracode.json").read_text())
        self.assertEqual(conf["instructions"], [])
        self.assertTrue((cfg.parent / "plugin" / "server.ts").exists())
        self.assertTrue((cfg / "node_modules" / "@opencode-ai" / "plugin" / "package.json").exists())
        flags = {Path(e[0]).stem: list(e[1].values())[0] for e in conf["plugin"] if isinstance(e, list)}
        self.assertEqual(flags, {"completion-gate": True, "soft-edit": False, "loop-detector": False, "task-card": True})

    def test_the_project_gets_the_rules_and_keeps_its_own_file(self):
        work = Path(self._tmp.name) / "work"
        work.mkdir()
        harness.write_project_rules(work)
        self.assertIn("./.tmp/", (work / "AGENTS.md").read_text())
        (work / "AGENTS.md").write_text("project's own\n")
        harness.write_project_rules(work)
        self.assertEqual((work / "AGENTS.md").read_text(), "project's own\n")


class ProjectChecks(unittest.TestCase):
    def log_with(self, commands):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        self.addCleanup(os.unlink, tmp.name)
        for c in commands:
            tmp.write(json.dumps({"type": "tool_use", "part": {"tool": "bash", "state": {"input": {"command": c}}}}) + "\n")
        tmp.close()
        return tmp.name

    def test_the_project_test_run_is_seen(self):
        r = harness.project_checks(self.log_with(["cd /w && python3 -m pytest -q"]))
        self.assertEqual(r, {"tests_run_project": True, "adhoc_checks": 0})

    def test_throwaway_scripts_are_counted_and_not_taken_for_tests(self):
        r = harness.project_checks(self.log_with(["python3 check.py", "node -e 'console.log(1)'", "node ./probe.js"]))
        self.assertFalse(r["tests_run_project"])
        self.assertEqual(r["adhoc_checks"], 3)


if __name__ == "__main__":
    unittest.main()
