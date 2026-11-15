"""AGTOOLS1b: run_experiment starts (the missing `import os` of d8cf2b4), judge tasks never stay pending silently, and the proxy can log
request bodies with image data URLs trimmed.
    cd bench/agentic-eval && python3 -B -m unittest test_agtools1b
"""
import http.server
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proxy  # noqa: E402
import run_experiment  # noqa: E402


class RunExperimentStarts(unittest.TestCase):
    def test_the_module_imports_and_help_runs(self):
        r = subprocess.run([sys.executable, "-B", str(HERE / "run_experiment.py"), "--help"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--judge", r.stdout)
        self.assertTrue(hasattr(run_experiment, "os"))   # d8cf2b4: os.environ was used without the import

    def test_the_answers_folder_is_set_for_the_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if k != "GEVAL_ANSWER_DIR"}
            arms = Path(tmp) / "arms.json"
            arms.write_text(json.dumps({"arms": [], "server": {}}))
            code = ("import os, sys; sys.argv=['x', %r, '--out', %r, '--tasks-root', %r]; import run_experiment; "
                    "run_experiment.main(); print('ANSWERS=' + os.environ['GEVAL_ANSWER_DIR'])" % (str(arms), tmp + "/out", tmp + "/no-tasks"))
            r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=60, env=env, cwd=str(HERE))
            self.assertIn(f"ANSWERS={tmp}/out/answers", r.stdout + r.stderr)


class JudgeNeverSilent(unittest.TestCase):
    def test_pending_tasks_are_counted_and_reported_without_answers(self):
        runs = [{"task": "gd_card-general-01", "arm": "a", "rep": 0, "test_tail": '{"score": 0.0, "judge_pending": true}'},
                {"task": "gq_x-general-01", "arm": "a", "rep": 0}]
        self.assertEqual([r["task"] for r in run_experiment.judge_pending(runs)], ["gd_card-general-01"])
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["GEVAL_ANSWER_DIR"] = tmp + "/answers"
            out = io.StringIO()
            with redirect_stdout(out):
                left = run_experiment.run_judge(tmp, runs)
            self.assertEqual(left, 1)
            self.assertIn("pending", out.getvalue())
            self.assertIn("no answers", out.getvalue())

    def test_the_report_names_pending_judge_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = {"task": "gd_card-general-01", "family": "gd_card", "kind": "general-01", "arm": "a", "rep": 0, "resolved": False, "score": 0.0,
                   "seconds": 10, "tokens": 1, "prompt_tokens": 1, "requests": 1, "test_tail": '{"judge_pending": true}'}
            (Path(tmp) / "results.jsonl").write_text(json.dumps(rec) + "\n")
            os.environ["GEVAL_ANSWER_DIR"] = tmp + "/answers"
            with redirect_stdout(io.StringIO()):
                run_experiment.finish(tmp, judge="off")
            self.assertIn("JUDGE PENDING: 1", (Path(tmp) / "REPORT.txt").read_text())


class _Upstream(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        body = json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ProxyBodies(unittest.TestCase):
    def test_request_bodies_are_logged_with_image_urls_trimmed_when_asked(self):
        up = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
        threading.Thread(target=up.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "p.jsonl"
            srv = proxy.serve(0, f"http://127.0.0.1:{up.server_port}", str(log))
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                png = "data:image/png;base64," + "A" * 5000
                body = {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "Attached media from tool result:"},
                                                                                 {"type": "image_url", "image_url": {"url": png}}]}]}
                for flag in ("0", "1"):
                    os.environ["AGENTIC_EVAL_LOG_REQUESTS"] = flag
                    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/v1/chat/completions", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=10) as r:
                        self.assertEqual(r.status, 200)
                bodies = Path(str(log) + ".bodies.jsonl")
                self.assertTrue(bodies.is_file())
                rows = [json.loads(l) for l in bodies.read_text().splitlines()]
                self.assertEqual(len(rows), 1)          # only the request made with the flag on
                url = rows[0]["body"]["messages"][0]["content"][1]["image_url"]["url"]
                self.assertTrue(url.startswith("data:image/png;base64,AAAA"))
                self.assertIn(f"({len(png)} chars)", url)
                self.assertLess(len(url), 80)
                self.assertEqual(rows[0]["body"]["messages"][0]["content"][0]["text"], "Attached media from tool result:")
            finally:
                os.environ.pop("AGENTIC_EVAL_LOG_REQUESTS", None)
                srv.shutdown()
                srv.server_close()
                up.shutdown()
                up.server_close()

    def test_trim_images_leaves_everything_else_alone(self):
        jpeg = "data:image/jpeg;base64,/9j/" + "B" * 100
        body = {"a": [{"url": "https://x/y.png"}, {"url": jpeg}], "text": "data:image stays in text"}
        out = proxy.trim_images(body)
        self.assertEqual(out["a"][0]["url"], "https://x/y.png")
        self.assertTrue(out["a"][1]["url"].endswith(f"({len(jpeg)} chars)"), out["a"][1]["url"])
        self.assertEqual(out["text"], "data:image stays in text")


if __name__ == "__main__":
    unittest.main()
