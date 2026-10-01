"""The measuring proxy and the statistics of the model comparison (bench/agentic-eval). Offline, no GPU, no model."""
import json
import socketserver
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench" / "agentic-eval"))
import analyze  # noqa: E402
import proxy  # noqa: E402


class Quick(HTTPServer):
    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


class FakeLlama(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        req = json.loads(raw)
        FakeLlama.seen.append(req)
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            events = [b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n',
                      b'data: {"choices":[{"delta":{"content":"llo"}}],"usage":{"prompt_tokens":11,"completion_tokens":2},'
                      b'"timings":{"predicted_per_second":40.5}}\n\n', b"data: [DONE]\n\n"]
            for e in events:
                self.wfile.write(f"{len(e):x}\r\n".encode() + e + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        else:
            out = json.dumps({"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)


class Proxy(unittest.TestCase):
    def setUp(self):
        FakeLlama.seen.clear()
        self.up = Quick(("127.0.0.1", 0), FakeLlama)
        threading.Thread(target=self.up.serve_forever, daemon=True).start()
        self.tmp = tempfile.TemporaryDirectory()
        self.log = str(Path(self.tmp.name) / "run.jsonl")
        self.px = proxy.serve(0, f"http://127.0.0.1:{self.up.server_port}", self.log,
                              {"chat_template_kwargs": {"terse": False}})
        threading.Thread(target=self.px.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.px.server_port}"

    def tearDown(self):
        self.px.shutdown()
        self.up.shutdown()
        self.tmp.cleanup()

    def post(self, body):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read()

    def test_spelling_extra_fields_and_usage_request_reach_the_model(self):
        self.post({"model": "m", "messages": [], "reasoningEffort": "medium", "stream": True})
        seen = FakeLlama.seen[0]
        self.assertEqual(seen["reasoning_effort"], "medium")
        self.assertNotIn("reasoningEffort", seen)
        self.assertIs(seen["chat_template_kwargs"]["terse"], False)
        self.assertTrue(seen["stream_options"]["include_usage"])

    def wait_logged(self, n=1):
        import time
        for _ in range(100):                                  # the proxy logs after it has sent the last chunk
            if proxy.totals(self.log)["requests"] >= n:
                break
            time.sleep(0.05)
        return proxy.totals(self.log)

    def test_a_stream_arrives_complete_and_its_usage_is_logged(self):
        data = self.post({"model": "m", "messages": [], "stream": True})
        self.assertIn(b"llo", data)
        self.assertIn(b"[DONE]", data)
        t = self.wait_logged()
        self.assertEqual((t["requests"], t["prompt_tokens"], t["completion_tokens"], t["errors"]), (1, 11, 2, 0))

    def test_a_plain_response_and_other_paths_pass_through(self):
        out = json.loads(self.post({"model": "m", "messages": []}))
        self.assertEqual(out["choices"][0]["message"]["content"], "hi")
        with urllib.request.urlopen(self.base + "/health", timeout=20) as r:
            self.assertEqual(json.load(r)["status"], "ok")
        self.assertEqual(self.wait_logged()["requests"], 1)                    # only completions are counted

    def test_a_dead_upstream_is_a_502_not_a_hang(self):
        self.up.shutdown()
        self.up.server_close()
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post({"model": "m", "messages": []})
        self.assertEqual(cm.exception.code, 502)

    def test_usage_scanner_handles_json_sse_and_garbage(self):
        self.assertEqual(proxy.scan_usage(b'{"usage":{"completion_tokens":4}}')["usage"]["completion_tokens"], 4)
        self.assertEqual(proxy.scan_usage(b"data: nonsense\n\ndata: [DONE]\n"), {})
        self.assertEqual(proxy.scan_usage(b""), {})


def runs_for(shifts, tasks=40, reps=2, seed=1):
    import random
    rng = random.Random(seed)
    out = []
    for t in range(tasks):
        p = rng.uniform(0.15, 0.85)
        for arm, d in shifts.items():
            for r in range(reps):
                out.append({"arm": arm, "task": f"t{t}", "rep": r, "resolved": rng.random() < min(0.99, max(0.01, p + d)),
                            "tokens": 10_000 + (0 if arm == "cheap" else 5_000), "seconds": 1.0})
    return out


class Statistics(unittest.TestCase):
    def test_an_obvious_winner_is_declared(self):
        res = analyze.compare(runs_for({"A": 0.30, "B": 0.0, "C": 0.0}, tasks=60), flips=4000, boot=800)
        self.assertEqual(analyze.verdict(res)["kind"], "winner")
        self.assertEqual(analyze.verdict(res)["arm"], "A")

    def test_equal_arms_are_a_tie_and_the_cheaper_one_wins_the_tie_break(self):
        runs = runs_for({"cheap": 0.0, "dear": 0.0}, tasks=40, seed=3)
        # make the two arms IDENTICAL in outcomes, so that the quality verdict is a tie by construction
        by = {(r["task"], r["rep"]): r["resolved"] for r in runs if r["arm"] == "cheap"}
        for r in runs:
            r["resolved"] = by[(r["task"], r["rep"])]
        text, res, ver = analyze.report(runs, flips=2000, boot=500)
        self.assertEqual(ver["kind"], "tie")
        self.assertIn("Tie-break", text)
        self.assertIn("cheap", text.split("Tie-break")[1].splitlines()[0])

    def test_holm_is_monotone_and_never_below_the_raw_p(self):
        adj = analyze.holm({"x": 0.01, "y": 0.04, "z": 0.03})
        self.assertAlmostEqual(adj["x"], 0.03)
        self.assertGreaterEqual(adj["z"], adj["x"])
        self.assertTrue(all(adj[k] >= p for k, p in {"x": 0.01, "y": 0.04, "z": 0.03}.items()))

    def test_repetitions_of_one_task_are_averaged_before_testing(self):
        runs = [{"arm": a, "task": "t", "rep": r, "resolved": a == "A", "tokens": 1, "seconds": 1}
                for a in ("A", "B") for r in range(5)]
        self.assertEqual(analyze.per_task_scores(runs), {"A": {"t": 1.0}, "B": {"t": 0.0}})

    def test_an_incomplete_design_is_an_error_not_a_silent_gap(self):
        with self.assertRaises(ValueError):
            analyze.per_task_scores([{"arm": "A", "task": "t1", "resolved": True},
                                     {"arm": "B", "task": "t2", "resolved": True}])

    def test_the_sign_flip_test_does_not_see_a_difference_that_is_not_there(self):
        self.assertEqual(analyze.sign_flip_p([0.0, 0.0, 0.0]), 1.0)
        self.assertGreater(analyze.sign_flip_p([0.5, -0.5, 0.5, -0.5, 0.5, -0.5], flips=4000), 0.5)
        self.assertLess(analyze.sign_flip_p([0.5] * 12, flips=4000), 0.01)


if __name__ == "__main__":
    unittest.main()
