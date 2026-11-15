"""Smoke checks against a stub HTTP server on 127.0.0.1 (plus the shura-qa fake llama-server if present, on its own port).
Run: python3 -B -m unittest tests.test_smoke_checks"""
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
from universal import smoke  # noqa: E402

FAKE = Path.home() / "Projects/shura-qa/fake_llama/fake_llama_server.py"


class Stub(BaseHTTPRequestHandler):
    cfg = {}

    def log_message(self, *a):
        pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        c = self.cfg
        if self.path == "/health":
            return self.send(200, {"status": "ok", "model": "tiel", "mode": "auto"})
        if self.path == "/vram" and c.get("vram", True):
            return self.send(200, {"free_mib": 4000})
        self.send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        c = self.cfg
        if self.path in ("/pause", "/resume") and c.get("pause", True):
            return self.send(200, {"ok": True})
        if self.path == "/v1/chat/completions":
            if body.get("tools") and c.get("tools", True):
                msg = {"content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}]}
                return self.send(200, {"choices": [{"message": msg, "finish_reason": "tool_calls"}]})
            timings = {"draft_n": 10, "draft_n_accepted": 7} if c.get("mtp", True) else {}
            return self.send(200, {"choices": [{"message": {"content": c.get("text", "hi there")}, "finish_reason": "stop"}], "timings": timings})
        self.send(404, {})


class Server:
    def __init__(self, **cfg):
        h = type("H", (Stub,), {"cfg": cfg})
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), h)
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()


MEM_OK = {"vram_free_mib": lambda: 3000, "ram_free_mib": lambda: 20000}
PROFILE = {"expected_model": "tiel", "expected_mode": "auto", "min_free_vram_mib": 1500, "min_free_ram_mib": 6000, "health_timeout_s": 2}


def by_name(res):
    return {r[0]: r for r in res["results"]}


class SmokeTests(unittest.TestCase):
    def test_a_all_ok(self):
        with Server() as s:
            res = smoke.run_smoke(s.url, PROFILE, MEM_OK)
        self.assertEqual(res["summary"], "ok", res)
        self.assertEqual(len(res["results"]), 7)
        self.assertTrue(all(r[1] == "ok" for r in res["results"]), res)

    def test_a2_wrong_mode_fails(self):
        with Server() as s:
            res = smoke.run_smoke(s.url, dict(PROFILE, expected_mode="gaming"), MEM_OK)
        self.assertEqual(by_name(res)["server_up"][1], "fail")

    def test_b_server_down(self):
        res = smoke.run_smoke("http://127.0.0.1:1", dict(PROFILE, health_timeout_s=1, exe_path="/x/llama-server"), MEM_OK)
        r = by_name(res)["server_up"]
        self.assertEqual(r[1], "fail")
        self.assertIn("не ответил", r[2])
        self.assertIn("exePath", r[2])
        self.assertEqual(res["summary"], "fail")

    def test_c_empty_reply(self):
        with Server(text="") as s:
            res = smoke.run_smoke(s.url, PROFILE, MEM_OK)
        r = by_name(res)["first_reply"]
        self.assertEqual(r[1], "fail")
        self.assertIn("пуст", r[2])

    def test_d_no_mtp_counters(self):
        with Server(mtp=False) as s:
            res = smoke.run_smoke(s.url, PROFILE, MEM_OK)
        self.assertEqual(by_name(res)["mtp_accepts"][1], "not_checked")
        self.assertNotEqual(res["summary"], "fail")

    def test_e_no_vram_endpoint(self):
        with Server(vram=False) as s:
            res = smoke.run_smoke(s.url, PROFILE, MEM_OK)
        self.assertEqual(by_name(res)["guardian_vram"][1], "not_checked")
        self.assertNotEqual(res["summary"], "fail")

    def test_f_memory_below_bound(self):
        with Server() as s:
            res = smoke.run_smoke(s.url, PROFILE, {"vram_free_mib": lambda: 500, "ram_free_mib": lambda: 20000})
        r = by_name(res)["memory_bounds"]
        self.assertEqual(r[1], "fail")
        self.assertIn("свободно меньше границы", r[2])

    def test_g_no_bounds_not_checked(self):
        with Server() as s:
            res = smoke.run_smoke(s.url, {"health_timeout_s": 2}, {})
        self.assertEqual(by_name(res)["memory_bounds"][1], "not_checked")

    def test_h_no_pause_no_tools(self):
        with Server(pause=False, tools=False) as s:
            res = smoke.run_smoke(s.url, PROFILE, MEM_OK)
        self.assertEqual(by_name(res)["pause_resume"][1], "not_checked")
        self.assertEqual(by_name(res)["agent_step"][1], "not_checked")

    def test_i_never_raises(self):
        res = smoke.run_smoke("not a url", {"health_timeout_s": 1}, {"ram_free_mib": lambda: 1 / 0})
        self.assertIn(res["summary"], ("fail", "not_checked"))


class HttpBaseTests(unittest.TestCase):
    def test_only_loopback_http(self):
        from unittest import mock
        for base in ("file:///etc/passwd", "http://example.invalid", "https://127.0.0.1:1", "ftp://localhost"):
            with self.subTest(base=base), mock.patch.object(smoke.urllib.request, "urlopen") as op:
                with self.assertRaises(ValueError) as cm:
                    smoke._http(base, "/x")
                self.assertIn("127.0.0.1", str(cm.exception))
                op.assert_not_called()


@unittest.skipUnless(FAKE.is_file(), "shura-qa fake_llama_server not found")
class FakeLlamaTests(unittest.TestCase):
    def test_fake_server(self):
        port = 18701
        env = dict(os.environ, FAKE_LLAMA_CONFIG="")
        proc = subprocess.Popen([sys.executable, "-B", str(FAKE), "--host", "127.0.0.1", "--port", str(port)], env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            res = smoke.run_smoke("http://127.0.0.1:%d" % port, {"health_timeout_s": 10}, MEM_OK)
        finally:
            proc.terminate()
            proc.wait(10)
        r = by_name(res)
        self.assertEqual(r["server_up"][1], "ok", res)
        self.assertEqual(r["first_reply"][1], "ok", res)
        self.assertEqual(r["guardian_vram"][1], "not_checked")
        self.assertEqual(r["pause_resume"][1], "not_checked")
        self.assertEqual(r["agent_step"][1], "not_checked")
        self.assertNotEqual(res["summary"], "fail", res)


if __name__ == "__main__":
    unittest.main()
