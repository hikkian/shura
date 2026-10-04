import gzip
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "external"))
import capture_proxy
import safety
import polyglot_harness


class EchoHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ExternalRunControlTests(unittest.TestCase):
    def test_tmpfs_guard_rejects_disk_paths(self):
        polyglot_harness._require_tmpfs("/dev/shm/run")
        with self.assertRaises(SystemExit):
            polyglot_harness._require_tmpfs("/tmp/run")

    def test_real_server_argv_must_explicitly_enable_mtp(self):
        polyglot_harness._require_draft_mtp(["llama-server", "--spec-type", "draft-mtp"])
        polyglot_harness._require_draft_mtp(["llama-server", "--spec-type=draft-mtp"])
        with self.assertRaises(ValueError):
            polyglot_harness._require_draft_mtp(["llama-server", "--ngl", "99"])

    def test_resume_recovers_complete_lines_and_drops_torn_tail(self):
        with tempfile.TemporaryDirectory(prefix="resume-", dir="/dev/shm") as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_bytes(
                b'{"task":"one","arm":"tiel-base","rep":0}\n'
                b'{"task":"two","arm":"tiel-base","rep":0}\n'
                b'{"task":"three","arm":"tiel-base"'
            )
            self.assertEqual(polyglot_harness._completed(path), {
                ("one", "tiel-base", 0), ("two", "tiel-base", 0),
            })
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_capture_proxy_records_request_body_and_preserves_response(self):
        with tempfile.TemporaryDirectory(prefix="body-capture-", dir="/dev/shm") as tmp:
            root = Path(tmp)
            upstream = ThreadingHTTPServer(("127.0.0.1", 0), EchoHandler)
            threading.Thread(target=upstream.serve_forever, daemon=True).start()
            capture_path = root / "request-bodies.jsonl.gz"
            usage_path = root / "usage.jsonl"
            server = capture_proxy.serve(
                0, f"http://127.0.0.1:{upstream.server_port}", str(usage_path), str(capture_path)
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            request_body = {"model": "stub", "max_tokens": 32768, "messages": [{"role": "user", "content": "test"}]}
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                    data=json.dumps(request_body).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=5) as response:
                    self.assertEqual(json.loads(response.read()), request_body)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=2)
                upstream.shutdown(); upstream.server_close()
            body_dir = root / "request-bodies"
            body_file = next(body_dir.glob("*.json.gz"))
            with gzip.open(body_file, "rt", encoding="utf-8") as stream:
                captured = json.loads(stream.readline())
            self.assertEqual(captured["request"]["max_tokens"], 32768)

    def test_s6_watchdog_can_be_exercised_without_nvml(self):
        safe_reading = lambda: {"gpu_c": 40, "cpu_c": 50, "vram_free": 4096,
                                "vram_used": 0, "ram": 8.0, "disk": 100.0}
        watchdog = safety.Watchdog("/", period=0.01, gpu=object(), reader=safe_reading)
        watchdog.start()
        time.sleep(0.04)
        watchdog.stop()
        watchdog.join(timeout=1)
        self.assertFalse(watchdog.tripped.is_set())


if __name__ == "__main__":
    unittest.main()
