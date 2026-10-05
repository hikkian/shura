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
import system_pressure


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
    def test_solution_stub_indicators(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            (work / "empty.py").write_text("  \n")
            (work / "todo.py").write_text("def solve():\n    raise NotImplementedError()\n")
            (work / "done.py").write_text("def solve():\n    return 42\n")
            self.assertEqual(polyglot_harness._solution_stub_indicators(
                work, ["empty.py", "todo.py", "done.py"]), {
                    "empty_solution_files": ["empty.py"],
                    "not_implemented_solution_files": ["todo.py"],
                    "has_empty_solution": True,
                    "has_not_implemented": True,
                })

    def test_tmpfs_guard_rejects_disk_paths(self):
        polyglot_harness._require_tmpfs("/dev/shm/run")
        with self.assertRaises(SystemExit):
            polyglot_harness._require_tmpfs("/tmp/run")

    def test_real_server_argv_must_explicitly_enable_mtp(self):
        polyglot_harness._require_draft_mtp(["llama-server", "--spec-type", "draft-mtp"])
        polyglot_harness._require_draft_mtp(["llama-server", "--spec-type=draft-mtp"])
        with self.assertRaises(ValueError):
            polyglot_harness._require_draft_mtp(["llama-server", "--ngl", "99"])

    def test_server_and_client_cpu_policies_are_separate(self):
        server = polyglot_harness._server_command([
            "/usr/bin/taskset", "-c", "0,1", "/usr/bin/nice", "-n", "19",
            "/opt/llama-server", "-t", "6",
        ])
        self.assertEqual(server, ["/usr/bin/nice", "-n", "10", "/opt/llama-server", "-t", "6"])
        client = polyglot_harness._client_command(["/opt/opencode", "run"], "0,1")
        self.assertEqual(client, ["/usr/bin/taskset", "-c", "0,1", "/usr/bin/nice", "-n", "19",
                                  "/opt/opencode", "run"])

    def test_server_log_decode_speed_reads_exact_eval_sample(self):
        with tempfile.TemporaryDirectory(prefix="server-rate-", dir="/dev/shm") as tmp:
            log = Path(tmp) / "server.log"
            log.write_text("load line\neval time = 4000.00 ms / 200 tokens (20.00 ms per token, 50.00 tokens per second)\n")
            self.assertEqual(polyglot_harness._server_log_decode_rate(log, 0), (
                200, 50.0, "eval time = 4000.00 ms / 200 tokens (20.00 ms per token, 50.00 tokens per second)"))
            self.assertEqual(polyglot_harness._server_log_decode_average(log, 0), 50.0)

    def test_server_monitor_detects_exit_and_repeated_unhealthy_probes(self):
        class Exited:
            def poll(self):
                return 17

        exited = polyglot_harness.ServerHealthMonitor(
            Exited(), "http://127.0.0.1:9/health", period=0.01)
        exited.start()
        exited.join(timeout=1)
        self.assertTrue(exited.failed.is_set())
        self.assertIn("exited with code 17", exited.reason)

        class Running:
            def poll(self):
                return None

        unhealthy = polyglot_harness.ServerHealthMonitor(
            Running(), "http://127.0.0.1:9/health", period=0.01, failures=2,
            health_check=lambda: False)
        unhealthy.start()
        unhealthy.join(timeout=1)
        self.assertTrue(unhealthy.failed.is_set())
        self.assertIn("missed 2 probes", unhealthy.reason)

    def test_checkpoint_warmup_is_due_every_twenty_completed_tasks_once(self):
        self.assertFalse(polyglot_harness._checkpoint_warmup_due(19, 0))
        self.assertTrue(polyglot_harness._checkpoint_warmup_due(20, 0))
        self.assertFalse(polyglot_harness._checkpoint_warmup_due(20, 20))
        self.assertTrue(polyglot_harness._checkpoint_warmup_due(40, 20))

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
            extra = {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                     "presence_penalty": 1.5, "min_p": 0.0, "max_tokens": 32768,
                     "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
            server = capture_proxy.serve(
                0, f"http://127.0.0.1:{upstream.server_port}", str(usage_path), str(capture_path), extra
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            request_body = {"model": "stub", "max_tokens": 32000, "temperature": 0.6,
                            "messages": [{"role": "user", "content": "test"}]}
            expected_upstream = {**request_body, **extra}
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                    data=json.dumps(request_body).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=5) as response:
                    self.assertEqual(json.loads(response.read()), expected_upstream)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=2)
                upstream.shutdown(); upstream.server_close()
            body_dir = root / "request-bodies"
            body_file = next(body_dir.glob("*.json.gz"))
            with gzip.open(body_file, "rt", encoding="utf-8") as stream:
                captured = json.loads(stream.readline())
            self.assertEqual(captured["request"]["max_tokens"], 32000)
            effective = captured["effective_request"]
            self.assertEqual(effective["max_tokens"], 32768)
            self.assertEqual((effective["temperature"], effective["top_p"], effective["top_k"]),
                             (1.0, 0.95, 20))
            self.assertEqual(effective["presence_penalty"], 1.5)
            self.assertEqual(effective["min_p"], 0.0)
            self.assertEqual(effective["chat_template_kwargs"], {
                "enable_thinking": True, "preserve_thinking": True,
            })

    def test_occamy_author_sampling_is_explicit(self):
        self.assertEqual(polyglot_harness.sampling("occamy"), {
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 32768,
            "presence_penalty": 1.5, "min_p": 0.0,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        })

    def test_s6_watchdog_can_be_exercised_without_nvml(self):
        safe_reading = lambda: {"gpu_c": 40, "cpu_c": 50, "vram_free": 4096,
                                "vram_used": 0, "ram": 8.0, "disk": 100.0}
        watchdog = safety.Watchdog("/", period=0.01, gpu=object(), reader=safe_reading)
        watchdog.start()
        time.sleep(0.04)
        watchdog.stop()
        watchdog.join(timeout=1)
        self.assertFalse(watchdog.tripped.is_set())
        self.assertEqual(watchdog.minimum["vram_free_mib"], 4096)
        self.assertEqual(watchdog.minimum["ram_available_gib"], 8.0)

    def test_zram_limit_trips_immediately(self):
        trips = []
        guard = system_pressure.SystemPressureGuard(
            trips.append, reader=lambda: {"zram_pct": 92.1, "psi_memory_full_avg10": 0.0}
        )
        guard.sample_once()
        self.assertEqual(len(trips), 1)
        self.assertIn("zram occupancy", trips[0])

    def test_psi_requires_two_consecutive_high_samples(self):
        trips = []
        samples = iter([
            {"zram_pct": 69.0, "psi_memory_full_avg10": 10.1},
            {"zram_pct": 69.0, "psi_memory_full_avg10": 9.9},
            {"zram_pct": 69.0, "psi_memory_full_avg10": 10.1},
            {"zram_pct": 69.0, "psi_memory_full_avg10": 10.2},
        ])
        def reader():
            return next(samples, {"zram_pct": 69.0, "psi_memory_full_avg10": 0.0})
        guard = system_pressure.SystemPressureGuard(
            trips.append,
            reader=lambda: {"zram_pct": 69.0, "psi_memory_full_avg10": 0.0},
        )
        guard.reader = reader
        for _ in range(3):
            guard.sample_once()
        self.assertEqual(trips, [])
        guard.sample_once()
        self.assertEqual(len(trips), 1)
        self.assertIn("PSI memory full", trips[0])


if __name__ == "__main__":
    unittest.main()
