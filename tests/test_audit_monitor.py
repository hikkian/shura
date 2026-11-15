"""Regression tests for the 2026-10-06 audit: monitor, monitor auth, crash report. Fake sysfs and handlers only."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))
import crash_report
import monitor
import test_gateway as fixture
from game_mode import GamePolicy, GameSignals
from monitor import Monitor


class MonitorAudit(unittest.TestCase):
    def put(self, root, relative, text):
        path = Path(root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_m4_zram_fill_is_uncompressed_data_over_disksize(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            disk = 1 << 30
            # orig 90 %, compressed 45 %, mem_used (with overhead) 47 %
            self.put(tmp, "block/zram0/mm_stat", f"{int(.9 * disk)} {int(.45 * disk)} {int(.47 * disk)} 0 0 0 0\n")
            self.put(tmp, "block/zram0/disksize", str(disk))
            z = monitor._read_zram(Path(tmp))
            self.assertAlmostEqual(z["used_percent"], 90.0, places=3)
            self.assertEqual(z["used_bytes"], int(.47 * disk))
            self.assertEqual(monitor._severity(z["used_percent"], 92.0, 80.0, True), "warn")

    def snapshot(self, m, events=None, cache=0):
        m.system_source = monitor.WindowsSystemSource()
        status = {"status": "UNLOADED", "default_model": "model", "llama_pid": None, "event_counters": events or {}}
        return m.snapshot(status=status, state=None, model_config={}, gpu={}, cache_seconds=cache)

    def finished(self, m, status, error=True, expected=False):
        ident = m.begin_request("model", "/v1/chat/completions")
        m.finish_request(ident, status=status, error=error, expected=expected)

    def test_m8_expected_503_and_client_errors_are_not_failures(self):
        m = monitor.Monitor()
        self.finished(m, 503, expected=True)       # game pause, AI off, memory pressure
        self.finished(m, 404)
        self.finished(m, 429)
        state = self.snapshot(m)
        self.assertEqual(state["reliability"]["request_errors"], 0)
        self.assertEqual(state["health"]["level"], "ok")
        self.assertEqual(state["block_status"]["reliability"], "ok")

    def test_m8_a_real_failure_is_critical_then_ages_out(self):
        clock = [1000.0]
        m = monitor.Monitor(wall_clock=lambda: clock[0], monotonic=lambda: clock[0])
        self.finished(m, 500)
        self.assertEqual(self.snapshot(m)["health"]["level"], "crit")
        self.assertEqual(self.snapshot(m)["reliability"]["request_errors"], 1)
        clock[0] += monitor.RELIABILITY_WINDOW_SECONDS + 1
        state = self.snapshot(m)
        self.assertEqual(state["health"]["level"], "ok")
        self.assertEqual(state["reliability"]["request_errors"], 1)    # lifetime counter stays visible

    def test_m8_a_request_with_no_status_is_still_an_error(self):
        m = monitor.Monitor()
        ident = m.begin_request("model", "/v1/chat/completions")
        m.finish_request(ident)
        self.assertEqual(m.request_errors, 1)

    def test_m8_length_cuts_also_age_out(self):
        clock = [1000.0]
        m = monitor.Monitor(wall_clock=lambda: clock[0], monotonic=lambda: clock[0])
        ident = m.begin_request("model", "/v1/chat/completions")
        m.observe_json(ident, b'{"choices":[{"finish_reason":"length"}]}')
        m.finish_request(ident, status=200)
        self.assertEqual(self.snapshot(m)["health"]["level"], "warn")
        clock[0] += monitor.RELIABILITY_WINDOW_SECONDS + 1
        self.assertEqual(self.snapshot(m)["health"]["level"], "ok")

    def test_l13d_reloads_are_not_restarts(self):
        m = monitor.Monitor()
        state = self.snapshot(m, {"server_started": 5})
        self.assertEqual(state["reliability"]["server_restarts"], 0)
        self.assertEqual(state["health"]["level"], "ok")
        state = self.snapshot(m, {"server_started": 5, "server_crashed": 1})
        self.assertEqual(state["reliability"]["server_restarts"], 1)
        state = self.snapshot(m, {"server_started": 1, "server_crashed": 1})
        self.assertEqual(state["reliability"]["server_restarts"], 0)

    def test_l12_non_ascii_authorization_header_is_refused_not_fatal(self):
        self.assertFalse(monitor.monitor_authorized({"Authorization": "Bearer é"}, {"apiKey": "x"}))
        self.assertTrue(monitor.monitor_authorized({"Authorization": "Bearer x"}, {"apiKey": "x"}))
        self.assertTrue(monitor.monitor_authorized({"Authorization": "Bearer ключ"}, {"apiKey": "ключ"}))
        self.assertFalse(monitor.monitor_authorized({}, {"apiKey": "x"}))
        self.assertTrue(monitor.monitor_authorized({"Authorization": "\udcff"}, {}))   # no key configured

    def test_l13a_leaked_active_requests_expire(self):
        clock = [0.0]
        m = monitor.Monitor(wall_clock=lambda: clock[0], monotonic=lambda: clock[0])
        ids = [m.begin_request("model", "/v1/chat/completions") for _ in range(64)]
        self.assertTrue(all(ids))
        self.assertIsNone(m.begin_request("model", "/v1/chat/completions"))   # table full
        self.assertTrue(self.snapshot(m)["now"]["generating"])
        clock[0] += monitor.ACTIVE_IDLE_SECONDS + 1
        self.assertFalse(self.snapshot(m)["now"]["generating"])
        self.assertIsNotNone(m.begin_request("model", "/v1/chat/completions"))

    def test_l13a_activity_keeps_a_long_request_alive(self):
        clock = [0.0]
        m = monitor.Monitor(wall_clock=lambda: clock[0], monotonic=lambda: clock[0])
        ident = m.begin_request("model", "/v1/chat/completions")
        for _ in range(4):
            clock[0] += monitor.ACTIVE_IDLE_SECONDS - 10
            m.observe_sse(ident, b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
        self.assertTrue(self.snapshot(m)["now"]["generating"])

    def route(self, m, provider, sleep_after=2):
        class Handler:
            command = "GET"

            def __init__(self):
                self.status, self.headers, self.sent_json = None, {}, []
                self.wfile = io.BytesIO()
                self.connection = Mock()

            def send_response(self, code): self.status = code
            def send_header(self, key, value): self.headers[key] = value
            def end_headers(self): pass
            def send_json(self, code, value): self.sent_json.append((code, value)); self.status = code

        handler = Handler()
        calls = []

        def sleep(_):
            calls.append(1)
            if len(calls) >= sleep_after:
                raise BrokenPipeError()

        original = m.stream
        m.stream = lambda p, s: original(p, s, sleep=sleep)
        monitor.monitor_route(handler, "/monitor/api/stream", provider, lambda: [], monitor_instance=m)
        return handler

    def test_l13b_stream_subscribers_are_capped(self):
        m = monitor.Monitor()
        m._subscribers = monitor.MAX_STREAM_SUBSCRIBERS
        handler = self.route(m, lambda: {"schema": 1})
        self.assertEqual(handler.sent_json[0][0], 503)
        self.assertEqual(handler.wfile.getvalue(), b"")
        self.assertEqual(m._subscribers, monitor.MAX_STREAM_SUBSCRIBERS)

    def test_l13b_a_stalled_reader_times_out(self):
        m = monitor.Monitor()
        handler = self.route(m, lambda: {"schema": 1})
        handler.connection.settimeout.assert_called_once_with(monitor.STREAM_WRITE_TIMEOUT_SECONDS)
        self.assertEqual(m._subscribers, 0)

    def test_l13b_a_write_timeout_ends_the_stream(self):
        m = monitor.Monitor()
        calls = []

        def send(state):
            calls.append(state)
            raise TimeoutError("client stopped reading")

        m.stream(lambda: {"schema": 1}, send, sleep=lambda _: None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(m._subscribers, 0)

    def test_l13c_provider_failure_after_headers_never_starts_a_second_response(self):
        m = monitor.Monitor()

        def provider():
            raise RuntimeError("sampler broke")

        handler = self.route(m, provider)
        self.assertEqual(handler.status, 200)
        self.assertEqual(handler.sent_json, [])
        body = handler.wfile.getvalue().decode()
        self.assertTrue(body.startswith("data: "))
        self.assertEqual(json.loads(body.split("data: ", 1)[1].split("\n\n")[0])["schema"], 1)

    def test_l13c_failure_before_headers_still_answers_once(self):
        class Handler:
            command = "GET"
            sent = []

            def send_json(self, code, value): self.sent.append(code)

        monitor.monitor_route(Handler(), "/monitor/api/state", lambda: 1 / 0, lambda: [])
        self.assertEqual(Handler.sent, [200])


class CrashReportAudit(unittest.TestCase):
    def test_l14_an_unserialisable_value_does_not_drop_the_record(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            path = crash_report.write_crash_report(tmp, [], 1, "x", lambda: {"a": {1, 2}, "p": Path("/x")})
            self.assertIsNotNone(path)
            record = json.loads(Path(path).read_text())
            self.assertIn("a", record["state"])

    def test_l14_a_circular_snapshot_degrades_to_an_error_marker(self):
        loop = {}
        loop["self"] = loop
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            path = crash_report.write_crash_report(tmp, [], 1, "x", lambda: loop)
            self.assertIsNotNone(path)
            self.assertEqual(json.loads(Path(path).read_text())["state"], {"snapshot_error": "state is not serialisable"})

    def test_l14_secrets_in_neutral_flag_values_are_redacted(self):
        args = ["llama-server", "--header", "Authorization: Bearer sk-abc", "--extra", "X-Api-Key: zzz",
                "--api-key", "k1", "--cookie=session=abc", "-m", "model.gguf"]
        shown = " ".join(crash_report._safe_args(args))
        for secret in ("sk-abc", "zzz", "k1", "session=abc"):
            self.assertNotIn(secret, shown)
        self.assertIn("model.gguf", shown)

    def test_l14_journal_lines_still_redacted(self):
        line = crash_report._redact("curl -H 'Authorization: Bearer tok123' --api-key=k2")
        self.assertNotIn("tok123", line)
        self.assertNotIn("k2", line)


if __name__ == "__main__":
    unittest.main()


class GatewayMonitorAudit(unittest.TestCase):
    """Gateway wiring of the monitor request accounting."""
    setUpClass = classmethod(fixture.GatewayGuard.setUpClass.__func__)
    tearDownClass = classmethod(fixture.GatewayGuard.tearDownClass.__func__)

    def setUp(self):
        fixture.GatewayGuard.setUp(self)
        self.patch_attr("wait_fresh_vram", Mock(return_value=True))
        self.patch_attr("event", Mock())

    patch_attr = fixture.GatewayGuard.patch_attr
    fake_llama = fixture.GatewayGuard.fake_llama

    patch_attr = fixture.GatewayGuard.patch_attr
    fake_llama = fixture.GatewayGuard.fake_llama

    def game(self, **extra):
        config = {**self.gw.G, "gameMode": True, "gameDetectNvml": True, **extra}
        self.patch_attr("G", config)
        self.patch_attr("GAME", GamePolicy(config))
        self.patch_attr("GAME_SIGNALS", GameSignals(config))


    def test_m8_deliberate_503s_are_not_counted_as_failures(self):
        monitor = Monitor()
        self.patch_attr("MONITOR", monitor)
        self.game()
        self.gw.GAME.active = True

        def finish(status, error=True):
            ident = monitor.begin_request("m", "/v1/chat/completions")
            self.gw._finish_monitor_request(SimpleNamespace(_monitor_request_id=ident), status=status, error=error)

        finish(503)                                      # game pause
        self.assertEqual(monitor.request_errors, 0)
        self.gw.GAME.active = False
        self.gw.GAME.ready = True
        self.gw.st.override = "OFF"
        finish(503)                                      # ai-off
        self.gw.st.override = "AUTO"
        self.gw.st.memory_pressure = True
        finish(503)                                      # memory pressure
        self.gw.st.memory_pressure = False
        finish(404)                                      # client error
        self.assertEqual(monitor.request_errors, 0)
        finish(503)                                      # nothing deliberate: a real outage
        finish(500)
        self.assertEqual(monitor.request_errors, 2)


if __name__ == "__main__":
    unittest.main()
