import json
import io
import re
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
import monitor
from response_recovery import forward


class MonitorParsers(unittest.TestCase):
    def test_memory_psi_zram_and_cpu_from_fixture_roots(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            root = Path(tmp)
            proc = root / "proc"
            sysroot = root / "sys"
            (proc / "pressure").mkdir(parents=True)
            (proc / "meminfo").write_text("MemTotal: 8192000 kB\nMemAvailable: 4096000 kB\n")
            (proc / "pressure/memory").write_text("some avg10=2.50 avg60=1.0 avg300=0.2 total=1\nfull avg10=0.25 avg60=0.1 avg300=0.0 total=1\n")
            (proc / "stat").write_text("cpu 1 2 3 4 5 6 7 8 9 10\ncpu0 10 0 5 80 5 0 0 0 0 0\ncpu1 10 0 5 80 5 0 0 0 0 0\n")
            zram = sysroot / "block/zram0"
            zram.mkdir(parents=True)
            (zram / "mm_stat").write_text("2097152 1048576 524288 0 0 0 0\n")
            (zram / "disksize").write_text(str(1048576 * 1024))
            for cpu in ("cpu0", "cpu1"):
                freq = sysroot / f"devices/system/cpu/{cpu}/cpufreq"
                freq.mkdir(parents=True)
                (freq / "scaling_cur_freq").write_text("3500000\n")
            hwmon = sysroot / "class/hwmon/hwmon0"
            hwmon.mkdir(parents=True)
            (hwmon / "name").write_text("k10temp\n")
            (hwmon / "temp1_input").write_text("72000\n")
            (hwmon / "temp1_label").write_text("Tctl\n")
            nvme = sysroot / "block/nvme0n1"
            nvme.mkdir(parents=True)
            (nvme / "stat").write_text("0 0 0 0 0 0 1024 0\n")
            self.assertEqual(monitor._read_meminfo(proc), (8192000, 4096000))
            self.assertEqual(monitor._read_psi(proc), {"some": 2.5, "full": 0.25})
            z = monitor._read_zram(sysroot)
            self.assertEqual(z["used_bytes"], 524288)
            # Fill is uncompressed data over disksize (2 MiB of 1 GiB), not the compressed footprint.
            self.assertAlmostEqual(z["used_percent"], 0.1953125)
            previous = {0: [8, 0, 4, 75, 5, 0, 0, 0, 0, 0], 1: [8, 0, 4, 75, 5, 0, 0, 0, 0, 0]}
            current, cores, freq = monitor._read_cpu(proc, sysroot, previous)
            self.assertEqual(len(current), 2)
            self.assertAlmostEqual(cores[0]["load_percent"], 37.5)
            self.assertEqual(freq, 3500.0)
            self.assertEqual(monitor._cpu_temperature(sysroot), 72.0)
            self.assertEqual(monitor._nvme_written(sysroot), 1024 * 512)

    def test_absent_sensors_are_unavailable_not_invented(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            root = Path(tmp)
            self.assertIsNone(monitor._read_meminfo(root)[0])
            self.assertIsNone(monitor._read_psi(root).get("full"))
            self.assertIsNone(monitor._read_zram(root)["used_percent"])

    def test_zram_aggregation_skips_a_device_until_all_fields_are_read(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            root = Path(tmp)
            disk = 2 * 1024**3
            good = root / "block/zram0"
            good.mkdir(parents=True)
            (good / "mm_stat").write_text(f"{800 * 1024**2} {300 * 1024**2} {500 * 1024**2} 0 0 0 0\n")
            (good / "disksize").write_text(str(disk))
            partial = root / "block/zram1"
            partial.mkdir(parents=True)
            (partial / "mm_stat").write_text(f"{500 * 1024**2} {250 * 1024**2} {300 * 1024**2} 0 0 0 0\n")
            (partial / "disksize").write_text("bogus\n")
            malformed = root / "block/zram2"
            malformed.mkdir(parents=True)
            (malformed / "mm_stat").write_text(f"{500 * 1024**2} invalid {300 * 1024**2} 0 0 0 0\n")
            (malformed / "disksize").write_text(str(disk))

            value = monitor._read_zram(root)

            self.assertEqual(value["original_bytes"], 800 * 1024**2)
            self.assertEqual(value["compressed_bytes"], 300 * 1024**2)
            self.assertEqual(value["used_bytes"], 500 * 1024**2)
            self.assertEqual(value["disk_bytes"], disk)
            self.assertAlmostEqual(value["used_percent"], 800 / 2048 * 100)


class MonitorRequests(unittest.TestCase):
    def test_sse_split_across_chunks_extracts_only_metadata(self):
        clock = [100.0]
        m = monitor.Monitor(wall_clock=lambda: clock[0], monotonic=lambda: clock[0])
        ident = m.begin_request("tiel-coder", "/v1/chat/completions?secret=x")
        self.assertEqual(ident, 1)
        clock[0] += 0.15
        m.observe_sse(ident, b'data: {"choices":[{"delta":{"content":"private answer"}}]}\n\n')
        clock[0] += 0.25
        m.observe_sse(ident, b'data: {"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":12,"completion_tokens":8,"prompt_tokens_details":{"cached_tokens":3}},"timings":{"prompt_per_second":40,"predicted_per_second":20,"draft_n":10,"draft_n_accepted":7}}\n\ndata: [DONE]\n\n')
        m.finish_request(ident, status=200)
        row = m.requests()[0]
        self.assertEqual(row["endpoint"], "/v1/chat/completions")
        self.assertEqual(row["prompt_tokens"], 12)
        self.assertEqual(row["completion_tokens"], 8)
        self.assertEqual(row["finish_reason"], "length")
        self.assertEqual(row["draft_acceptance_percent"], 70)
        self.assertEqual(row["prefix_reuse_percent"], 25)
        self.assertAlmostEqual(row["ttft_seconds"], 0.15)
        self.assertNotIn("private answer", json.dumps(row))
        self.assertNotIn("secret", json.dumps(row))

    def test_non_stream_json_does_not_claim_full_response_time_as_ttft(self):
        m = monitor.Monitor()
        ident = m.begin_request("occamy", "/v1/chat/completions")
        m.observe_json(ident, b'{"choices":[{"message":{"content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1}}')
        m.finish_request(ident, status=200)
        self.assertIsNone(m.requests()[0]["ttft_seconds"])

    def test_request_model_can_be_resolved_after_routing(self):
        m = monitor.Monitor()
        ident = m.begin_request("default", "/v1/chat/completions")
        m.set_request_model(ident, "occamy")
        m.finish_request(ident, status=200)
        self.assertEqual(m.requests()[0]["model"], "occamy")

    def test_system_sampling_does_not_hold_the_stream_metadata_lock(self):
        m = monitor.Monitor()
        ident = m.begin_request("tiel", "/v1/chat/completions")
        started = threading.Event()
        release = threading.Event()

        def slow_sample():
            started.set()
            release.wait(1)
            return {}

        thread = threading.Thread(target=lambda: m.get_or_sample(slow_sample))
        thread.start()
        self.assertTrue(started.wait(1))
        before = time.monotonic()
        m.observe_sse(ident, b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
        self.assertLess(time.monotonic() - before, 0.1)
        release.set()
        thread.join(1)
        self.assertFalse(thread.is_alive())

    def test_request_history_is_bounded_and_newest_first(self):
        m = monitor.Monitor()
        for i in range(105):
            ident = m.begin_request("tiel", "/v1/completions")
            m.finish_request(ident, status=200)
        rows = m.requests()
        self.assertEqual(len(rows), 100)

    def test_event_history_masks_credentials_and_is_bounded(self):
        m = monitor.Monitor()
        for i in range(35):
            m.add_event({"reason": "server_started", "detail": f"api_key=secret{i}"})
        events = m.events()
        self.assertEqual(len(events), 30)
        self.assertNotIn("secret", json.dumps(events))


class MonitorState(unittest.TestCase):
    def test_page_shell_allows_auth_prompt_but_api_remains_protected(self):
        class Handler:
            command = "GET"
            def __init__(self):
                self.status = None
                self.headers = {}
                self.wfile = io.BytesIO()
            def send_response(self, code): self.status = code
            def send_header(self, key, value): self.headers[key] = value
            def end_headers(self): pass
            def send_json(self, code, payload): self.status = code
        page = Handler()
        self.assertTrue(monitor.monitor_route(page, "/monitor", lambda: {}, lambda: [], authorized=False))
        self.assertEqual(page.status, 200)
        self.assertIn("Shura Monitor", page.wfile.getvalue().decode())
        api = Handler()
        self.assertTrue(monitor.monitor_route(api, "/monitor/api/state", lambda: {}, lambda: [], authorized=False))
        self.assertEqual(api.status, 401)
        self.assertEqual(api.headers.get("WWW-Authenticate"), "Bearer")

    def test_unloaded_state_has_zero_context_but_unavailable_gpu(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            m = monitor.Monitor(proc_root=Path(tmp) / "proc", sys_root=Path(tmp) / "sys", home=tmp)
            state = m.snapshot(status={"status": "UNLOADED", "default_model": "tiel-coder", "llama_pid": None},
                               state=None, model_config={"ctxSize": 200000}, gpu={"free_mib": 9000, "used_mib": 3000},
                               slots=None)
            self.assertEqual(state["context"]["used_tokens"], 0)
            self.assertEqual(state["context"]["used_percent"], 0.0)
            self.assertEqual(state["gpu"]["model_process_mib"], None)
            self.assertEqual(state["gpu"]["other_processes_mib"], 3000)
            self.assertEqual(state["header"]["state"], "UNLOADED")

    def test_active_request_is_not_idle_while_prefill_has_no_sse_chunk_yet(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            m = monitor.Monitor(proc_root=Path(tmp) / "proc", sys_root=Path(tmp) / "sys", home=tmp)
            ident = m.begin_request("tiel-coder", "/v1/chat/completions")
            state = m.snapshot(status={"status": "READY", "model": "tiel-coder", "llama_pid": 123},
                               state=None, model_config={}, gpu={}, slots=None)
            self.assertEqual(state["now"]["request_number"], ident)
            self.assertTrue(state["now"]["generating"])
            self.assertEqual(state["now"]["live_tokens_per_second"], 0.0)

    def test_schema_capabilities_health_status_and_monitor_cost_are_additive(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            root = Path(tmp)
            m = monitor.Monitor(proc_root=root / "proc", sys_root=root / "sys", home=tmp)
            state = m.snapshot(status={"status": "READY", "model": "tiel", "llama_pid": None},
                state=None, model_config={"ctxSize": 200000},
                gpu={"source": "gpu_amd", "total_mib": 8192, "free_mib": 300, "used_mib": 7892, "temperature_c": 84},
                slots=[{"id": 0, "n_ctx": 187000, "n_past": 180000}], expert_cache={"slots": 26, "hit_percent": 50},
                gpu_capabilities={"gpu_amd": True})
            self.assertEqual(state["schema"], 1)
            self.assertTrue(state["capabilities"]["gpu_amd"])
            self.assertTrue(state["capabilities"]["slots"])
            self.assertTrue(state["capabilities"]["expert_cache"])
            self.assertEqual(state["context"]["limit_tokens"], 187000)
            self.assertEqual(state["health"]["level"], "crit")
            self.assertEqual(state["health"]["reasons"][0]["code"], "gpu_critical")
            self.assertEqual(state["status_line"]["code"], "neutral")
            self.assertIn("stream_subscribers", state["monitor_self"])

    def test_health_ignores_sources_that_are_not_available(self):
        state = monitor._health({"gpu": "warn", "memory": "warn", "cpu": "warn", "reliability": "ok", "context": "warn"},
            {"gpu_nvidia": False, "gpu_amd": False, "gpu_intel": False, "system_linux": False, "zram": False, "psi": False, "slots": False},
            {"ram_available_mib": None, "zram_used_percent": None, "psi_full_avg10": None, "cpu_temperature_c": None,
             "ssd_free_gb": None, "context_percent": None, "reliability_available": False})
        self.assertEqual(state, {"level": "ok", "reasons": [], "hint": "No action needed."})

    def test_windows_system_adapter_returns_schema_with_unavailable_metrics(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            m = monitor.Monitor(proc_root=Path(tmp) / "proc", sys_root=Path(tmp) / "sys", home=tmp)
            m.system_source = monitor.WindowsSystemSource()
            state = m.snapshot(status={"status": "UNLOADED", "default_model": "model", "llama_pid": None},
                               state=None, model_config={}, gpu={})
            self.assertEqual(state["schema"], 1)
            self.assertFalse(state["capabilities"]["system_linux"])
            self.assertFalse(state["capabilities"]["zram"])
            self.assertIsNone(state["memory"]["ram_available_mib"])
            self.assertEqual(state["health"]["level"], "ok")

    def test_gateway_status_line_only_names_observed_conditions(self):
        self.assertEqual(monitor._status_line({}, True, 0, None, None, {}, [])["code"], "waiting_first_token")
        self.assertEqual(monitor._status_line({}, True, 25, None, None, {}, [])["code"], "generating")
        polite = {"enabled": True, "other_busy_cpu_threshold": 2.5, "sample": {"other_busy_cpus": 3.0}}
        self.assertEqual(monitor._status_line({}, False, 0, None, polite, {}, [])["code"], "polite_busy")
        self.assertEqual(monitor._status_line({}, False, 0, None, None, {}, [])["code"], "neutral")

    def test_sse_stream_counts_subscribers_and_recovers_after_disconnect(self):
        m = monitor.Monitor()
        sent = []
        calls = []
        def state():
            calls.append(1)
            return {"seq": len(calls), "subscribers": m.stream_subscribers}
        def send(item):
            sent.append(item)
            if len(sent) == 3:
                raise BrokenPipeError("client disconnected")
        m.stream(state, send, sleep=lambda _seconds: None)
        self.assertEqual([x["seq"] for x in sent], [1, 2, 3])
        self.assertTrue(all(x["subscribers"] == 1 for x in sent))
        self.assertEqual(m.stream_subscribers, 0)
        ident = m.begin_request("tiel", "/v1/chat/completions")
        self.assertIsNotNone(ident)

    def test_no_sse_subscriber_means_no_stream_sampling(self):
        m = monitor.Monitor()
        samples = []
        self.assertEqual(m.stream_subscribers, 0)
        self.assertEqual(samples, [])

    def test_sse_route_emits_same_json_and_uses_subscriber(self):
        class Handler:
            command = "GET"
            def __init__(self):
                self.headers = {}
                self.status = None
                self.wfile = io.BytesIO()
            def send_response(self, code): self.status = code
            def send_header(self, key, value): self.headers[key] = value
            def end_headers(self): pass
            def send_json(self, code, value): self.status = code
        m = monitor.Monitor()
        def one_event(provider, send):
            send(provider())
        m.stream = one_event
        handler = Handler()
        state = {"schema": 1, "health": {"level": "ok"}}
        self.assertTrue(monitor.monitor_route(handler, "/monitor/api/stream", lambda: state, lambda: [], monitor_instance=m))
        self.assertEqual(handler.status, 200)
        self.assertEqual(handler.headers["Content-Type"], "text/event-stream; charset=utf-8")
        self.assertEqual(json.loads(handler.wfile.getvalue().decode().split(": ", 1)[1]), state)

    def test_expert_metrics_only_report_explicit_values(self):
        class Reply:
            status = 200
            def read(self, _):
                return b"# HELP llama_expert_cache_hits\nllama_expert_cache_hits 7\nllama_expert_cache_misses 3\nllama_expert_cache_slots 26\n"
        class Conn:
            def __init__(self, *args, **kwargs):
                pass
            def request(self, *args, **kwargs):
                pass
            def getresponse(self):
                return Reply()
            def close(self):
                pass
        with patch.object(monitor.http.client, "HTTPConnection", Conn):
            self.assertEqual(monitor.read_expert_cache_metrics("127.0.0.1", 1), {"slots": 26, "hit_percent": 70})

    def test_html_is_self_contained_and_preview_data_is_marked_demo(self):
        demo = monitor.demo_data()
        html = monitor.render_page(demo)
        self.assertIn("DEMO DATA", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("http://", html)
        self.assertIn("setInterval(pollState,interval)", html)
        self.assertIn("new URLSearchParams(location.search).get('view')==='compact'", html)
        self.assertIn("sessionStorage.setItem(AUTH_KEY,value)", html)
        self.assertIn("body.is-offline.compact #card-speed", html)
        self.assertIn("document.title=compact?T.compact_title:T.page_title", html)
        self.assertIn("if(compact||document.hidden||DEMO)return", html)
        self.assertIn("/monitor/api/requests", html)
        self.assertIn("/monitor/api/stream", html)
        self.assertIn("$('health').title=d.health?.hint||T.health_tip", html)
        self.assertIn("serverState('context')", html)
        self.assertIn("const worst=['ok','warn','crit'].includes(d.health?.level)?d.health.level:'warn'", html)
        self.assertNotIn("const all=[bs.gpu", html)
        self.assertIn("caps.gpu_nvidia||caps.gpu_amd||caps.gpu_intel", html)
        self.assertIn("monitor-self", html)
        self.assertIn("stream_subscribers", html)
        self.assertIn("prefers-color-scheme:light", html)
        self.assertIn('<span id="updated"></span> · <span id="footer"></span>', html)
        self.assertNotIn('id="updated"></span> · <span id="refresh"></span> ·', html)
        self.assertIn('data-scenario="high"', html)
        self.assertIn('data-scenario="unloaded"', html)
        self.assertIn("speedHistory.length>120", html)
        self.assertEqual(monitor.THRESHOLDS["context_warn_percent"], 80)
        self.assertEqual(monitor.THRESHOLDS["context_crit_percent"], 92)
        self.assertEqual(monitor.THRESHOLDS["cpu_core_warn_percent"], 60)
        self.assertEqual(monitor.THRESHOLDS["cpu_core_crit_percent"], 85)
        self.assertIn('id="card-vram"><div class="card-head"><h2 class="eyebrow" data-text="vram"', monitor.PAGE)
        self.assertIn("ctxState=serverState('context')", monitor.PAGE)
        self.assertIn("load>LIMITS.cpu_core_warn_percent", monitor.PAGE)
        self.assertIn("load>LIMITS.cpu_core_crit_percent", monitor.PAGE)
        self.assertIn('#system-meters>#card-reliability{grid-column:span 2}', monitor.PAGE)
        meter_start = monitor.PAGE.index('<section class="meter-grid"')
        meter_end = monitor.PAGE.index('</section>', meter_start)
        self.assertLess(meter_start, monitor.PAGE.index('id="card-cpu"', meter_start))
        self.assertLess(monitor.PAGE.index('id="card-cpu"', meter_start), monitor.PAGE.index('id="card-reliability"', meter_start))
        self.assertLess(monitor.PAGE.index('id="card-reliability"', meter_start), meter_end)
        self.assertLess(meter_end, monitor.PAGE.index('<section class="feeds"', meter_end))
        self.assertEqual(demo["scenarios"]["high"]["state"]["gpu"]["free_mib"], 300)
        self.assertEqual(demo["scenarios"]["high"]["state"]["gpu"]["temperature_c"], 84)
        self.assertEqual(demo["scenarios"]["high"]["state"]["memory"]["zram_used_percent"], 90)
        self.assertEqual(demo["scenarios"]["high"]["state"]["memory"]["psi_full_avg10"], 12)
        self.assertNotEqual(demo["scenarios"]["normal"]["state"]["cpu"]["cores"], demo["scenarios"]["high"]["state"]["cpu"]["cores"])
        self.assertNotEqual(demo["scenarios"]["normal"]["state"]["context"]["used_percent"], demo["scenarios"]["high"]["state"]["context"]["used_percent"])
        self.assertIn(".fill{display:block;", monitor.PAGE)
        self.assertIn(".core-bar{display:block;width:100%;", monitor.PAGE)
        self.assertTrue(set(re.findall(r'data-text="([^"]+)"', monitor.PAGE)) <= monitor.TEXTS.keys())
        self.assertTrue(set(re.findall(r'data-unit="([^"]+)"', monitor.PAGE)) <= monitor.TEXTS.keys())

    def test_telemetry_callback_failure_cannot_interrupt_streaming(self):
        reply = Mock(status=200)
        reply.getheader.return_value = "text/event-stream"
        reply.getheaders.return_value = [("Content-Type", "text/event-stream")]
        reply.read1.side_effect = [b"data: [DONE]\n\n", b""]
        conn = Mock()
        conn.getresponse.return_value = reply
        handler = Mock(command="POST", path="/v1/chat/completions", headers={}, hop_by_hop=set())
        handler.wfile = io.BytesIO()

        def broken_observer(*_args):
            raise RuntimeError("synthetic telemetry failure")

        forward(handler, b"{}", lambda: conn, lambda: False, observe=broken_observer)
        self.assertIn(b"[DONE]", handler.wfile.getvalue())


if __name__ == "__main__":
    unittest.main()
