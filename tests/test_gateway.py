"""Offline guard and lifecycle tests; no GPU, model, live config or service is touched."""
import http.client
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parent.parent


class GatewayGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.import_root = tempfile.TemporaryDirectory(prefix="shura-guard-import-", dir="/tmp")
        root = Path(cls.import_root.name)
        for name in ("guardian", "model-launch"):
            (root / f"{name}.json").write_text((REPO / "config" / f"{name}.example.json").read_text())
        spec = importlib.util.spec_from_file_location("test_gateway_module", REPO / "gateway" / "ai_gateway.py")
        cls.gw = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, AI_GATEWAY_CONFIG=str(root), AI_GATEWAY_STATE=str(root / "state")):
            spec.loader.exec_module(cls.gw)
        cls.real_vram_free_gb = staticmethod(cls.gw.vram_free_gb)

    @classmethod
    def tearDownClass(cls):
        cls.import_root.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="shura-guard-test-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.disk = self.root / "disk"
        self.disk.mkdir()
        self.ram = self.root / "ram"
        self.ram.mkdir()
        self.patch_attr("SLOT_DIR", self.disk)
        self.patch_attr("RAM_SLOT_DIR", self.ram)
        self.patch_attr("st", self.gw.State())
        self.patch_attr("G", {**self.gw.G, "idleUnloadSeconds": 3600, "vramGuard": True})
        self.patch_attr("log", Mock())
        self.patch_attr("notify_vram_yield", Mock())
        self.patch_attr("mem_available_gb", Mock(return_value=24))
        self.patch_attr("vram_free_gb", Mock(return_value=12))
        self.patch_attr("process_vram_mib", Mock(return_value=10970))
        self.patch_attr("write_server_lease", Mock())
        self.patch_attr("backend_port_in_use", Mock(return_value=False))
        self.proc = Mock(pid=1234)
        self.proc.poll.return_value = None
        self.proc.terminate.side_effect = lambda: setattr(self.proc.poll, "return_value", 0)
        self.gw.st.proc = self.proc
        self.gw.st.vram_free_mib = 12 * 1024
        self.gw.st.vram_sample_at = time.time()
        self.gw.st.status = "READY"
        self.gw.st.model_id = self.gw.M["defaultModel"]
        self.gw.st.override = "AUTO"
        self.gw.st.ram_backend = True
        self.gw.st.last_request = time.time() - 91
        self.api = Mock(side_effect=self.fake_llama)
        self.patch_attr("llama_call", self.api)

    def patch_attr(self, name, value):
        p = patch.object(self.gw, name, value)
        p.start()
        self.addCleanup(p.stop)

    def fake_llama(self, method, path, body=None, timeout=10):
        if path == "/slots":
            return 200, [{"n_prompt_tokens": 187000, "n_slot_save_estimated_bytes": 2 * 1024 * 1024}]
        if "action=save" in path:
            (self.ram / body["filename"]).write_bytes(b"test checkpoint")
            return 200, {"n_saved": 187000}
        if "action=restore" in path:
            return 200, {"n_restored": 187000}
        return 200, {}

    def tick(self, now, free=100):
        self.gw.monitor_tick(now, free, 24)

    def yield_model(self):
        self.tick(10)
        self.tick(15)

    def test_idle_pressure_saves_only_in_ram_then_unloads(self):
        self.tick(10)
        self.tick(14.9)
        self.proc.terminate.assert_not_called()
        self.tick(15)
        self.proc.terminate.assert_called_once()
        self.assertEqual(self.gw.st.last_yield_reason, "vram_pressure")
        self.assertIsNotNone(self.gw.st.last_yield_at)
        self.assertEqual(list(self.disk.iterdir()), [])
        self.assertEqual({p.name for p in self.ram.iterdir()}, {"active.bin", "checkpoint.json"})
        self.assertEqual(self.gw.notify_vram_yield.call_count, 2)

    def test_busy_response_is_not_interrupted(self):
        self.gw.st.busy = 1
        self.yield_model()
        self.assertTrue(self.gw.st.vram_pressure)
        self.proc.terminate.assert_not_called()
        self.api.assert_not_called()
        self.gw.st.busy = 0
        self.gw.st.last_request = time.time() - 91
        self.tick(16)
        self.proc.terminate.assert_called_once()

    def test_recovery_before_response_finishes_cancels_yield(self):
        self.gw.st.busy = 1
        self.yield_model()
        self.tick(16, 250)
        self.gw.st.busy = 0
        self.tick(17, 250)
        self.assertFalse(self.gw.st.vram_pressure)
        self.proc.terminate.assert_not_called()

    def test_flapping_requires_a_new_full_sustained_interval(self):
        self.tick(10)
        self.tick(14, 121)
        self.tick(15)
        self.tick(19.9)
        self.proc.terminate.assert_not_called()
        self.tick(20)
        self.proc.terminate.assert_called_once()

    def test_failed_vram_reader_breaks_the_consecutive_interval(self):
        self.tick(10)
        self.tick(14, -1024)
        self.tick(15)
        self.tick(19)
        self.proc.terminate.assert_not_called()

    def test_threshold_boundary_is_not_pressure(self):
        self.tick(10, 120)
        self.tick(30, 120)
        self.proc.terminate.assert_not_called()

    def test_failed_save_keeps_the_intact_live_session(self):
        self.api.side_effect = lambda *args, **kwargs: (500, {})
        self.yield_model()
        self.proc.terminate.assert_not_called()
        self.assertGreater(self.gw.notify_vram_yield.call_count, 0)
        self.assertIn("checkpoint failed", self.gw.st.last_error)
        self.assertFalse(self.gw.st.ram_checkpoint)

    def test_ram_or_tmpfs_exhaustion_does_not_destroy_session(self):
        with patch.object(self.gw, "ram_slot_space_ok", return_value=False):
            self.yield_model()
        self.assertEqual(self.api.call_count, 1)  # slot inspection precedes space check
        self.proc.terminate.assert_not_called()

    def test_short_sessions_are_saved_for_pressure(self):
        original = self.fake_llama
        def api(m, p, *a, **k):
            result = original(m, p, *a, **k)
            if p == "/slots":
                return 200, [{"n_prompt_tokens": 50, "n_slot_save_estimated_bytes": 2 * 1024 * 1024}]
            if "action=save" in p:
                return 200, {"n_saved": 50}
            return result
        self.api.side_effect = api
        self.yield_model()
        self.assertIsNotNone(self.gw.st.ram_checkpoint)

    def test_ordinary_unload_still_saves_to_disk(self):
        self.gw.stop_llama_locked("manual unload")
        disk = self.disk / self.gw.slot_filename(self.gw.M["defaultModel"], False)
        self.assertEqual(disk.read_bytes(), b"test checkpoint")
        self.assertEqual(list(self.ram.iterdir()), [])

    def test_short_ordinary_slot_still_skips_disk(self):
        self.api.side_effect = lambda *a, **k: (200, [{"n_prompt_tokens": 50}])
        self.gw.stop_llama_locked("idle")
        self.assertEqual(list(self.disk.iterdir()), [])

    def test_ram_restore_wins_over_an_older_disk_slot(self):
        self.yield_model()
        self.gw.st.model_id = self.gw.M["defaultModel"]
        disk = self.disk / self.gw.slot_filename(self.gw.st.model_id, False)
        disk.write_bytes(b"old disk snapshot")
        self.api.reset_mock()
        self.assertEqual(self.gw.restore_slot_locked(), "")
        self.assertEqual(self.api.call_args.args[2], {"filename": "active.bin"})
        self.assertIsNone(self.gw.st.ram_checkpoint)
        self.assertEqual(disk.read_bytes(), b"old disk snapshot")
        self.assertEqual(list(self.ram.iterdir()), [])

    def test_ram_restore_failure_is_not_silent_fallback(self):
        self.yield_model()
        self.gw.st.model_id = self.gw.M["defaultModel"]
        self.api.side_effect = lambda *a, **k: (500, {})
        self.assertEqual(self.gw.restore_slot_locked(), "restart_empty")
        self.assertIsNone(self.gw.st.ram_checkpoint)
        self.assertFalse((self.ram / "active.bin").exists())
        self.assertIn("re-read", self.gw.st.guard_fallback)

    def test_text_checkpoint_is_not_restored_into_vision(self):
        self.yield_model()
        self.gw.st.model_id = self.gw.M["defaultModel"]
        self.gw.st.vision = True
        self.api.reset_mock()
        self.assertEqual(self.gw.restore_slot_locked(), "")
        self.api.assert_not_called()

    def test_old_disk_checkpoint_is_staged_in_ram_and_not_deleted(self):
        name = self.gw.slot_filename(self.gw.st.model_id, False)
        (self.disk / name).write_bytes(b"disk snapshot")
        self.assertEqual(self.gw.restore_slot_locked(), "")
        self.assertEqual(self.api.call_args.args[2], {"filename": "restore.bin"})
        self.assertEqual(list(self.ram.iterdir()), [])
        self.assertTrue((self.disk / name).exists())

    def test_ready_request_reserves_busy_before_monitor_can_unload(self):
        self.assertEqual(self.gw.acquire_model(self.gw.st.model_id, False), "")
        self.assertEqual(self.gw.st.busy, 1)
        self.yield_model()
        self.proc.terminate.assert_not_called()
        self.gw.st.busy = 0
        self.gw.st.last_request = time.time() - 91
        self.tick(16)
        self.proc.terminate.assert_called_once()

    def test_low_vram_times_out_without_starting_a_model(self):
        self.gw.st.proc = None
        self.gw.st.status = "UNLOADED"
        self.gw.G["vramReloadWaitSeconds"] = 0
        self.gw.st.vram_free_mib = 2 * 1024
        with patch.object(self.gw, "start_llama") as start:
            self.assertIn("VRAM", self.gw.acquire_model(self.gw.M["defaultModel"], False))
        start.assert_not_called()
        self.assertEqual(self.gw.st.busy, 0)

    def test_waiting_for_vram_does_not_hold_lifecycle_lock(self):
        self.gw.st.proc = None
        self.gw.st.status = "UNLOADED"
        self.gw.G["vramReloadWaitSeconds"] = 2
        self.gw.st.vram_free_mib = 2 * 1024
        self.gw.st.vram_sample_at = time.time()
        self.gw.vram_free_gb.side_effect = [2, 12]
        def sleep(_):
            self.gw.st.vram_free_mib = 12 * 1024
            self.gw.st.vram_sample_at = time.time()
            acquired = []
            def other_thread():
                with self.gw.st.lock:
                    acquired.append(True)
            t = threading.Thread(target=other_thread)
            t.start()
            t.join(timeout=1)
            self.assertEqual(acquired, [True])
        with patch.object(self.gw.time, "sleep", side_effect=sleep), patch.object(self.gw, "start_llama", return_value=""):
            self.assertEqual(self.gw.acquire_model(self.gw.M["defaultModel"], False), "")

    def test_reader_failure_uses_original_load_policy(self):
        self.gw.st.vram_free_mib = None
        self.gw.st.vram_sample_at = None
        self.assertEqual(self.gw.preload_check(self.gw.model_config(self.gw.st.model_id, False)), "")
        self.assertIn("NVML", self.gw.st.guard_fallback)

    def test_stale_nvml_fallback_clears_when_monitor_recovers(self):
        self.gw.st.vram_free_mib = None
        self.gw.st.vram_sample_at = None
        self.gw.preload_check(self.gw.model_config(self.gw.st.model_id, False))
        self.gw.monitor_tick(10, 9000, 24)
        self.assertEqual(self.gw.st.monitor_error, "")
        self.assertEqual(self.gw.st.guard_fallback, "")
        self.assertEqual(self.gw.st.event_counters["telemetry_recovered"], 1)

    def test_spawn_failure_releases_request_reservation(self):
        self.gw.st.proc = None
        self.gw.st.status = "UNLOADED"
        with patch.object(self.gw.subprocess, "Popen", side_effect=OSError("test spawn failed")):
            self.assertIn("spawn", self.gw.acquire_model(self.gw.M["defaultModel"], False))
        self.assertEqual(self.gw.st.busy, 0)
        self.assertEqual(self.gw.st.status, "UNLOADED")

    def test_unexpected_start_failure_does_not_leak_busy(self):
        self.gw.st.proc = None
        self.gw.st.status = "UNLOADED"
        with patch.object(self.gw, "start_llama", side_effect=RuntimeError("failed")), self.assertRaises(RuntimeError):
            self.gw.acquire_model(self.gw.M["defaultModel"], False)
        self.assertEqual(self.gw.st.busy, 0)

    def test_switching_vision_cannot_interrupt_an_active_response(self):
        self.gw.st.busy = 1
        error = self.gw.acquire_model(self.gw.st.model_id, True)
        self.assertIn("response is in progress", error)
        self.proc.terminate.assert_not_called()
        self.assertEqual(self.gw.st.busy, 1)

    def test_invalid_ram_path_keeps_current_answer_available(self):
        self.gw.RAM_SLOT_DIR = None
        self.gw.G["vramPressureSlotDir"] = "/proc"  # procfs stays non-tmpfs even in a RAM checkout
        self.assertEqual(self.gw.acquire_model(self.gw.st.model_id, False), "")
        self.assertEqual(self.gw.st.busy, 1)
        self.assertIn("ordinary lifecycle", self.gw.st.guard_fallback)

    def test_restore_failure_starts_one_fresh_backend_before_request(self):
        self.yield_model()
        first, second = Mock(pid=4321), Mock(pid=4322)
        for proc in (first, second):
            proc.poll.return_value = None
            proc.terminate.side_effect = lambda p=proc: setattr(p.poll, "return_value", 0)
        def api(method, path, body=None, timeout=10):
            return (200, {}) if path == "/health" else (500, {})
        with patch.object(self.gw.subprocess, "Popen", side_effect=[first, second]) as spawn, patch.object(self.gw.time, "sleep"), patch.object(self.gw, "llama_call", side_effect=api):
            self.assertEqual(self.gw.acquire_model(self.gw.M["defaultModel"], False), "")
        self.assertEqual(spawn.call_count, 2)
        first.terminate.assert_called_once()
        self.assertEqual(self.gw.st.busy, 1)
        self.assertEqual(self.gw.st.status, "READY")
        self.assertIsNone(self.gw.st.ram_checkpoint)

    def test_status_exposes_yield_fields(self):
        self.yield_model()
        status = self.gw.status_snapshot()
        self.assertTrue(status["vram_pressure"])
        self.assertEqual(status["last_yield_reason"], "vram_pressure")
        self.assertTrue(status["ram_session_saved"])

    def test_gateway_restart_finds_ram_checkpoint(self):
        self.gw.RAM_SLOT_DIR = None
        self.gw.G["vramPressureSlotDir"] = str(self.root)
        self.ram = self.gw.prepare_ram_slot_dir()
        self.yield_model()
        self.gw.RAM_SLOT_DIR = None
        self.gw.st = self.gw.State()
        self.gw.st.ram_backend = True
        self.gw.prepare_ram_slot_dir()
        self.assertEqual(self.gw.st.ram_checkpoint["tokens"], 187000)
        self.gw.st.model_id = self.gw.M["defaultModel"]
        self.assertEqual(self.gw.restore_slot_locked(), "")

    def test_incompatible_checkpoint_is_not_restored_after_restart(self):
        self.gw.RAM_SLOT_DIR = None
        self.gw.G["vramPressureSlotDir"] = str(self.root)
        self.ram = self.gw.prepare_ram_slot_dir()
        self.yield_model()
        metadata = self.ram / "checkpoint.json"
        value = json.loads(metadata.read_text())
        value["signature"] = "old config"
        metadata.write_text(json.dumps(value))
        self.gw.RAM_SLOT_DIR = None
        self.gw.st = self.gw.State()
        self.gw.st.ram_backend = True
        self.gw.prepare_ram_slot_dir()
        self.assertIsNone(self.gw.st.ram_checkpoint)

    def test_stream_finishes_before_pressure_unload(self):
        entered, finish = threading.Event(), threading.Event()
        test = self
        class StreamingHandler(self.gw.Handler):
            def proxy(self, body):
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"6\r\nfirst \r\n")
                self.wfile.flush()
                entered.set()
                if not finish.wait(timeout=3):
                    raise RuntimeError("test stream did not finish")
                self.wfile.write(b"4\r\nlast\r\n0\r\n\r\n")
                self.wfile.flush()
        server = ThreadingHTTPServer(("127.0.0.1", 0), StreamingHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        output, errors = [], []
        def client():
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            try:
                conn.request("POST", "/v1/chat/completions", body=b"{}")
                response = conn.getresponse()
                output.append((response.status, response.read()))
            except Exception as e:  # noqa: BLE001 - collect every client-side stream error for this test.
                errors.append(e)
            finally:
                conn.close()
        request = threading.Thread(target=client)
        request.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            self.assertEqual(test.gw.st.busy, 1)
            self.yield_model()
            self.proc.terminate.assert_not_called()
            finish.set()
            request.join(timeout=3)
            self.assertEqual(errors, [])
            self.assertEqual(output, [(200, b"first last")])
            # Closing the request thread releases its lifecycle reservation.
            server.shutdown()
            thread.join(timeout=2)
            self.assertEqual(self.gw.st.busy, 0)
            self.tick(16)
            self.proc.terminate.assert_not_called()
            self.gw.st.last_request = time.time() - 91
            self.tick(17)
            self.proc.terminate.assert_called_once()
        finally:
            finish.set()
            request.join(timeout=3)
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_reload_guard_accounts_for_actual_model_footprint(self):
        self.gw.G["vramFreeCriticalMiB"] = 11000  # pressure trigger must not become a load headroom
        self.yield_model()
        self.assertAlmostEqual(self.gw.st.vram_resume_min_gb, (10970 + 250) / 1024)
        self.gw.G["vramReloadWaitSeconds"] = 0
        self.gw.st.vram_free_mib = int(10.5 * 1024)
        with patch.object(self.gw, "start_llama") as start:
            self.assertIn("VRAM", self.gw.acquire_model(self.gw.M["defaultModel"], False))
        start.assert_not_called()
        self.assertGreater(self.gw.st.vram_resume_min_gb, 10.5)

    def test_unknown_process_memory_uses_preload_headroom(self):
        self.gw.st.preload_vram_gb = 11.2
        self.gw.process_vram_mib.return_value = -1
        self.yield_model()
        self.assertEqual(self.gw.st.vram_resume_min_gb, 11.2)

    def test_ram_directory_rejects_disk_and_is_private_on_tmpfs(self):
        self.gw.RAM_SLOT_DIR = None
        self.gw.G["vramPressureSlotDir"] = "/proc"  # procfs stays non-tmpfs even in a RAM checkout
        with self.assertRaisesRegex(RuntimeError, "tmpfs"):
            self.gw.prepare_ram_slot_dir()
        self.gw.G["vramPressureSlotDir"] = str(self.root)
        ram = self.gw.prepare_ram_slot_dir()
        self.assertEqual(ram.stat().st_mode & 0o777, 0o700)
        self.assertTrue(ram.is_relative_to(self.root))


if __name__ == "__main__":
    unittest.main()
