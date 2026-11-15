"""Model lifecycle: loading, restoring a saved session, crashes, hangs and shutdown.

The black-box tests run the real gateway against the fake llama-server; the in-process tests exercise single functions with the
same fixtures the older gateway tests use.
"""
import os
import signal
import socket
import sys
import json
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "support"))
sys.path.insert(0, str(HERE.parent / "gateway"))
from gateway_harness import GatewayHarness  # noqa: E402
import test_gateway as baseline  # noqa: E402

FULL = {"slotSaveCheckpoints": 1}


class Loading(unittest.TestCase):
    def test_the_status_answers_during_a_load_and_says_loading(self):
        with GatewayHarness(behaviour={"load_seconds": 5, "n_tokens": 3}, guardian=FULL) as gw:
            worker = threading.Thread(target=lambda: gw.chat("x"))
            worker.start()
            gw.wait_for(lambda: gw.proc and gw.fake_cfg.exists() and gw.pids_on(gw.llama_port), what="the server to start")
            started = time.monotonic()
            status = gw.status()
            self.assertLess(time.monotonic() - started, 1.5)          # it used to block for the whole load
            self.assertEqual(status["status"], "LOADING")
            self.assertIsNotNone(status["loading_seconds"])
            self.assertEqual(gw.request("GET", "/monitor/api/state", timeout=5)[0], 200)
            worker.join(30)
            self.assertEqual(gw.status()["status"], "READY")

    def test_a_refused_switch_off_stops_a_load_that_is_in_progress(self):
        with GatewayHarness(behaviour={"load_seconds": 20, "n_tokens": 3}, guardian={**FULL, "healthCheckTimeoutSeconds": 60}) as gw:
            worker = threading.Thread(target=lambda: gw.chat("x"))
            worker.start()
            gw.wait_for(lambda: gw.pids_on(gw.llama_port), what="the server to start")
            started = time.monotonic()
            self.assertEqual(gw.request("POST", "/guardian/ai-off")[0], 200)
            self.assertLess(time.monotonic() - started, 8)
            worker.join(10)
            gw.wait_for(lambda: not gw.pids_on(gw.llama_port), what="the half-loaded server to be stopped")
            self.assertEqual(gw.status()["override"], "OFF")


class SavedSessions(unittest.TestCase):
    def saved_session(self, gw):
        """A session of 30000 tokens, saved by the gateway the normal way (unload)."""
        self.assertEqual(gw.chat("x")[0], 200)
        self.assertEqual(gw.request("POST", "/guardian/unload")[0], 200)
        slots = sorted(p.name for p in (gw.tmp / "slots").iterdir())
        self.assertEqual(len(slots), 2, slots)                         # <model>-text.bin and its sidecar
        return gw.tmp / "slots"

    def test_a_session_the_server_refuses_costs_one_restart_and_never_blocks_the_model(self):
        # The old code restarted, then refused the second load because telemetry had gone stale during the first.
        guardian = {**FULL, "slotSaveMinTokens": 1000, "vramTelemetryStaleSeconds": 1}
        with GatewayHarness(behaviour={"n_tokens": 3, "prompt_tokens": 30000, "load_seconds": 2.5}, guardian=guardian) as gw:
            slots = self.saved_session(gw)
            gw.set_behaviour(restore_status=500)
            gw.request("GET", "/guardian/status")
            # The restore call is refused by the (restarted) fake as well, so make the first server refuse and the second accept.
            starts_before = gw.log().count("Starting llama-server")
            code, _ = gw.chat("again")
            self.assertEqual(code, 200, gw.log()[-1500:])
            self.assertEqual(gw.status()["status"], "READY")
            self.assertGreaterEqual(gw.log().count("Starting llama-server") - starts_before, 1)
            self.assertEqual([p for p in slots.iterdir() if p.suffix == ".bin"], [])   # the refused session was discarded

    def test_a_session_file_the_gateway_cannot_verify_is_dropped_without_a_restart(self):
        with GatewayHarness(behaviour={"n_tokens": 3, "load_seconds": 1}, guardian=FULL) as gw:
            slots = gw.tmp / "slots"
            slots.mkdir(exist_ok=True)
            (slots / "tiel-coder-text.bin").write_bytes(b"left by an older gateway: no sidecar")
            self.assertEqual(gw.chat("x")[0], 200)
            self.assertEqual(gw.log().count("Starting llama-server"), 1)
            self.assertFalse((slots / "tiel-coder-text.bin").exists())

    def test_an_unchanged_session_is_not_written_to_disk_again(self):
        guardian = {**FULL, "slotSaveMinTokens": 1000}
        with GatewayHarness(behaviour={"n_tokens": 3, "prompt_tokens": 30000}, guardian=guardian) as gw:
            slots = self.saved_session(gw)
            first = {p.name: p.stat().st_mtime_ns for p in slots.iterdir()}
            self.assertEqual(gw.request("GET", "/v1/models")[0], 200)
            time.sleep(0.05)
            # restore (a request that only reads state would not even load) and unload without generating anything
            self.assertEqual(gw.chat("restore")[0], 200)
            gw.request("POST", "/guardian/unload")
            self.assertEqual({p.name for p in slots.iterdir()}, set(first))

    def test_leftovers_of_an_interrupted_save_are_swept_at_start(self):
        harness = GatewayHarness(guardian=FULL)
        slots = harness.tmp / "slots"
        slots.mkdir()
        (slots / "tiel-coder-text.bin.pending").write_bytes(b"half")
        (slots / ".slot-abc123").write_bytes(b"half")
        with harness as gw:
            gw.status()
            self.assertEqual(sorted(p.name for p in slots.iterdir()), [])

    def test_state_and_slot_directories_are_private(self):
        with GatewayHarness(behaviour={"n_tokens": 3}, guardian=FULL) as gw:
            gw.chat("x")
            for directory in (gw.tmp / "state", gw.tmp / "slots"):
                self.assertEqual(directory.stat().st_mode & 0o777, 0o700, directory)


class Crashes(unittest.TestCase):
    def test_a_model_that_dies_on_every_answer_ends_in_error_instead_of_restarting_for_ever(self):
        guardian = {**FULL, "crashRestartMaxAttempts": 3, "errorCooldownSeconds": 0}
        with GatewayHarness(behaviour={"n_tokens": 30, "token_delay_ms": 2, "crash_after_tokens": 2}, guardian=guardian) as gw:
            for _ in range(8):
                gw.chat("x", stream=True)
                if gw.status()["status"] == "ERROR":
                    break
                time.sleep(0.3)
            status = gw.status()
            self.assertEqual(status["status"], "ERROR", status)
            self.assertGreaterEqual(status["crash_count"], 3)
            self.assertEqual(gw.chat("x")[0], 503)                      # refused with the reason, no new server
            self.assertEqual(gw.request("POST", "/guardian/unload")[0], 200)
            self.assertEqual(gw.status()["status"], "ERROR")            # an unrelated stop does not clear the error
            self.assertEqual(gw.request("POST", "/guardian/ai-on")[0], 200)
            self.assertEqual(gw.status()["status"], "UNLOADED")         # only ai-on does

    def test_the_error_state_ends_by_itself_after_the_cooldown(self):
        guardian = {**FULL, "crashRestartMaxAttempts": 2, "errorCooldownSeconds": 3}
        with GatewayHarness(behaviour={"n_tokens": 30, "token_delay_ms": 2, "crash_after_tokens": 2}, guardian=guardian) as gw:
            for _ in range(6):
                gw.chat("x", stream=True)
                if gw.status()["status"] == "ERROR":
                    break
                time.sleep(0.3)
            self.assertEqual(gw.status()["status"], "ERROR")
            gw.wait_for(lambda: gw.status()["status"] == "UNLOADED", timeout=10, what="the cooldown")
            gw.set_behaviour(crash_after_tokens=None)
            self.assertEqual(gw.chat("x")[0], 200)

    def test_a_server_that_never_gets_healthy_counts_as_failing(self):
        guardian = {**FULL, "healthCheckTimeoutSeconds": 2, "crashRestartMaxAttempts": 2, "errorCooldownSeconds": 0}
        with GatewayHarness(behaviour={"health_status": 500}, guardian=guardian) as gw:
            for _ in range(3):
                gw.chat("x")
                if gw.status()["status"] == "ERROR":
                    break
            self.assertEqual(gw.status()["status"], "ERROR")

    def test_a_server_that_stops_answering_health_is_declared_hung_and_replaced(self):
        guardian = {**FULL, "healthProbeSeconds": 1, "healthProbeFailures": 2}
        with GatewayHarness(behaviour={"n_tokens": 3}, guardian=guardian) as gw:
            self.assertEqual(gw.chat("x")[0], 200)
            first = gw.status()["llama_pid"]
            gw.set_behaviour(health_status=500)
            gw.wait_for(lambda: gw.status()["llama_pid"] is None, timeout=15, what="the hung server to be stopped")
            self.assertIn("hung", gw.status()["last_error"])
            gw.set_behaviour(health_status=None)
            self.assertEqual(gw.chat("x")[0], 200)
            self.assertNotEqual(gw.status()["llama_pid"], first)

    def test_an_orphaned_server_of_a_dead_gateway_is_taken_over(self):
        with GatewayHarness(behaviour={"n_tokens": 3}, guardian=FULL) as gw:
            self.assertEqual(gw.chat("x")[0], 200)
            orphan = gw.status()["llama_pid"]
            os.kill(gw.proc.pid, signal.SIGKILL)                       # the gateway dies, its server lives on
            gw.proc.wait(10)
            time.sleep(0.3)
            self.assertTrue(gw.pids_on(gw.llama_port))
            gw.start()
            self.assertEqual(gw.chat("again")[0], 200, gw.log()[-1500:])
            self.assertNotEqual(gw.status()["llama_pid"], orphan)
            gw.wait_for(lambda: gw.pids_on(gw.llama_port) == [gw.status()["llama_pid"]], what="only the new server to listen")


class DeadButNotGone(unittest.TestCase):
    """A server killed with a large GPU footprint closes its socket at once but is reaped only after the driver let go of the memory."""

    def vanish(self, gw, seconds):
        gw.request("POST", "/__fake/vanish", {"seconds": seconds}, port=gw.llama_port)
        gw.wait_for(lambda: not gw.pids_on(gw.llama_port), timeout=5, what="the listening socket to close")

    def test_the_next_request_waits_for_the_dying_server_and_restarts_it(self):
        for guardian in (FULL, {"slotSaveCheckpoints": 0}):
            with GatewayHarness(behaviour={"n_tokens": 3}, guardian=guardian) as gw:
                self.assertEqual(gw.chat("warm")[0], 200)
                first = gw.status()["llama_pid"]
                self.vanish(gw, 3)
                started = time.monotonic()
                code, body = gw.chat("again")
                self.assertEqual(code, 200, (body, gw.log()[-800:]))             # not a 502 into a refused connection
                self.assertGreaterEqual(time.monotonic() - started, 2)            # it really waited for the old process to be gone
                status = gw.status()
                self.assertNotEqual(status["llama_pid"], first)
                self.assertEqual(status["crash_count"], 1)                       # and the failure was counted

    def test_a_server_that_stops_listening_but_never_exits_is_declared_hung(self):
        with GatewayHarness(behaviour={"n_tokens": 3}, guardian={**FULL, "deadBackendWaitSeconds": 2}) as gw:
            self.assertEqual(gw.chat("warm")[0], 200)
            pid = gw.status()["llama_pid"]
            self.vanish(gw, 60)
            started = time.monotonic()
            code, _ = gw.chat("again")
            self.assertEqual(code, 200, gw.log()[-800:])
            self.assertLess(time.monotonic() - started, 20)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)                                                   # the lingering process was killed, not left behind
            self.assertIn("hung", gw.log())

    def test_a_stopped_server_is_killed_at_once_not_after_a_sigterm_that_cannot_land(self):
        with GatewayHarness(behaviour={"n_tokens": 400, "token_delay_ms": 20}, guardian={**FULL, "upstreamIdleSeconds": 2}) as gw:
            self.assertEqual(gw.chat("warm", stream=False)[0], 200)
            pid = gw.status()["llama_pid"]
            result = {}

            def client():
                result["r"] = gw.chat("x", stream=True)
            worker = threading.Thread(target=client)
            worker.start()
            gw.wait_for(lambda: gw.status()["busy_requests"] == 1, what="the answer in progress")
            time.sleep(0.5)
            os.kill(pid, signal.SIGSTOP)
            worker.join(30)
            self.assertIn("error", " ".join(result["r"][1]["lines"]))
            started = time.monotonic()
            for _ in range(80):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except ProcessLookupError:
                    break
            self.assertLess(time.monotonic() - started, 5, "the stopped server was still there")
            self.assertEqual(gw.chat("again")[0], 200)


class OrphanAtStart(unittest.TestCase):
    def test_a_server_left_by_a_dead_gateway_is_stopped_when_the_new_gateway_starts_not_at_the_first_request(self):
        for guardian in (FULL, {"slotSaveCheckpoints": 0}):
            with GatewayHarness(behaviour={"n_tokens": 3}, guardian=guardian) as gw:
                self.assertEqual(gw.chat("x")[0], 200)
                orphan = gw.status()["llama_pid"]
                os.kill(gw.proc.pid, signal.SIGKILL)
                gw.proc.wait(10)
                time.sleep(0.3)
                self.assertTrue(gw.pids_on(gw.llama_port))
                gw.start()                                                  # no request is sent
                gw.wait_for(lambda: not gw.pids_on(gw.llama_port), timeout=15, what="the orphan to be stopped by the new gateway")
                with self.assertRaises(ProcessLookupError):
                    os.kill(orphan, 0)
                self.assertIn("orphan_recovered", gw.log())

    def test_the_memory_check_cannot_stand_between_a_request_and_the_orphan(self):
        # the live machine refuses a load with less than ramFreeMinGBToLoad free, and the orphan holds that memory
        with GatewayHarness(behaviour={"n_tokens": 3}, guardian={**FULL, "ramFreeMinGBToLoad": 0}) as gw:
            self.assertEqual(gw.chat("x")[0], 200)
            os.kill(gw.proc.pid, signal.SIGKILL)
            gw.proc.wait(10)
            gw.start()
            gw.wait_for(lambda: not gw.pids_on(gw.llama_port), timeout=15, what="the orphan to be stopped")
            self.assertEqual(gw.chat("again")[0], 200)

    def test_the_llama_server_cors_option_is_one_origin_not_a_list(self):
        with GatewayHarness(behaviour={"n_tokens": 3}, guardian=FULL) as gw:
            gw.chat("x")
            argv = gw.request("GET", "/__fake/args", port=gw.llama_port)[1]
            value = argv[argv.index("--cors-origins") + 1]
            self.assertEqual(value, f"http://127.0.0.1:{gw.llama_port}")
            self.assertNotIn(",", value)


class Shutdown(unittest.TestCase):
    def test_a_stop_signal_during_a_load_ends_the_load_and_leaves_no_server_behind(self):
        with GatewayHarness(behaviour={"load_seconds": 30, "n_tokens": 3}, guardian={**FULL, "healthCheckTimeoutSeconds": 120}) as gw:
            worker = threading.Thread(target=lambda: gw.chat("x"), daemon=True)
            worker.start()
            gw.wait_for(lambda: gw.pids_on(gw.llama_port), what="the server to start")
            started = time.monotonic()
            os.killpg(gw.proc.pid, signal.SIGTERM)
            gw.proc.wait(20)
            self.assertLess(time.monotonic() - started, 15)             # systemd gives 60 s; the old code waited for the whole load
            self.assertEqual(gw.pids_on(gw.llama_port), [])

    def test_unload_while_an_answer_is_running_asks_first(self):
        with GatewayHarness(behaviour={"n_tokens": 200, "token_delay_ms": 20}, guardian=FULL) as gw:
            worker = threading.Thread(target=lambda: gw.chat("x", stream=True))
            worker.start()
            gw.wait_for(lambda: gw.status()["busy_requests"] == 1, what="an answer in progress")
            self.assertEqual(gw.request("POST", "/guardian/unload")[0], 409)
            self.assertEqual(gw.status()["status"], "READY")
            self.assertEqual(gw.request("POST", "/guardian/unload?force=1")[0], 200)
            worker.join(20)


# --- single functions --------------------------------------------------------------------------------------------------

class LifecycleUnit(unittest.TestCase):
    setUpClass = classmethod(baseline.GatewayGuard.setUpClass.__func__)
    tearDownClass = classmethod(baseline.GatewayGuard.tearDownClass.__func__)
    setUp = baseline.GatewayGuard.setUp
    patch_attr = baseline.GatewayGuard.patch_attr
    fake_llama = baseline.GatewayGuard.fake_llama

    def test_crashes_older_than_the_window_stop_counting_and_ready_does_not_reset_them(self):
        st = self.gw.st
        st.crash_count, st.crash_times = 0, deque()
        self.gw.G.update(crashWindowSeconds=100, crashRestartMaxAttempts=3)
        self.assertFalse(self.gw.note_crash(1000))
        self.assertFalse(self.gw.note_crash(1050))
        self.assertEqual(st.crash_count, 2)
        st.status = "READY"                                             # reaching READY changes nothing
        self.assertEqual(st.crash_count, 2)
        self.assertFalse(self.gw.note_crash(1200))                      # the first two are older than 100 s now
        self.assertEqual(st.crash_count, 1)
        self.assertFalse(self.gw.note_crash(1210))
        self.assertTrue(self.gw.note_crash(1220))
        self.assertIsNotNone(st.error_since)

    def test_stopping_a_server_that_was_already_dead_counts_a_crash_and_error_survives_a_stop(self):
        st = self.gw.st
        st.crash_count, st.crash_times, st.status = 0, deque(), "READY"
        self.gw.G["crashRestartMaxAttempts"] = 1
        self.proc.poll.return_value = 137
        st.expected_exit = False
        self.assertTrue(self.gw.stop_llama_locked("unload", save=False))
        self.assertEqual((st.status, st.crash_count), ("ERROR", 1))
        st.proc = None
        self.assertTrue(self.gw.stop_llama_locked("unload again", save=False))
        self.assertEqual(st.status, "ERROR")

    def test_memory_pressure_unloads_even_when_the_session_cannot_be_saved(self):
        st = self.gw.st
        self.gw.GAME.enabled = True                                     # game mode refuses an unload whose save failed
        self.addCleanup(setattr, self.gw.GAME, "enabled", False)
        st.status, st.busy = "READY", 0
        st.model_id = self.gw.M["defaultModel"]
        self.gw.G.update(memoryPressureSustainPolls=1, ramFreeMinGB=100, vramGuard=False, idleUnloadSeconds=10 ** 6)
        st.last_ram_poll = -1e9
        with patch.object(self.gw, "save_slot_locked", return_value=False):
            self.gw.monitor_tick(1000.0, 8000, 1.0)
        self.proc.terminate.assert_called_once()
        self.assertEqual(st.status, "MEMORY_PRESSURE")
        self.assertIsNone(st.proc)

    def test_a_dead_server_noticed_by_the_monitor_is_forgotten_completely(self):
        st = self.gw.st
        st.status = "READY"
        st.model_id = self.gw.M["defaultModel"]
        self.proc.poll.return_value = 1
        st.expected_exit = False
        self.gw.G["vramGuard"] = False
        self.gw.monitor_tick(10.0, 8000, 24)
        self.assertEqual((st.model_id, st.vision, st.proc), ("", False, None))

    def test_a_vram_sample_of_a_process_that_has_been_replaced_is_not_stored(self):
        st = self.gw.st
        old = Mock(pid=111)
        st.proc_vram_mib = None
        self.gw.monitor_tick(10.0, 8000, 24, process_mib=7000, proc=old)      # st.proc is another process
        self.assertIsNone(st.proc_vram_mib)
        self.gw.monitor_tick(11.0, 8000, 24, process_mib=7000, proc=st.proc)
        self.assertEqual(st.proc_vram_mib, 7000)

    def test_the_error_cooldown_leaves_error_by_itself(self):
        st = self.gw.st
        st.status, st.error_since, st.last_error = "ERROR", 100.0, "crashed"
        st.proc = None
        self.gw.G["errorCooldownSeconds"] = 300
        self.gw.monitor_tick(399.0, 8000, 24)
        self.assertEqual(st.status, "ERROR")
        self.gw.monitor_tick(400.0, 8000, 24)
        self.assertEqual((st.status, st.crash_count, st.last_error), ("UNLOADED", 0, ""))

    def test_a_refused_model_switch_does_not_unload_the_running_model(self):
        st = self.gw.st
        st.status, st.vision = "READY", False
        st.model_id = self.gw.M["defaultModel"]
        other = next((m for m in self.gw.M["models"] if m != st.model_id), None)
        if other is None:
            self.skipTest("one model configured")
        with patch.object(self.gw, "preload_check", return_value="Not enough free VRAM"), \
                patch.object(self.gw, "stop_llama_locked") as stop:
            self.assertEqual(self.gw.start_llama(other, False), "Not enough free VRAM")
        stop.assert_not_called()
        self.assertEqual(st.status, "READY")

    def test_a_bad_configuration_does_not_leave_loading_behind(self):
        st = self.gw.st
        st.proc, st.status, st.model_id = None, "UNLOADED", ""
        with patch.object(self.gw, "build_args", side_effect=ValueError("poll must be between 0 and 100")):
            error = self.gw.start_llama(self.gw.M["defaultModel"], False)
        self.assertIn("Invalid model configuration", error)
        self.assertEqual(st.status, "UNLOADED")

    def test_an_unexpected_exception_during_load_stops_the_child_and_answers(self):
        st = self.gw.st
        st.proc, st.status, st.model_id = None, "UNLOADED", ""
        child = Mock(pid=4321)
        child.poll.return_value = None
        child.terminate.side_effect = lambda: setattr(child.poll, "return_value", 0)
        with patch.object(self.gw.subprocess, "Popen", return_value=child), \
                patch.object(self.gw, "write_server_lease"), \
                patch.object(self.gw, "llama_call", side_effect=KeyError("boom")), \
                patch.object(self.gw.POLITE, "attach"), \
                patch.object(self.gw.time, "sleep"):
            error = self.gw.start_llama(self.gw.M["defaultModel"], False)
        self.assertIn("KeyError", error)
        self.assertNotEqual(st.status, "LOADING")
        child.terminate.assert_called()

    def test_a_stop_signal_does_not_wait_for_a_loading_thread_and_kills_the_child(self):
        st = self.gw.st
        st.shutting_down.clear()
        self.addCleanup(st.shutting_down.clear)
        release = threading.Event()
        taken = threading.Event()

        def hold_the_lock():
            with st.lock:
                taken.set()
                release.wait(10)
        holder = threading.Thread(target=hold_the_lock, daemon=True)
        holder.start()
        taken.wait(2)
        self.gw.G["shutdownLockSeconds"] = 0.2
        with patch.object(self.gw.sys, "exit", side_effect=SystemExit) as exit_, self.assertRaises(SystemExit):
            self.gw.shutdown(signal.SIGTERM, None)
        release.set()
        exit_.assert_called_once_with(0)
        self.proc.terminate.assert_called()                             # the child does not outlive the gateway
        self.assertTrue(st.shutting_down.is_set())
        self.assertEqual(self.gw.start_llama(self.gw.M["defaultModel"], False), "Gateway is shutting down")

    def test_game_cancel_survives_a_lease_that_has_no_connection_yet(self):
        gw = self.gw
        gw.G["gameDrainSeconds"] = 0
        gw.GAME.enabled = True
        gw.GAME.active = True
        gw.GAME.ready = True
        gw.GAME.drain_since = 0
        self.addCleanup(lambda: (setattr(gw.GAME, "enabled", False), setattr(gw.GAME, "active", False)))
        lease = {"cancel": threading.Event(), "conn": None}
        gw.st.busy = 1
        with patch.object(gw, "GAME_REQUESTS", {1: lease}), patch.object(gw.GAME, "update", return_value=False):
            gw.game_tick(100.0, ["game"])
        self.assertTrue(lease["cancel"].is_set())

    def test_the_api_key_goes_to_a_private_file_and_the_log_line_hides_it(self):
        gw = self.gw
        with patch.dict(gw.M, {"apiKey": "k-123"}), patch.object(gw, "server_supports", return_value=True), \
                patch.object(gw.common, "server_supports", return_value=True), \
                patch.object(gw, "STATE_DIR", self.root / "state"), patch.object(gw, "server_slot_dir", return_value=self.disk):
            args = gw.build_args(gw.M["models"][gw.M["defaultModel"]], False)
            self.assertNotIn("k-123", args)
            path = Path(args[args.index("--api-key-file") + 1])
            self.assertEqual(path.read_text().strip(), "k-123")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertIn("--cors-origins", args)
        self.assertNotIn("k-123", gw.redacted_command(["llama", "--api-key", "k-123", "--x"]))

    def test_a_server_without_the_option_is_not_started_with_it(self):
        gw = self.gw
        with patch.dict(gw.M, {"apiKey": "k-123"}), patch.object(gw, "server_supports", return_value=False), \
                patch.object(gw.common, "server_supports", return_value=False), \
                patch.object(gw, "server_slot_dir", return_value=self.disk):
            args = gw.build_args(gw.M["models"][gw.M["defaultModel"]], False)
        self.assertNotIn("--cors-origins", args)
        self.assertNotIn("--api-key-file", args)
        self.assertEqual(args[args.index("--api-key") + 1], "k-123")     # the older way still works

    def test_status_answers_while_another_thread_holds_the_lifecycle_lock(self):
        st = self.gw.st
        release, taken = threading.Event(), threading.Event()

        def hold():
            with st.lock:
                taken.set()
                release.wait(10)
        threading.Thread(target=hold, daemon=True).start()
        taken.wait(2)
        started = time.monotonic()
        snapshot = self.gw.status_snapshot()
        release.set()
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertIn("status", snapshot)


class ReviewFindings(unittest.TestCase):
    def test_a_launcher_script_as_the_server_binary_loads_with_a_lease_on_the_ram_backend(self):
        import stat
        import tempfile
        shim = Path(tempfile.mkdtemp(prefix="shura-launcher-", dir="/dev/shm"))
        self.addCleanup(__import__("shutil").rmtree, shim, True)
        script = shim / "server.sh"
        fake = HERE / "support" / "fake-llama-bin"
        script.write_text(f'#!/bin/sh\nexec "{fake}" "$@"\n')
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        guardian = {"crashRecovery": True, "slotSaveCheckpoints": 1}
        with GatewayHarness(behaviour={"n_tokens": 3}, guardian=guardian, launch={"exePath": str(script)}) as gw:
            self.assertEqual(gw.chat("x")[0], 200, gw.log()[-1200:])
            leases = list(gw.tmp.glob("shura-slots-*/server.json"))
            self.assertEqual(len(leases), 1, gw.log()[-800:])
            orphan = gw.status()["llama_pid"]
            os.kill(gw.proc.pid, signal.SIGKILL)                       # the gateway dies, the launcher-started server lives on
            gw.proc.wait(10)
            time.sleep(0.3)
            gw.start()
            self.assertEqual(gw.chat("again")[0], 200, gw.log()[-1200:])
            self.assertNotEqual(gw.status()["llama_pid"], orphan)       # the old server was recognised as ours and replaced

    def test_the_model_list_is_the_same_loaded_or_not(self):
        launch = json.loads((HERE.parent / "config" / "model-launch.example.json").read_text())
        expected = list(launch["models"])
        for guardian in ({"slotSaveCheckpoints": 0}, {"slotSaveCheckpoints": 1}):
            with GatewayHarness(behaviour={"n_tokens": 3}, guardian=guardian) as gw:
                before = gw.request("GET", "/v1/models")[1]
                gw.chat("x")
                self.assertEqual(gw.request("GET", "/v1/models")[1], before)
                self.assertEqual([m["id"] for m in before["data"]], expected)

    def test_a_request_for_another_model_does_not_stop_a_server_that_is_answering(self):
        import copy
        example = json.loads((HERE.parent / "config" / "model-launch.example.json").read_text())
        first = copy.deepcopy(example["models"]["tiel-coder"])
        for key in ("mmprojPath", "visionOverrides"):
            first.pop(key, None)
        first["vramFreeMinGBToLoad"] = 0
        models = {"tiel-coder": first, "second": copy.deepcopy(first)}
        for guardian in ({"slotSaveCheckpoints": 0}, {"slotSaveCheckpoints": 1}):
            with GatewayHarness(behaviour={"n_tokens": 120, "token_delay_ms": 20}, guardian=guardian,
                                launch={"models": models, "defaultModel": "tiel-coder"}) as gw:
                results = {}

                def long_stream():
                    results["a"] = gw.request("POST", "/v1/chat/completions", {"model": "tiel-coder", "stream": True,
                                                                              "messages": [{"role": "user", "content": "x"}]}, stream=True)
                worker = threading.Thread(target=long_stream)
                worker.start()
                gw.wait_for(lambda: gw.status()["busy_requests"] == 1, what="the first answer in progress")
                code, body = gw.request("POST", "/v1/chat/completions", {"model": "second", "messages": [{"role": "user", "content": "y"}]})
                worker.join(30)
                self.assertEqual(code, 503, body)
                self.assertIn("in progress", json.dumps(body))
                self.assertTrue(results["a"][1]["complete"])             # the first answer was not cut by the switch
                self.assertEqual(results["a"][1]["error"], None)

    def test_a_read_only_request_never_blocks_an_unload(self):
        with GatewayHarness(behaviour={"n_tokens": 3}, guardian=FULL) as gw:
            gw.chat("x")
            self.assertEqual(gw.request("GET", "/props")[0], 200)
            self.assertEqual(gw.status()["busy_requests"], 0)
            self.assertEqual(gw.request("POST", "/guardian/unload")[0], 200)


class ReviewUnits(unittest.TestCase):
    setUpClass = classmethod(baseline.GatewayGuard.setUpClass.__func__)
    tearDownClass = classmethod(baseline.GatewayGuard.tearDownClass.__func__)
    setUp = baseline.GatewayGuard.setUp
    patch_attr = baseline.GatewayGuard.patch_attr
    fake_llama = baseline.GatewayGuard.fake_llama
    tick = baseline.GatewayGuard.tick
    yield_model = baseline.GatewayGuard.yield_model

    def test_memory_pressure_recovery_does_not_clear_the_error_state(self):
        st = self.gw.st
        st.status, st.memory_pressure, st.proc = "ERROR", True, None
        st.last_ram_poll = -1e9
        self.gw.G.update(vramGuard=False, idleUnloadSeconds=10 ** 6)
        self.gw.monitor_tick(1000.0, 8000, 64.0)                       # plenty of RAM again
        self.assertEqual(st.status, "ERROR")
        self.assertFalse(st.memory_pressure)

    def test_a_rejected_ram_session_does_not_make_the_next_start_skip_a_good_disk_session(self):
        self.yield_model()
        (self.ram / "active.bin").write_bytes(b"damaged!")
        self.gw.st.model_id = self.gw.M["defaultModel"]
        self.assertEqual(self.gw.restore_slot_locked(), "")
        self.assertFalse(self.gw.st.skip_disk_restore)
        self.assertIsNone(self.gw.st.ram_checkpoint)

    def test_a_game_cancelled_json_answer_is_a_503_not_a_502(self):
        sys.path.insert(0, str(HERE))
        import test_gateway_streams as streams
        cancel = threading.Event()
        lease = {"cancel": cancel, "conn": None, "retry_after": 5}

        def script(conn, up):
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 500\r\n\r\n" + b'{"part')
            time.sleep(4)
        upstream = streams.Upstream(script)
        front = streams.Front(upstream.port, lease=lease)
        self.addCleanup(upstream.close)
        self.addCleanup(front.close)

        def game_monitor_cuts_the_request():
            time.sleep(0.5)
            cancel.set()
            lease["sock"].shutdown(socket.SHUT_RDWR)
        threading.Thread(target=game_monitor_cuts_the_request, daemon=True).start()
        status, headers, data, _ = streams.fetch(front.port)
        self.assertEqual(status, 503)
        self.assertEqual(headers.get("Retry-After"), "5")
        self.assertIn(b"SHURA_GAME_PAUSE", data)

    def test_a_key_file_that_cannot_be_written_falls_back_to_the_command_line(self):
        gw = self.gw
        with patch.dict(gw.M, {"apiKey": "k-1"}), patch.object(gw, "server_supports", return_value=True), \
                patch.object(gw.common, "write_api_key_file", side_effect=OSError("read-only")), \
                patch.object(gw, "server_slot_dir", return_value=self.disk):
            args = gw.build_args(gw.M["models"][gw.M["defaultModel"]], False)
        self.assertEqual(args[args.index("--api-key") + 1], "k-1")
        self.assertNotIn("--api-key-file", args)

    def test_a_watcher_survives_a_descriptor_above_the_select_limit(self):
        import resource
        sys.path.insert(0, str(HERE.parent / "gateway"))
        from proxy_core import ClientWatch
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard < 1200 and hard != resource.RLIM_INFINITY:
            self.skipTest("descriptor limit too low")
        resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 1200), hard))
        self.addCleanup(resource.setrlimit, resource.RLIMIT_NOFILE, (soft, hard))
        spare = [socket.socket() for _ in range(1100)]
        self.addCleanup(lambda: [x.close() for x in spare])
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        left = []
        watch = ClientWatch(b, lambda: left.append(1))
        self.assertGreater(b.fileno(), 1023)
        b_peer_close = a.close
        b_peer_close()
        time.sleep(0.8)
        watch.stop()
        b.close()
        self.assertEqual(left, [1])                                   # recognised, instead of failing as "gone" for the wrong reason


class ClockAndPaths(unittest.TestCase):
    setUpClass = classmethod(baseline.GatewayGuard.setUpClass.__func__)
    tearDownClass = classmethod(baseline.GatewayGuard.tearDownClass.__func__)
    setUp = baseline.GatewayGuard.setUp
    patch_attr = baseline.GatewayGuard.patch_attr
    fake_llama = baseline.GatewayGuard.fake_llama

    def test_a_suspend_does_not_make_a_busy_model_look_idle(self):
        st = self.gw.st
        st.clock_ref = None
        st.last_request = 1000.0
        self.assertEqual(self.gw.common.compensate_wall_clock(st, 1000.0, 50.0), 0.0)
        # the machine slept for 3 hours: the wall clock jumped, the monotonic clock did not
        self.assertAlmostEqual(self.gw.common.compensate_wall_clock(st, 1000.0 + 10800 + 2, 52.0), 10800.0)
        self.assertAlmostEqual(st.last_request, 1000.0 + 10800)
        # ordinary time passing, and a small drift, change nothing
        self.assertEqual(self.gw.common.compensate_wall_clock(st, 1000.0 + 10800 + 12, 62.0), 0.0)
        self.assertEqual(self.gw.common.compensate_wall_clock(st, 1000.0 + 10800 + 15, 62.5), 0.0)

    def test_a_shared_tmpfs_name_taken_by_someone_else_moves_the_ram_directory_to_the_runtime_dir(self):
        gw = self.gw
        runtime = self.shm_root / "run"
        runtime.mkdir()
        runtime.chmod(0o700)
        shared = self.shm_root / "shared"
        shared.mkdir()
        gw.RAM_SLOT_DIR = None
        with patch.object(gw, "RAM_SLOT_DIR", None), patch.dict(gw.G, {"vramPressureSlotDir": str(shared)}), \
                patch.dict(os.environ, {"XDG_RUNTIME_DIR": str(runtime)}), patch.object(gw, "STATE_DIR", self.root / "state"), \
                patch.object(gw, "recover_orphan"), patch.object(gw, "recover_gear_transaction"):
            (self.root / "state").mkdir(exist_ok=True)
            first = gw.prepare_ram_slot_dir()
            self.assertEqual(first.parent, shared.resolve())
            gw.RAM_SLOT_DIR = None
            first.chmod(0o755)                                         # a pre-created / tampered directory: not private any more
            second = gw.prepare_ram_slot_dir()
            self.assertEqual(second.parent, runtime.resolve())
            self.assertEqual(second.stat().st_mode & 0o777, 0o700)
            gw.RAM_SLOT_DIR = None
            self.assertTrue(any(e["reason"] == "ram_path_moved" for e in gw.st.events))


if __name__ == "__main__":
    unittest.main()
