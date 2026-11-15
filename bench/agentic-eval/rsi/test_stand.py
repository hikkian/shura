"""RSIL1 fixes (2026-10-22): the server's memory rule before a run, the own-server check, and the stop on a signal. Offline: no GPU, no
server is started. Run: cd bench/agentic-eval && python3 -B -m unittest discover -s rsi -p 'test_stand.py'"""
import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rsi import executor  # noqa: E402

SMAPS = """\
7f0000000000-7f0000100000 rw-p 00000000 00:00 0
Rss:                4096 kB
7f0000100000-7f0000200000 rw-p 00000000 00:00 0                          [heap]
Rss:                1048576 kB
7f0000200000-7f0000300000 rw-s 00000000 00:06 1234                       /dev/zero
Rss:                2097152 kB
"""
# port 8091 is 0x1F9B; a LISTEN socket (state 0A) on 8091 with inode 555; a connection on 8091 (state 01) with inode 777; a LISTEN on 8092 (0x1F9C) with inode 888
NET_TCP = """\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:1F9B 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 555 1 0000000000000000 100 0 0 10 0
   1: 0100007F:1F9B 0100007F:C350 01 00000000:00000000 00:00000000 00000000  1000        0 777 1 0000000000000000 100 0 0 10 0
   2: 0100007F:1F9C 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 888 1 0000000000000000 100 0 0 10 0
"""


class MemoryRule(unittest.TestCase):
    def test_heap_and_dev_zero_are_summed_and_other_mappings_are_ignored(self):
        self.assertEqual(executor.parse_smaps_kib(SMAPS), (1048576, 2097152))

    def test_the_limit_is_strictly_above_six_gib(self):
        self.assertTrue(executor.over_memory_limit(3.5, 2.6))        # 6.1
        self.assertFalse(executor.over_memory_limit(3.0, 3.0))       # 6.0 is not above 6
        self.assertFalse(executor.over_memory_limit(2.0, 3.0))
        self.assertTrue(executor.over_memory_limit(4.95, 2.24))      # the RSIL1 reading of 06:21

    def test_server_memory_is_read_from_its_smaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "123").mkdir()
            (Path(tmp) / "123" / "smaps").write_text(SMAPS, encoding="utf-8")
            self.assertEqual(executor.server_memory_gib(123, proc=tmp), (1.0, 2.0))


class OwnServerCheck(unittest.TestCase):
    def test_only_listen_sockets_on_the_port_count(self):
        self.assertEqual(executor.listen_inodes(8091, [NET_TCP]), {555})

    def test_the_listener_pid_is_found_through_the_socket_inode(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, net = Path(tmp, "proc"), Path(tmp, "tcp")
            for pid, inode in ((123, 555), (124, 777)):
                (proc / str(pid) / "fd").mkdir(parents=True)
                (proc / str(pid) / "fd" / "3").symlink_to(f"socket:[{inode}]")
            net.write_text(NET_TCP, encoding="utf-8")
            self.assertEqual(executor.listener_pids(8091, proc=str(proc), net_files=(str(net),)), {123})

    def test_the_server_is_found_by_its_exact_argv(self):
        argv = ["/x/llama-server", "--port", "8091"]
        with tempfile.TemporaryDirectory() as tmp:
            for pid, raw in ((200, b"/x/llama-server\0--port\0" + b"8091\0"), (201, b"/x/llama-server\0--port\0" + b"8092\0")):
                (Path(tmp) / str(pid)).mkdir()
                (Path(tmp) / str(pid) / "cmdline").write_bytes(raw)
            self.assertEqual(executor.find_server_pids(argv, proc=tmp), [200])

    def test_a_zombie_is_not_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "300").mkdir()
            (Path(tmp) / "300" / "stat").write_text("300 (llama-server) Z 1 0 0", encoding="utf-8")
            self.assertFalse(executor.pid_running(300, proc=tmp))
            (Path(tmp) / "300" / "stat").write_text("300 (llama-server) S 1 0 0", encoding="utf-8")
            self.assertTrue(executor.pid_running(300, proc=tmp))

    def test_alive_is_false_when_another_process_holds_the_port(self):
        stand = executor.GpuStand.__new__(executor.GpuStand)
        stand.port = 8091
        with mock.patch.object(executor.GpuStand, "_own_pid", return_value=123), \
                mock.patch.object(executor, "listener_pids", return_value={999}), \
                mock.patch.object(executor, "health_ok", return_value=True) as health:
            self.assertFalse(stand.alive())
            health.assert_not_called()

    def test_alive_needs_the_own_pid_on_the_port_and_health(self):
        stand = executor.GpuStand.__new__(executor.GpuStand)
        stand.port = 8091
        with mock.patch.object(executor.GpuStand, "_own_pid", return_value=123), \
                mock.patch.object(executor, "listener_pids", return_value={123}), \
                mock.patch.object(executor, "health_ok", return_value=True):
            self.assertTrue(stand.alive())
        with mock.patch.object(executor.GpuStand, "_own_pid", return_value=None):
            self.assertFalse(stand.alive())

    def test_a_start_refuses_a_port_held_by_another_process(self):
        stand = executor.GpuStand.__new__(executor.GpuStand)
        stand.port = 8091
        with mock.patch.object(executor, "listener_pids", return_value={999}):
            with self.assertRaises(SystemExit) as cm:
                stand._wait_port_free(timeout=0)
        self.assertIn("999", str(cm.exception))


class StopByPid(unittest.TestCase):
    def test_stop_pid_ends_the_process_it_is_given(self):
        p = subprocess.Popen(["sleep", "30"])
        executor.stop_pid(p.pid, grace=3)
        self.assertEqual(p.wait(timeout=5), -signal.SIGTERM)


class PlannedRestartInRunner(unittest.TestCase):
    class FakeStand:
        fake = False

        proxy_port, proxy_log = 1, None

        def __init__(self, alive=True, over=False):
            self._alive, self._over = alive, over
            self.restarts = []

        def alive(self):
            return self._alive

        def memory_over_limit(self):
            return self._over

        def restart(self, reason=""):
            self.restarts.append(reason)
            self._alive, self._over = True, False

    def run_once(self, stand):
        calls = []

        def fake_run_task(task, root, model, arm, rep, port, proxy_log, out, scratch):
            calls.append(task["id"])
            return {"requests": 1, "resolved": True}

        with mock.patch.object(executor, "run_task", fake_run_task):
            rec = executor.Runner(stand, tempfile.mkdtemp()).run({"id": "t1"}, ".", "tiel", "base", 1, tempfile.mkdtemp())
        return rec, calls

    def test_a_server_over_the_limit_is_restarted_before_the_run(self):
        stand = self.FakeStand(alive=True, over=True)
        rec, calls = self.run_once(stand)
        self.assertEqual(calls, ["t1"])
        self.assertEqual(len(stand.restarts), 1)
        self.assertIn("planned before t1", stand.restarts[0])
        self.assertEqual(rec["requests"], 1)

    def test_a_dead_server_is_restarted_before_the_run(self):
        stand = self.FakeStand(alive=False, over=False)
        self.run_once(stand)
        self.assertEqual(len(stand.restarts), 1)
        self.assertIn("not alive", stand.restarts[0])

    def test_a_healthy_server_under_the_limit_is_not_restarted(self):
        stand = self.FakeStand(alive=True, over=False)
        self.run_once(stand)
        self.assertEqual(stand.restarts, [])


class SignalStopsOwnServer(unittest.TestCase):
    def test_a_signal_stops_the_own_server_first_and_ends_the_with_block(self):
        stand = executor.GpuStand.__new__(executor.GpuStand)
        calls = []
        stand.stop_server = lambda: calls.append("stop")
        with self.assertRaises(SystemExit) as cm:
            stand._on_signal(signal.SIGTERM, None)
        self.assertEqual(calls, ["stop"])
        self.assertEqual(cm.exception.code, 128 + signal.SIGTERM)


class ServerArgvOnFile(unittest.TestCase):
    def test_the_pid_file_is_read_only_while_the_argv_matches(self):
        argv = ["/x/llama-server", "--port", "8091"]
        with tempfile.TemporaryDirectory() as tmp:
            argv_file = Path(tmp, "argv.json")
            argv_file.write_text(json.dumps(argv), encoding="utf-8")
            stand = executor.GpuStand(argv_file, Path(tmp, "gpu"))
            (Path(tmp, "gpu")).mkdir()
            (Path(tmp, "gpu") / executor.PID_FILE).write_text(json.dumps({"pid": 2 ** 22 + 7}), encoding="utf-8")
            self.assertIsNone(stand._own_pid())   # no such process


class DipRecording(unittest.TestCase):
    """VRAMDIP1: a dip shorter than the guard window leaves a record: length, minimum, trace, who held the memory."""

    def feed(self, rec, samples):
        out = []
        for t, free in samples:
            out += rec.step(free, t, wall=f"w{t}")
        return out

    def test_a_short_dip_is_recorded_and_does_not_trigger_the_guard(self):
        rec = executor.DipRecorder(snapshot=lambda: [{"pid": 1, "comm": "llama-server", "vram_MiB": 12000}])
        samples = [(0.0, 2.0), (0.1, 1.9), (0.2, 1.4), (0.3, 1.32), (0.4, 1.45), (0.5, 1.8), (0.6, 2.0)]
        dips = self.feed(rec, samples)
        self.assertEqual(len(dips), 1)
        d = dips[0]
        self.assertEqual((d["min_free_GiB"], d["duration_s"], d["ended"], d["wall"]), (1.32, 0.3, True, "w0.2"))
        self.assertEqual(d["by_process"][0]["comm"], "llama-server")
        self.assertEqual(d["trace"][0], (-0.2, 2.0))                   # seconds from the start of the dip, free GiB: first the samples before it
        self.assertIn((0.0, 1.4), d["trace"])
        self.assertEqual(d["trace"][-1], (0.3, 1.8))                   # the sample that ended it
        # the same samples never trigger the stop of guard_step (a window of 1 s)
        ls = lm = None
        for t, free in samples:
            ls, lm, trig = executor.guard_step(ls, lm, free, t)
            self.assertIsNone(trig)

    def test_the_trace_keeps_the_samples_before_the_dip(self):
        rec = executor.DipRecorder(trace_before=3)
        d = self.feed(rec, [(0.0, 2.0), (0.1, 2.1), (0.2, 2.2), (0.3, 2.3), (0.4, 1.0), (0.5, 2.0)])[0]
        self.assertEqual([f for _, f in d["trace"]][:4], [2.1, 2.2, 2.3, 1.0])
        self.assertEqual(d["trace"][3][0], 0.0)

    def test_each_dip_is_its_own_record_and_an_open_one_is_flushed(self):
        rec = executor.DipRecorder()
        dips = self.feed(rec, [(0.0, 1.0), (0.1, 2.0), (0.2, 2.0), (0.3, 1.2)])
        self.assertEqual(len(dips), 1)
        flushed = rec.flush(0.6)
        self.assertEqual((len(flushed), flushed[0]["ended"], flushed[0]["duration_s"], flushed[0]["min_free_GiB"]), (1, False, 0.3, 1.2))
        self.assertEqual(rec.flush(0.7), [])

    def test_a_failing_snapshot_does_not_stop_the_recorder(self):
        def boom():
            raise OSError("no /proc")
        d = self.feed(executor.DipRecorder(snapshot=boom), [(0.0, 1.0), (0.1, 2.0)])[0]
        self.assertIn("error", d["by_process"])

    def test_vram_by_process_reads_amdgpu_clients_from_fdinfo(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def proc(pid, comm, fds):
                (root / str(pid) / "fdinfo").mkdir(parents=True)
                (root / str(pid) / "comm").write_text(comm + "\n")
                for n, text in fds.items():
                    (root / str(pid) / "fdinfo" / str(n)).write_text(text)
            drm = "pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t%d\ndrm-memory-vram:\t%d KiB\n"
            proc(10, "llama-server", {5: drm % (7, 12582912), 6: drm % (7, 12582912)})      # two fds of one client: counted once
            proc(11, "brave", {3: drm % (8, 307200), 4: drm % (9, 102400)})                  # two clients: summed
            proc(12, "bash", {0: "pos:\t0\n"})                                              # no DRM at all
            (root / "self").mkdir()
            out = executor.vram_by_process(str(root))
            self.assertEqual([(r["comm"], r["vram_MiB"]) for r in out], [("llama-server", 12288), ("brave", 400)])


class CrashDuringARun(unittest.TestCase):
    """RSI1 night 1: the llama-server crashed inside a run; the agent hung to the deadline and the run was recorded as the model's timeout."""

    class Stand:
        fake = False
        client_strict = True
        proxy_port, proxy_log = 1, "x"

        def __init__(self, crashes_in_run):
            self.n, self.crashes_in_run, self.restarts, self.current_run = 0, crashes_in_run, [], ""

        def alive(self): return True
        def memory_over_limit(self): return False
        def restart(self, reason=""): self.restarts.append(reason)
        def crashes(self): return self.n

    def runner(self, stand, results):
        r = executor.Runner(stand, "/nonexistent")
        calls = []

        def fake_run_task(task, root, model, arm, rep, port, log, out, scratch, sampling=None, abort=None):
            calls.append(abort)
            if len(calls) <= stand.crashes_in_run:
                stand.n += 1                                  # the gateway logs server_crashed during this run
                assert abort is not None and abort()          # the agent would be ended at once
                return {"requests": 3, "client": {"ok": True, "bad_total": 0}, "timed_out": True, "score": 0.4}
            return results
        self.saved = executor.run_task
        executor.run_task = fake_run_task
        self.addCleanup(lambda: setattr(executor, "run_task", self.saved))
        return r, calls

    def test_a_run_with_a_server_crash_is_discarded_and_repeated_not_counted_as_a_failure_of_the_model(self):
        stand = self.Stand(1)
        r, calls = self.runner(stand, {"requests": 5, "client": {"ok": True, "bad_total": 0}, "score": 1.0})
        rec = r.run({"id": "t"}, "root", "tiel", "base", 1, "out")
        self.assertEqual(rec["score"], 1.0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(r.discarded), 1)
        self.assertTrue(r.discarded[0]["server_crashed"])
        self.assertEqual(len(stand.restarts), 1)

    def test_a_run_whose_agent_engine_crashed_is_discarded_and_repeated(self):
        # RSI1, 2026-10-31 16:17: Bun segfaulted inside the agent engine: an empty agent log, agent_exit 1, a partly scored run
        stand = self.Stand(0)
        good = {"requests": 5, "client": {"ok": True, "bad_total": 0}, "score": 1.0}
        r, calls = self.runner(stand, good)
        runs = []
        inner = executor.run_task

        def with_engine_crash(*a, **k):
            runs.append(1)
            if len(runs) == 1:
                return {"requests": 7, "client": {"ok": True, "bad_total": 0}, "score": 0.5, "agent_exit": 1, "engine_failed": "the engine run failed: [Errno 111] Connection refused"}
            return inner(*a, **k)
        executor.run_task = with_engine_crash
        rec = r.run({"id": "t"}, "root", "tiel", "base", 1, "out")
        self.assertEqual(rec["score"], 1.0)
        self.assertEqual(len(r.discarded), 1)
        self.assertIn("Connection refused", r.discarded[0]["engine_failed"])
        self.assertNotIn("server_crashed", r.discarded[0])

    def test_a_run_without_a_crash_is_taken_as_before(self):
        stand = self.Stand(0)
        r, calls = self.runner(stand, {"requests": 5, "client": {"ok": True, "bad_total": 0}, "score": 1.0})
        self.assertEqual(r.run({"id": "t"}, "root", "tiel", "base", 1, "out")["score"], 1.0)
        self.assertEqual((len(calls), r.discarded, stand.restarts), (1, [], []))

    def test_a_server_that_keeps_crashing_stops_the_night_instead_of_counting_noise(self):
        stand = self.Stand(9)
        r, calls = self.runner(stand, {})
        with self.assertRaises(SystemExit):
            r.run({"id": "t"}, "root", "tiel", "base", 1, "out")
        self.assertEqual(len(r.discarded), executor.RUN_ATTEMPTS)


class AbortEndsTheAgentRun(unittest.TestCase):
    def test_the_abort_check_is_thread_local_and_cleared(self):
        import harness
        self.assertFalse(harness._aborted())
        with harness.run_abort(lambda: True):
            self.assertTrue(harness._aborted())
        self.assertFalse(harness._aborted())
        with harness.run_abort(lambda: 1 / 0):                    # a failing check never aborts a run by itself
            self.assertFalse(harness._aborted())


class GatewayCrashCounter(unittest.TestCase):
    def test_server_crashed_lines_are_counted_once_each_and_cheaply(self):
        import tempfile
        from rsi import gateway_stand
        work = Path(tempfile.mkdtemp(prefix="rsi1-crash-", dir="/dev/shm"))
        self.addCleanup(__import__("shutil").rmtree, work, True)
        stand = object.__new__(gateway_stand.GatewayStand)
        stand.work = work
        self.assertEqual(stand.crashes(), 0)                      # no log yet
        log = work / "gateway.log"
        log.write_text("2026 Guardian server_started\n")
        self.assertEqual(stand.crashes(), 0)
        with open(log, "a") as f:
            f.write("2026 Guardian server_crashed: llama-server crashed (exit -6)\n")
        self.assertEqual(stand.crashes(), 1)
        self.assertEqual(stand.crashes(), 1)                      # the same line is not counted twice
        with open(log, "a") as f:
            f.write("x\n2026 Guardian server_crashed: again\n")
        self.assertEqual(stand.crashes(), 2)


class GuardStopReason(unittest.TestCase):
    """RSIL0f: the VRAM guard writes why it stopped the server (threshold, minimum, duration, desktop share, context, run)."""

    def test_the_window_triggers_only_after_the_guard_seconds_below_the_threshold(self):
        ls, lm, trig = executor.guard_step(None, None, 1.2, 0.0)
        self.assertEqual((ls, lm, trig), (0.0, 1.2, None))
        ls, lm, trig = executor.guard_step(ls, lm, 1.0, 0.5)
        self.assertIsNone(trig)
        ls, lm, trig = executor.guard_step(ls, lm, 1.7, 1.2)            # back above the threshold: the window is reset
        self.assertEqual((ls, lm, trig), (None, None, None))
        ls, lm, trig = executor.guard_step(ls, lm, 1.4, 1.5)            # a new window starts at 1.5 s
        self.assertEqual((ls, lm, trig), (1.5, 1.4, None))
        ls, lm, trig = executor.guard_step(ls, lm, 1.1, 3.0)
        self.assertEqual(trig, (1.5, 1.1))                                # 1.5 s below 1.5 GiB, minimum 1.1 GiB

    def test_a_reading_above_the_threshold_resets_the_window(self):
        self.assertEqual(executor.guard_step(0.0, 1.0, 2.0, 5.0), (None, None, None))

    def test_the_record_has_every_field_of_the_card(self):
        rec = executor.guard_stop_record(t="t", min_free_gib=1.23456, low_s=2.5, desktop_mib=2100, context_tokens=17015,
                                         run="py_x dev 2 (attempt 1)")
        self.assertEqual(rec["threshold_GiB"], 1.5)
        self.assertEqual(rec["min_free_GiB"], 1.235)
        self.assertEqual(rec["low_s"], 2.5)
        self.assertEqual(rec["desktop_used_MiB_before_start"], 2100)
        self.assertEqual(rec["context_tokens"], 17015)
        self.assertEqual(rec["run"], "py_x dev 2 (attempt 1)")
        self.assertEqual(rec["reason"], "VRAM guard")

    def test_the_context_is_the_last_request_in_the_proxy_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp, "proxy.jsonl")
            log.write_text('{"status": 200, "usage": {"prompt_tokens": 1000}}\n{"status": 200}\n{"status": 200, "usage": {"prompt_tokens": 17015}}\n',
                           encoding="utf-8")
            self.assertEqual(executor.last_prompt_tokens(log), 17015)
            self.assertIsNone(executor.last_prompt_tokens(Path(tmp, "missing.jsonl")))

    def test_a_stop_is_written_to_the_stand_folder_and_the_restart_note_keeps_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp, "gpu")
            work.mkdir()
            log = Path(tmp, "proxy.jsonl")
            log.write_text('{"status": 200, "usage": {"prompt_tokens": 17015}}\n', encoding="utf-8")
            stand = executor.GpuStand.__new__(executor.GpuStand)
            stand.work, stand.proxy_log = work, log
            stand.desktop_mib_before, stand.current_run = 2100, "py_x dev 2 (attempt 1)"
            stand.guard_stops, stand.stop_reason, stand.restarts = [], None, []
            stand.lock = __import__("threading").RLock()
            stand._record_guard_stop(1.23456, 2.5)
            lines = (work / "guard_stops.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertEqual((rec["context_tokens"], rec["desktop_used_MiB_before_start"], rec["min_free_GiB"]), (17015, 2100, 1.235))
            self.assertIn("VRAM guard stop", (work / "server.log").read_text(encoding="utf-8"))
            self.assertIn("context 17015 tokens", stand.stop_reason)
            # the next restart carries the reason once, then the reason is cleared
            stand.server_memory = lambda: None
            stand.stop_server = lambda: None
            stand._start = lambda label: None
            stand.restart(reason="after x")
            self.assertIn("VRAM guard", stand.restarts[0]["reason"])
            self.assertIsNone(stand.stop_reason)


if __name__ == "__main__":
    unittest.main()
