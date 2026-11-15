"""Regression tests for the 2026-10-06 audit: game mode, NVML helper. Fakes only: no GPU, no model."""
import ctypes
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))
import gpu_nvml
import test_gateway as fixture
from game_mode import GamePolicy, GameSignals


def active_policy(**config):
    p = GamePolicy({"gameMode": True, **config})
    for t in range(0, 10):
        p.update(t, ["foreign GPU"], [])
    assert p.active
    return p


class GamePolicyAudit(unittest.TestCase):
    def test_h1_persistent_detector_error_cannot_park_forever(self):
        p = active_policy()
        # Short error bursts still keep the pause (a driver reload is not the end of a game) ...
        for t in range(10, 200):
            p.update(t, [], ["NVML: boom"])
        self.assertTrue(p.active)
        # ... but a detector that never works releases it after the bounded hold plus the quiet time.
        for t in range(200, 10 + 300 + 60 + 5):
            p.update(t, [], ["NVML: boom"])
        self.assertFalse(p.active)

    def test_h1_hold_restarts_when_errors_stop_and_reasons_return(self):
        p = active_policy(gameErrorHoldSeconds=100)
        for t in range(10, 90):
            p.update(t, [], ["NVML: boom"])
        p.update(90, ["foreign GPU"], [])      # the game is seen again: hold timer must reset
        for t in range(91, 180):
            p.update(t, [], ["NVML: boom"])
        self.assertTrue(p.active)

    def test_h1_manual_resume_overrides_detector_errors(self):
        p = active_policy()
        p.update(10, [], ["gamemoded unavailable"])
        self.assertTrue(p.active)
        p.manual_switch(False, 11)
        self.assertFalse(p.active)
        self.assertEqual(p.reason, "")

    def test_h1_manual_resume_still_respects_a_live_game_signal(self):
        p = active_policy()
        p.manual_switch(False, 11)
        self.assertTrue(p.active)

    def test_l2_resume_clears_the_on_debounce(self):
        p = GamePolicy({"gameMode": True})
        p.manual_switch(True, 10)
        self.assertTrue(p.active)
        p.manual_switch(False, 11)
        self.assertFalse(p.active)
        p.update(100, ["game"])
        self.assertFalse(p.active)             # one fresh reason is debounced again
        p.update(105, ["game"])
        self.assertTrue(p.active)


class GameSignalsAudit(unittest.TestCase):
    def proc(self, root, pid, comm, cmdline=b"", environ=b""):
        d = Path(root) / str(pid)
        d.mkdir(exist_ok=True)
        (d / "comm").write_text(comm)
        (d / "cmdline").write_bytes(cmdline)
        (d / "environ").write_bytes(environ)
        return d

    def test_l3_truncated_comm_matches_desktop_allow_list(self):
        n = Mock()
        n.processes.return_value = {5: 4000}
        n.process_utilization.return_value = ({}, 0)
        sensor = GameSignals({"gameDetectNvml": True})
        # /proc/PID/comm keeps 15 characters of "kscreenlocker_greet"
        with patch.object(sensor, "process", return_value="kscreenlocker_g"):
            self.assertEqual(sensor.collect(None, n)[0], [])

    def test_l3_reaper_is_steam_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.proc(tmp, 100, "reaper", b"/opt/REAPER/reaper\0")
            self.proc(tmp, 101, "reaper", b"reaper\0SteamLaunch\0AppId=7\0--\0game\0")
            reasons, _ = GameSignals({"gameDetectReaper": True}, tmp).collect(None)
            self.assertEqual(reasons, ["Steam reaper PID101"])

    def test_l4_environ_is_read_once_per_process(self):
        reads = []
        original = Path.open

        def spy(path, *args, **kwargs):
            if path.name == "environ" and args[:1] == ("rb",):
                reads.append(str(path))
            return original(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            d = self.proc(tmp, 200, "sleep", environ=b"SteamAppId=5\0")
            sensor = GameSignals({"gameDetectSteam": True}, tmp)
            with patch.object(Path, "open", spy):
                for _ in range(5):
                    self.assertEqual(sensor.collect(None)[0], ["SteamAppId PID200"])
                self.assertEqual(len(reads), 1)
                # exec of another program in the same pid (new comm) is re-examined
                (d / "comm").write_text("game")
                (d / "environ").write_bytes(b"X=1\0")
                self.assertEqual(sensor.collect(None)[0], [])
                self.assertEqual(len(reads), 2)
            # a recycled pid is a new process: the old verdict must not stick
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
            self.proc(tmp, 200, "game", environ=b"SteamAppId=9\0")
            with patch.object(Path, "open", spy):
                self.assertEqual(sensor.collect(None)[0], ["SteamAppId PID200"])

    def test_l5_empty_gpu_poll_does_not_break_the_debounce(self):
        clock = [0.0]
        n = Mock()
        n.processes.return_value = {3: 100}
        polls = [({3: 95}, 1), ({}, 1), ({}, 1), ({3: 90}, 2), ({}, 2)]
        n.process_utilization.side_effect = polls
        sensor = GameSignals({"gameDetectNvml": True}, clock=lambda: clock[0])
        seen = []
        with patch.object(sensor, "process", return_value="game"):
            for i in range(len(polls)):
                clock[0] = float(i)
                seen.append(sensor.collect(None, n)[0])
        self.assertTrue(all(r == ["foreign GPU PID3 game 95%"] or r == ["foreign GPU PID3 game 90%"] for r in seen), seen)

    def test_l5_stale_utilisation_expires_and_warning_clears(self):
        clock = [0.0]
        n = Mock()
        n.processes.return_value = {3: 100}
        n.process_utilization.side_effect = [({3: 95}, 1), RuntimeError("util boom"), ({}, 1), ({}, 1)]
        sensor = GameSignals({"gameDetectNvml": True}, clock=lambda: clock[0])
        with patch.object(sensor, "process", return_value="game"):
            sensor.collect(None, n)
            clock[0] = 1.0
            sensor.collect(None, n)
            self.assertEqual(sensor.util_warning, "util boom")
            clock[0] = 2.0
            sensor.collect(None, n)
            self.assertEqual(sensor.util_warning, "")   # cleared once NVML answers again
            clock[0] = 20.0
            self.assertEqual(sensor.collect(None, n)[0], [])

    def test_h1_gdbus_uint32_form_is_parsed(self):
        with patch("game_mode.shutil.which", return_value="/bin/x"), patch(
            "game_mode.subprocess.run", return_value=Mock(returncode=0, stdout="(<uint32 2>,)\n")
        ):
            reasons, errors = GameSignals({"gameDetectGameMode": True}).collect(None)
        self.assertEqual((reasons, errors), (["gamemoded registered game"], []))


class FakeNvmlLib:
    def __init__(self, init=0, handle=0):
        self.init, self.handle_rc, self.shutdowns = init, handle, 0

    def nvmlInit_v2(self):
        return self.init

    def nvmlDeviceGetHandleByIndex_v2(self, index, ref):
        return self.handle_rc

    def nvmlShutdown(self):
        self.shutdowns += 1


class NvmlAudit(unittest.TestCase):
    def device(self, lib):
        n = gpu_nvml.NVML.__new__(gpu_nvml.NVML)
        n.lib, n.handle, n.needs_reinit = lib, ctypes.c_void_p(), False
        return n

    def test_i2_failed_init_releases_the_init_count(self):
        lib = FakeNvmlLib(handle=999)
        with patch.object(gpu_nvml.ctypes, "CDLL", return_value=lib):
            with self.assertRaises(RuntimeError):
                gpu_nvml.NVML()
        self.assertEqual(lib.shutdowns, 1)

    def test_i2_lost_gpu_marks_the_handle_for_replacement(self):
        n = self.device(FakeNvmlLib())
        with self.assertRaises(RuntimeError):
            n._check(15, "x")
        self.assertTrue(n.needs_reinit)
        n.needs_reinit = False
        with self.assertRaises(RuntimeError):
            n._check(3, "x")            # NOT_SUPPORTED is permanent, not a dead handle
        self.assertFalse(n.needs_reinit)

    def test_i2_proc_mib_grows_its_buffer(self):
        lib = FakeNvmlLib()

        def processes(handle, count_ref, buf):
            count = count_ref._obj
            if count.value < 100:
                count.value = 100
                return gpu_nvml.NVML_ERROR_INSUFFICIENT_SIZE
            for i in range(100):
                buf[i].pid, buf[i].usedGpuMemory = 1000 + i, (i + 1) << 20
            count.value = 100
            return 0

        lib.nvmlDeviceGetComputeRunningProcesses_v3 = processes
        self.assertEqual(self.device(lib).proc_mib(1099), 100)

    def test_i2_missing_v3_symbol_is_a_runtime_error(self):
        n = self.device(FakeNvmlLib())     # has no nvmlDeviceGetComputeRunningProcesses_v3
        with self.assertRaises(RuntimeError):
            n.proc_mib(1)
        with self.assertRaises(RuntimeError):
            n.processes()

    def test_l5_utilisation_keeps_the_peak_per_pid(self):
        lib = FakeNvmlLib()
        samples = [(7, 1, 95), (7, 2, 10), (8, 2, 5)]     # not chronological per pid

        def util(handle, buf, count_ref, last_seen):
            for i, (pid, stamp, sm) in enumerate(samples):
                buf[i].pid, buf[i].timeStamp, buf[i].smUtil = pid, stamp, sm
            count_ref._obj.value = len(samples)
            return 0

        lib.nvmlDeviceGetProcessUtilization = util
        values, latest = self.device(lib).process_utilization(0)
        self.assertEqual(values, {7: 95, 8: 5})
        self.assertEqual(latest, 2)


class Once:
    """A stop event that lets a `while not stop.is_set()` loop run exactly one iteration."""
    def __init__(self):
        self.calls = 0

    def is_set(self):
        self.calls += 1
        return self.calls > 1

    def wait(self, _seconds):
        return False


class GatewayGameAudit(unittest.TestCase):
    """Gateway wiring of the game-mode fixes."""
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


    def test_l1_missing_nvml_is_an_error_sample_not_a_missing_sample(self):
        self.game()
        self.patch_attr("nvml_device", Mock(side_effect=OSError("libnvidia-ml.so.1 not found")))
        self.assertFalse(self.gw.GAME.ready)
        self.gw.game_monitor(Once())
        self.assertTrue(self.gw.GAME.ready)             # requests are served instead of 503 forever
        self.assertIn("NVML: libnvidia-ml", self.gw.GAME.errors[0])
        self.assertFalse(self.gw.GAME.active)

    def test_l1_any_collect_failure_still_reaches_the_policy(self):
        self.game(gameDetectNvml=False)
        self.gw.GAME_SIGNALS.collect = Mock(side_effect=ValueError("boom"))
        self.gw.game_monitor(Once())
        self.assertTrue(self.gw.GAME.ready)
        self.assertIn("boom", self.gw.GAME.errors[0])

    def test_l1_model_process_vanishing_between_check_and_read_is_harmless(self):
        self.game(gameDetectNvml=False)

        class Flaky:           # st.proc is a process the first time it is read and None the second time
            lock = __import__("threading").RLock()
            reads = 0

            @property
            def proc(self):
                self.reads += 1
                return Mock(pid=1) if self.reads == 1 else None

        self.patch_attr("st", Flaky())
        self.patch_attr("game_tick", Mock(return_value=[]))
        self.gw.GAME_SIGNALS.collect = Mock(return_value=([], []))
        self.gw.game_monitor(Once())
        self.gw.game_tick.assert_called_once()


    def test_i2_a_dead_nvml_handle_is_replaced_on_the_next_request(self):
        gw = self.gw
        dead, fresh = Mock(needs_reinit=True), Mock(needs_reinit=False)
        self.patch_attr("_NVML_DEVICE", dead)
        self.patch_attr("NVML", Mock(return_value=fresh))
        self.assertIs(gw.nvml_device(), fresh)
        dead.close.assert_called_once()
        self.assertIs(gw.nvml_device(), fresh)           # a healthy handle is kept
        gw.NVML.assert_called_once()

    def test_i2_a_mock_or_healthy_handle_is_never_replaced(self):
        gw = self.gw
        healthy = Mock()                                  # needs_reinit is a truthy Mock, not True
        self.patch_attr("_NVML_DEVICE", healthy)
        self.patch_attr("NVML", Mock())
        self.assertIs(gw.nvml_device(), healthy)
        gw.NVML.assert_not_called()


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
