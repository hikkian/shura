"""GPURESET1: a GPU reset is told apart from a crash of the model. The kernel log of the real event of 2026-10-25 14:12 (RX 7800 XT,
kernel 7.2.11) is the fixture; the gateway part runs with a mocked server process and a patched kernel-log reader."""
import io
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_gateway as fixture
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
import gpu_reset  # noqa: E402

REAL = """\
1761390753.100000 hkkfedora kernel: amdgpu 0000:2d:00.0: ring gfx_0.0.0 timeout, signaled seq=47755563, emitted seq=47755565
1761390753.100100 hkkfedora kernel: amdgpu 0000:2d:00.0:  Process kwin_wayland pid 3862 thread kwin_wayla:cs0 pid 3904
1761390753.200000 hkkfedora kernel: [drm:gfx_v11_0_bad_op_irq [amdgpu]] *ERROR* Illegal opcode in command stream
1761390755.000000 hkkfedora kernel: amdgpu 0000:2d:00.0: MES failed to respond to msg=RESET
1761390755.100000 hkkfedora kernel: amdgpu 0000:2d:00.0: GPU reset begin!. Source:  1
1761390755.200000 hkkfedora kernel: amdgpu: Failed to suspend process pid 1020885
1761390758.300000 hkkfedora kernel: amdgpu 0000:2d:00.0: VRAM is lost due to GPU reset!
1761390759.400000 hkkfedora kernel: amdgpu 0000:2d:00.0: GPU reset(1) succeeded!
"""
RING_ONLY = """\
1761390753.1 h kernel: amdgpu 0000:2d:00.0: ring gfx_0.0.0 timeout, signaled seq=1, emitted seq=2
1761390753.2 h kernel: amdgpu 0000:2d:00.0: Ring gfx_0.0.0 reset succeeded
"""
PAGE_FAULT = "1761390753.1 h kernel: amdgpu 0000:2d:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:660)\n"


@patch.dict(os.environ, {"AI_GATEWAY_KERNEL_LOG": "on"})
class Parse(unittest.TestCase):
    def test_the_real_event(self):
        info = gpu_reset.parse_kernel_log(REAL, pid=1020885)
        self.assertEqual((info["ring"], info["culprit"], info["vram_lost"], info["recovered"], info["ours"]),
                         ("gfx_0.0.0", "kwin_wayland", True, True, True))
        self.assertAlmostEqual(info["at"], 1761390755.1)
        self.assertLessEqual(len(info["lines"]), 6)
        text = gpu_reset.label(-6, info)
        self.assertIn("GPU reset", text)
        self.assertIn("kwin_wayland", text)
        self.assertIn("exit -6", text)
        self.assertIn("not at fault", text)

    def test_a_reset_that_does_not_name_our_server_is_still_a_reset(self):
        self.assertFalse(gpu_reset.parse_kernel_log(REAL, pid=4242)["ours"])
        self.assertIsNotNone(gpu_reset.parse_kernel_log(REAL, pid=4242))

    def test_a_recovered_ring_timeout_or_a_page_fault_is_not_a_card_reset(self):
        self.assertIsNone(gpu_reset.parse_kernel_log(RING_ONLY))
        self.assertIsNone(gpu_reset.parse_kernel_log(PAGE_FAULT))
        self.assertIsNone(gpu_reset.parse_kernel_log(""))
        self.assertIsNone(gpu_reset.parse_kernel_log("garbage\nmore garbage\n"))

    def test_check_after_exit_asks_the_kernel_log_since_the_start_and_never_raises(self):
        calls = []

        def run(cmd, timeout=3.0):
            calls.append(cmd)
            return 0, REAL
        info = gpu_reset.check_after_exit(run, started_wall=1761390000.0, now_wall=1761390800.0, pid=1)
        self.assertIsNotNone(info)
        self.assertEqual(calls[0][:2], ["journalctl", "-k"])
        self.assertIn("@1761389995", calls[0])
        self.assertIsNone(gpu_reset.check_after_exit(lambda cmd, timeout=3.0: (1, ""), 0, 10, 1))

        def boom(cmd, timeout=3.0):
            raise RuntimeError("journal exploded")
        self.assertIsNone(gpu_reset.check_after_exit(boom, 0, 10, 1))

    def test_the_window_is_bounded(self):
        calls = []
        gpu_reset.check_after_exit(lambda cmd, timeout=3.0: calls.append(cmd) or (0, ""), started_wall=0.0, now_wall=1761390800.0, pid=1)
        since = int(calls[0][-1][1:])
        self.assertGreaterEqual(since, 1761390800 - 6 * 3600)


class Ledger(unittest.TestCase):
    def test_limit_in_a_window(self):
        ledger = gpu_reset.ResetLedger({"gpuResetMaxPerWindow": 3, "gpuResetWindowSeconds": 900})
        self.assertEqual([ledger.note({}, t) for t in (0, 100, 200)], [False, False, True])
        self.assertTrue(ledger.note({}, 300))                  # still three inside the window
        fresh = gpu_reset.ResetLedger({"gpuResetMaxPerWindow": 3, "gpuResetWindowSeconds": 900})
        self.assertEqual([fresh.note({}, t) for t in (0, 1000, 2000, 3000)], [False] * 4)   # spread out: never a loop
        self.assertEqual(fresh.snapshot()["total"], 4)

    def test_config(self):
        gpu_reset.validate_config({})
        for bad in ({"gpuResetRetry": "yes"}, {"gpuResetMaxPerWindow": 0}, {"gpuResetWindowSeconds": "900"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                gpu_reset.validate_config(bad)


@patch.dict(os.environ, {"AI_GATEWAY_KERNEL_LOG": "on"})
class GatewayPart(unittest.TestCase):
    setUpClass = classmethod(fixture.GatewayGuard.setUpClass.__func__)
    tearDownClass = classmethod(fixture.GatewayGuard.tearDownClass.__func__)
    setUp = fixture.GatewayGuard.setUp
    patch_attr = fixture.GatewayGuard.patch_attr
    fake_llama = fixture.GatewayGuard.fake_llama

    def kernel(self, text):
        p = patch.object(gpu_reset, "run_command", return_value=(0, text))
        p.start()
        self.addCleanup(p.stop)

    def dead(self, code=-6, pid=1234):
        self.proc.pid = pid
        self.proc.returncode = code
        self.proc.poll.return_value = code
        self.proc.args = ["llama-server"]
        return self.proc

    def test_a_reset_is_labelled_counted_apart_and_not_a_strike_against_the_model(self):
        gw = self.gw
        self.kernel(REAL)
        proc = self.dead()
        gw.st.expected_exit = False
        gw.monitor_tick(100.0, 12000, 24)
        self.assertIn("GPU reset", gw.st.last_error)
        self.assertIn("not at fault", gw.st.last_error)
        self.assertEqual(gw.st.crash_count, 0)
        self.assertEqual(gw.st.gpu_resets.snapshot()["total"], 1)
        self.assertEqual(gw.st.event_counters.get("gpu_reset"), 1)
        self.assertEqual(gw.st.status, "UNLOADED")            # restarts on the next request, no ERROR state
        self.assertIsNone(gw.st.proc)
        self.assertEqual(gw.status_snapshot()["gpu_resets"]["total"], 1)

    def test_without_a_reset_in_the_kernel_log_it_is_a_crash_as_before(self):
        gw = self.gw
        self.kernel(PAGE_FAULT)
        self.dead()
        gw.st.expected_exit = False
        gw.monitor_tick(100.0, 12000, 24)
        self.assertEqual(gw.st.last_error, "llama-server crashed (exit -6)")
        self.assertEqual(gw.st.crash_count, 1)
        self.assertEqual(gw.st.gpu_resets.snapshot()["total"], 0)

    def test_the_governor_exit_75_and_a_clean_exit_never_look_at_the_kernel_log(self):
        gw = self.gw
        with patch.object(gpu_reset, "run_command", side_effect=AssertionError("must not run")):
            for code in (75, 0):
                self.assertIsNone(gw.detect_gpu_reset(Mock(pid=77, poll=Mock(return_value=code))))

    def test_a_card_that_keeps_resetting_stops_the_restarts(self):
        gw = self.gw
        self.kernel(REAL)
        limited = []
        for n in range(3):
            proc = self.dead(pid=2000 + n)
            gw.st.proc = proc
            gw.st.expected_exit = False
            gw.record_server_crash(proc, "test")
            limited.append(gw.note_crash(proc=proc))
        self.assertEqual(limited, [False, False, True])
        self.assertEqual(gw.st.crash_count, 0)

    def test_one_look_at_the_kernel_log_per_process(self):
        gw = self.gw
        self.kernel(REAL)
        proc = self.dead()
        gw.detect_gpu_reset(proc)
        gw.detect_gpu_reset(proc)
        self.assertEqual(gpu_reset.run_command.call_count, 1)

    def test_status_shows_the_reset_count(self):
        self.assertEqual(self.gw.status_snapshot()["gpu_resets"]["total"], 0)


@patch.dict(os.environ, {"AI_GATEWAY_KERNEL_LOG": "on"})
class Replay(unittest.TestCase):
    """gpuResetRetry: a request that had no token yet is replayed once on the restarted server, only when a GPU reset killed it."""
    setUpClass = classmethod(fixture.GatewayGuard.setUpClass.__func__)
    tearDownClass = classmethod(fixture.GatewayGuard.tearDownClass.__func__)
    setUp = fixture.GatewayGuard.setUp
    patch_attr = fixture.GatewayGuard.patch_attr
    fake_llama = fixture.GatewayGuard.fake_llama

    PROGRESS = b'data: {"choices":[{"delta":{}}],"prompt_progress":{"total":1000,"processed":500}}\n\n'
    TOKEN = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
    DONE = b"data: [DONE]\n\n"

    def conn(self, chunks):
        r = Mock(status=200)
        r.getheader.return_value = "text/event-stream"
        r.getheaders.return_value = [("Content-Type", "text/event-stream")]
        r.read1.side_effect = chunks
        c = Mock()
        c.getresponse.return_value = r
        return c

    def run_proxy(self, conns, kernel_text, vram_retry=False, reset_retry=True):
        gw = self.gw
        gw.G.update({"vramRetry": vram_retry, "gpuResetRetry": reset_retry})
        gw.st.busy = 1
        self.proc.poll.return_value = -6
        self.proc.returncode = -6
        self.proc.args = ["llama-server"]
        h = object.__new__(gw.Handler)
        h.command, h.path, h.headers, h.wfile = "POST", "/v1/chat/completions", {}, io.BytesIO()
        for method in ["send_response", "send_header", "end_headers", "send_json"]:
            setattr(h, method, Mock())

        def start(model, vision, layout="short"):
            gw.st.proc = Mock(pid=4321)
            gw.st.proc.poll.return_value = None
            gw.st.status = "READY"
            return ""

        with (patch.object(gpu_reset, "run_command", return_value=(0, kernel_text)),
              patch.object(gw.http.client, "HTTPConnection", side_effect=conns) as opened,
              patch.object(gw, "wait_fresh_vram", return_value=True),
              patch.object(gw, "start_llama", side_effect=start) as load):
            h.proxy_recoverable(b"{}", gw.st.model_id, False)
        return h, load, opened

    def test_a_reset_before_the_first_token_is_replayed_once(self):
        h, load, opened = self.run_proxy([self.conn([self.PROGRESS, ConnectionResetError()]), self.conn([self.TOKEN, self.DONE, b""])], REAL)
        load.assert_called_once()
        self.assertEqual(opened.call_count, 2)
        self.assertIn(b'"content":"hi"', h.wfile.getvalue())
        self.assertEqual(self.gw.st.crash_count, 0)
        self.assertEqual(self.gw.st.gpu_resets.snapshot()["total"], 1)
        self.assertEqual(self.gw.st.event_counters["response_replayed_before_content"], 1)

    def test_an_ordinary_crash_is_not_replayed_by_this_switch(self):
        h, load, opened = self.run_proxy([self.conn([self.PROGRESS, ConnectionResetError()])], PAGE_FAULT)
        load.assert_not_called()
        self.assertEqual(opened.call_count, 1)
        self.assertIn(b"SHURA_RESPONSE_INTERRUPTED", h.wfile.getvalue())

    def test_after_a_token_nothing_is_replayed_even_after_a_reset(self):
        h, load, opened = self.run_proxy([self.conn([self.PROGRESS, self.TOKEN, ConnectionResetError()])], REAL)
        load.assert_not_called()
        self.assertIn(b"SHURA_RESPONSE_INTERRUPTED", h.wfile.getvalue())

    def test_the_switch_off_means_no_replay(self):
        h, load, _ = self.run_proxy([self.conn([self.PROGRESS, ConnectionResetError()])], REAL, reset_retry=False)
        load.assert_not_called()

    def test_vram_retry_still_replays_any_stop(self):
        h, load, opened = self.run_proxy([self.conn([self.PROGRESS, ConnectionResetError()]), self.conn([self.TOKEN, self.DONE, b""])],
                                         PAGE_FAULT, vram_retry=True, reset_retry=False)
        load.assert_called_once()
        self.assertEqual(opened.call_count, 2)


if __name__ == "__main__":
    unittest.main()
