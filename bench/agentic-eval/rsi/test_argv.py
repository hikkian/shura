"""RSIL0c tests: the stand's llama-server command line comes from the gateway's build_args (offline: nothing is started, no GPU).
Run: cd bench/agentic-eval && python3 -B -m unittest discover -s rsi -p 'test_argv.py'"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rsi import server_argv  # noqa: E402


class StandCommandLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.argv = server_argv.build("tiel-coder")
        # the expected values come from the candidate file itself, so a new layout or binary in the candidate does not break the test
        cls.cand = json.loads(server_argv.CANDIDATE.read_text())

    def test_it_is_the_gateway_command_for_the_candidate_profile(self):
        a, mc = self.argv, self.cand["models"]["tiel-coder"]
        self.assertEqual(a[0], self.cand["exePath"])
        self.assertEqual(a[a.index("--port") + 1], "8091")
        self.assertEqual(a[a.index("-m") + 1], mc["modelPath"])
        self.assertEqual(a[a.index("-c") + 1], "32768")
        self.assertEqual(a[a.index("-np") + 1], "1")
        self.assertEqual(a[a.index("--cache-ram") + 1], "1024")
        self.assertEqual(a[a.index("--spec-type") + 1], "draft-mtp")

    def test_one_n_cpu_moe_flag_only(self):
        self.assertEqual(sum(1 for x in self.argv if x in ("-ncmoe", "--n-cpu-moe")), 1)
        self.assertEqual(self.argv[self.argv.index("-ncmoe") + 1], str(self.cand["models"]["tiel-coder"]["nCpuMoe"]))

    def test_the_live_state_folder_is_not_used(self):
        for x in self.argv:
            self.assertNotIn(".local/state/ai-gateway", x)

    def test_the_moe_profile_is_the_real_data_file(self):
        profile = self.argv[self.argv.index("--moe-cache-profile") + 1]
        self.assertTrue(Path(profile).exists(), profile)
        self.assertNotIn("rsil0c-argv-", profile)

    def test_the_slot_folder_is_under_dev_shm(self):
        self.assertEqual(self.argv[self.argv.index("--slot-save-path") + 1], "/dev/shm/RSIL0c/slots")

    def test_differences_to_the_reference_are_listed_not_hidden(self):
        ref = ("/home/alisher/Desktop/AI/shura-bin/x/llama-server --port 8091 -ncmoe 10 --moe-cache-live-resize -c 32768 "
               "--slot-save-path /dev/shm/LAYOUT2/slots")
        diff = server_argv.differences(self.argv, ref)
        self.assertIn("--moe-cache-live-resize", diff["missing_in_gateway"])
        self.assertIn("--slot-save-path", diff["changed"])


class ReleaseWait(unittest.TestCase):
    """RSIL0c: after the stop the real stand waits, with a timeout, until VRAM comes back, and keeps both numbers. The VRAM counters are
    patched to temporary files: the test does not touch the GPU."""

    def test_the_stand_waits_for_vram_to_come_back_and_keeps_both_numbers(self):
        import shutil
        import subprocess
        import tempfile
        import threading
        import time
        from unittest import mock
        from rsi import executor
        tmp = Path(tempfile.mkdtemp(prefix="rsil0c-vram-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        total, used = tmp / "total", tmp / "used"
        total.write_text(str(16 * 2 ** 30))
        used.write_text(str(14 * 2 ** 30))           # the server holds 14 GiB until it is stopped

        class Srv:
            def shutdown(self):
                pass

            def server_close(self):
                pass

        (tmp / "argv.json").write_text(json.dumps(["llama-server"]))
        stand = executor.GpuStand(tmp / "argv.json", tmp / "work")
        stand.free_before = 14.0
        stand.srv = Srv()
        stand.proc = subprocess.Popen(["true"], start_new_session=True)
        stand.proc.wait()

        def give_back():                             # VRAM is released half a second after the process has ended
            time.sleep(0.5)
            used.write_text(str(2 * 2 ** 30))
        threading.Thread(target=give_back, daemon=True).start()
        with mock.patch.object(executor, "VRAM_USED", used), mock.patch.object(executor, "VRAM_TOTAL", total):
            stand.__exit__(None, None, None)
        self.assertLess(stand.free_after_stop, 3, "right after the stop the memory was still held")
        self.assertTrue(stand.released)
        self.assertGreaterEqual(stand.release_wait_s, 0.4)
        self.assertGreaterEqual(stand.free_released, stand.release_target)


if __name__ == "__main__":
    unittest.main()
