"""RAM valve actions through the whole gateway with the fake llama-server (tests/support) and substituted memory readings.

The readings come from a fake /proc tree (ramGuardProcRoot), so the levels move on demand. Nothing real is exhausted.
"""
import sys
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
from gateway_harness import GatewayHarness  # noqa: E402

REQUEST_SHORT = "hello"


def write_meminfo(proc_dir, avail_gb):
    proc_dir.mkdir(parents=True, exist_ok=True)
    (proc_dir / "meminfo").write_text(f"MemTotal:       32000000 kB\nMemAvailable:   {int(avail_gb * 1024 * 1024)} kB\n")


class RamValveActions(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = Path(tempfile.mkdtemp(prefix="ram1b-", dir="/dev/shm"))
        self.proc = self.temp / "proc"
        self.sysroot = self.temp / "sys"
        self.sysroot.mkdir()
        self.gw = None

    def tearDown(self):
        if self.gw is not None:
            self.gw.__exit__(None, None, None)
        import shutil
        shutil.rmtree(self.temp, ignore_errors=True)

    def start(self, extra=None):
        guardian = {"slotSaveCheckpoints": 1, "ramGuardProcRoot": str(self.proc), "ramGuardSysRoot": str(self.sysroot),
                    "ramGuardIdleSeconds": 0, "pollIntervalSeconds": 1}
        guardian.update(extra or {})
        self.gw = GatewayHarness(guardian=guardian)
        self.gw.__enter__()
        return self.gw

    def wait_until(self, predicate, timeout=30, what="condition"):
        end = time.time() + timeout
        while time.time() < end:
            if predicate():
                return True
            time.sleep(0.5)
        self.fail(f"timed out waiting for {what}")

    def status_is(self, value):
        return lambda: self.gw.status()["status"] == value

    def test_critical_unloads_with_the_session_in_ram_and_returns_when_memory_eases(self):
        write_meminfo(self.proc, 8.0)
        self.start({"ramGuard": True})
        status, _ = self.gw.chat(content=REQUEST_SHORT, timeout=120)
        self.assertEqual(status, 200)
        self.assertEqual(self.gw.status()["status"], "READY")
        write_meminfo(self.proc, 1.0)                        # critical
        self.wait_until(self.status_is("UNLOADED"), what="the model to unload")
        log = self.gw.log()
        self.assertIn("ram_valve_yield", log)
        status, payload = self.gw.chat(content=REQUEST_SHORT, timeout=30)
        self.assertEqual(status, 503)                         # the load waits while memory is critical
        self.assertIn("RAM valve", str(payload))
        write_meminfo(self.proc, 8.0)                        # back to ok
        status, _ = self.wait_until_chat_ok(timeout=60)
        self.assertEqual(status, 200)
        self.assertLess(log.index("ram_guard_level"), log.index("ram_valve_yield"))

    def wait_until_chat_ok(self, timeout):
        end = time.time() + timeout
        last = (None, None)
        while time.time() < end:
            last = self.gw.chat(content=REQUEST_SHORT, timeout=60)
            if last[0] == 200:
                return last
            time.sleep(1)
        return last

    def test_default_off_leaves_the_model_loaded_under_critical_readings(self):
        write_meminfo(self.proc, 8.0)
        self.start({"ramGuard": False})
        self.assertEqual(self.gw.chat(content=REQUEST_SHORT, timeout=120)[0], 200)
        write_meminfo(self.proc, 1.0)
        time.sleep(3)
        self.assertEqual(self.gw.status()["status"], "READY")
        self.assertNotIn("ram_valve_yield", self.gw.log())

    def test_tight_defers_a_large_new_prompt_only_while_the_desktop_floor_would_not_hold(self):
        # RAMFIX1: tight (< 6 GiB) alone no longer answers 503: a large prompt goes on while MemAvailable stays >= ramDeskFloorGB (4);
        # below that the gateway waits itself and goes on when it holds; only a wait that never ends in room gives the 503
        write_meminfo(self.proc, 8.0)
        self.start({"ramGuard": True, "ramValveDeferChars": 200, "ramValveDeferWaitSeconds": 3})
        self.assertEqual(self.gw.chat(content=REQUEST_SHORT, timeout=120)[0], 200)
        write_meminfo(self.proc, 5.5)                       # tight, the floor holds
        self.wait_until(lambda: "ram_guard_level" in self.gw.log(), what="the tight level event")
        self.assertEqual(self.gw.chat(content="x" * 1000, timeout=30)[0], 200)
        self.assertNotIn("ram_valve_defer", self.gw.log())
        write_meminfo(self.proc, 4.02)                      # tight, and the prompt would take the desktop below 4 GiB
        status, payload = self.gw.chat(content="y" * 2000, timeout=30)
        self.assertEqual(status, 503)                        # 3 s of waiting never made the room: the honest answer
        self.assertIn("System memory is tight", str(payload))
        self.assertIn("ram_valve_defer_wait", self.gw.log())
        self.assertEqual(self.gw.chat(content=REQUEST_SHORT, timeout=60)[0], 200)      # a small one goes through
        threading.Timer(1.0, write_meminfo, (self.proc, 5.5)).start()                 # memory comes back while it waits
        self.assertEqual(self.gw.chat(content="z" * 3000, timeout=30)[0], 200)
        self.assertIn("ram_valve_defer_done", self.gw.log())

if __name__ == "__main__":
    unittest.main()
