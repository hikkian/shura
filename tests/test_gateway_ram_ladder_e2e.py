"""RAMLOGIC2: the RAM ladder in the real gateway against the fake server, with a fake /proc (FAKE_PROC_ROOT) whose free memory
the test changes while the gateway runs. The example model keeps 26 expert layers on the CPU: the server's fixed part is
9736 MiB and its prompt cache in RAM 4096; with deskNeed off the desktop's reserve is the ordinary 2048 MiB."""
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
sys.path.insert(0, str(HERE))

from gateway_harness import GatewayHarness  # noqa: E402

FIXED, CACHE, RESERVE, HARD_MIN = 9736, 4096, 2048, 1024


def write_mem(root, free_mib, swap_used_mib=0):
    kb = lambda mib: int(mib * 1024)  # noqa: E731
    (root / "meminfo").write_text(f"MemTotal: {kb(32768)} kB\nMemFree: {kb(free_mib)} kB\nMemAvailable: {kb(free_mib)} kB\n"
                                  f"Inactive(file): 0 kB\nShmem: 0 kB\nSwapTotal: {kb(16384)} kB\n"
                                  f"SwapFree: {kb(16384 - swap_used_mib)} kB\n")
    (root / "pressure").mkdir(exist_ok=True)
    (root / "pressure" / "memory").write_text("some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
                                              "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")


def cache_ram_of_starts(gw):
    return [int(re.search(r"--cache-ram (\d+)", line).group(1)) for line in gw.full_log().splitlines()
            if "Starting llama-server" in line]


class RamLadderE2E(unittest.TestCase):
    def setUp(self):
        self.proc = Path(tempfile.mkdtemp(prefix="ramlogic2-proc-", dir="/dev/shm" if Path("/dev/shm").is_dir() else None))
        self.addCleanup(shutil.rmtree, self.proc, True)

    def harness(self, free_mib, wait_s, **behaviour):
        write_mem(self.proc, free_mib)
        return GatewayHarness(guardian={"ramPolicy": "honest", "ramWaitSeconds": wait_s, "deskNeed": False},
                              behaviour=behaviour or None, env={"FAKE_PROC_ROOT": str(self.proc)})

    def chat_in_background(self, gw):
        result = {}
        thread = threading.Thread(target=lambda: result.update(answer=gw.chat("hello"), at=time.monotonic()), daemon=True)
        result["sent"] = time.monotonic()
        thread.start()
        return thread, result

    def test_a_shortage_waits_with_the_numbers_and_loads_when_memory_comes_back(self):
        with self.harness(FIXED + RESERVE - 1500, wait_s=15) as gw:
            thread, result = self.chat_in_background(gw)
            ladder = gw.wait_for(lambda: gw.status()["ram_ladder"]["state"] == "waiting" and gw.status()["ram_ladder"],
                                 timeout=10, what="the wait")
            self.assertEqual(ladder["need_mib"], FIXED + RESERVE)
            self.assertEqual(ladder["free_mib"], FIXED + RESERVE - 1500)
            self.assertEqual(ladder["reserve_mib"], RESERVE)
            self.assertIn("GiB free", ladder["reason"])
            self.assertTrue(thread.is_alive(), "the request waits instead of failing")
            write_mem(self.proc, FIXED + CACHE + RESERVE + 2000)         # the desktop let go of memory
            thread.join(30)
            self.assertEqual(result["answer"][0], 200, gw.log()[-1500:])
            self.assertLess(result["at"] - result["sent"], 15, "it loaded as soon as memory came back, not at the end of the wait")
            self.assertEqual(cache_ram_of_starts(gw), [CACHE])
            self.assertEqual(gw.status()["ram_ladder"]["state"], "idle")

    def test_a_smaller_prompt_cache_loads_at_once(self):
        with self.harness(FIXED + RESERVE + 700, wait_s=15) as gw:
            started = time.monotonic()
            self.assertEqual(gw.chat("hello")[0], 200, gw.log()[-1500:])
            self.assertLess(time.monotonic() - started, 10)
            self.assertEqual(cache_ram_of_starts(gw), [512])            # whole 256 MiB steps of what fits
            self.assertEqual(gw.status()["ram_ladder"]["state"], "light")

    def test_after_the_wait_a_guarded_load_above_the_hard_minimum(self):
        with self.harness(FIXED + HARD_MIN + 300, wait_s=2) as gw:
            started = time.monotonic()
            self.assertEqual(gw.chat("hello")[0], 200, gw.log()[-1500:])
            self.assertGreaterEqual(time.monotonic() - started, 2)
            self.assertEqual(cache_ram_of_starts(gw), [0])
            self.assertEqual(gw.status()["ram_ladder"]["state"], "guarded")

    def test_below_the_hard_minimum_an_honest_refusal_that_asks_nobody_to_close_anything(self):
        with self.harness(FIXED + HARD_MIN - 300, wait_s=2) as gw:
            code, body = gw.chat("hello")
            self.assertEqual(code, 503, body)
            message = body["error"]["message"]
            self.assertIn("not enough memory for the model yet", message)
            self.assertIn("GiB free", message)
            for word in ("close", "quit", "закро"):
                self.assertNotIn(word, message.lower())
            self.assertEqual(cache_ram_of_starts(gw), [])
            self.assertEqual(gw.status()["ram_ladder"]["state"], "refused")

    def test_desk_growth_during_a_load_stops_the_next_loads(self):
        with self.harness(FIXED + CACHE + RESERVE + 4000, wait_s=2, load_seconds=2.0) as gw:
            thread, result = self.chat_in_background(gw)
            gw.wait_for(lambda: gw.status()["status"] == "LOADING", timeout=10, what="the load")
            write_mem(self.proc, FIXED + CACHE + RESERVE + 2000, swap_used_mib=1600)   # the desktop grows and pushes 1.6 GiB to swap
            thread.join(30)
            self.assertEqual(result["answer"][0], 200, gw.log()[-1500:])      # the load that was running finishes
            self.assertIn("ram_swap_growth", gw.full_log())
            self.assertEqual(gw.request("POST", "/guardian/unload")[0], 200)
            gw.wait_for(lambda: gw.status()["status"] != "READY", timeout=15, what="the unload")
            code, body = gw.chat("again")
            self.assertEqual(code, 503, body)
            self.assertIn("swap grew", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()
