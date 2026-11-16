"""RAMFIX1 in the real gateway against the fake server and a fake /proc: the scenario of 2026-11-16. A long session is saved to
the slot folder in RAM at an unload; the desktop is busy; the next request must load the model, not refuse it. The fake /proc
gives the snapshot's 2 GiB back to honest free once the file has left RAM (as tmpfs does). The disk folder for moved sessions
must not be tmpfs, so it lives under ~/.cache (the fake snapshots are a few KB) and is removed afterwards."""
import os
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

from gateway_harness import GatewayHarness
from test_gateway_ram_ladder_e2e import (
    CACHE,
    FIXED,
    HARD_MIN,
    RESERVE,
    write_mem,
)

SNAPSHOT_MIB = 2048


class RamFixE2E(unittest.TestCase):
    def setUp(self):
        self.proc = Path(tempfile.mkdtemp(prefix="ramfix1-proc-", dir="/dev/shm"))
        self.addCleanup(shutil.rmtree, self.proc, True)
        self.addCleanup(shutil.rmtree, self.proc.parent / (self.proc.name + ".next"), True)
        disk_root = Path.home() / ".cache"
        disk_root.mkdir(exist_ok=True)
        self.spill = Path(tempfile.mkdtemp(prefix="ramfix1-spill-", dir=disk_root))
        self.addCleanup(shutil.rmtree, self.spill, True)
        self.stop = threading.Event()
        self.addCleanup(self.stop.set)

    def harness(self, free_mib):
        write_mem(self.proc, free_mib)
        return GatewayHarness(guardian={"ramPolicy": "honest", "ramWaitSeconds": 4, "deskNeed": False, "vramGuard": True,
                                        "slotSaveMinTokens": 0, "slotSpillDir": str(self.spill),
                                        "spillPrefillTps": 0.01, "ssdWriteDailyLimitGB": 8},
                              env={"FAKE_PROC_ROOT": str(self.proc)})

    def follow_the_snapshot(self, slots, base_free):
        """The fake kernel: while a saved session sits in the RAM slot folder it holds SNAPSHOT_MIB of memory."""
        def run():
            while not self.stop.is_set():
                held = any(p.suffix == ".bin" for p in slots.glob("*.bin")) if slots.is_dir() else False
                staging = self.proc.parent / (self.proc.name + ".next")   # atomic: the gateway never reads a half-written file
                staging.mkdir(exist_ok=True)
                write_mem(staging, base_free - (SNAPSHOT_MIB if held else 0))
                os.replace(staging / "meminfo", self.proc / "meminfo")
                time.sleep(0.005)
        threading.Thread(target=run, daemon=True).start()

    def save_a_session(self, gw):
        self.assertEqual(gw.chat("a long session")[0], 200, gw.log()[-1500:])
        self.assertEqual(gw.request("POST", "/guardian/unload")[0], 200)
        gw.wait_for(lambda: gw.status()["status"] == "UNLOADED", timeout=20, what="the unload")
        slots = Path(gw.tmp) / "slots"
        saved = list(slots.glob("*.bin"))
        self.assertEqual(len(saved), 1, f"the unload saved the session in the RAM slot folder: {list(slots.iterdir())}")
        return slots, saved[0]

    def test_the_own_snapshot_moves_to_disk_the_model_loads_and_the_session_is_restored(self):
        # without the snapshot honest free is FIXED + CACHE + RESERVE + 500: a plain load; with it 1.5 GiB short
        base = FIXED + CACHE + RESERVE + 500
        with self.harness(base + SNAPSHOT_MIB) as gw:
            slots, saved = self.save_a_session(gw)
            self.follow_the_snapshot(slots, base)
            time.sleep(0.2)
            started = time.monotonic()
            code, body = gw.chat("the next question")
            self.assertEqual(code, 200, body)
            self.assertLess(time.monotonic() - started, 4, "loaded at once, without the RAM wait")
            self.assertFalse(saved.exists(), "the snapshot left RAM")
            self.assertIn("ram_spill", gw.full_log())
            self.assertIn("moved to the disk", gw.full_log())
            self.assertTrue(any("action=restore" in str(e.get("path", "")) for e in gw.fake_log()), "the session was restored")
            self.assertFalse((self.spill / saved.name).exists(), "read back from the SSD: deleted there at once")
            self.assertNotIn("checkpoint_rejected", gw.full_log())
            # "light" only when the re-read raced the fake kernel's 5 ms update; never a wait, a guarded load or a refusal
            self.assertIn(gw.status()["ram_ladder"]["state"], ("idle", "light"))

    def test_a_real_shortage_still_waits_and_loads_guarded_without_touching_the_session_folder_twice(self):
        # even with the snapshot gone memory is short: the ladder (wait, then guarded above the hard minimum) still runs
        base = FIXED + HARD_MIN + 300
        with self.harness(base + SNAPSHOT_MIB) as gw:
            slots, saved = self.save_a_session(gw)
            self.follow_the_snapshot(slots, base)
            time.sleep(0.2)
            code, body = gw.chat("the next question")
            self.assertEqual(code, 200, body)
            self.assertFalse(saved.exists())
            self.assertIn("moved to the disk", gw.full_log())
            self.assertEqual(gw.status()["ram_ladder"]["state"], "guarded")


if __name__ == "__main__":
    unittest.main()
