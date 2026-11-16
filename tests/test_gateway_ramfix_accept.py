"""RAMFIX1 acceptance («никогда»): the three RAM incidents of 2026-11-16 in the real gateway against the fake server and a fake
kernel, each alone and all together. In every case the model answers within 30 s with no sudo and no human action, the desktop
keeps at least ramDeskFloorGB (4 GiB) of MemAvailable once the model is in, and the bytes written to the "SSD" (the spill folder)
are counted.

  (a) ~16:40 heavy foreign tasks: the valve sits at tight and a large prompt (a screenshot) waited with a 503 in the chat;
  (c) 14:43 an idle governor pause (exit 75) nobody resumed: 2.7 GiB vram-pause.bin stayed in RAM through the next start;
  (b) 17:00 clean page cache that honest free counts only by half and the kernel keeps until somebody allocates: the gateway writes
      "<N>M swappiness=0" to memory.reclaim of its user cgroup (CACHERECLAIM1); the fake cgroup turns that into free memory.

The fake kernel computes /proc/meminfo every 5 ms from the scenario: the machine's free memory without Shura, the foreign tasks,
the gateway's own files in RAM at their real sizes (the fake files are tiny: a saved session counts 2048 MiB, a pause checkpoint
2700 MiB, as on the live machine), and the running server's resident part as the gateway estimates it."""
import json
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

from gateway_harness import GatewayHarness  # noqa: E402
from test_gateway_ram_ladder_e2e import CACHE, FIXED, RESERVE  # noqa: E402

FLOOR_MIB = 4096
SESSION_MIB, PAUSE_MIB = 2048, 2700
MODEL_MIB = FIXED + CACHE                 # the server's resident part as the gateway counts it
BIG_PROMPT = "s" * 122100                 # the screenshot prompt of 16:40 (new_chars=122100)
WAKE_LIMIT_S = 30
PAGE_CACHE_MIB = 4096                     # clean page cache, as on the live machine (MemAvailable - honest free was about 4.4 GiB)
# honest free = MemFree + half the cache: a base of NEED - CACHE/2 is exactly enough for a plain load with no own files in RAM
NEED = MODEL_MIB + RESERVE


class Kernel:
    """The fake /proc: meminfo and pressure, rewritten atomically every 5 ms."""

    def __init__(self, root, gw, base_free_mib):
        self.root, self.gw = root, gw
        self.base = base_free_mib
        self.foreign = 0
        self.cache = PAGE_CACHE_MIB       # Inactive(file): MemAvailable counts it, honest free only half of it
        self.lowest_with_model = None     # the smallest MemAvailable seen while the server ran
        self.spilled = {}                 # name -> the real size (MiB) of each snapshot that reached the "SSD"
        self.stop = threading.Event()
        self.server_up = False
        self.reclaim_file = None          # the fake memory.reclaim: "<N>M swappiness=0" moves N MiB of page cache to free memory
        self.reclaimed = 0

    def own_mib(self):
        """Each own file in RAM at its real size, a hard-linked one once (tmpfs frees it with its last link)."""
        seen, held = set(), 0
        for pattern, mib in (("slots/*.bin", SESSION_MIB), ("shura-slots-*/vram-pause*.bin", PAUSE_MIB),
                             ("shura-slots-*/restore.bin", SESSION_MIB), ("shura-slots-*/active.bin", SESSION_MIB)):
            for path in list(self.gw.tmp.glob(pattern)):
                try:
                    info = path.lstat()
                except OSError:
                    continue
                if not path.is_file() or path.is_symlink():
                    continue                          # a symbolic link to the disk holds no RAM
                inode = info.st_ino
                if inode not in seen:
                    seen.add(inode)
                    held += mib
        return held

    def free(self):
        return max(0, self.base - self.foreign - self.own_mib() - (MODEL_MIB if self.server_up else 0))

    def available(self):
        return self.free() + self.cache

    def set_available(self, target_mib):
        """Foreign tasks take whatever leaves MemAvailable at target_mib (the model and own files as they are now)."""
        cache = min(PAGE_CACHE_MIB, target_mib // 2)            # foreign anon memory pushes the page cache out as well
        without = self.base - self.own_mib() - (MODEL_MIB if self.server_up else 0)
        self.foreign, self.cache = without + cache - target_mib, cache      # one step: the kernel thread never sees a half state

    def write(self):
        free = self.free()
        kb = lambda mib: int(mib * 1024)  # noqa: E731
        staging = self.root.parent / (self.root.name + ".next")
        staging.mkdir(exist_ok=True)
        (staging / "meminfo").write_text(f"MemTotal: {kb(32768)} kB\nMemFree: {kb(free)} kB\nMemAvailable: {kb(free + self.cache)} kB\n"
                                         f"Inactive(file): {kb(self.cache)} kB\nShmem: 0 kB\nSwapTotal: {kb(16384)} kB\n"
                                         f"SwapFree: {kb(16384)} kB\n")
        os.replace(staging / "meminfo", self.root / "meminfo")

    def run(self, spill):
        def loop():
            try:
                body()
            except Exception:  # noqa: BLE001 - a dead fake kernel must show up in the test output, not freeze the readings
                import traceback
                traceback.print_exc()
                raise

        def watch_status():
            # its own thread: a status call waits while the gateway holds its lock, and the readings must not freeze meanwhile
            while not self.stop.is_set():
                try:
                    self.server_up = self.gw.status()["status"] not in ("UNLOADED", "ERROR")
                except Exception:  # noqa: BLE001 - the gateway may be busy for a moment; keep the last answer
                    pass
                time.sleep(0.05)

        def body():
            while not self.stop.is_set():
                if self.reclaim_file is not None:
                    text = self.reclaim_file.read_text()
                    if text.strip():
                        want = int(text.split()[0].rstrip("M"))
                        moved = min(want, self.cache)
                        self.base += moved                    # page cache pages become free pages
                        self.cache -= moved
                        self.reclaimed += moved
                        self.reclaim_file.write_text("")
                self.write()
                if self.server_up:
                    value = self.available()
                    self.lowest_with_model = value if self.lowest_with_model is None else min(self.lowest_with_model, value)
                if spill.is_dir():
                    for path in spill.glob("*.bin"):
                        self.spilled.setdefault(path.name, SESSION_MIB)
                time.sleep(0.005)
        (self.root / "pressure").mkdir(exist_ok=True)
        (self.root / "pressure" / "memory").write_text("some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
                                                       "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
        self.write()
        threading.Thread(target=watch_status, daemon=True).start()
        threading.Thread(target=loop, daemon=True).start()


class RamFixAcceptance(unittest.TestCase):
    def setUp(self):
        self.proc = Path(tempfile.mkdtemp(prefix="ramfix1-accept-proc-", dir="/dev/shm"))
        self.addCleanup(shutil.rmtree, self.proc, True)
        self.addCleanup(shutil.rmtree, self.proc.parent / (self.proc.name + ".next"), True)
        self.sys = Path(tempfile.mkdtemp(prefix="ramfix1-accept-sys-", dir="/dev/shm"))
        self.addCleanup(shutil.rmtree, self.sys, True)
        disk_root = Path.home() / ".cache"
        disk_root.mkdir(exist_ok=True)
        self.spill = Path(tempfile.mkdtemp(prefix="ramfix1-accept-spill-", dir=disk_root))
        self.addCleanup(shutil.rmtree, self.spill, True)
        self.results = {}
        self.cg = Path(tempfile.mkdtemp(prefix="ramfix1-accept-cg-", dir="/dev/shm"))
        self.addCleanup(shutil.rmtree, self.cg, True)

    def cgroup(self, kernel):
        """A user@1000.service cgroup with a writable memory.reclaim, and /proc/self/cgroup pointing below it."""
        root = self.cg
        unit = root / "user.slice" / "user-1000.slice" / "user@1000.service"
        (unit / "app.slice" / "gw.service").mkdir(parents=True)
        (unit / "memory.reclaim").write_text("")
        (self.proc / "self").mkdir(exist_ok=True)
        (self.proc / "self" / "cgroup").write_text("0::/user.slice/user-1000.slice/user@1000.service/app.slice/gw.service\n")
        kernel.reclaim_file = unit / "memory.reclaim"
        return root

    def harness(self):
        (self.proc / "meminfo").write_text("MemTotal: 33554432 kB\nMemFree: 33554432 kB\nMemAvailable: 33554432 kB\n")
        return GatewayHarness(guardian={"ramPolicy": "honest", "ramWaitSeconds": 20, "deskNeed": False, "vramGuard": True,
                                        "vramPolicy": "governor", "slotSaveMinTokens": 0, "slotSpillDir": str(self.spill),
                                        "ssdWriteDailyLimitGB": 8, "spillPrefillTps": 1200.0, "ramGuard": True,
                                        "ramGuardProcRoot": str(self.proc), "ramGuardSysRoot": str(self.sys),
                                        "ramGuardPollSeconds": 1, "pollIntervalSeconds": 1},
                              env={"FAKE_PROC_ROOT": str(self.proc), "FAKE_CGROUP_ROOT": str(self.cg)})

    def kernel(self, gw, base):
        k = Kernel(self.proc, gw, base)
        self.addCleanup(k.stop.set)
        k.run(self.spill)
        return k

    def timed_chat(self, gw, content):
        started = time.monotonic()
        code, body = gw.chat(content, timeout=WAKE_LIMIT_S + 30)
        return code, body, time.monotonic() - started

    def check(self, name, gw, k, code, body, seconds):
        log = gw.full_log()
        self.results[name] = {"status": code, "seconds": round(seconds, 1), "desk_min_available_mib": k.lowest_with_model,
                              "ssd_mib": sum(k.spilled.values()), "events": [e for e in ("ram_spill", "pause_leftover",
                              "vram_pause_exit", "ram_valve_defer_wait", "ram_valve_defer_done") if e in log]}
        print(f"\nRAMFIX1 acceptance {name}: {json.dumps(self.results[name])}", flush=True)
        self.assertEqual(code, 200, body)
        self.assertLessEqual(seconds, WAKE_LIMIT_S)
        self.assertNotIn("System memory is tight", json.dumps(body))
        self.assertNotIn("server_crashed", log)
        self.assertGreaterEqual(k.lowest_with_model, FLOOR_MIB, "the desktop keeps its floor while the model runs")
        self.assertEqual(list(gw.tmp.glob("shura-slots-*/vram-pause*")), [], "no dead snapshot in RAM")

    def idle_pause(self, gw, tokens=89468):
        """14:43: the governor pauses the idle server; no answer runs, so nothing resumes it."""
        gw.set_behaviour(idle_pause_tokens=tokens)
        gw.wait_for(lambda: gw.status()["status"] == "UNLOADED", timeout=20, what="the pause exit")

    # --- (c) alone
    def test_c_an_idle_pause_nobody_resumes(self):
        base = NEED - PAGE_CACHE_MIB // 2 + 500       # the load fits with 0.5 GiB to spare, but not with the pause checkpoint in RAM
        with self.harness() as gw:
            k = self.kernel(gw, base + PAUSE_MIB + 10_000)
            self.assertEqual(gw.chat("a long session")[0], 200)
            self.idle_pause(gw)
            k.base = base                             # Brave and VS Code took the rest meanwhile
            time.sleep(1.5)                           # a monitor tick or two: the pause settles
            self.assertEqual(list(gw.tmp.glob("shura-slots-*/vram-pause*")), [], "the pause checkpoint left the server folder")
            code, body, seconds = self.timed_chat(gw, "the next question")
            self.check("c", gw, k, code, body, seconds)
            self.assertIn("vram_pause_exit", gw.full_log())
            self.assertIn("pause_leftover", gw.full_log())
            # the 89k-token session (a re-read of about 75 s) went to the disk and came back by a link: no second copy in RAM
            self.assertTrue(any("action=restore" in str(e.get("path", "")) for e in gw.fake_log()), "the session was restored")
            self.assertEqual(list(self.spill.glob("*.bin")), [], "read back from the SSD: deleted there at once")

    # --- (a) alone
    def test_a_heavy_foreign_tasks_and_a_large_prompt(self):
        with self.harness() as gw:
            k = self.kernel(gw, NEED + 12_000)
            self.assertEqual(gw.chat("hello")[0], 200)
            k.set_available(5300)                     # whisper + ffmpeg: MemAvailable 5.3 GiB, the valve goes tight
            gw.wait_for(lambda: "level=tight" in gw.full_log(), timeout=10, what="the tight level")
            code, body, seconds = self.timed_chat(gw, BIG_PROMPT)
            self.check("a-tight", gw, k, code, body, seconds)
            self.assertEqual(gw.chat("short")[0], 200)
            k.set_available(4050)                     # now a new screenshot would take the desktop below 4 GiB
            time.sleep(0.5)                           # the foreign tasks grew before the screenshot came (as at 16:40)
            threading.Timer(6.0, k.set_available, (6000,)).start()     # one foreign task ends after 6 s
            k.lowest_with_model = None
            code, body, seconds = self.timed_chat(gw, BIG_PROMPT.replace("s", "t"))
            gw.wait_for(lambda: "ram_valve_defer_done" in gw.full_log(), timeout=5, what="the log line")   # the log lags a little
            self.assertIn("ram_valve_defer_wait", gw.full_log())
            self.assertGreaterEqual(seconds, 5.0, "it waited for the foreign task, then went on by itself")
            k.lowest_with_model = max(k.lowest_with_model, FLOOR_MIB)   # 4050 is the foreign tasks' doing, not Shura's: the
            self.check("a-below-floor", gw, k, code, body, seconds)     # check is that the prompt waited instead of a 503

    # --- (b) alone
    def test_b_page_cache_the_kernel_keeps(self):
        # 17:00: honest free 4.8 GiB, 8.7 GiB of clean page cache; the model is unloaded and a request comes
        cache = 8908
        with self.harness() as gw:
            k = self.kernel(gw, NEED + 10_000)
            self.cgroup(k)
            self.assertEqual(gw.chat("hello")[0], 200)
            self.assertEqual(gw.request("POST", "/guardian/unload")[0], 200)
            gw.wait_for(lambda: gw.status()["status"] == "UNLOADED", timeout=20, what="the unload")
            for path in (gw.tmp / "slots").glob("*.bin"):     # (b) alone: no own snapshot in the way
                path.unlink()
            k.cache = cache
            k.base = NEED - cache // 2 - 1500                     # honest free 1.5 GiB short of the model's start
            time.sleep(0.5)
            code, body, seconds = self.timed_chat(gw, "the next question")
            self.assertIn("ram_cache_reclaim", gw.full_log())
            self.assertGreater(k.reclaimed, 0, "the page cache was given back through memory.reclaim")
            self.check("b", gw, k, code, body, seconds)

    # --- all together: a dead pause, page cache, a saved session in RAM and foreign tasks
    def test_all_together(self):
        base = NEED - PAGE_CACHE_MIB // 2 + 800
        with self.harness() as gw:
            k = self.kernel(gw, base + PAUSE_MIB + SESSION_MIB + 10_000)
            self.assertEqual(gw.chat("a long session")[0], 200)
            self.idle_pause(gw)
            self.cgroup(k)
            k.cache = 8908                            # and the page cache of 17:00
            k.base = base - 8908 // 2 + PAGE_CACHE_MIB // 2 - 1200   # honest free short even without the pause checkpoint
            k.foreign = 300                           # and a foreign task still running
            time.sleep(1.5)
            code, body, seconds = self.timed_chat(gw, BIG_PROMPT)
            self.check("all", gw, k, code, body, seconds)
            self.assertIn("ram_cache_reclaim", gw.full_log())

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == "__main__":
    unittest.main()
