"""RAM valve policy (gateway/ram_valve.py): levels with hysteresis on fake /proc files, oom_score_adj writes, cache budget,
daily SSD write budget. Nothing here touches the real /proc or /sys or the GPU."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gateway"))
sys.path.insert(0, str(HERE / "support"))
import ram_valve as rv  # noqa: E402


def write_meminfo(proc_dir, avail_gb):
    proc_dir.mkdir(parents=True, exist_ok=True)
    (proc_dir / "meminfo").write_text(f"MemTotal:       32000000 kB\nMemAvailable:   {int(avail_gb * 1024 * 1024)} kB\n")


def meminfo(avail_gb):
    return f"MemTotal:       32000000 kB\nMemAvailable:   {int(avail_gb * 1024 * 1024)} kB\n"


class ParseReadings(unittest.TestCase):
    def test_meminfo_available_in_gb(self):
        self.assertAlmostEqual(rv.parse_meminfo_available_gb(meminfo(4.0)), 4.0, places=3)

    def test_meminfo_missing_or_bad_is_none(self):
        self.assertIsNone(rv.parse_meminfo_available_gb("MemTotal: 1 kB\n"))
        self.assertIsNone(rv.parse_meminfo_available_gb("MemAvailable: abc kB\n"))

    def test_psi_full_avg10(self):
        text = "some avg10=1.00 avg60=0.50 avg300=0.10 total=1\nfull avg10=7.25 avg60=3.00 avg300=1.00 total=9\n"
        self.assertEqual(rv.parse_psi_full_avg10(text), 7.25)
        self.assertIsNone(rv.parse_psi_full_avg10("some avg10=1.00\n"))
        self.assertIsNone(rv.parse_psi_full_avg10("full avg10=-3\n"))

    def test_read_readings_from_fake_trees(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            proc = Path(temp) / "proc"
            (proc / "pressure").mkdir(parents=True)
            (proc / "meminfo").write_text(meminfo(5.0))
            (proc / "pressure" / "memory").write_text("full avg10=2.5 avg60=0 avg300=0 total=1\n")
            mem, psi, zram = rv.read_readings(proc_root=proc, sys_root=Path(temp) / "sys-missing")
            self.assertAlmostEqual(mem, 5.0, places=3)
            self.assertEqual(psi, 2.5)
            self.assertIsNone(zram)

    def test_read_readings_never_raises_on_missing_files(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            self.assertEqual(rv.read_readings(proc_root=Path(temp) / "nope", sys_root=Path(temp) / "nope"), (None, None, None))


class LevelStateMachine(unittest.TestCase):
    def test_starts_ok_and_enters_tight_immediately(self):
        valve = rv.RamValve()
        self.assertEqual(valve.update(8.0, 0.0, 10.0), ("ok", False))
        self.assertEqual(valve.update(5.5, 0.0, 10.0), ("tight", True))

    def test_no_flapping_around_the_threshold(self):
        valve = rv.RamValve()
        valve.update(5.5, 0.0, 10.0)                       # tight
        changes = [valve.update(x, 0.0, 10.0)[1] for x in (5.9, 6.0, 5.9, 6.2, 5.8, 6.1)]
        self.assertEqual(changes, [False] * 6)             # inside the band: no change at all
        self.assertEqual(valve.level, "tight")

    def test_leaving_needs_the_hysteresis(self):
        valve = rv.RamValve()
        valve.update(5.5, 0.0, 10.0)
        self.assertEqual(valve.update(6.4, 0.0, 10.0), ("tight", False))   # threshold 6.0 + 0.5 not reached yet
        self.assertEqual(valve.update(6.5, 0.0, 10.0), ("ok", True))

    def test_escalates_at_once_and_steps_down_one_level(self):
        valve = rv.RamValve()
        self.assertEqual(valve.update(1.0, 0.0, 10.0), ("critical", True))   # jump straight to the worst level
        self.assertEqual(valve.update(4.0, 0.0, 10.0), ("pressing", True))   # one step down per update, not to ok
        self.assertEqual(valve.update(4.0, 0.0, 10.0), ("tight", True))      # 4 GB is what "tight" asks for
        self.assertEqual(valve.update(4.0, 0.0, 10.0), ("tight", False))
        self.assertEqual(valve.update(6.5, 0.0, 10.0), ("ok", True))         # 6.0 + 0.5 hysteresis reached
        self.assertEqual(valve.update(6.5, 0.0, 10.0), ("ok", False))

    def test_psi_and_zram_also_drive_the_level(self):
        valve = rv.RamValve()
        self.assertEqual(valve.update(8.0, 20.0, 10.0)[0], "pressing")     # PSI full avg10 20 >= 15
        valve = rv.RamValve()
        self.assertEqual(valve.update(8.0, 0.0, 92.0)[0], "critical")      # zram 92% >= 90
        valve = rv.RamValve()
        self.assertEqual(valve.update(8.0, 0.0, 65.0)[0], "tight")         # zram 65% >= 60

    def test_missing_readings_do_not_raise_the_level(self):
        valve = rv.RamValve()
        self.assertEqual(valve.update(None, None, None), ("ok", False))

    def test_thresholds_are_configurable(self):
        valve = rv.RamValve({"ramGuardTightMemAvailableGB": 10.0})
        self.assertEqual(valve.update(8.0, 0.0, 10.0), ("tight", True))


class OomScoreAdj(unittest.TestCase):
    def test_off_by_default_writes_nothing(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            self.assertEqual(rv.oom_score_adj_write(123, 0, proc_root=temp), (False, None))
            self.assertFalse((Path(temp) / "123").exists())

    def test_writes_the_value_for_our_child(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            (Path(temp) / "4321").mkdir()
            self.assertEqual(rv.oom_score_adj_write(4321, 800, proc_root=temp), (True, None))
            self.assertEqual((Path(temp) / "4321" / "oom_score_adj").read_text(), "800")

    def test_failed_write_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            ok, error = rv.oom_score_adj_write(999, 800, proc_root=temp)   # no such pid directory
            self.assertFalse(ok)
            self.assertTrue(error)

    def test_rejects_out_of_range_and_bool(self):
        self.assertEqual(rv.oom_score_adj_write(1, 5000, proc_root="/nonexistent")[0], False)
        self.assertEqual(rv.oom_score_adj_write(1, True, proc_root="/nonexistent")[0], False)


class CacheBudget(unittest.TestCase):
    def test_no_budget_keeps_the_requested_values(self):
        self.assertEqual(rv.cache_budget(31.0, 0, 200000, 32, 1024), (1024, 32))

    def test_fits_the_budget_by_reducing_checkpoints_first(self):
        # 31 GB machine, 10% = about 3174 MiB. Per checkpoint at 200k tokens: 63 + 2*200000/1024 = 454 MiB.
        cache, ckpt = rv.cache_budget(31.0, 10, 200000, 32, 1024)
        self.assertLessEqual(cache + ckpt * (63 + 2 * 200000 / 1024), 31.0 * 1024 * 10 / 100 + 1e-6)
        self.assertEqual(cache, 1024)                       # the prompt cache is kept while checkpoints give way
        self.assertLess(ckpt, 32)

    def test_cache_is_reduced_when_no_checkpoints_fit(self):
        cache, ckpt = rv.cache_budget(16.0, 5, 200000, 0, 4096)   # 5% of 16 GB = 819 MiB
        self.assertEqual(ckpt, 0)
        self.assertLessEqual(cache, 819)

    def test_unknown_total_keeps_values(self):
        self.assertEqual(rv.cache_budget(None, 10, 200000, 32, 1024), (1024, 32))


class SsdDailyBudget(unittest.TestCase):
    def test_no_limit_always_allows(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            budget = rv.SsdDailyBudget(0, Path(temp) / "state.json")
            self.assertTrue(budget.allowed(10 ** 12))

    def test_counts_per_day_and_refuses_over_the_limit(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            day = {"t": 1_700_000_000.0}
            budget = rv.SsdDailyBudget(1, Path(temp) / "state.json", clock=lambda: day["t"])   # 1 GiB
            gib = 1024 ** 3
            self.assertTrue(budget.allowed(gib // 2))
            budget.record(gib // 2)
            self.assertTrue(budget.allowed(gib // 2))
            budget.record(gib // 2)
            self.assertFalse(budget.allowed(1))                       # the day's 1 GiB is used
            day["t"] += 86400                                          # the next day starts from zero
            self.assertTrue(budget.allowed(gib // 2))

    def test_corrupt_state_is_treated_as_empty(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            path = Path(temp) / "state.json"
            path.write_text("{not json")
            budget = rv.SsdDailyBudget(1, path)
            self.assertTrue(budget.allowed(1024))
            budget.record(1024)
            self.assertEqual(json.loads(path.read_text())["bytes"], 1024)


class GatewayRamValve(unittest.TestCase):
    """The whole gateway with the fake llama-server: the switches reach the child process."""

    def start_and_pid(self, guardian):
        from gateway_harness import GatewayHarness
        # slotSaveCheckpoints 1 selects the current gateway, as the live guardian.json does (the legacy gateway has no valve)
        h = GatewayHarness(guardian={"slotSaveCheckpoints": 1, **guardian})
        h.__enter__()
        self.addCleanup(h.__exit__, None, None, None)
        status, _ = h.chat(content="hello", stream=False, timeout=120)
        self.assertEqual(status, 200)
        pids = GatewayHarness.pids_on(h.llama_port)
        self.assertTrue(pids, "no llama-server pid found")
        return pids[0]

    def test_child_gets_the_oom_score_when_configured(self):
        pid = self.start_and_pid({"llamaOomScoreAdj": 800})
        self.assertEqual(Path(f"/proc/{pid}/oom_score_adj").read_text().strip(), "800")

    def test_child_keeps_its_own_score_by_default(self):
        pid = self.start_and_pid({})
        self.assertNotEqual(Path(f"/proc/{pid}/oom_score_adj").read_text().strip(), "800")

    def test_cache_budget_reaches_the_command_line(self):
        pid = self.start_and_pid({"ramCacheBudgetPct": 1})
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        cache = int(argv[argv.index(b"--cache-ram") + 1])
        total_mib = (ram_valve_total_gb() or 31.0) * 1024
        self.assertLessEqual(cache, total_mib * 0.01 + 1)
        self.assertLess(cache, 4096)                        # the configured 4096 MiB was reduced


def ram_valve_total_gb():
    return rv.read_mem_total_gb()


class SnapshotCountsAgainstTheBudget(unittest.TestCase):
    def test_snapshot_in_ram_is_subtracted_before_the_checkpoints_and_cache_are_chosen(self):
        plain = rv.cache_budget(31.0, 10, 200000, 32, 1024, 0.0)
        with_snapshot = rv.cache_budget(31.0, 10, 200000, 32, 1024, 1000.0)
        self.assertLessEqual(with_snapshot[0] + with_snapshot[1] * (63 + 2 * 200000 / 1024), 31 * 1024 * 0.10 - 1000 + 1e-6)
        self.assertLessEqual(with_snapshot[1], plain[1])


class PressingHalvesTheNextStart(unittest.TestCase):
    """The gateway side: after a pressing unload, the next server command line carries half the cache and checkpoints."""

    def test_pressing_reduces_the_cache_of_the_next_start(self):
        import tempfile
        import shutil
        import time
        from gateway_harness import GatewayHarness
        temp = Path(tempfile.mkdtemp(prefix="ram1b-p-", dir="/dev/shm"))
        proc = temp / "proc"
        sysroot = temp / "sys"
        sysroot.mkdir()
        write_meminfo(proc, 8.0)
        gw = GatewayHarness(guardian={"slotSaveCheckpoints": 1, "ramGuardProcRoot": str(proc), "ramGuardSysRoot": str(sysroot),
                                      "ramGuardIdleSeconds": 0, "pollIntervalSeconds": 1, "ramGuard": True})
        gw.__enter__()
        try:
            def cache_arg():
                pids = GatewayHarness.pids_on(gw.llama_port)
                argv = Path(f"/proc/{pids[0]}/cmdline").read_bytes().split(b"\0")
                return int(argv[argv.index(b"--cache-ram") + 1])
            self.assertEqual(gw.chat(content="hello", timeout=120)[0], 200)
            before = cache_arg()
            write_meminfo(proc, 2.5)                          # pressing: idle 0 s, so it acts at once
            end = time.time() + 30
            while time.time() < end and gw.status()["status"] != "UNLOADED":
                time.sleep(0.5)
            self.assertEqual(gw.status()["status"], "UNLOADED")
            write_meminfo(proc, 8.0)
            end = time.time() + 60
            while time.time() < end:
                status, _ = gw.chat(content="hello", timeout=60)
                if status == 200:
                    break
                time.sleep(1)
            self.assertEqual(status, 200)
            self.assertEqual(cache_arg(), before // 2)
        finally:
            gw.__exit__(None, None, None)
            shutil.rmtree(temp, ignore_errors=True)


class DefaultsAreOff(unittest.TestCase):
    def test_every_new_switch_is_off_by_default(self):
        self.assertFalse(rv.DEFAULTS["ramGuard"])
        self.assertEqual(rv.DEFAULTS["llamaOomScoreAdj"], 0)
        self.assertEqual(rv.DEFAULTS["ramCacheBudgetPct"], 0)
        self.assertEqual(rv.DEFAULTS["ssdWriteDailyLimitGB"], 0)


if __name__ == "__main__":
    unittest.main()
