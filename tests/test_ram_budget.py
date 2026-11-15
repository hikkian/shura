"""RAMGUARD1 (R0-R1): honest free memory, the ladder, load admission and the server scope, on fake /proc trees. No GPU."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gateway"))

import ram_budget as rb  # noqa: E402


def meminfo(free_gb, inactive_file_gb, swap_total_gb=16.0, swap_used_gb=0.0, available_gb=21.0):
    """A /proc/meminfo text in kB. MemAvailable is set separately: the tests show it is not the metric."""
    kb = lambda gb: int(gb * 1024 * 1024)  # noqa: E731
    return (f"MemTotal: {kb(32)} kB\nMemFree: {kb(free_gb)} kB\nMemAvailable: {kb(available_gb)} kB\n"
            f"Inactive(file): {kb(inactive_file_gb)} kB\nShmem: {kb(0.5)} kB\n"
            f"SwapTotal: {kb(swap_total_gb)} kB\nSwapFree: {kb(swap_total_gb - swap_used_gb)} kB\n")


def pressure(some10=0.0, some60=0.0, full10=0.0):
    return (f"some avg10={some10:.2f} avg60={some60:.2f} avg300=0.00 total=1\n"
            f"full avg10={full10:.2f} avg60=0.00 avg300=0.00 total=1\n")


class ReadingTests(unittest.TestCase):
    def test_meminfo_and_pressure_parse_from_a_fake_proc_tree(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "pressure").mkdir()
            Path(root, "meminfo").write_text(meminfo(free_gb=2.0, inactive_file_gb=10.0))
            Path(root, "pressure", "memory").write_text(pressure(some10=3.0, some60=4.0, full10=1.0))
            mem = rb.read_meminfo(root)
            psi = rb.read_pressure(root)
            self.assertAlmostEqual(mem["MemFree"], 2.0 * 1024, places=0)
            self.assertAlmostEqual(mem["Inactive(file)"], 10.0 * 1024, places=0)
            self.assertEqual(psi["some"]["avg60"], 4.0)
            self.assertEqual(psi["full"]["avg10"], 1.0)

    def test_missing_files_give_none_not_an_exception(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(rb.read_meminfo(root))
            self.assertIsNone(rb.read_pressure(root))


class HonestFreeTests(unittest.TestCase):
    def test_memavailable_is_not_the_metric(self):
        # MemAvailable says 21 GB (it counts the whole page cache); honest free only counts half of the inactive file cache
        mem = rb.parse_meminfo_mib(meminfo(free_gb=2.0, inactive_file_gb=15.0, available_gb=21.0))
        honest = rb.honest_free_mib(mem)
        self.assertAlmostEqual(honest, (2.0 + 0.5 * 15.0) * 1024, delta=2)
        self.assertLess(honest, 21.0 * 1024)

    def test_zram_level_does_not_penalise(self):
        # this machine's zram sits at 55-60 % all the time: 9 of 16 GB used must not cost honest free
        normal = rb.parse_meminfo_mib(meminfo(free_gb=8.0, inactive_file_gb=0.0, swap_used_gb=9.0))
        self.assertAlmostEqual(rb.honest_free_mib(normal), 8.0 * 1024, delta=2)

    def test_swap_penalty_starts_above_sixty_percent_full(self):
        full = rb.parse_meminfo_mib(meminfo(free_gb=8.0, inactive_file_gb=0.0, swap_used_gb=11.0))       # 69 %: penalty 1.4 GB
        self.assertAlmostEqual(rb.honest_free_mib(full), (8.0 - 1.4) * 1024, delta=2)


class LevelTests(unittest.TestCase):
    R = rb.DEFAULT_DESK_RESERVE_MIB

    def mem(self, honest_gb, swap_used_gb=0.0):
        free = honest_gb
        return rb.parse_meminfo_mib(meminfo(free_gb=free, inactive_file_gb=0.0, swap_used_gb=swap_used_gb))

    def test_each_level_by_free_memory(self):
        self.assertEqual(rb.level_of(self.R + 4096, None), "ok")
        self.assertEqual(rb.level_of(self.R + 1024, None), "tight")
        self.assertEqual(rb.level_of(self.R - 512, None), "pressing")
        self.assertEqual(rb.level_of(self.R / 2 - 512, None), "critical")

    def test_pressure_and_swap_raise_the_level(self):
        self.assertEqual(rb.level_of(self.R + 8192, {"some": {"avg10": 6.0}}), "tight")
        self.assertEqual(rb.level_of(self.R + 8192, {"some": {"avg10": 16.0}}), "pressing")
        self.assertEqual(rb.level_of(self.R + 8192, {"full": {"avg10": 41.0}}), "critical")
        self.assertEqual(rb.level_of(self.R + 8192, None, swap_fraction=0.55), "ok")       # zram: no longer a level signal
        self.assertEqual(rb.level_of(self.R + 8192, None, swap_fraction=0.85), "ok")
        self.assertEqual(rb.level_of(self.R + 8192, None, swap_fraction=0.95), "critical")

    def test_ladder_enters_at_once_and_leaves_one_step_at_a_time_with_hysteresis(self):
        R = self.R
        ladder = rb.RamLadder()
        gb = lambda mib: mib / 1024.0  # noqa: E731
        self.assertEqual(ladder.update(self.mem(gb(R / 2 - 512)), None), "critical")       # enters at once
        # just above the critical threshold (R/2): not clear enough to leave it (needs HYSTERESIS_MIB more)
        self.assertEqual(ladder.update(self.mem(gb(R / 2 + 100)), None), "critical")
        # well above it: leaves the critical level, but only one step, to pressing
        self.assertEqual(ladder.update(self.mem(gb(R / 2 + rb.HYSTERESIS_MIB + 200)), None), "pressing")
        # clear of the pressing threshold (R) by the hysteresis: one more step, to tight
        self.assertEqual(ladder.update(self.mem(gb(R + rb.HYSTERESIS_MIB + 200)), None), "tight")

    def test_ladder_does_not_flap_around_a_boundary(self):
        ladder = rb.RamLadder()
        R = self.R
        self.assertEqual(ladder.update(rb.parse_meminfo_mib(meminfo(free_gb=(R + 1900) / 1024, inactive_file_gb=0.0)), None), "tight")
        changes_after_entry = ladder.changes
        # oscillating just around the tight threshold (R + 2048): inside the hysteresis, the level holds
        for free_mib in [R + 2100, R + 1900, R + 2100, R + 1900, R + 2100]:
            level = ladder.update(rb.parse_meminfo_mib(meminfo(free_gb=free_mib / 1024, inactive_file_gb=0.0)), None)
            self.assertEqual(level, "tight")
        self.assertEqual(ladder.changes, changes_after_entry)


class AdmissionTests(unittest.TestCase):
    def mem(self, honest_gb, swap_used_gb=0.0):
        return rb.parse_meminfo_mib(meminfo(free_gb=honest_gb, inactive_file_gb=0.0, swap_used_gb=swap_used_gb))

    def test_admits_when_the_desktop_keeps_its_reserve(self):
        ok, reason = rb.admit_load(self.mem(20.0), {"some": {"avg60": 0.5}}, resident_fixed_mib=6000)
        self.assertTrue(ok, reason)

    def test_refuses_when_the_resident_part_would_eat_the_reserve(self):
        ok, reason = rb.admit_load(self.mem(7.0), {"some": {"avg60": 0.5}}, resident_fixed_mib=6000)
        self.assertFalse(ok)
        self.assertIn("desktop reserve", reason)

    def test_swap_at_sixty_percent_does_not_refuse(self):
        ok, reason = rb.admit_load(self.mem(20.0, swap_used_gb=10.0), {"some": {"avg60": 0.5}}, resident_fixed_mib=6000)
        self.assertTrue(ok, reason)

    def test_refuses_on_swap_and_on_pressure(self):
        ok, reason = rb.admit_load(self.mem(20.0, swap_used_gb=15.0), {"some": {"avg60": 0.5}}, resident_fixed_mib=6000)
        self.assertFalse(ok)
        self.assertIn("swap", reason)
        ok, reason = rb.admit_load(self.mem(20.0), {"some": {"avg60": 7.0}}, resident_fixed_mib=6000)
        self.assertFalse(ok)
        self.assertIn("pressure", reason)

    def test_missing_readings_refuse(self):
        self.assertFalse(rb.admit_load(None, {}, 1000)[0])
        self.assertFalse(rb.admit_load(self.mem(20.0), None, 1000)[0])

    def test_shared_memory_planned_for_snapshots_counts_against_the_budget(self):
        self.assertTrue(rb.admit_load(self.mem(14.0), {"some": {"avg60": 0.0}}, 4000, shm_planned_mib=0)[0])
        self.assertFalse(rb.admit_load(self.mem(7.0), {"some": {"avg60": 0.0}}, 4000, shm_planned_mib=2000)[0])

    def test_resident_estimate_grows_with_cpu_experts(self):
        self.assertGreater(rb.resident_fixed_estimate_mib(14, 1024), rb.resident_fixed_estimate_mib(10, 1024))
        self.assertAlmostEqual(rb.resident_fixed_estimate_mib(0, 0, draft=False), rb.SERVER_OVERHEAD_MIB, delta=1)


class DeskReserveTests(unittest.TestCase):
    """RAMLOGIC2: the reserve is the desktop's room to grow, from its need, not its size."""

    def test_reserve_follows_the_need(self):
        self.assertEqual([rb.desk_reserve_for(n) for n in (0.0, 0.3, 0.6, 1.0)], [1024, 2048, 3072, 4608])
        self.assertEqual(rb.DEFAULT_DESK_RESERVE_MIB, rb.desk_reserve_for(0.3))

    def test_reserve_is_clamped_and_never_shrinks_as_need_grows(self):
        self.assertEqual(rb.desk_reserve_for(-1), 1024)
        self.assertEqual(rb.desk_reserve_for(7), 4608)
        values = [rb.desk_reserve_for(n / 100) for n in range(101)]
        self.assertEqual(values, sorted(values))
        self.assertTrue(all(v % 256 == 0 for v in values))

    def test_the_sage1_refusals_would_have_loaded(self):
        # SAGE1 memory-gate.jsonl: 135 of 149 checks refused at 6144; the tightest had honest free 9235 MiB with Tiel's 7160 MiB
        # resident estimate (prompt cache 1024 included). With the desktop's growth as the reserve it loads as configured.
        mem = rb.parse_meminfo_mib(meminfo(free_gb=9235 / 1024, inactive_file_gb=0.0))
        plan = rb.plan_load(mem, {"some": {"avg60": 0.0}}, 7160, 1024, reserve_mib=2048)
        self.assertEqual(plan["step"], "load", plan)
        self.assertFalse(rb.admit_load(mem, {"some": {"avg60": 0.0}}, 7160, desk_reserve_mib=6144)[0])


class LoadLadderTests(unittest.TestCase):
    """RAMLOGIC2: plan_load climbs a ladder instead of refusing. Example: 6000 MiB of the server + 4096 of prompt cache."""
    CALM = {"some": {"avg60": 0.0}}
    FIXED, CACHE, R = 6000, 4096, 2048

    def plan(self, free_mib, waited_out=False, psi=None, swap_used_gb=0.0, credit_mib=0.0):
        mem = rb.parse_meminfo_mib(meminfo(free_gb=free_mib / 1024, inactive_file_gb=0.0, swap_used_gb=swap_used_gb))
        return rb.plan_load(mem, psi or self.CALM, self.FIXED + self.CACHE, self.CACHE, reserve_mib=self.R, waited_out=waited_out,
                            credit_mib=credit_mib)

    def test_room_for_everything_loads_as_configured(self):
        plan = self.plan(self.FIXED + self.CACHE + self.R)
        self.assertEqual((plan["step"], plan["cache_ram_mib"]), ("load", self.CACHE))

    def test_a_smaller_prompt_cache_comes_first(self):
        plan = self.plan(self.FIXED + self.R + 1000)
        self.assertEqual((plan["step"], plan["cache_ram_mib"]), ("light", 768))       # whole 256 MiB steps that fit
        plan = self.plan(self.FIXED + self.R + 10)
        self.assertEqual((plan["step"], plan["cache_ram_mib"]), ("light", 0))
        self.assertIn("prompt cache", plan["reason"])

    def test_then_a_short_wait_that_says_what_it_waits_for(self):
        plan = self.plan(self.FIXED + self.R - 500)
        self.assertEqual(plan["step"], "wait")
        self.assertEqual(plan["need_mib"], self.FIXED + self.R)
        self.assertEqual(plan["free_mib"], self.FIXED + self.R - 500)
        self.assertIn("GiB free", plan["reason"])
        self.assertIn("GiB needed", plan["reason"])

    def test_after_the_wait_a_guarded_load_above_the_hard_minimum(self):
        plan = self.plan(self.FIXED + rb.RAM_HARD_MIN_MIB + 100, waited_out=True)
        self.assertEqual((plan["step"], plan["cache_ram_mib"]), ("guarded", 0))

    def test_below_the_hard_minimum_after_the_wait_an_honest_refusal(self):
        plan = self.plan(self.FIXED + rb.RAM_HARD_MIN_MIB - 100, waited_out=True)
        self.assertEqual(plan["step"], "refuse")
        self.assertIn("GiB free", plan["reason"])

    def test_pressure_waits_and_is_never_loaded_through(self):
        busy = {"some": {"avg60": 7.0}}
        self.assertEqual(self.plan(30000, psi=busy)["step"], "wait")
        self.assertEqual(self.plan(30000, psi=busy, waited_out=True)["step"], "refuse")
        self.assertEqual(self.plan(30000, swap_used_gb=15.0, waited_out=True)["step"], "refuse")

    def test_a_switch_counts_the_memory_the_running_server_gives_back(self):
        short = self.FIXED + self.R - 500
        self.assertEqual(self.plan(short)["step"], "wait")
        self.assertEqual(self.plan(short, credit_mib=self.CACHE + 600)["step"], "load")

    def test_missing_readings_refuse(self):
        self.assertEqual(rb.plan_load(None, self.CALM, 1000, 0)["step"], "refuse")

    def test_no_step_ever_asks_to_close_anything(self):
        for free in range(0, 20000, 333):
            for waited_out in (False, True):
                reason = self.plan(free, waited_out=waited_out)["reason"].lower()
                for word in ("close", "quit", "kill", "закро"):
                    self.assertNotIn(word, reason)

    def test_the_steps_never_get_worse_as_memory_grows(self):
        order = {"refuse": 0, "wait": 1, "guarded": 2, "light": 3, "load": 4}
        for waited_out in (False, True):
            steps = [order[self.plan(free, waited_out=waited_out)["step"]] for free in range(0, 20000, 64)]
            self.assertEqual(steps, sorted(steps))


class SwapGrowthTests(unittest.TestCase):
    """RAMGUARD1 in-flight: a load may grow swap by SWAP_GROWTH_LIMIT_MIB (1 GiB); more refuses the next loads."""

    def swap(self, used_gb):
        return rb.parse_meminfo_mib(meminfo(free_gb=8.0, inactive_file_gb=0.0, swap_used_gb=used_gb))

    def test_limit_is_one_gib(self):
        self.assertEqual(rb.SWAP_GROWTH_LIMIT_MIB, 1024)

    def test_growth_within_the_limit_is_fine(self):
        self.assertFalse(rb.swap_growth_too_big(self.swap(2.0), self.swap(2.5)))     # +512 MiB

    def test_growth_over_the_limit_is_flagged(self):
        self.assertTrue(rb.swap_growth_too_big(self.swap(2.0), self.swap(3.5)))      # +1536 MiB

    def test_shrinking_swap_is_never_flagged(self):
        self.assertFalse(rb.swap_growth_too_big(self.swap(9.0), self.swap(2.0)))


class ScopeTests(unittest.TestCase):
    def test_the_server_runs_in_its_own_scope_with_limits_and_no_swap(self):
        cmd = rb.build_scope_command(["llama-server", "--port", "8091"], memory_high_mib=7000, memory_max_mib=9000, available=True)
        self.assertEqual(cmd[0], "systemd-run")
        self.assertIn("--scope", cmd)
        self.assertIn("--slice=shura.slice", cmd)
        self.assertIn("MemoryHigh=7000M", cmd)
        self.assertIn("MemoryMax=9000M", cmd)
        self.assertIn("MemorySwapMax=0", cmd)
        self.assertIn("OOMScoreAdjust=800", cmd)
        self.assertEqual(cmd[-3:], ["llama-server", "--port", "8091"])

    def test_without_systemd_run_the_command_is_unchanged(self):
        argv = ["llama-server", "--port", "8091"]
        self.assertEqual(rb.build_scope_command(argv, 7000, 9000, available=False), argv)
        self.assertFalse(rb.scope_available(which=lambda name: None))
        self.assertTrue(rb.scope_available(which=lambda name: "/usr/bin/systemd-run"))

    def test_limits_follow_the_measured_resident_part(self):
        high, maximum = rb.scope_limits(5000)
        self.assertEqual(high, 5000 + 1536)
        self.assertEqual(maximum, high + 2048)


class WriteRuleTests(unittest.TestCase):
    """Opus's rule: no disk write for the cache under pressure; a snapshot on disk only at critical; never periodic."""

    def test_only_critical_writes_to_disk(self):
        for level in ("ok", "tight", "pressing"):
            self.assertFalse(rb.LADDER_ACTIONS[level]["writes_disk"], level)
        self.assertTrue(rb.LADDER_ACTIONS["critical"]["writes_disk"])

    def test_pressing_resets_instead_of_writing(self):
        actions = rb.LADDER_ACTIONS["pressing"]["actions"]
        self.assertIn("reset_prompt_cache", actions)
        self.assertNotIn("snapshot_to_nvme_once", actions)

    def test_snapshot_size_from_the_measured_per_token_size(self):
        self.assertAlmostEqual(rb.snapshot_gb(32000), 0.448, places=2)           # the 460 MB measured in SESS2b
        self.assertAlmostEqual(rb.snapshot_gb(100000), 1.4, places=2)

    def test_disk_snapshot_refused_below_critical(self):
        for level in ("ok", "tight", "pressing"):
            self.assertFalse(rb.may_write_snapshot(level, 0.0, 8.0, 32000)[0], level)

    def test_disk_snapshot_refused_when_the_daily_budget_is_used(self):
        self.assertTrue(rb.may_write_snapshot("critical", 0.0, 8.0, 32000)[0])
        allowed, reason = rb.may_write_snapshot("critical", 7.8, 8.0, 32000)
        self.assertFalse(allowed)
        self.assertIn("daily", reason)


if __name__ == "__main__":
    unittest.main()
