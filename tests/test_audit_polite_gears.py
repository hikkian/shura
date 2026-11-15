"""Regression tests for the 2026-10-06 audit: polite CPU controller, cgroup scheduler, wakeup probe, slot gears.
Fake /proc, fake systemd runner, mocked servers and probes. No GPU, no model, no live unit."""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))
import cpu_cgroup
import test_gateway as fixture
import wakeup_probe
from polite_model import Controller, Sample, Sensors
from slot_gears import Budget, Ratchet, RollbackFailed, apply_slots


def fake_proc_task(root, tid, start="5"):
    d = Path(root) / "42/task" / str(tid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "stat").write_text(f"{tid} (t) " + " ".join(["S"] + ["0"] * 11 + ["0", "0"] + ["0"] * 5 + [start] + ["0"] * 3))


class FakeSystem:
    def __init__(self, masks):
        self.masks = {tid: set(mask) for tid, mask in masks.items()}

    def sched_getaffinity(self, tid):
        if tid not in self.masks:
            raise ProcessLookupError(tid)
        return set(self.masks[tid])

    def sched_setaffinity(self, tid, mask):
        if tid not in self.masks:
            raise ProcessLookupError(tid)
        self.masks[tid] = set(mask)


class AffinityAudit(unittest.TestCase):
    FULL = {0, 1, 2, 3}
    REDUCED = {0, 1}

    def scheduler(self, tids=(1, 2)):
        tmp = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for tid in tids:
            fake_proc_task(root, tid)
        (root / "42").mkdir(exist_ok=True)
        (root / "42/cgroup").write_text("0::/unit")
        system = FakeSystem({tid: self.FULL for tid in tids})
        s = cpu_cgroup.Scheduler({}, root=root, system=system)
        s.path = "/unit"
        s.weights = [100, 50, 20, 5]
        patchers = [patch.object(s, "change"), patch.object(s, "cpus_for_l3", return_value=self.REDUCED),
                    patch("polite_model.proc_ticks", side_effect=lambda p: (0, 7))]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)
        return s, system, root

    def test_m9_thread_born_under_confinement_is_restored_to_full(self):
        s, system, root = self.scheduler()
        s.apply(42, 7, 3, self.FULL)
        self.assertEqual(system.masks, {1: self.REDUCED, 2: self.REDUCED})
        system.masks[3] = set(self.REDUCED)         # created by a confined thread: inherits its mask
        fake_proc_task(root, 3)
        s.apply(42, 7, 3, self.FULL)
        s.apply(42, 7, 0, self.FULL)
        self.assertEqual(system.masks, {1: self.FULL, 2: self.FULL, 3: self.FULL})

    def test_m9_exited_threads_are_pruned_while_confined(self):
        s, system, root = self.scheduler((1, 2, 3))
        s.apply(42, 7, 3, self.FULL)
        self.assertEqual(len(s.affinity_before), 3)
        (root / "42/task/3/stat").unlink()
        (root / "42/task/3").rmdir()
        s.apply(42, 7, 3, self.FULL)
        self.assertEqual({key[1] for key in s.affinity_before}, {1, 2})

    def test_h2_failed_weight_change_cannot_leave_threads_pinned(self):
        s, system, _ = self.scheduler()
        s.apply(42, 7, 3, self.FULL)
        s.change.side_effect = subprocess.TimeoutExpired("systemctl", 2)
        with self.assertRaises(subprocess.TimeoutExpired):
            s.apply(42, 7, 2, self.FULL)
        self.assertEqual(system.masks, {1: self.FULL, 2: self.FULL})
        self.assertFalse(s.affinity_active)

    def test_h2_restore_affinity_widens_without_touching_the_weight(self):
        s, system, _ = self.scheduler()
        s.apply(42, 7, 3, self.FULL)
        s.change.reset_mock()
        s.restore_affinity(42, 7, self.FULL)
        self.assertEqual(system.masks, {1: self.FULL, 2: self.FULL})
        s.change.assert_not_called()
        self.assertFalse(s.affinity_active)

    def test_h2_restore_affinity_skips_a_replaced_process(self):
        s, system, _ = self.scheduler()
        s.apply(42, 7, 3, self.FULL)
        with patch("polite_model.proc_ticks", side_effect=lambda p: (0, 99)):
            s.restore_affinity(42, 7, self.FULL)
        self.assertEqual(system.masks, {1: self.REDUCED, 2: self.REDUCED})
        self.assertFalse(s.affinity_active)

    def test_m2_a_thread_that_exits_mid_apply_is_not_an_error(self):
        s, system, root = self.scheduler((1, 2, 3))
        del system.masks[2]                          # listed in /proc, gone by the time we ask
        s.apply(42, 7, 3, self.FULL)
        self.assertEqual(system.masks, {1: self.REDUCED, 3: self.REDUCED})


class LeaseAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "proc/self").mkdir(parents=True)
        (self.root / "proc/self/cgroup").write_text("0::/unit")
        (self.root / "cg/unit").mkdir(parents=True)
        (self.root / "cg/unit/cpu.weight").write_text("100")
        self.weight = "[not set]"
        self.s = cpu_cgroup.Scheduler({"politeUnit": "test.service"}, root=self.root / "proc",
                                      runner=self.fake_run, cgroup_root=self.root / "cg")
        self.s.lease_dir = self.root / "lease"
        self.s.lease = self.s.lease_dir / "weight.json"

    def fake_run(self, args, **kwargs):
        if "set-property" in args:
            self.weight = args[-1].split("=", 1)[1] or "[not set]"
        return Mock(stdout="ControlGroup=/unit\nCPUWeight=" + self.weight + "\n")

    def test_l8_crash_between_restore_and_unlink_is_recoverable(self):
        self.s.reversible()
        self.s.change(20)
        self.s.write_lease(5)                        # lease: applied 5, previous 20, original "[not set]"
        self.weight = "[not set]"                    # the process died right after restoring the original
        self.s.original = self.s.weight = None
        self.s.recover()
        self.assertEqual(self.weight, "[not set]")
        self.assertFalse(self.s.lease.exists())

    def test_l8_a_foreign_weight_is_still_refused(self):
        self.s.reversible()
        self.s.change(20)
        self.s.write_lease(5)
        self.weight = "77"
        self.s.original = self.s.weight = None
        with self.assertRaises(RuntimeError):
            self.s.recover()

    def test_reversible_never_overwrites_the_recorded_original(self):
        self.s.reversible()
        self.s.change(20)
        self.assertEqual(self.s.original, "[not set]")
        self.s.reversible()                           # CPUWeight now reads 20: that is our override
        self.assertEqual(self.s.original, "[not set]")


class FakeScheduler:
    """Records what the controller asks for; `broken` makes every weight change fail like a stalled systemctl."""
    root = Path("/proc")

    def __init__(self, level_weights=(100, 50, 20, 5)):
        self.weights = list(level_weights)
        self.weight = None
        self.broken = False
        self.closed = 0
        self.restored = []
        self.system = Mock()
        self.system.sched_getaffinity.return_value = {0, 1}

    def reversible(self):
        return True

    def apply(self, pid, identity, level, full):
        if self.broken:
            raise subprocess.TimeoutExpired("systemctl", 2)
        self.weight = self.weights[level]

    def restore_affinity(self, pid, identity, full):
        self.restored.append(pid)

    def close(self):
        self.closed += 1
        self.weight = None


class PoliteAudit(unittest.TestCase):
    def setUp(self):
        p = patch("polite_model.proc_ticks", return_value=(0, 7))
        p.start()
        self.addCleanup(p.stop)

    def controller(self, mode="static", sensors=None, **config):
        self.sched = FakeScheduler()
        if sensors is None:
            sensors = Mock()
            sensors.sample.return_value = Sample(0, 0, 0)
        return Controller({"politeModel": True, "politeMode": mode, **config}, scheduler=self.sched, sensors=sensors)

    def test_h2_failed_apply_still_restores_the_weight(self):
        c = self.controller()
        c.tick(0, 42, 0.0)
        self.assertEqual(self.sched.weight, 20)
        self.sched.broken = True
        c.tick(1, 42, 0.0)                     # the L0 re-apply of the old rollback would fail here too
        self.assertIsNone(self.sched.weight)
        self.assertEqual(self.sched.closed, 1)
        self.assertEqual(self.sched.restored, [42])
        self.assertFalse(c.available)
        self.assertIn("systemctl", c.error)

    def test_m1_stale_probe_without_a_live_model_restores_the_weight(self):
        c = self.controller("dynamic")
        c.tick(0, 42, 0.1)
        self.sched.weight = 20                  # as if level 2 had been applied
        c.stale(None)
        self.assertIsNone(self.sched.weight)
        self.assertEqual(self.sched.closed, 1)
        self.assertFalse(c.available)

    def test_m2_one_transient_error_does_not_end_the_controller(self):
        sensors = Mock()
        sensors.sample.side_effect = [OSError("pressure file busy"), Sample(0, 0, 0), Sample(0, 0, 0)]
        c = self.controller("dynamic", sensors)
        c.tick(0, 42, 0.1)
        self.assertFalse(c.available)
        self.assertTrue(c.recoverable)
        self.assertFalse(c.retry_due(1))
        c.tick(1, 42, 0.1)                      # inside the back-off: nothing happens
        self.assertEqual(sensors.sample.call_count, 1)
        self.assertTrue(c.retry_due(5))
        c.tick(5, 42, 0.1)
        self.assertTrue(c.available)
        self.assertEqual(c.error, "")
        self.assertEqual(self.sched.weight, 100)

    def test_m2_backoff_grows_then_resets_after_a_quiet_period(self):
        c = self.controller()
        self.sched.broken = True
        now = 0
        waits = []
        for _ in range(5):
            c.tick(now, 42, 0.0)
            waits.append(c.retry_at - now)
            now = c.retry_at
        self.assertEqual(waits, [5, 10, 20, 40, 80])
        self.sched.broken = False
        c.tick(now, 42, 0.0)
        self.assertTrue(c.available)
        self.sched.broken = True
        later = now + 600
        c.tick(later, 42, 0.0)
        self.assertEqual(c.retry_at - later, 5)

    def test_m2_a_stale_probe_suspension_recovers_when_the_probe_is_back(self):
        c = self.controller("dynamic")
        c.tick(0, 42, 0.1)
        c.stale(42)
        self.assertFalse(c.available)
        self.assertTrue(c.recoverable)
        self.assertTrue(c.retry_due(time.monotonic() + 10))

    def test_l7_invalid_static_weight_is_refused_up_front(self):
        for bad in (0, 100000, "20", 2.5, True):
            c = self.controller(politeStaticWeight=bad)
            self.assertFalse(c.available, bad)
            self.assertIn("politeStaticWeight", c.error)
            self.assertFalse(c.recoverable)       # a config error never retries
            c.tick(0, 42, 0.0)
            self.assertIsNone(self.sched.weight)

    def test_l7_string_true_is_reported_not_silently_ignored(self):
        c = Controller({"politeModel": "true"}, scheduler=FakeScheduler())
        self.assertFalse(c.enabled)
        self.assertIn("JSON value true", c.error)
        self.assertFalse(Controller({"politeModel": False}, scheduler=FakeScheduler()).error)

    def test_l7_unknown_mode_is_refused(self):
        c = self.controller("turbo")
        self.assertFalse(c.available)
        self.assertIn("politeMode", c.error)

    def test_l9_nothing_is_applied_after_shutdown(self):
        c = self.controller()
        c.tick(0, 42, 0.0)
        self.assertEqual(self.sched.weight, 20)
        c.shutdown()
        self.assertIsNone(self.sched.weight)
        c.tick(1, 42, 0.0)                      # the monitor thread is still ticking at exit
        self.assertIsNone(self.sched.weight)
        self.assertFalse(c.recoverable)

    def test_l9_shutdown_restores_affinity_too(self):
        c = self.controller()
        c.tick(0, 42, 0.0)
        c.shutdown()
        self.assertEqual(self.sched.restored, [42])

    def test_l9_scheduler_calls_are_serialised_by_a_lock(self):
        s = cpu_cgroup.Scheduler({})
        self.assertTrue(hasattr(s.lock, "acquire"))


class SensorsAudit(unittest.TestCase):
    def write(self, root, pid, ticks, start):
        fields = ["S"] + ["0"] * 21
        fields[11], fields[19] = str(ticks), str(start)
        (root / str(pid)).mkdir(exist_ok=True)
        (root / str(pid) / "stat").write_text(f"{pid} (llama) " + " ".join(fields))

    def test_l6_restarted_model_is_not_charged_to_other(self):
        hz = os.sysconf("SC_CLK_TCK")
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            root = Path(tmp)
            (root / "pressure").mkdir()
            (root / "pressure/cpu").write_text("some avg10=1.0 avg60=0 avg300=0 total=0\n")
            self.write(root, 42, 100, 7)
            (root / "stat").write_text("cpu 1000 0 0 100 0 0 0 0 0 0\n")
            clock = [0.0]
            sensors = Sensors(root, lambda: clock[0])
            sensors.sample(42, 0.1)
            # The server restarts (new pid) and spends 3 busy cores for 20 s before the next sample.
            self.write(root, 43, 60 * hz, 900)
            (root / "stat").write_text(f"cpu {1000 + 60 * hz} 0 0 100 0 0 0 0 0 0\n")
            clock[0] = 20.0
            self.assertEqual(sensors.sample(43, 0.1).other_busy_cpus, 0)
            # The next window is normal again.
            self.write(root, 43, 60 * hz + hz, 900)
            (root / "stat").write_text(f"cpu {1000 + 61 * hz + hz} 0 0 100 0 0 0 0 0 0\n")
            clock[0] = 21.0
            self.assertAlmostEqual(sensors.sample(43, 0.1).other_busy_cpus, 1.0)


class WakeupProbeAudit(unittest.TestCase):
    def test_m3_p99_is_not_the_maximum(self):
        samples = [0.1] * 99 + [3.0]
        self.assertLess(wakeup_probe.percentile(samples, 0.99), 3.0)
        self.assertEqual(wakeup_probe.percentile([1, 2, 3], 0.99), 3)

    def test_m3_one_hiccup_per_second_does_not_trip_the_dynamic_policy(self):
        window = wakeup_probe.Window()
        for second in range(60):
            samples = [0.1] * 96 + [3.0 if second % 5 == 4 else 0.1]
            report = window.add_second(samples)
            self.assertEqual(report["max_ms"], max(samples))
            if second >= 4:
                self.assertLess(report["p99_ms"], 2.0, second)
        # while a sustained problem (a slow wakeup in 3% of samples) is still reported
        window = wakeup_probe.Window()
        report = None
        for _ in range(5):
            report = window.add_second([0.1] * 94 + [4.0] * 3)
        self.assertGreaterEqual(report["p99_ms"], 2.0)

    def test_m3_window_is_bounded(self):
        window = wakeup_probe.Window(seconds=3)
        for _ in range(10):
            window.add_second([0.1] * 10)
        self.assertEqual(len(window.chunks), 3)


class SlotGearAudit(unittest.TestCase):
    def test_m6_bad_level_entries_do_not_disable_the_controller(self):
        self.assertEqual(Ratchet({"slotGearLevels": ["x", None, 24, 20]}, 22).levels, [22, 20])
        self.assertEqual(Ratchet({"slotGearLevels": ["x"]}, 18).levels, [18])

    def test_m6_a_floor_violation_and_a_bad_initial_are_a_clean_value_error(self):
        for initial in (12, None, "18", 15.5):
            with self.assertRaises(ValueError):
                Ratchet({}, initial)

    def test_m6_current_gear_missing_from_the_level_list_does_not_raise(self):
        gear = Ratchet({}, 26)
        gear.current = 22                       # the level list was edited between runs
        self.assertIsNone(gear.target(0, 100))
        self.assertIn(22, gear.levels)
        self.assertEqual(gear.target(61, 100), 20)

    def test_l10_refused_snapshot_is_none_not_a_failed_change(self):
        server = Mock()
        server.save.return_value = None
        self.assertIsNone(apply_slots(24, server))
        server.restart.assert_not_called()

    def test_l10_deferred_costs_no_budget(self):
        gear = Ratchet({}, 26)
        gear.target(0, 100)
        gear.deferred(61, "change deferred")
        self.assertEqual(gear.history, [])
        self.assertIsNone(gear.target(70, 100))     # short pause
        self.assertEqual(gear.target(91, 100), 24)
        self.assertEqual(gear.reason, "sustained desktop pressure")

    def test_l11_timers_do_not_run_while_the_server_is_not_ready(self):
        gear = Ratchet({}, 26)
        gear.target(0, 100)
        gear.target(30, 100, ready=False)           # vision/loading readings are not representative
        self.assertIsNone(gear.target(61, 100))
        self.assertEqual(gear.target(121, 100), 24)

    def test_l11_busy_still_lets_pressure_accumulate(self):
        gear = Ratchet({}, 26)
        gear.target(0, 100)
        self.assertIsNone(gear.target(60, 100, busy=True))
        self.assertEqual(gear.target(61, 100), 24)

    def test_l11_stale_reason_clears_when_data_returns(self):
        gear = Ratchet({}, 26)
        gear.target(0, None)
        self.assertEqual(gear.reason, "fresh VRAM data unavailable")
        gear.target(1, 400)
        self.assertNotEqual(gear.reason, "fresh VRAM data unavailable")

    def test_l11_wrong_per_slot_estimate_cannot_ping_pong(self):
        """The real cost is 40 MiB/slot, the estimate 36: every up-shift lands below the corridor."""
        gear = Ratchet({}, 26, Budget())
        base = 325 + 24 * 40                          # free MiB with zero slots
        moves = []
        for now in range(0, 6 * 3600, 5):
            target = gear.target(now, base - gear.current * 40)
            if target is not None:
                moves.append((now, "up" if target > gear.current else "down"))
                gear.completed(now, target, True)
        self.assertLessEqual(len(moves), 3, moves)
        self.assertLessEqual(sum(1 for _, d in moves if d == "up"), 1, moves)
        self.assertIn("ping-pong", gear.reason)

    def test_l11_a_correct_estimate_is_not_penalised(self):
        gear = Ratchet({}, 26, Budget())
        base = 400 + 20 * 36
        moves = []
        for now in range(0, 3 * 3600, 5):
            target = gear.target(now, base - gear.current * 36)
            if target is not None:
                moves.append(target)
                gear.completed(now, target, True)
        self.assertEqual(len(moves), len(set(moves)), moves)    # no level is visited twice


class GatewayPoliteGearsAudit(unittest.TestCase):
    """Gateway wiring of the polite-controller and slot-gear fixes."""
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

    def gears(self):
        gw = self.gw
        gw.G.update(slotGears=True, slotSaveCheckpoints=1)
        self.patch_attr("GEARS", {})
        patcher = patch.object(gw, "GEARS_UNAVAILABLE", set(), create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.patch_attr("GEAR_BUDGET", Budget())
        self.patch_attr("GEAR_BUDGET_LOADED", True)
        gw.st.vram_free_mib = 100                        # far below the corridor
        gw.st.vram_sample_at = time.time()

    def pressure(self, now):
        self.gw.st.vram_sample_at = time.time()
        self.gw.reliability_tick(now)

    def polite(self, mode="static"):
        gw = self.gw
        sched = Mock(root=gw.Path("/proc"), weight=None)
        sched.reversible.return_value = True
        sched.system.sched_getaffinity.return_value = {0, 1}
        sensors = Mock()
        controller = Controller({"politeModel": True, "politeMode": mode}, scheduler=sched, sensors=sensors)
        self.patch_attr("POLITE", controller)
        gw.G["politeMode"] = mode
        gw.G["slotGears"] = False
        patcher = patch("polite_model.proc_ticks", return_value=(0, 7))
        patcher.start()
        self.addCleanup(patcher.stop)
        return controller, sched

    def test_m6_a_model_that_cannot_have_gears_does_not_break_the_tick(self):
        gw = self.gw
        mid = gw.st.model_id
        models = {**gw.M["models"], mid: {**gw.M["models"][mid], "moeCacheSlots": 12}}
        self.patch_attr("M", {**gw.M, "models": models})
        self.gears()
        gw.st.monitor_error = ""
        for now in range(3):
            self.pressure(now)                           # must not raise: raising wipes the VRAM guard timers
        self.assertTrue(gw.G["slotGears"])
        self.assertIn(mid, gw.GEARS_UNAVAILABLE)
        self.assertEqual([c.args[0] for c in gw.event.call_args_list].count("gears_unavailable"), 1)
        self.assertEqual(gw.st.monitor_error, "")

    def test_m6_missing_slot_setting_is_also_contained(self):
        gw = self.gw
        mid = gw.st.model_id
        models = {**gw.M["models"], mid: {k: v for k, v in gw.M["models"][mid].items() if k != "moeCacheSlots"}}
        self.patch_attr("M", {**gw.M, "models": models})
        self.gears()
        self.pressure(0)
        self.assertIn(mid, gw.GEARS_UNAVAILABLE)

    def test_m6_vram_emergency_still_fires_with_a_bad_gear_config(self):
        gw = self.gw
        mid = gw.st.model_id
        models = {**gw.M["models"], mid: {**gw.M["models"][mid], "moeCacheSlots": 12}}
        self.patch_attr("M", {**gw.M, "models": models})
        self.gears()
        gw.G["vramGuard"] = True
        for now in (100, 110):
            gw.monitor_tick(now, 59, 24)                 # below vramEmergencyMiB for longer than 10 s
            gw.reliability_tick(now)
        self.proc.terminate.assert_called_once()

    def test_l10_a_refused_change_is_not_charged(self):
        gw = self.gw
        self.gears()
        self.patch_attr("apply_slots", Mock(return_value=None))
        self.pressure(0)
        self.pressure(61)
        gear = gw.GEARS[gw.st.model_id]
        self.assertGreaterEqual(gw.apply_slots.call_count, 1)
        self.assertEqual(gear.history, [])
        self.assertNotIn("restart failed", gear.reason)
        calls = gw.apply_slots.call_count
        self.pressure(62)                                # short pause, no hammering every second
        self.assertEqual(gw.apply_slots.call_count, calls)

    def test_l10_unexpected_exception_is_charged_once_not_every_second(self):
        gw = self.gw
        self.gears()
        self.patch_attr("apply_slots", Mock(side_effect=KeyError("surprise")))
        self.pressure(0)
        self.pressure(61)
        gear = gw.GEARS[gw.st.model_id]
        self.assertEqual(len(gear.history), 1)
        self.pressure(62)
        self.assertEqual(gw.apply_slots.call_count, 1)

    def test_m7_rollback_gets_longer_limits_than_the_forward_step(self):
        gw = self.gw
        gw.G.update(slotSaveCheckpoints=1, slotGearRestartSeconds=20)
        with patch.object(gw, "RUNTIME_SLOTS", {}):
            backend = gw.GearBackend()
            snapshot = backend.save()
            with patch.object(gw, "start_llama", return_value="injected") as start:
                backend.restart(20, snapshot)
                forward = start.call_args.kwargs
                backend.restart(backend.original_slots, snapshot)
                rollback = start.call_args.kwargs
            backend.release(snapshot)
        self.assertLessEqual(forward["health_timeout"], 20)
        self.assertLessEqual(forward["restore_timeout"], 5)
        self.assertGreater(rollback["health_timeout"], 60)
        self.assertGreater(rollback["restore_timeout"], 5)

    def test_m7_failed_rollback_leaves_the_door_open_for_a_normal_load(self):
        gw = self.gw
        self.gears()

        def doomed(target):
            gw.st.proc = None                            # both restarts failed, nothing is running
            raise RollbackFailed("slot restart and rollback both failed; A2 snapshot retained")

        self.patch_attr("apply_slots", Mock(side_effect=doomed))
        self.pressure(0)
        self.pressure(61)
        self.assertEqual(gw.st.status, "UNLOADED")       # not the dead-end ERROR
        self.assertIn("rollback both failed", gw.st.last_error)
        self.assertFalse(gw.G["slotGears"])


    def test_m2_gateway_keeps_ticking_a_suspended_controller_until_it_recovers(self):
        gw = self.gw
        controller, sched = self.polite()
        sched.apply.side_effect = [subprocess.TimeoutExpired("systemctl", 2), None]
        gw.reliability_tick(0)
        self.assertFalse(controller.available)
        gw.reliability_tick(1)                           # still inside the back-off
        self.assertEqual(sched.apply.call_count, 1)
        gw.reliability_tick(6)
        self.assertEqual(sched.apply.call_count, 2)
        self.assertTrue(controller.available)
        self.assertGreaterEqual([c.args[0] for c in gw.event.call_args_list].count("cpu_policy_changed"), 2)

    def test_m1_gateway_restores_the_weight_when_the_probe_goes_stale_with_no_model(self):
        gw = self.gw
        controller, sched = self.polite("dynamic")
        gw.st.proc = None
        self.patch_attr("WAKEUP", {"proc": None, "sample_at": 0.0, "p99_ms": 0.1, "started_at": 0.0})
        gw.reliability_tick(100)                         # last sample is 100 s old
        sched.close.assert_called_once_with()
        self.assertFalse(controller.available)

    def test_m2_a_dead_wakeup_probe_is_restarted(self):
        gw = self.gw
        dead = Mock()
        dead.poll.return_value = 1
        wakeup = {"proc": dead, "sample_at": 3.0, "p99_ms": 0.1, "started_at": 0.0}
        self.patch_attr("WAKEUP", wakeup)
        self.patch_attr("stop_wakeup_probe", Mock())
        self.patch_attr("start_wakeup_probe", Mock())
        self.polite("dynamic")
        gw.ensure_wakeup_probe(100)
        gw.stop_wakeup_probe.assert_called_once()
        gw.start_wakeup_probe.assert_called_once()
        self.assertIsNone(wakeup["sample_at"])
        gw.ensure_wakeup_probe(101)                      # rate limited
        gw.start_wakeup_probe.assert_called_once()

    def test_m2_a_live_wakeup_probe_is_left_alone(self):
        alive = Mock()
        alive.poll.return_value = None
        self.patch_attr("WAKEUP", {"proc": alive, "sample_at": 0.0, "p99_ms": 0.1, "started_at": 0.0})
        self.patch_attr("start_wakeup_probe", Mock())
        self.gw.ensure_wakeup_probe(100)
        self.gw.start_wakeup_probe.assert_not_called()

    def test_m2_probe_start_is_allowed_for_a_suspended_but_recoverable_controller(self):
        gw = self.gw
        controller, sched = self.polite("dynamic")
        controller.available, controller.retry_at = False, 5.0
        with patch.object(gw.subprocess, "Popen") as popen, patch.object(gw.threading, "Thread"):
            self.patch_attr("WAKEUP", {"proc": None, "sample_at": None, "p99_ms": None, "started_at": None})
            gw.start_wakeup_probe()
            popen.assert_called_once()

    def test_l9_exit_handler_shuts_the_controller_down_for_good(self):
        gw = self.gw
        controller, sched = self.polite()
        gw.reliability_tick(0)
        gw.close_cpu_policy()
        sched.close.assert_called()
        before = sched.apply.call_count
        gw.reliability_tick(10)                          # the monitor thread outlives atexit
        self.assertEqual(sched.apply.call_count, before)

    def test_l9_exit_handler_reports_a_failed_restore(self):
        gw = self.gw
        controller, sched = self.polite()
        sched.close.side_effect = RuntimeError("External CPUWeight change; not overwritten")
        gw.close_cpu_policy()
        self.assertEqual(gw.event.call_args.args[0], "cpu_restore_failed")


if __name__ == "__main__":
    unittest.main()
