"""Offline CPU / ratchet fault tests. No model, NVML, GPU, live files or priorities."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))
from polite_model import Controller, Policy, Sample, Scheduler, Sensors
from slot_gears import Budget, Ratchet, apply_slots
import cpu_launch


class PoliteTests(unittest.TestCase):
    def test_aimd_and_five_second_hysteresis(self):
        p = Policy({})
        high = Sample(15, 0.1, 0.1)
        self.assertEqual(p.target(0, high), 1)
        p.committed(0, 1)
        self.assertEqual(p.target(4.9, high), 1)
        self.assertEqual(p.target(5, high), 2)
        p.committed(5, 2)
        self.assertEqual(p.target(10, high), 3)

    def test_recovery_one_step_per_30_seconds(self):
        p = Policy({})
        p.committed(0, 3)
        calm = Sample(0, 0.1, 0.5)
        self.assertEqual(p.target(5, calm), 3)
        self.assertEqual(p.target(34.9, calm), 3)
        self.assertEqual(p.target(35, calm), 2)
        p.committed(35, 2)
        self.assertEqual(p.target(36, calm), 2)
        self.assertEqual(p.target(66, calm), 1)

    def test_middle_band_breaks_calm(self):
        p = Policy({})
        p.committed(0, 2)
        p.target(1, Sample(0, 0, 0))
        p.target(20, Sample(3, 1, 1.5))
        self.assertEqual(p.target(31, Sample(0, 0, 0)), 2)

    def test_unprivileged_host_refuses_without_touching_pid(self):
        scheduler = Mock()
        scheduler.reversible.return_value = False
        c = Controller({"politeModel": True}, scheduler=scheduler)
        c.tick(0, 123, 100)
        scheduler.apply.assert_not_called()
        self.assertFalse(c.snapshot()["available"])
        self.assertIn("RLIMIT_NICE", c.snapshot()["error"])

    def test_default_off_does_not_even_check_capabilities(self):
        scheduler = Mock()
        Controller({}, scheduler=scheduler).tick(0, 123, 100)
        scheduler.reversible.assert_not_called()
        scheduler.apply.assert_not_called()

    def test_fake_proc_counts_only_non_model_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pressure").mkdir()
            (root / "pressure/cpu").write_text("some avg10=1.2 avg60=0.1 avg300=0.1 total=0\n")
            (root / "42").mkdir()
            fields = ["S"] + ["0"] * 21
            fields[11], fields[12], fields[19] = "100", "0", "7"
            (root / "42/stat").write_text("42 (model ) worker) " + " ".join(fields))
            (root / "stat").write_text("cpu 1000 0 0 100 0 0 0 0 0 0\n")
            clock = Mock(side_effect=[0, 1, 2])
            sensors = Sensors(root, clock)
            self.assertEqual(sensors.sample(42, 0.1).psi, 1.2)
            hz = os.sysconf("SC_CLK_TCK")
            fields[11] = str(100 + hz)
            (root / "42/stat").write_text("42 (model) " + " ".join(fields))
            (root / "stat").write_text(f"cpu {1000+2*hz} 0 0 100 0 0 0 0 0 0\n")
            self.assertEqual(sensors.sample(42, 0.1).other_busy_cpus, 1)
            # PID reuse cannot subtract ticks belonging to a new process.
            fields[19] = "8"
            (root / "42/stat").write_text("42 (replacement) " + " ".join(fields))
            (root / "stat").write_text(f"cpu {1000+3*hz} 0 0 100 0 0 0 0 0 0\n")
            self.assertEqual(sensors.sample(42, 0.1).other_busy_cpus, 1)

    def test_stale_wakeup_probe_restores_l0_and_disables(self):
        scheduler = Mock()
        scheduler.reversible.return_value = True
        c = Controller({"politeModel": True}, scheduler=scheduler)
        c.identity, c.full_cpus = (42, 7), {0, 1}
        c.policy.level = 3
        c.stale(42)
        scheduler.apply.assert_called_once_with(42, 7, 0, {0, 1})
        self.assertFalse(c.available)
        self.assertEqual(c.policy.level, 0)

    def test_sensor_failure_is_not_a_calm_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OSError):
                Sensors(tmp).sample(None, 0)

    def test_scheduler_refuses_pid_reuse(self):
        scheduler = Scheduler(system=Mock())
        with patch.object(scheduler, "reversible", return_value=True), patch("polite_model.proc_ticks", return_value=(0, 2)):
            with self.assertRaises(ProcessLookupError):
                scheduler.apply(123, 1, 3, {0, 1})
        scheduler.system.sched_setscheduler.assert_not_called()

    def test_launch_sets_policy_before_exec(self):
        with patch.object(cpu_launch.os, "setpriority") as nice, patch.object(cpu_launch.os, "sched_setscheduler") as sched, \
                patch.object(cpu_launch.os, "sched_getaffinity", return_value={0, 1, 2, 3}), \
                patch.object(cpu_launch.os, "sched_setaffinity") as affinity, patch.object(cpu_launch.os, "execvp") as execute:
            cpu_launch.launch(3, ["llama-server", "--port", "8091"])
        nice.assert_called_once_with(os.PRIO_PROCESS, 0, 19)
        self.assertEqual(sched.call_args.args[1], os.SCHED_IDLE)
        affinity.assert_called_once_with(0, {0, 1})
        execute.assert_called_once_with("llama-server", ["llama-server", "--port", "8091"])

    def test_scheduler_synchronizes_new_threads_and_restores_all_cores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "42/task"
            tasks.mkdir(parents=True)
            for tid in (42, 43):
                (tasks / str(tid)).mkdir()
            system = Mock()
            system.sched_getscheduler.return_value = os.SCHED_OTHER
            system.getpriority.return_value = 10
            system.sched_getaffinity.return_value = {0, 1, 2, 3}
            scheduler = Scheduler(root, system)
            with patch.object(scheduler, "reversible", return_value=True), patch("polite_model.proc_ticks", return_value=(0, 7)):
                scheduler.apply(42, 7, 3, {0, 1, 2, 3})
                self.assertEqual(system.sched_setaffinity.call_count, 2)
                (tasks / "44").mkdir()
                system.reset_mock()
                scheduler.apply(42, 7, 0, {0, 1, 2, 3})
                self.assertEqual(system.sched_setaffinity.call_count, 3)
                for call in system.sched_setaffinity.call_args_list:
                    self.assertEqual(call.args[1], {0, 1, 2, 3})

    def test_partial_scheduler_failure_restores_previous_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "42/task"
            tasks.mkdir(parents=True)
            (tasks / "42").mkdir()
            system = Mock()
            system.sched_getscheduler.return_value = os.SCHED_OTHER
            system.getpriority.return_value = 10
            system.sched_getaffinity.return_value = {0, 1}
            system.sched_setaffinity.side_effect = [PermissionError("injected"), None]
            scheduler = Scheduler(root, system)
            with patch.object(scheduler, "reversible", return_value=True), patch("polite_model.proc_ticks", return_value=(0, 7)):
                with self.assertRaises(PermissionError):
                    scheduler.apply(42, 7, 3, {0, 1})
            self.assertEqual(system.sched_setscheduler.call_args.args[1], os.SCHED_OTHER)
            self.assertEqual(system.setpriority.call_args.args[2], 10)


class GearTests(unittest.TestCase):
    def test_default_levels_and_floor(self):
        gear = Ratchet({}, 26)
        self.assertEqual(gear.levels, [26, 24, 20, 16])
        with self.assertRaises(ValueError):
            Ratchet({}, 12)
        gear = Ratchet({"slotGearLevels": [12, 8, 16]}, 16)
        gear.target(0, 100)
        self.assertIsNone(gear.target(60, 100))

    def test_down_sustained_and_only_between_responses(self):
        gear = Ratchet({}, 26)
        self.assertIsNone(gear.target(0, 240))
        self.assertIsNone(gear.target(59, 240))
        self.assertIsNone(gear.target(60, 240, busy=True))
        self.assertEqual(gear.target(61, 240), 24)

    def test_up_requires_30_minutes_and_enough_for_next_gear(self):
        gear = Ratchet({}, 26)
        gear.current = 20
        self.assertIsNone(gear.target(0, 400))
        self.assertIsNone(gear.target(1799, 400))
        self.assertEqual(gear.target(1800, 400), 24)
        gear.target(1801, 300)
        self.assertIsNone(gear.target(1802, 400))

    def test_hourly_budget_then_daily_budget(self):
        gear = Ratchet({}, 26)
        gear.completed(0, 24, True)
        gear.completed(60, 20, True)
        gear.target(61, 100)
        self.assertIsNone(gear.target(121, 100))
        self.assertIn("budget", gear.reason)
        self.assertEqual(gear.target(3661, 100), 16)
        gear.history = [4000, 5000, 6000, 7000, 8000, 9000]
        gear.current = 20
        self.assertIsNone(gear.target(13000, 100))
        self.assertIn("budget", gear.reason)

    def test_stale_sample_resets_pressure_and_calm(self):
        gear = Ratchet({}, 26)
        gear.target(0, 100)
        gear.target(60, None)
        self.assertIsNone(gear.target(61, 100))

    def test_models_share_one_change_budget(self):
        budget = Budget()
        tiel, occamy = Ratchet({}, 26, budget), Ratchet({}, 26, budget)
        tiel.completed(0, 24, True)
        occamy.completed(60, 24, True)
        tiel.target(61, 100)
        self.assertIsNone(tiel.target(121, 100))
        self.assertIn("budget", tiel.reason)

    def test_flapping_does_not_restart(self):
        gear = Ratchet({}, 26)
        for now in range(0, 600, 20):
            self.assertIsNone(gear.target(now, 200 if (now // 20) % 2 else 500))

    def test_failed_restart_preserves_gear_and_backoff(self):
        gear = Ratchet({}, 26)
        gear.target(0, 100)
        gear.completed(60, 24, False)
        self.assertEqual(gear.current, 26)
        self.assertEqual(gear.history, [60])
        self.assertIsNone(gear.target(89, 100))
        self.assertEqual(gear.target(90, 100), 24)

    def test_restart_failure_restores_old_server(self):
        server = Mock()
        server.current_slots.return_value = 26
        server.save.return_value = "snapshot"
        server.restart.side_effect = [False, True]
        self.assertFalse(apply_slots(24, server))
        self.assertEqual([call.args for call in server.restart.call_args_list], [(24, "snapshot"), (26, "snapshot")])
        server.release.assert_called_once_with("snapshot")

    def test_failed_save_never_changes_server(self):
        server = Mock()
        server.save.return_value = None
        self.assertFalse(apply_slots(24, server))
        server.restart.assert_not_called()
        server.release.assert_not_called()

    def test_double_restart_failure_keeps_recovery_snapshot(self):
        server = Mock()
        server.save.return_value = "snapshot"
        server.restart.return_value = False
        with self.assertRaisesRegex(RuntimeError, "rollback both failed"):
            apply_slots(24, server)
        server.release.assert_not_called()

    def test_successful_change_releases_snapshot(self):
        server = Mock()
        server.save.return_value = "snapshot"
        server.restart.return_value = True
        self.assertTrue(apply_slots(24, server))
        server.release.assert_called_once_with("snapshot")


if __name__ == "__main__":
    unittest.main()
