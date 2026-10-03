"""CPU property and response faults use fake cgroups and fake HTTP; no GPU."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))
from cpu_cgroup import Scheduler
from response_recovery import forward, interrupted_event


class CgroupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "proc/self").mkdir(parents=True)
        (self.root / "proc/self/cgroup").write_text("0::/unit")
        (self.root / "cg/unit").mkdir(parents=True)
        (self.root / "cg/unit/cpu.weight").write_text("100")
        self.weight = "[not set]"
        self.calls = []
        self.s = Scheduler(
            {"politeUnit": "test.service"},
            root=self.root / "proc",
            runner=self.fake_run,
            cgroup_root=self.root / "cg",
        )
        self.s.lease_dir = self.root / "lease"
        self.s.lease = self.s.lease_dir / "weight.json"

    def fake_run(self, args, **kwargs):
        self.calls.append(args)
        if "set-property" in args:
            self.weight = args[-1].split("=", 1)[1] or "[not set]"
        return Mock(stdout="ControlGroup=/unit\nCPUWeight=" + self.weight + "\n")

    def test_reversible_and_restores_original(self):
        self.assertTrue(self.s.reversible())
        self.s.change(20)
        self.assertEqual(self.weight, "20")
        self.s.change(100)
        self.assertEqual(self.weight, "100")
        self.s.close()
        self.assertEqual(self.weight, "[not set]")
        self.assertFalse(self.s.lease.exists())

    def test_no_property_call_when_unit_does_not_own_gateway(self):
        (self.root / "proc/self/cgroup").write_text("0::/elsewhere")
        with self.assertRaises(PermissionError):
            self.s.change(20)
        self.assertFalse(any("set-property" in call for call in self.calls))

    def test_same_weight_does_not_repeat_property_writes(self):
        self.s.change(20)
        self.calls = []
        self.s.change(20)
        self.assertFalse(any("set-property" in call for call in self.calls))

    def test_property_failure_rolls_back(self):
        self.s.reversible()
        self.s.runner = Mock(side_effect=[RuntimeError("injected"), Mock(stdout="")])
        with self.assertRaises(RuntimeError):
            self.s.change(20)
        self.assertEqual(self.s.runner.call_args_list[-1].args[0][-1], "CPUWeight=")
        self.assertFalse(self.s.lease.exists())

    def test_crash_recovery_handles_pre_write_and_post_write(self):
        self.s.reversible()
        self.s.change(20)
        self.s.write_lease(5)  # gateway died before the property changed
        self.s.recover()
        self.assertEqual(self.weight, "[not set]")
        self.s.change(5)
        self.s.recover()
        self.assertEqual(self.weight, "[not set]")

    def test_crash_before_first_property_write_is_harmless(self):
        self.s.reversible()
        self.s.write_lease(20)
        self.s.recover()
        self.assertEqual(self.weight, "[not set]")
        self.assertFalse(self.s.lease.exists())

    def test_external_property_change_is_not_overwritten(self):
        self.s.change(20)
        self.weight = "77"
        with self.assertRaises(RuntimeError):
            self.s.close()
        self.assertEqual(self.weight, "77")

    def test_unsafe_lease_refused(self):
        self.s.lease_dir.mkdir(mode=0o700)
        self.s.lease.write_text("x" * 5000)
        with self.assertRaises(PermissionError):
            self.s.recover()


class ResponseTests(unittest.TestCase):
    def handler(self):
        h = Mock(
            command="POST", path="/v1/chat/completions", headers={}, hop_by_hop=set()
        )
        h.wfile = io.BytesIO()
        return h

    def conn(self, chunks):
        response = Mock(status=200)
        response.getheader.return_value = "text/event-stream"
        response.getheaders.return_value = [("Content-Type", "text/event-stream")]
        response.read1.side_effect = chunks
        c = Mock()
        c.getresponse.return_value = response
        return c

    def test_no_bytes_safe_replay_once(self):
        h = self.handler()
        bad = self.conn([ConnectionResetError()])
        good = self.conn([b"data: [DONE]\n\n", b""])

        def recover():
            h.end_headers.assert_not_called()
            self.assertEqual(h.wfile.getvalue(), b"")
            return True

        call = Mock(side_effect=[bad, good])
        forward(h, b"{}", call, recover)
        self.assertEqual(call.call_count, 2)
        self.assertIn(b"[DONE]", h.wfile.getvalue())

    def test_partial_stream_is_not_replayed(self):
        h = self.handler()
        recover = Mock()
        c = self.conn(
            [
                b'data: {"choices":[{"delta":{"content":"part"}}]}\n\n',
                ConnectionResetError(),
            ]
        )
        forward(h, b"{}", lambda: c, recover)
        recover.assert_not_called()
        self.assertIn(b"SHURA_RESPONSE_INTERRUPTED", h.wfile.getvalue())
        self.assertNotIn(b"[DONE]", h.wfile.getvalue())

    def test_headers_are_first_byte_boundary(self):
        h = self.handler()
        h.end_headers.side_effect = BrokenPipeError()
        recover = Mock()
        forward(h, b"{}", lambda: self.conn([b"data: x\n\n"]), recover)
        recover.assert_not_called()

    def test_retry_failure_returns_503_before_headers(self):
        h = self.handler()
        forward(h, b"{}", lambda: self.conn([ConnectionResetError()]), lambda: False)
        self.assertEqual(h.send_json.call_args.args[0], 503)
        h.end_headers.assert_not_called()

    def test_game_pause_does_not_replay(self):
        cancel = Mock()
        cancel.is_set.return_value = True
        h = self.handler()
        recover = Mock()
        forward(h, b"{}", lambda: self.conn([b"data: x\n\n"]), recover, cancel)
        recover.assert_not_called()

    def test_normal_done_contains_no_error(self):
        h = self.handler()
        forward(h, b"{}", lambda: self.conn([b"data: [DONE]\n\n", b""]), lambda: False)
        self.assertNotIn(b"SHURA_RESPONSE_INTERRUPTED", h.wfile.getvalue())

    def test_error_message_is_not_retryable_phrase(self):
        value = json.loads(interrupted_event().decode()[6:])
        self.assertEqual(value["error"]["code"], "SHURA_RESPONSE_INTERRUPTED")
        self.assertNotIn("503", value["error"]["message"])
        self.assertNotIn("retry", value["error"]["message"])


class AffinityTests(unittest.TestCase):
    def test_cgroup_executor_never_changes_nice_or_scheduler(self):
        import cpu_cgroup
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "42/task"
            tasks.mkdir(parents=True)
            (tasks / "42").mkdir()
            (tasks / "43").mkdir()
            (root / "42/cgroup").write_text("0::/unit")
            system = Mock()
            system.sched_getaffinity.return_value = {0, 1, 2, 3}
            scheduler = cpu_cgroup.Scheduler({}, root=root, system=system)
            scheduler.path = "/unit"
            with (
                patch.object(scheduler, "change") as change,
                patch.object(scheduler, "cpus_for_l3", return_value={0, 1}),
                patch("polite_model.proc_ticks", return_value=(0, 7)),
            ):
                scheduler.apply(42, 7, 3, {0, 1, 2, 3})
                self.assertEqual(system.sched_setaffinity.call_count, 2)
                system.sched_setaffinity.assert_any_call(42, {0, 1})
                scheduler.apply(42, 7, 0, {0, 1, 2, 3})
                self.assertEqual(system.sched_setaffinity.call_count, 4)
                system.sched_setaffinity.assert_any_call(42, {0, 1, 2, 3})
                self.assertEqual(change.call_args.args[0], 100)
            system.sched_setscheduler.assert_not_called()
            system.setpriority.assert_not_called()


class ControllerModeTests(unittest.TestCase):
    def test_constant_weight_does_not_depend_on_sensors(self):
        from polite_model import Controller
        from unittest.mock import patch

        scheduler = Mock(root=Path("/proc"))
        scheduler.reversible.return_value = True
        scheduler.system.sched_getaffinity.return_value = {0, 1}
        sensors = Mock()
        controller = Controller(
            {"politeModel": True, "politeMode": "static", "politeStaticWeight": 20},
            scheduler=scheduler,
            sensors=sensors,
        )
        with patch("polite_model.proc_ticks", return_value=(0, 7)):
            controller.tick(1, 42, 100)
        sensors.sample.assert_not_called()
        self.assertEqual(scheduler.weights, [20] * 4)
        scheduler.apply.assert_called_once_with(42, 7, 0, {0, 1})

    def test_dynamic_mode_retains_aimd(self):
        from polite_model import Controller, Sample
        from unittest.mock import patch

        scheduler = Mock(root=Path("/proc"))
        scheduler.reversible.return_value = True
        scheduler.system.sched_getaffinity.return_value = {0, 1}
        sensors = Mock()
        sensors.sample.return_value = Sample(15, 0.1, 0.1)
        controller = Controller(
            {"politeModel": True, "politeMode": "dynamic"},
            scheduler=scheduler,
            sensors=sensors,
        )
        with patch("polite_model.proc_ticks", return_value=(0, 7)):
            controller.tick(1, 42, 0.1)
        self.assertEqual(controller.policy.level, 1)
        scheduler.apply.assert_called_once_with(42, 7, 1, {0, 1})
