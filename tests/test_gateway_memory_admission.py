"""Offline tests for zram / memory PSI admission; pressure readers are always stubbed."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
REPO_TESTS = HERE  # the repository's tests folder, wherever the checkout lives
sys.path.insert(0, str(REPO_TESTS))
sys.path.insert(0, str(REPO_TESTS / 'support'))
from gateway_harness import GatewayHarness
from vram_mode import legacy_only

import test_gateway as baseline


class MemoryAdmission(unittest.TestCase):
    """Reuse the isolated gateway fixture; never touch host /proc or /sys."""

    patch_attr = baseline.GatewayGuard.patch_attr
    fake_llama = baseline.GatewayGuard.fake_llama

    @classmethod
    def setUpClass(cls):
        cls.created_import = not hasattr(baseline.GatewayGuard, "gw")
        if cls.created_import:
            baseline.GatewayGuard.setUpClass()
        cls.gw = baseline.GatewayGuard.gw

    @classmethod
    def tearDownClass(cls):
        if cls.created_import:
            baseline.GatewayGuard.tearDownClass()

    def setUp(self):
        baseline.GatewayGuard.setUp(self)
        self.gw = type(self).gw
        self.mc = self.gw.model_config(self.gw.st.model_id, False)
        self.gw.G.pop('zramMaxPctToLoad', None)
        self.gw.G.pop('psiMemoryFullMaxToLoad', None)
        self.sample = {"zram_used_percent": 42.0, "zram_used_bytes": 512 * 1024**2,
                       "full_avg10": 2.0, "full_avg60": 1.5}
        self.real_read_load_pressure = self.gw.read_load_pressure
        self.reader = Mock(return_value=self.sample)
        self.patch_attr("read_load_pressure", self.reader)

    def test_missing_keys_keep_previous_ram_vram_messages_and_skip_reader(self):
        with patch.object(self.gw, "read_load_pressure", side_effect=AssertionError("must stay disabled")):
            self.assertEqual(self.gw.preload_check(self.mc), "")
            self.gw.G['ramFreeMinGBToLoad'] = 25
            self.assertEqual(self.gw.preload_check(self.mc),
                             "Not enough free RAM to load the model (24.0 GB available, need 25 GB)")
            self.gw.G['ramFreeMinGBToLoad'] = 0
            self.gw.st.vram_free_mib = 100
            self.mc['vramFreeMinGBToLoad'] = 2
            self.assertEqual(self.gw.preload_check(self.mc), "Not enough free VRAM (0.1 GB free, need 2.00 GB)")

    def test_nonpositive_thresholds_disable_reads(self):
        self.gw.G.update(zramMaxPctToLoad=0, psiMemoryFullMaxToLoad=-1)
        with patch.object(self.gw, "read_load_pressure", side_effect=AssertionError("disabled checks must not read")):
            self.assertEqual(self.gw.preload_check(self.mc), "")

    def test_pressure_gate_false_skips_only_pressure_not_ram_or_vram(self):
        self.gw.G.update(zramMaxPctToLoad=60, psiMemoryFullMaxToLoad=5)
        self.sample.update(zram_used_percent=80, full_avg10=10, full_avg60=10)
        self.assertIn("zram", self.gw.preload_check(self.mc).lower())
        self.reader.reset_mock()
        self.assertEqual(self.gw.preload_check(self.mc, pressure_gate=False), "")
        self.reader.assert_not_called()
        self.gw.G["ramFreeMinGBToLoad"] = 25
        self.assertIn("Not enough free RAM", self.gw.preload_check(self.mc, pressure_gate=False))
        self.gw.G["ramFreeMinGBToLoad"] = 0
        self.mc["vramFreeMinGBToLoad"] = 13
        self.assertIn("Not enough free VRAM", self.gw.preload_check(self.mc, pressure_gate=False))

    def test_pressure_refusal_logs_once_per_episode_and_counts_retries(self):
        self.gw.G["zramMaxPctToLoad"] = 60
        self.sample["zram_used_percent"] = 61
        self.assertIn("zram", self.gw.preload_check(self.mc).lower())
        self.sample["zram_used_percent"] = 62
        self.assertIn("zram", self.gw.preload_check(self.mc).lower())
        self.assertEqual(self.gw.log.call_count, 1)
        self.assertEqual(self.gw.status_snapshot()["event_counters"]["load_deferred_zram"], 2)
        self.sample["zram_used_percent"] = 40
        self.assertEqual(self.gw.preload_check(self.mc), "")
        self.sample["zram_used_percent"] = 61
        self.assertIn("zram", self.gw.preload_check(self.mc).lower())
        self.assertEqual(self.gw.log.call_count, 2)
        self.assertEqual(self.gw.status_snapshot()["event_counters"]["load_deferred_zram"], 3)

    def test_request_precheck_is_not_repeated_inside_start_llama(self):
        self.gw.G.update(zramMaxPctToLoad=60, psiMemoryFullMaxToLoad=5)
        self.gw.st.proc = None
        self.gw.st.status = "UNLOADED"
        with patch.object(self.gw, "load_server_locked", return_value=""):
            self.assertEqual(self.gw.acquire_model(self.gw.st.model_id, False), "")
        self.reader.assert_called_once()

    def test_gear_rollback_restarts_under_high_zram_or_psi(self):
        gw = self.gw
        model = gw.st.model_id
        gw.G.update(slotSaveCheckpoints=1, zramMaxPctToLoad=60, psiMemoryFullMaxToLoad=5)
        real_start = gw.start_llama
        pressures = (
            {"zram_used_percent": 80, "zram_used_bytes": 900 * 1024**2, "full_avg10": 1, "full_avg60": 1},
            {"zram_used_percent": 40, "zram_used_bytes": 400 * 1024**2, "full_avg10": 8, "full_avg60": 8},
        )
        for index, pressure in enumerate(pressures):
            with self.subTest(pressure=pressure):
                self.reader.return_value = pressure
                self.reader.reset_mock()
                gw.st.model_id, gw.st.vision, gw.st.status, gw.st.proc, gw.st.busy = model, False, "READY", self.proc, 0

                def loaded(*_args, **_kwargs):
                    gw.st.proc = Mock(pid=5100 + index)
                    gw.st.proc.poll.return_value = None
                    gw.st.status = "READY"
                    return ""

                def start_without_gate(*args, **kwargs):
                    self.assertFalse(gw.PRESSURE_GATE_OVERRIDE.get())
                    return real_start(*args, **kwargs)

                with patch.object(gw, "RUNTIME_SLOTS", {}), \
                        patch.object(gw, "stop_llama_locked", return_value=True), \
                        patch.object(gw, "wait_fresh_vram", return_value=True), \
                        patch.object(gw, "load_server_locked", side_effect=loaded), \
                        patch.object(gw, "start_llama", side_effect=start_without_gate) as start:
                    backend = gw.GearBackend()
                    self.assertTrue(backend.restart(backend.original_slots, {"empty": True}))
                start.assert_called_once()
                self.reader.assert_not_called()

    def test_crash_replay_restarts_under_high_zram_or_psi(self):
        gw = self.gw
        model = gw.st.model_id
        gw.G.update(crashRecovery=True, slotSaveCheckpoints=1,
                    zramMaxPctToLoad=60, psiMemoryFullMaxToLoad=5)
        real_start = gw.start_llama
        pressures = (
            {"zram_used_percent": 80, "zram_used_bytes": 900 * 1024**2, "full_avg10": 1, "full_avg60": 1},
            {"zram_used_percent": 40, "zram_used_bytes": 400 * 1024**2, "full_avg10": 8, "full_avg60": 8},
        )
        for index, pressure in enumerate(pressures):
            with self.subTest(pressure=pressure):
                self.reader.return_value = pressure
                self.reader.reset_mock()
                gw.st.model_id, gw.st.vision, gw.st.status, gw.st.proc, gw.st.busy = model, False, "READY", self.proc, 1
                gw.st.crash_count = 0
                handler = object.__new__(gw.Handler)
                handler.command = "POST"
                handler.path = "/v1/chat/completions"
                handler.headers = {}
                handler._monitor_request_id = None

                def invoke_recovery(_handler, _body, _open, recover, *_args, **_kwargs):
                    self.assertTrue(recover())

                def loaded(*_args, **_kwargs):
                    gw.st.proc = Mock(pid=5200 + index)
                    gw.st.proc.poll.return_value = None
                    gw.st.status = "READY"
                    return ""

                def start_without_gate(*args, **kwargs):
                    self.assertFalse(gw.PRESSURE_GATE_OVERRIDE.get())
                    return real_start(*args, **kwargs)

                with patch.object(gw, "forward_recovery", side_effect=invoke_recovery), \
                        patch.object(gw, "wait_for_process_exit", return_value=True), \
                        patch.object(gw, "record_server_crash"), \
                        patch.object(gw, "note_crash", return_value=False), \
                        patch.object(gw, "wait_fresh_vram", return_value=True), \
                        patch.object(gw, "load_server_locked", side_effect=loaded), \
                        patch.object(gw, "_finish_monitor_request"), \
                        patch.object(gw, "start_llama", side_effect=start_without_gate) as start:
                    gw.Handler.proxy_recoverable(handler, b"{}", model, False)
                start.assert_called_once()
                self.reader.assert_not_called()
                gw.st.busy = 0

    def test_reader_parses_fixture_psi_and_uses_existing_zram_reader(self):
        psi = "some avg10=1.00 avg60=0.50 avg300=0.25 total=3\nfull avg10=4.25 avg60=2.75 avg300=1.25 total=7\n"
        with patch.object(Path, "read_text", return_value=psi), \
                patch.object(self.gw, "_read_zram", return_value={"used_percent": 58.0,
                                                                   "used_bytes": 456 * 1024**2}) as zram:
            self.assertEqual(self.real_read_load_pressure(), {
                "full_avg10": 4.25, "full_avg60": 2.75, "zram_used_percent": 58.0,
                "zram_used_bytes": 456 * 1024**2})
            zram.assert_called_once_with(Path("/sys"))

    def test_missing_or_malformed_psi_fixture_is_unavailable(self):
        with patch.object(Path, "read_text", side_effect=OSError("fixture absent")), \
                patch.object(self.gw, "_read_zram", return_value={"used_percent": 40.0,
                                                                    "used_bytes": 100}):
            sample = self.real_read_load_pressure()
        self.assertIsNone(sample["full_avg10"])
        self.assertIsNone(sample["full_avg60"])
        self.assertEqual(sample["zram_used_percent"], 40.0)
        with patch.object(Path, "read_text", return_value="full avg10=oops avg60=0 total=0\n"), \
                patch.object(self.gw, "_read_zram", return_value={"used_percent": 40.0,
                                                                    "used_bytes": 100}):
            sample = self.real_read_load_pressure()
        self.assertIsNone(sample["full_avg10"])

    def test_zram_under_or_at_threshold_allows_and_above_defers(self):
        self.gw.G['zramMaxPctToLoad'] = 60
        for value in (59.9, 60.0):
            with self.subTest(value=value):
                self.sample['zram_used_percent'] = value
                self.assertEqual(self.gw.preload_check(self.mc), "")
        self.sample['zram_used_percent'] = 60.1
        msg = self.gw.preload_check(self.mc)
        self.assertIn("zram", msg.lower())
        self.assertIn("60.1%", msg)
        self.assertIn("60.0%", msg)
        self.assertIn("512 MiB RAM used", msg)

    def test_psi_under_or_at_threshold_allows_and_above_defers(self):
        self.gw.G['psiMemoryFullMaxToLoad'] = 5
        for value in (4.9, 5.0):
            with self.subTest(value=value):
                self.sample['full_avg10'] = value
                self.assertEqual(self.gw.preload_check(self.mc), "")
        self.sample['full_avg10'] = 5.1
        msg = self.gw.preload_check(self.mc)
        self.assertIn("PSI", msg)
        self.assertIn("5.1%", msg)
        self.assertIn("5.0%", msg)

    def test_both_high_uses_stable_zram_message_and_records_event(self):
        self.gw.G.update(zramMaxPctToLoad=60, psiMemoryFullMaxToLoad=5)
        self.sample.update(zram_used_percent=70, full_avg10=8)
        msg = self.gw.preload_check(self.mc)
        self.assertIn("zram", msg.lower())
        self.assertNotIn("PSI", msg)
        self.gw.log.assert_called_once()
        self.assertIn("load_deferred_zram", self.gw.log.call_args.args[0])

    def test_unavailable_reader_fails_open_and_logs_once_until_recovery(self):
        self.gw.G.update(zramMaxPctToLoad=60, psiMemoryFullMaxToLoad=5)
        self.reader.return_value = None
        self.assertEqual(self.gw.preload_check(self.mc), "")
        self.assertEqual(self.gw.preload_check(self.mc), "")
        self.assertEqual(self.gw.log.call_count, 1)
        self.reader.return_value = self.sample
        self.assertEqual(self.gw.preload_check(self.mc), "")
        self.reader.return_value = None
        self.gw.preload_check(self.mc)
        self.assertEqual(self.gw.log.call_count, 3)  # first unavailable, recovery, second unavailable episode

    def test_invalid_thresholds_are_startup_configuration_errors(self):
        for value in ("bad", True, float('nan'), float('inf'), 101):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "zramMaxPctToLoad"):
                    self.gw.validate_load_pressure_config({"zramMaxPctToLoad": value})
        for value in ("oops", True, float('nan'), 101):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "psiMemoryFullMaxToLoad"):
                    self.gw.validate_load_pressure_config({"psiMemoryFullMaxToLoad": value})
        self.gw.validate_load_pressure_config({"zramMaxPctToLoad": None, "psiMemoryFullMaxToLoad": False})
        self.gw.validate_load_pressure_config({"zramMaxPctToLoad": 0, "psiMemoryFullMaxToLoad": -1})
        with self.assertRaisesRegex(ValueError, "memoryAdmissionRetryAfterSeconds"):
            self.gw.validate_load_pressure_config({"memoryAdmissionRetryAfterSeconds": 0})

    def test_retry_after_pressure_falls_and_refusal_never_enters_loading(self):
        self.gw.G['zramMaxPctToLoad'] = 60
        self.sample['zram_used_percent'] = 80
        self.gw.st.proc = None
        self.gw.st.status = "UNLOADED"
        load = Mock(return_value="")
        with patch.object(self.gw, "load_server_locked", load):
            error = self.gw.start_llama(self.gw.st.model_id, False)
            self.assertIn("zram", error.lower())
            self.assertEqual(self.gw.st.status, "UNLOADED")
            load.assert_not_called()
            self.sample['zram_used_percent'] = 55
            self.assertEqual(self.gw.start_llama(self.gw.st.model_id, False), "")
            load.assert_called_once()

    def test_pressure_refusal_is_returned_to_the_client_as_503(self):
        message = "Model load deferred: zram is 75.0% occupied (limit 60.0%). Wait, then retry."
        self.gw.G.pop("memoryAdmissionRetryAfterSeconds", None)
        handler = Mock()
        handler.command = "POST"
        handler.send_api_error.return_value = "sent"
        model = self.gw.st.model_id
        with patch.object(self.gw, "_monitor_call", return_value=None), \
                patch.object(self.gw, "inspect_body", return_value=(False, model, {"messages": []})), \
                patch.object(self.gw, "apply_terse", side_effect=lambda body, *_: body), \
                patch.object(self.gw, "acquire_model", return_value=message) as acquire:
            result = self.gw.Handler.handle_generation(handler, "/v1/chat/completions", b"{}")
        self.assertEqual(result, "sent")
        self.assertEqual(handler.send_api_error.call_args.args[:3],
                         (503, f"Could not start model '{model}': {message}", "service_unavailable"))
        self.assertEqual(handler.send_api_error.call_args.args[4], 30)
        acquire.assert_called_once_with(model, False, "short")

        handler.send_api_error.reset_mock()
        self.gw.G["memoryAdmissionRetryAfterSeconds"] = 45
        with patch.object(self.gw, "_monitor_call", return_value=None), \
                patch.object(self.gw, "inspect_body", return_value=(False, model, {"messages": []})), \
                patch.object(self.gw, "apply_terse", side_effect=lambda body, *_: body), \
                patch.object(self.gw, "acquire_model", return_value=message):
            self.gw.Handler.handle_generation(handler, "/v1/chat/completions", b"{}")
        self.assertEqual(handler.send_api_error.call_args.args[4], 45)

        handler.send_api_error.reset_mock()
        with patch.object(self.gw, "_monitor_call", return_value=None), \
                patch.object(self.gw, "inspect_body", return_value=(False, model, {"messages": []})), \
                patch.object(self.gw, "apply_terse", side_effect=lambda body, *_: body), \
                patch.object(self.gw, "acquire_model", return_value="Not enough free RAM"):
            self.gw.Handler.handle_generation(handler, "/v1/chat/completions", b"{}")
        self.assertIsNone(handler.send_api_error.call_args.args[4])

    def test_explicit_admitted_load_keeps_existing_ram_bypass_semantics(self):
        self.gw.G['zramMaxPctToLoad'] = 60
        self.sample['zram_used_percent'] = 90
        self.gw.st.proc = None
        load = Mock(return_value="")
        with patch.object(self.gw, "load_server_locked", load):
            self.assertEqual(self.gw.start_llama(self.gw.st.model_id, False, admitted=True), "")
        self.reader.assert_not_called()
        load.assert_called_once()


class MemoryAdmissionHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.created_import = not hasattr(baseline.GatewayGuard, "gw")
        if cls.created_import:
            baseline.GatewayGuard.setUpClass()
        cls.gw = baseline.GatewayGuard.gw

    @classmethod
    def tearDownClass(cls):
        if cls.created_import:
            baseline.GatewayGuard.tearDownClass()

    def test_missing_pressure_keys_still_load_the_fake_server(self):
        with GatewayHarness(guardian={"ramFreeMinGBToLoad": 0}) as gw:
            self.assertEqual(gw.chat("memory-admission-default")[0], 200)
            self.assertEqual(gw.status()['status'], 'READY')

    def test_legacy_selector_keeps_the_existing_feature_conditions(self):
        config = dict(self.gw.G, vramGuard=False, slotSaveCheckpoints=0,
                      politeModel=False, slotGears=False, crashRecovery=False)
        self.assertTrue(self.gw.legacy_gateway_selected(config, game_enabled=False))
        self.assertFalse(self.gw.legacy_gateway_selected({**config, "vramGuard": True}, False))
        self.assertFalse(self.gw.legacy_gateway_selected({**config, "slotSaveCheckpoints": 1}, False))
        self.assertFalse(self.gw.legacy_gateway_selected(config, game_enabled=True))
        self.assertFalse(self.gw.legacy_gateway_selected({**config, "ramPolicy": "honest"}, False))   # RAMLOGIC2
        self.assertTrue(self.gw.legacy_gateway_selected({**config, "ramPolicy": "off"}, False))

    def test_invalid_pressure_config_fails_at_gateway_startup(self):
        gw = GatewayHarness(guardian={"vramGuard": True, "psiMemoryFullMaxToLoad": "not-a-number"})
        try:
            with self.assertRaisesRegex(RuntimeError, "psiMemoryFullMaxToLoad"):
                gw.start(timeout=3)
        finally:
            gw.stop()

    @legacy_only("the legacy entrypoint's pressure warnings do not run under the governor (new entrypoint validates strictly)")
    def test_legacy_path_warns_and_continues_with_bad_pressure_values(self):
        gw = GatewayHarness(guardian={"zramMaxPctToLoad": 60, "psiMemoryFullMaxToLoad": "bad"})
        try:
            gw.start()
            log = gw.log()
            self.assertEqual(log.count("Guardian memory_admission_legacy_ignored"), 1)
            self.assertIn("zramMaxPctToLoad", log)
            self.assertIn("psiMemoryFullMaxToLoad", log)
            self.assertIn("legacy gateway", log)
            self.assertEqual(gw.request("GET", "/health")[0], 200)
        finally:
            gw.stop()

    def test_model_launch_pressure_keys_are_warned_as_misplaced(self):
        with GatewayHarness(launch={"zramMaxPctToLoad": 60},
                            model={"psiMemoryFullMaxToLoad": 5}) as gw:
            log = gw.log()
            self.assertEqual(log.count("Guardian memory_admission_misplaced_config"), 1)
            self.assertIn("model-launch.json", log)
            self.assertIn("guardian.json", log)
