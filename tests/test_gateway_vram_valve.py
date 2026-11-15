"""Gateway valve tests use the fake server and a manually published NVML sample."""
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
import test_gateway as fixture


class GatewayVramValve(unittest.TestCase):
    patch_attr = fixture.GatewayGuard.patch_attr
    fake_llama = fixture.GatewayGuard.fake_llama

    @classmethod
    def setUpClass(cls):
        cls.created_import = not hasattr(fixture.GatewayGuard, "gw")
        if cls.created_import:
            fixture.GatewayGuard.setUpClass()
        cls.gw = fixture.GatewayGuard.gw

    @classmethod
    def tearDownClass(cls):
        if cls.created_import:
            fixture.GatewayGuard.tearDownClass()

    def setUp(self):
        self.gw = type(self).gw
        fixture.GatewayGuard.setUp(self)
        self.gw = type(self).gw
        self.mid = self.gw.M["defaultModel"]
        self.patch_attr("G_RAW", {**self.gw.G_RAW, "vramCardReserveMiB": 194, "vramValve": True,
                                   "vramGuard": True, "slotSaveCheckpoints": 1})
        self.patch_attr("M", {**self.gw.M, "models": {mid: dict(mc) for mid, mc in self.gw.M["models"].items()}})
        self.patch_attr("RUNTIME_SLOTS", {self.mid: 18})
        self.patch_attr("VALVES", {})
        self.patch_attr("SLOT_COSTS", self.gw.SlotCostEstimator())
        self.patch_attr("SLOT_COST_WARNED", set())
        self.patch_attr("valve_vram_probe_window", Mock(return_value=None))
        self.gw.G.update(vramValve=True, vramGuard=True, slotSaveCheckpoints=1,
                         vramCardReserveMiB=194, vramValveLevels=[18, 16, 8, 0],
                         vramValveHysteresisMiB=200, vramValveUpSeconds=60,
                         vramValveParkSeconds=2, slotGearMiBPerSlot=36, slotGears=False)
        self.gw.M["models"][self.mid]["moeCacheSlots"] = 18
        self.gw.M["models"][self.mid]["moeCacheProfile"] = "/tmp/fake-profile.csv"
        self.gw.M["models"][self.mid].pop("moeCacheSlotMiB", None)
        self.gw.M["models"][self.mid].pop("vramValveLevels", None)
        self.gw.st.status = "READY"
        self.gw.st.model_id = self.mid
        self.gw.st.vision = False
        self.gw.st.proc = Mock(pid=9876)
        self.gw.st.proc.poll.return_value = None
        self.gw.st.busy = 0
        self.gw.st.vram_free_mib = 100
        self.gw.st.vram_sample_at = time.time()
        self.gw.st.monitor_error = ""

    def test_disabled_valve_keeps_server_command_and_policy_unchanged(self):
        self.gw.G["vramValve"] = False
        self.gw.G["slotSaveCheckpoints"] = 0
        self.gw.G_RAW.pop("vramValve", None)
        self.gw.G_RAW["slotSaveCheckpoints"] = 0
        args = self.gw.build_args(self.gw.model_config(self.mid, False), False)
        self.assertNotIn("--moe-cache-live-resize", args)
        self.gw.M["models"][self.mid]["moeCacheSlotMiB"] = 31.3
        self.gw.M["models"][self.mid]["vramValveLevels"] = [54, 40]
        with_model_valve_metadata = self.gw.build_args(self.gw.model_config(self.mid, False), False)
        self.assertEqual(args, with_model_valve_metadata)
        self.assertEqual(self.gw.status_snapshot()["vram_valve"]["enabled"], False)
        with patch.object(self.gw, "llama_call") as call:
            self.gw.reliability_tick(0)
        call.assert_not_called()

    def test_low_sample_shrinks_live_server_and_updates_status(self):
        def resize(method, path, body, timeout=10):
            self.assertEqual((method, path, body["slots"]), ("POST", "/moe-cache/slots", 8))
            return 200, {"success": True, "old_slots": 18, "new_slots": 8,
                         "released_bytes": 360 * 1024 * 1024}
        with patch.object(self.gw, "llama_call", side_effect=resize):
            self.gw.reliability_tick(time.monotonic())
        valve = self.gw.VALVES[self.mid]
        self.assertEqual(valve.current, 8)
        self.assertEqual(self.gw.RUNTIME_SLOTS[self.mid], 8)
        self.assertEqual(self.gw.st.events[-1]["reason"], "vram_valve_resized")
        status = self.gw.status_snapshot()["vram_valve"]
        self.assertEqual(status["models"][self.mid]["slots"], 8)

    def test_busy_request_waits_for_server_request_boundary(self):
        self.gw.st.busy = 1
        with patch.object(self.gw, "llama_call") as call:
            self.gw.reliability_tick(time.monotonic())
        call.assert_not_called()
        self.assertEqual(self.gw.valve_for(self.mid).current, 18)
        self.assertIn("request boundary", self.gw.valve_for(self.mid).reason)

    def test_resize_helper_refuses_direct_call_while_request_is_active(self):
        self.gw.st.busy = 1
        with patch.object(self.gw, "llama_call") as call:
            self.assertIsNone(self.gw.apply_valve_resize_locked(self.mid, 16, time.monotonic()))
        call.assert_not_called()
        self.assertEqual(self.gw.RUNTIME_SLOTS[self.mid], 18)
        self.assertIn("idle request boundary", self.gw.valve_for(self.mid).reason)

    def test_failed_resize_retains_old_level_and_retries_after_backoff(self):
        with patch.object(self.gw, "llama_call", return_value=(500, {
                "success": False, "old_slots": 18, "new_slots": 18, "message": "allocation failed"})) as call:
            self.gw.reliability_tick(10)
            self.assertEqual(self.gw.valve_for(self.mid).current, 18)
            self.gw.reliability_tick(11)
            self.assertEqual(call.call_count, 1)
            self.gw.reliability_tick(13)
            self.assertEqual(call.call_count, 2)
        self.assertEqual(self.gw.st.events[-1]["reason"], "vram_valve_resize_failed")

    def test_double_failure_truthfully_records_server_zero(self):
        with patch.object(self.gw, "llama_call", return_value=(500, {
                "success": False, "old_slots": 18, "new_slots": 0, "message": "rollback failed; cache disabled"})):
            self.gw.reliability_tick(1)
        self.assertEqual(self.gw.valve_for(self.mid).current, 0)
        self.assertEqual(self.gw.RUNTIME_SLOTS[self.mid], 0)

    def test_stale_or_missing_sample_does_not_resize_and_is_visible(self):
        self.gw.st.vram_sample_at = time.time() - 60
        with patch.object(self.gw, "llama_call") as call:
            self.gw.reliability_tick(time.monotonic())
        call.assert_not_called()
        self.assertEqual(self.gw.st.events[-1]["reason"], "vram_valve_telemetry_unavailable")
        self.gw.st.vram_free_mib = None
        self.gw.st.vram_sample_at = time.time()
        with patch.object(self.gw, "llama_call") as call:
            self.gw.reliability_tick(time.monotonic() + 1)
        call.assert_not_called()

    def test_zero_slots_below_reserve_uses_existing_a2_parking(self):
        valve = self.gw.valve_for(self.mid)
        valve.current = 0
        self.gw.RUNTIME_SLOTS[self.mid] = 0
        self.gw.st.vram_free_mib = 180
        now = time.monotonic()
        with patch.object(self.gw, "llama_call") as call, \
                patch.object(self.gw, "save_slot_locked", return_value=True) as save, \
                patch.object(self.gw, "stop_llama_locked", return_value=True) as stop:
            self.gw.reliability_tick(now)
            self.gw.reliability_tick(now + 3)
        call.assert_not_called()
        save.assert_called_once_with(ram_only=True)
        stop.assert_called_once_with("VRAM valve minimum reached; vramGuard parking", save=False)
        self.assertEqual(self.gw.st.last_yield_reason, "vram_valve_exhausted")

    def test_recovery_requires_full_hysteresis_window_then_grows_one_step(self):
        valve = self.gw.valve_for(self.mid)
        valve.current = 0
        self.gw.RUNTIME_SLOTS[self.mid] = 0
        self.gw.st.vram_free_mib = 900
        start = time.monotonic()
        call = Mock(return_value=(200, {"success": True, "old_slots": 0, "new_slots": 8}))
        with patch.object(self.gw, "llama_call", side_effect=call):
            self.gw.reliability_tick(start)
            self.assertEqual(call.call_count, 0)
            self.gw.reliability_tick(start + 59)
            self.assertEqual(call.call_count, 0)
            self.gw.reliability_tick(start + 60)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args[2], {"slots": 8})
        self.assertEqual(valve.current, 8)

    def test_slot_zero_command_keeps_profile_and_adds_live_resize_flag(self):
        self.gw.RUNTIME_SLOTS[self.mid] = 0
        self.gw.G["vramValve"] = True
        with patch.object(self.gw, "server_supports", side_effect=lambda flag: flag == "--ctx-checkpoints"):
            args = self.gw.build_args(self.gw.model_config(self.mid, False), False)
        self.assertEqual(args[args.index("--moe-cache-slots") + 1], "0")
        self.assertIn("--moe-cache-profile", args)
        self.assertIn("--moe-cache-live-resize", args)

    def test_checkpoint_count_falls_back_to_upstream_flag_only_when_opted_in(self):
        self.gw.G["slotSaveCheckpoints"] = 1
        with patch.object(self.gw, "server_supports", side_effect=lambda flag: flag == "--ctx-checkpoints"):
            args = self.gw.build_args(self.gw.model_config(self.mid, False), False)
        self.assertIn("--ctx-checkpoints", args)
        self.assertEqual(args[args.index("--ctx-checkpoints") + 1], "1")
        self.assertNotIn("--slot-save-checkpoints", args)

    def test_a2_signature_ignores_slot_count_only_while_opt_in(self):
        self.gw.G["vramValve"] = True
        self.gw.RUNTIME_SLOTS[self.mid] = 18
        enabled = self.gw.checkpoint_signature(self.mid, False)
        self.gw.RUNTIME_SLOTS[self.mid] = 0
        self.assertEqual(self.gw.checkpoint_signature(self.mid, False), enabled)
        self.gw.G["vramValve"] = False
        self.gw.G_RAW.pop("vramCardReserveMiB", None)
        configured = self.gw.checkpoint_signature(self.mid, False)
        self.gw.RUNTIME_SLOTS[self.mid] = 18
        self.assertNotEqual(self.gw.checkpoint_signature(self.mid, False), configured)

    def test_startup_validation_rejects_missing_guard_or_conflicting_slot_gears(self):
        good = {"vramValve": True, "vramCardReserveMiB": 194, "vramGuard": True,
                "slotSaveCheckpoints": 1, "slotGears": False}
        self.gw.validate_vram_valve_config(good)
        with self.assertRaisesRegex(ValueError, "vramGuard"):
            self.gw.validate_vram_valve_config({**good, "vramGuard": False})
        with self.assertRaisesRegex(ValueError, "slotGears"):
            self.gw.validate_vram_valve_config({**good, "slotGears": True})
        self.assertTrue(self.gw.legacy_gateway_selected({"vramValve": False}, False))

    def test_startup_reserve_defers_instead_of_selecting_zero_for_valve(self):
        self.gw.st.vram_free_mib = 10444
        self.gw.M["models"][self.mid]["vramFreeMinGBToLoad"] = 10.0
        self.gw.G["vramValveLevels"] = [18, 16, 8, 0]
        mc = self.gw.model_config(self.mid, False)
        result = self.gw.choose_startup_slots(self.mid, False, mc)
        self.assertIn("minimum cache 8 slots", result)
        self.assertEqual(mc["moeCacheSlots"], 18)
        self.assertEqual(self.gw.RUNTIME_SLOTS[self.mid], 18)
        self.assertNotIn(self.mid, self.gw.STARTUP_VRAM_SELECTIONS)

    def test_valve_startup_levels_are_bounded_and_positive_for_apex_maxima(self):
        self.gw.G["vramValveLevels"] = [60, 40, 0]
        self.assertEqual(self.gw.startup_slot_levels(self.mid, 54), [54, 40])
        self.assertEqual(self.gw.startup_slot_levels(self.mid, 60), [60, 40])

    def test_model_specific_ladders_and_slot_costs_survive_model_switches(self):
        tiel = self.mid
        occamy = "occamy-1.0"
        models = {mid: dict(mc) for mid, mc in self.gw.M["models"].items()}
        models[tiel].update(moeCacheSlots=54, vramValveLevels=[54, 40])
        models[tiel].pop("moeCacheSlotMiB", None)
        models[occamy] = {**dict(models[tiel]), "moeCacheSlots": 12}
        models[occamy].pop("vramValveLevels", None)  # uses shared [54,40], clipped to [12]
        self.patch_attr("M", {**self.gw.M, "models": models})
        # Tiel's record overrides the shared IQ4-style ladder. Occamy inherits
        # the APEX shared list below and gets only its own configured maximum.
        self.gw.G["vramValveLevels"] = [24, 18, 16, 8]
        self.patch_attr("RUNTIME_SLOTS", {})
        self.patch_attr("VALVES", {})
        foreign = {111: 128, 222: 64}
        self.gw.SLOT_COSTS.observe_resize(tiel, 54, 40, [500, 500], [938.2, 938.2], foreign, foreign)
        self.gw.SLOT_COSTS.observe_resize(occamy, 12, 10, [500, 500], [572.4, 572.4], foreign, foreign)

        self.gw.st.model_id = tiel
        tiel_valve = self.gw.valve_for(tiel)
        self.assertEqual(tiel_valve.levels, [54, 40])
        self.assertAlmostEqual(tiel_valve.slot_cost_mib, 31.3, places=1)
        self.assertEqual(tiel_valve.slot_cost_source, "measured")

        self.gw.G["vramValveLevels"] = [54, 40]
        self.gw.st.model_id = occamy
        occamy_valve = self.gw.valve_for(occamy)
        self.assertEqual(occamy_valve.levels, [12])
        self.assertAlmostEqual(occamy_valve.slot_cost_mib, 36.2, places=1)
        self.assertEqual(occamy_valve.slot_cost_source, "measured")
        with patch.object(self.gw, "llama_call") as call:
            self.gw.st.vram_free_mib = 100
            self.gw.vram_valve_tick(time.monotonic())
        call.assert_not_called()
        self.assertFalse(any(e.get("reason") == "vram_valve_config_error" for e in self.gw.st.events))

        self.gw.st.model_id = tiel
        self.assertIs(self.gw.valve_for(tiel), tiel_valve)
        self.assertIs(self.gw.valve_for(occamy), occamy_valve)
        self.assertAlmostEqual(self.gw.valve_for(tiel).slot_cost_mib, 31.3, places=1)
        self.assertAlmostEqual(self.gw.valve_for(occamy).slot_cost_mib, 36.2, places=1)

    def test_configured_model_price_wins_and_appears_in_status_and_resize_log(self):
        self.gw.M["models"][self.mid]["moeCacheSlotMiB"] = 31.3
        valve = self.gw.valve_for(self.mid)
        self.assertEqual((valve.slot_cost_mib, valve.slot_cost_source), (31.3, "config"))
        before = {"free_samples": [500, 502], "foreign": {11: 100}}
        after = {"free_samples": [562, 563], "foreign": {11: 100}}
        with patch.object(self.gw, "valve_vram_probe_window", side_effect=[before, after]), \
                patch.object(self.gw, "llama_call", return_value=(200, {
                    "success": True, "old_slots": 18, "new_slots": 16})):
            self.gw.apply_valve_resize_locked(self.mid, 16, time.monotonic())
        self.assertEqual((valve.slot_cost_mib, valve.slot_cost_source), (31.3, "config"))
        status = self.gw.status_snapshot()["vram_valve"]["models"][self.mid]
        self.assertEqual(status["slot_cost_source"], "config")
        self.assertEqual(status["slot_cost_mib"], 31.3)
        details = " ".join(e["detail"] for e in self.gw.st.events if e.get("reason") == "vram_valve_resized")
        self.assertIn("source=config", details)

    def test_vision_valve_is_inactive_and_status_reports_eight_slots(self):
        self.gw.M["models"][self.mid]["moeCacheSlots"] = 54
        self.gw.M["models"][self.mid]["visionOverrides"] = {"moeCacheSlots": 8}
        self.gw.st.vision = True
        self.gw.st.model_id = self.mid
        self.assertIsNone(self.gw.valve_for(self.mid))
        status = self.gw.status_snapshot()["vram_valve"]
        self.assertFalse(status["active"])
        self.assertEqual(status["models"][self.mid]["slots"], 8)
        self.assertNotEqual(status["models"][self.mid]["slots"], 54)

    def test_model_ladder_config_validation_allows_levels_above_per_model_max(self):
        self.gw.validate_vram_valve_model_configs({"tiel": {"moeCacheSlots": 54, "vramValveLevels": [54, 40]},
                                                   "occamy": {"moeCacheSlots": 12, "vramValveLevels": [54, 40]}})

    def test_apex_startup_can_use_positive_twenty_rung_when_configured(self):
        self.gw.G["vramValveLevels"] = [60, 40, 20, 0]
        self.assertEqual(self.gw.startup_slot_levels(self.mid, 54), [54, 40, 20])
        self.assertEqual(self.gw.startup_slot_levels(self.mid, 60), [60, 40, 20])

    def test_startup_selects_smallest_positive_rung_when_only_that_fits(self):
        self.gw.G["vramValveLevels"] = [60, 40, 0]
        self.gw.st.vram_free_mib = 11634
        self.gw.M["models"][self.mid]["moeCacheSlots"] = 60
        self.gw.M["models"][self.mid]["vramFreeMinGBToLoad"] = 10.0
        mc = self.gw.model_config(self.mid, False)
        self.assertEqual(self.gw.choose_startup_slots(self.mid, False, mc), "")
        self.assertEqual(mc["moeCacheSlots"], 40)
        self.assertEqual(self.gw.STARTUP_VRAM_SELECTIONS[self.mid]["slots"], 40)


if __name__ == "__main__":
    unittest.main()
