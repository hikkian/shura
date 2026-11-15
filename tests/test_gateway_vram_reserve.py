"""Startup card-VRAM reserve selection, using the isolated fake server and NVML sample."""
import time
import unittest
from unittest.mock import Mock, patch

import test_gateway as fixture


class GatewayVramReserve(unittest.TestCase):
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
        self.patch_attr("G_RAW", dict(self.gw.G_RAW))
        self.patch_attr("M", {**self.gw.M, "models": {mid: dict(mc) for mid, mc in self.gw.M["models"].items()}})
        self.patch_attr("RUNTIME_SLOTS", {})
        self.patch_attr("STARTUP_VRAM_SELECTIONS", {})
        self.gw.G["slotGearMiBPerSlot"] = 36
        self.gw.G["slotGearLevels"] = [26, 24, 20, 16]
        self.gw.G["vramTelemetryStaleSeconds"] = 5
        self.gw.M["models"][self.mid]["moeCacheSlots"] = 26
        self.gw.M["models"][self.mid]["vramFreeMinGBToLoad"] = 0
        self.gw.st.proc = None
        self.gw.st.status = "UNLOADED"
        self.gw.st.model_id = ""
        self.gw.st.vram_free_mib = 2000
        self.gw.st.vram_sample_at = time.time()
        self.gw.st.monitor_error = ""

    def enable(self, reserve=194):
        self.gw.G_RAW["vramCardReserveMiB"] = reserve
        self.gw.G["vramCardReserveMiB"] = reserve

    def test_missing_key_preserves_configured_slots_without_nvml_or_override(self):
        mc = self.gw.model_config(self.mid, False)
        with patch.object(self.gw, "vram_card_reserve_enabled", return_value=False), \
                patch.object(self.gw, "process_vram_mib", side_effect=AssertionError("no process-peak gate")):
            self.assertEqual(self.gw.choose_startup_slots(self.mid, False, mc), "")
        self.assertNotIn(self.mid, self.gw.RUNTIME_SLOTS)
        self.assertEqual(mc["moeCacheSlots"], 26)

    def test_acquire_uses_configured_slots_when_reserve_key_is_zero(self):
        self.gw.G_RAW["vramCardReserveMiB"] = 0
        seen = []
        def fake_start(mid, vision, **_kwargs):
            seen.extend(self.gw.build_args(self.gw.model_config(mid, vision), vision))
            return ""
        with patch.object(self.gw, "start_llama", side_effect=fake_start):
            self.assertEqual(self.gw.acquire_model(self.mid, False), "")
        self.assertIn("26", seen[seen.index("--moe-cache-slots") + 1:])
        self.assertEqual(seen[seen.index("--moe-cache-slots") + 1], "26")

    def test_plenty_of_card_vram_keeps_configured_maximum_and_status_logs_choice(self):
        self.enable()
        mc = self.gw.model_config(self.mid, False)
        self.assertEqual(self.gw.choose_startup_slots(self.mid, False, mc), "")
        self.assertEqual(mc["moeCacheSlots"], 26)
        self.assertEqual(self.gw.RUNTIME_SLOTS[self.mid], 26)
        self.assertEqual(self.gw.st.events[-1]["reason"], "vram_card_reserve_selected")
        status = self.gw.status_snapshot()
        self.assertEqual(status["vram_card_reserve"]["models"][self.mid]["slots"], 26)
        self.assertTrue(status["slot_gears"]["enabled"])
        self.assertIn("reserve 194 MiB", status["slot_gears"]["models"][self.mid]["reason"])

    def test_low_free_vram_drops_multiple_known_levels(self):
        self.enable()
        self.gw.st.vram_free_mib = 1000  # N=24 leaves 136; N=20 leaves 280.
        mc = self.gw.model_config(self.mid, False)
        self.assertEqual(self.gw.choose_startup_slots(self.mid, False, mc), "")
        self.assertEqual(mc["moeCacheSlots"], 20)
        self.assertEqual(self.gw.RUNTIME_SLOTS[self.mid], 20)

    def test_lowered_choice_reaches_fake_server_command(self):
        self.enable()
        self.gw.st.vram_free_mib = 1000
        seen = []
        def fake_start(mid, vision, **_kwargs):
            seen.extend(self.gw.build_args(self.gw.model_config(mid, vision), vision))
            return ""
        with patch.object(self.gw, "start_llama", side_effect=fake_start):
            self.assertEqual(self.gw.acquire_model(self.mid, False), "")
        self.assertEqual(seen[seen.index("--moe-cache-slots") + 1], "20")

    def test_very_low_free_vram_defers_before_fake_server_launch(self):
        self.enable()
        self.gw.st.vram_free_mib = 500
        with patch.object(self.gw, "start_llama") as start:
            error = self.gw.acquire_model(self.mid, False)
        start.assert_not_called()
        self.assertIn("reserve 194 MiB cannot be kept", error)
        self.assertEqual(self.gw.st.events[-1]["reason"], "vram_card_reserve_deferred")
        self.assertEqual(self.gw.st.status, "UNLOADED")

    def test_model_switch_credits_fresh_vram_released_by_current_server(self):
        self.enable()
        self.gw.G["vramValve"] = True
        self.gw.M["models"]["occamy"] = {"moeCacheSlots": 12, "moeCacheSlotMiB": 36.0,
                                         "vramValveLevels": [12], "vramFreeMinGBToLoad": 8.0}
        self.gw.st.model_id = self.mid
        self.gw.st.vision = False
        self.gw.st.status = "READY"
        self.gw.st.proc = self.proc
        self.gw.st.vram_free_mib = 500
        mc = self.gw.model_config("occamy", False)

        self.assertEqual(self.gw.choose_startup_slots("occamy", False, mc), "")
        self.assertEqual(mc["moeCacheSlots"], 12)
        selected = self.gw.STARTUP_VRAM_SELECTIONS["occamy"]
        self.assertGreater(selected["estimated_after_mib"], 194)
        self.assertGreater(selected["free_mib"], self.gw.st.vram_free_mib)

    def test_model_switch_does_not_credit_stale_current_process_vram(self):
        self.enable()
        self.gw.G["vramValve"] = True
        self.gw.M["models"]["occamy"] = {"moeCacheSlots": 12, "moeCacheSlotMiB": 36.0,
                                         "vramValveLevels": [12], "vramFreeMinGBToLoad": 8.0}
        self.gw.st.model_id = self.mid
        self.gw.st.proc = self.proc
        self.gw.st.status = "READY"
        self.gw.st.vram_free_mib = 500
        with patch.object(self.gw, "process_vram_mib", return_value=-1):
            error = self.gw.choose_startup_slots("occamy", False, self.gw.model_config("occamy", False))
        self.assertIn("card VRAM reserve", error)
        self.assertNotIn("occamy", self.gw.RUNTIME_SLOTS)

    def test_exact_boundary_is_admitted(self):
        self.enable()
        self.gw.st.vram_free_mib = 194 + 24 * 36
        mc = self.gw.model_config(self.mid, False)
        self.assertEqual(self.gw.choose_startup_slots(self.mid, False, mc), "")
        self.assertEqual(mc["moeCacheSlots"], 24)
        self.assertEqual(self.gw.STARTUP_VRAM_SELECTIONS[self.mid]["estimated_after_mib"], 194)

    def test_stale_or_unavailable_sample_falls_through_to_existing_fresh_sample_gate(self):
        self.enable()
        mc = self.gw.model_config(self.mid, False)
        self.gw.st.vram_sample_at = time.time() - 10
        self.assertEqual(self.gw.choose_startup_slots(self.mid, False, mc), "")
        self.assertIn("Fresh NVML", self.gw.preload_check(mc))
        self.assertEqual(self.gw.st.events[-1]["reason"], "telemetry_failed")
        self.gw.st.vram_sample_at = time.time()
        self.gw.st.vram_free_mib = None
        self.assertEqual(self.gw.choose_startup_slots(self.mid, False, mc), "")
        self.assertIn("Fresh NVML", self.gw.preload_check(mc))

    def test_no_slot_levels_uses_safe_supported_fallback(self):
        self.enable()
        self.gw.G.pop("slotGearLevels", None)
        self.gw.st.vram_free_mib = 1000
        mc = self.gw.model_config(self.mid, False)
        self.assertEqual(self.gw.choose_startup_slots(self.mid, False, mc), "")
        self.assertEqual(mc["moeCacheSlots"], 20)

    def test_slot_gears_start_at_selected_level_then_keep_the_existing_ratchet(self):
        self.enable()
        self.gw.G["slotGears"] = True
        self.gw.st.vram_free_mib = 1000
        mc = self.gw.model_config(self.mid, False)
        self.assertEqual(self.gw.choose_startup_slots(self.mid, False, mc), "")
        gear = self.gw.gear_for(self.mid)
        self.assertEqual(gear.current, 20)
        self.assertIsNone(gear.target(0, 100, ready=True))
        self.assertEqual(gear.target(61, 100, ready=True), 16)

    def test_config_validation_and_reserve_selects_new_gateway_path(self):
        self.gw.validate_vram_card_reserve_config({})
        for value in (0, None, False):
            with self.subTest(value=value):
                self.gw.validate_vram_card_reserve_config({"vramCardReserveMiB": value})
        for value in (True, -1, 1.5, "194", 1_000_001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.gw.validate_vram_card_reserve_config({"vramCardReserveMiB": value})
        self.assertTrue(self.gw.legacy_gateway_selected({}, False))
        self.assertTrue(self.gw.legacy_gateway_selected({"vramCardReserveMiB": 0}, False))
        self.assertFalse(self.gw.legacy_gateway_selected({"vramCardReserveMiB": 194}, False))
        self.assertFalse(self.gw.legacy_gateway_selected({"vramCardReserveMiB": "bad"}, False))


if __name__ == "__main__":
    unittest.main()
