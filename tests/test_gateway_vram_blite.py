"""VRAMCOR4: the long layout has a lite twin (longlite = longContextProfile + longLiteProfile): a desktop that holds VRAM no longer
turns the A->B switch into a 503 or a killed server. Same rules as the short lite layout (VRAMCOR2): admission fallback, pressure on
the running layout, return only after the hysteresis. Isolated fake server, no GPU."""
import json
import time
import unittest
from unittest.mock import patch

import test_gateway_vram_lite as lite


class GatewayVramBLite(lite.GatewayVramLite):
    def setUp(self):
        super().setUp()
        mc = self.gw.M["models"][self.mid]
        mc.update({"longContextSwitchTokens": 28000, "longContextHoldSeconds": 300,
                   "longContextProfile": {"ctxSize": 200000, "nCpuMoe": 20, "ubatchSize": 512, "moeCacheSlots": 2,
                                          "vramFreeMinGBToLoad": 12.0},
                   "longLiteProfile": {"nCpuMoe": 22, "moeCacheSlots": 2, "vramFreeMinGBToLoad": 11.45}})
        self.gw.LAYOUT_STATE.update(switched_at=-float("inf"), last_len=0)
        self.gw.st.context_mode = "auto"

    # the inherited short-lite tests run again with a long profile present: nothing of VRAMCOR2 may change
    def test_the_vision_mode_and_other_errors_never_fall_back(self):
        # VRAMCOR2's test, adapted: with a longLiteProfile the long layout may fall back, vision and foreign reasons still may not
        deferral = "Model load deferred: card VRAM reserve 1536 MiB cannot be kept; free 1 MiB"
        mc = self.gw.model_config(self.mid, True, "long")
        self.assertEqual(self.gw.lite_fallback(self.mid, True, "long", mc, deferral), ("long", mc, deferral))
        mc = self.gw.model_config(self.mid, False, "long")
        other = "Model load deferred: system memory is critical (RAM valve)"
        self.assertEqual(self.gw.lite_fallback(self.mid, False, "long", mc, other), ("long", mc, other))
        self.assertEqual(self.gw.lite_fallback(self.mid, False, "longlite", mc, deferral), ("longlite", mc, deferral))

    # --- the profile ------------------------------------------------------------------------------------------------------
    def test_longlite_keeps_the_long_window_and_frees_vram(self):
        long = self.gw.model_config(self.mid, False, "long")
        longlite = self.gw.model_config(self.mid, False, "longlite")
        self.assertEqual(longlite["ctxSize"], 200000)
        self.assertEqual((longlite["nCpuMoe"], longlite["moeCacheSlots"], longlite["vramFreeMinGBToLoad"], longlite["ubatchSize"]),
                         (22, 2, 11.45, 512))
        self.assertEqual(long["nCpuMoe"], 20)

    def test_without_a_long_lite_profile_there_is_no_long_lite_layout(self):
        self.gw.M["models"][self.mid].pop("longLiteProfile")
        self.assertFalse(self.gw.lite_enabled(self.mid, False, "long"))
        self.assertEqual(self.gw.model_config(self.mid, False, "longlite")["nCpuMoe"], 20)

    # --- admission --------------------------------------------------------------------------------------------------------
    def test_a_deferral_of_the_long_layout_becomes_a_longlite_start_at_once(self):
        self.gw.st.vram_free_mib = 13400   # long: 13400-12000-26 = 1374 < 1536; longlite: 13400-11450-26 = 1924
        seen = []
        with patch.object(self.gw, "start_llama", side_effect=lambda mid, vision, **k: seen.append(k.get("layout")) or ""):
            self.assertEqual(self.gw.acquire_model(self.mid, False, "long"), "")
        self.assertEqual(seen, ["longlite"])
        self.assertIn("vram_lite_selected", self.events())

    def test_plenty_of_vram_keeps_the_long_layout(self):
        self.gw.st.vram_free_mib = 15000
        seen = []
        with patch.object(self.gw, "start_llama", side_effect=lambda mid, vision, **k: seen.append(k.get("layout")) or ""):
            self.assertEqual(self.gw.acquire_model(self.mid, False, "long"), "")
        self.assertEqual(seen, ["long"])

    def test_the_deferral_stands_when_even_longlite_does_not_fit(self):
        self.gw.st.vram_free_mib = 12500
        with patch.object(self.gw, "start_llama") as start:
            error = self.gw.acquire_model(self.mid, False, "long")
        start.assert_not_called()
        self.assertIn("card VRAM reserve", error)

    # --- the running layout under pressure --------------------------------------------------------------------------------
    def choose(self, now=None, **kw):
        with patch.object(self.gw, "pure_request_layout", return_value=kw.get("size", "long")):
            return self.gw.choose_request_layout(self.mid, False, self.BODY, 100, now or time.time())

    def test_pressure_on_the_running_long_layout_moves_to_longlite(self):
        self.run_loaded("long")
        self.gw.st.vram_free_mib = 1400   # < reserve; longlite gives back (12.0-11.45)*1000 = 550 -> 1950
        self.assertEqual(self.choose(), "longlite")
        self.assertGreater(self.gw.LITE_STATE["since"], 0)

    def test_no_pressure_keeps_the_long_layout_and_hopeless_pressure_does_too(self):
        self.run_loaded("long")
        for free in (3000, 600):          # plenty / even longlite would not keep the reserve
            with self.subTest(free=free):
                self.gw.st.vram_free_mib = free
                self.assertEqual(self.choose(), "long")

    def test_longlite_returns_to_long_only_after_the_hysteresis(self):
        self.run_loaded("longlite")
        now = time.time()
        self.gw.LITE_STATE.update(since=now - 1000, relief_since=None, relief_checked=-float("inf"))
        self.gw.st.vram_free_mib = 1536 + 550 + 600 + 50     # the long layout would keep the reserve plus the margin
        self.assertEqual(self.choose(now), "longlite")          # the first check only starts the count
        self.assertEqual(self.choose(now + 60), "longlite")
        self.assertEqual(self.choose(now + 125), "long")
        self.gw.st.vram_free_mib = 1536 + 550 + 100             # pressure again: the count restarts
        self.assertEqual(self.choose(now + 130), "longlite")

    def test_a_small_request_while_in_longlite_stays_in_the_long_family_during_the_hold(self):
        self.run_loaded("longlite")
        self.gw.LAYOUT_STATE.update(switched_at=time.time() - 10, last_len=0)
        self.gw.LITE_STATE.update(since=time.time(), relief_since=None, relief_checked=-float("inf"))
        self.gw.st.vram_free_mib = 1400
        self.assertEqual(self.choose(size="short"), "longlite")

    def test_extended_context_also_gets_the_lite_twin(self):
        self.run_loaded("long")
        self.gw.st.context_mode = "extended"
        self.gw.st.vram_free_mib = 1400
        self.assertEqual(self.gw.choose_request_layout(self.mid, False, self.BODY, 100, time.time()), "longlite")

    # --- the config -------------------------------------------------------------------------------------------------------
    def test_validation_of_the_long_lite_profile(self):
        good = {self.mid: {"nCpuMoe": 12, "longContextProfile": {"ctxSize": 200000, "nCpuMoe": 20},
                           "longLiteProfile": {"nCpuMoe": 22, "moeCacheSlots": 2, "vramFreeMinGBToLoad": 11.4}}}
        self.gw.validate_lite_profile_config(good, {"vramCardReserveMiB": 1536})
        bad = {
            "no long profile": {"nCpuMoe": 12, "longLiteProfile": {"nCpuMoe": 22}},
            "fewer cpu experts than the long profile": {"nCpuMoe": 12, "longContextProfile": {"nCpuMoe": 20}, "longLiteProfile": {"nCpuMoe": 18}},
            "unknown key": {"nCpuMoe": 12, "longContextProfile": {"nCpuMoe": 20}, "longLiteProfile": {"ctxSize": 1000}},
            "empty": {"nCpuMoe": 12, "longContextProfile": {"nCpuMoe": 20}, "longLiteProfile": {}},
        }
        for name, settings in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.gw.validate_lite_profile_config({self.mid: settings}, {"vramCardReserveMiB": 1536})
        with self.assertRaises(ValueError):
            self.gw.validate_lite_profile_config(good, {})

    def test_status_publishes_the_long_lite_layout(self):
        self.run_loaded("longlite")
        status = self.gw.status_snapshot()
        self.assertEqual(status["layout"], "longlite")
        self.assertTrue(status["lite"]["available"])


if __name__ == "__main__":
    unittest.main()
