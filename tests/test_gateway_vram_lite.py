"""VRAMCOR2: the lite layout. When the short layout cannot keep the card's VRAM reserve (the desktop took VRAM), the gateway starts the
lite layout at once (more experts on the CPU, a small expert cache) instead of answering 503 for 40-50 s, and goes back to the short
layout only after the reserve plus a margin has been free for a while. Isolated fake server, no GPU."""
import json
import math
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_gateway as fixture

MiB = 1


class GatewayVramLite(unittest.TestCase):
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
        self.patch_attr("LITE_STATE", {"since": -math.inf, "relief_since": None, "relief_checked": -math.inf})
        self.gw.G["slotGearMiBPerSlot"] = 13
        self.gw.G["slotGearLevels"] = [20, 16]
        self.gw.G["vramTelemetryStaleSeconds"] = 5
        self.gw.G["vramValve"] = False
        self.gw.G["vramLiteReturnSeconds"] = 120
        self.gw.G["vramLiteReturnMarginMiB"] = 600
        mc = self.gw.M["models"][self.mid]
        mc.pop("longContextProfile", None)
        mc.pop("longContextSwitchTokens", None)
        mc.update({"moeCacheSlots": 20, "vramFreeMinGBToLoad": 12.1, "nCpuMoe": 12,
                   "liteProfile": {"nCpuMoe": 14, "moeCacheSlots": 2, "vramFreeMinGBToLoad": 11.5}})
        self.gw.G_RAW["vramCardReserveMiB"] = 1536
        self.gw.G["vramCardReserveMiB"] = 1536
        self.gw.st.proc = None
        self.gw.st.status = "UNLOADED"
        self.gw.st.model_id = ""
        self.gw.st.layout = "short"
        self.gw.st.vision = False
        self.gw.st.vram_sample_at = time.time()
        self.gw.st.monitor_error = ""
        self.gw.st.vram_free_mib = 14000

    def events(self):
        return [e["reason"] for e in self.gw.st.events]

    def run_loaded(self, layout):
        proc = Mock()
        proc.poll.return_value = None
        self.gw.st.proc, self.gw.st.status, self.gw.st.model_id, self.gw.st.layout = proc, "READY", self.mid, layout

    BODY = json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode()

    # --- the profile -------------------------------------------------------------------------------------------------------
    def test_lite_profile_changes_only_what_frees_vram_and_keeps_the_window(self):
        short = self.gw.model_config(self.mid, False, "short")
        lite = self.gw.model_config(self.mid, False, "lite")
        self.assertEqual((lite["nCpuMoe"], lite["moeCacheSlots"], lite["vramFreeMinGBToLoad"]), (14, 2, 11.5))
        self.assertEqual(lite["ctxSize"], short["ctxSize"])
        self.assertEqual(short["nCpuMoe"], 12)
        self.assertEqual(self.gw.model_config(self.mid, True, "lite").get("nCpuMoe"), self.gw.model_config(self.mid, True)["nCpuMoe"])

    def test_without_a_profile_the_lite_layout_is_the_short_one(self):
        self.gw.M["models"][self.mid].pop("liteProfile")
        self.assertEqual(self.gw.model_config(self.mid, False, "lite"), self.gw.model_config(self.mid, False, "short"))
        self.assertFalse(self.gw.lite_enabled(self.mid))

    def test_lite_startup_slots_come_from_the_lite_profile_and_baseline(self):
        self.gw.st.vram_free_mib = 13700   # short: 13700-12100-208 = 1392 < 1536; lite: 13700-11500-26 = 2174
        mc = self.gw.model_config(self.mid, False, "lite")
        self.assertEqual(self.gw.choose_startup_slots(self.mid, False, mc, "lite"), "")
        self.assertEqual(mc["moeCacheSlots"], 2)

    # --- the fallback at admission ----------------------------------------------------------------------------------------
    def test_a_deferral_of_the_short_layout_becomes_a_lite_start_at_once(self):
        self.gw.st.vram_free_mib = 13700
        seen = []

        def fake_start(mid, vision, **kwargs):
            seen.append(kwargs.get("layout"))
            return ""
        with patch.object(self.gw, "start_llama", side_effect=fake_start):
            self.assertEqual(self.gw.acquire_model(self.mid, False, "short"), "")
        self.assertEqual(seen, ["lite"])
        self.assertIn("vram_lite_selected", self.events())
        self.assertGreater(self.gw.LITE_STATE["since"], 0)

    def test_plenty_of_vram_keeps_the_short_layout(self):
        self.gw.st.vram_free_mib = 15000
        seen = []
        with patch.object(self.gw, "start_llama", side_effect=lambda mid, vision, **k: seen.append(k.get("layout")) or ""):
            self.assertEqual(self.gw.acquire_model(self.mid, False, "short"), "")
        self.assertEqual(seen, ["short"])
        self.assertNotIn("vram_lite_selected", self.events())

    def test_the_original_deferral_stands_when_there_is_no_profile_or_lite_does_not_fit_or_vision(self):
        cases = {"no profile": (13700, False, True), "lite does not fit": (11000, False, False)}
        for name, (free, vision, drop_profile) in cases.items():
            with self.subTest(name):
                self.gw.st.vram_free_mib = free
                saved = self.gw.M["models"][self.mid].get("liteProfile")
                if drop_profile:
                    self.gw.M["models"][self.mid].pop("liteProfile")
                try:
                    with patch.object(self.gw, "start_llama") as start:
                        error = self.gw.acquire_model(self.mid, vision, "short")
                finally:
                    if saved is not None:
                        self.gw.M["models"][self.mid]["liteProfile"] = saved
                start.assert_not_called()
                self.assertIn("card VRAM reserve", error)

    def test_the_vision_mode_and_other_errors_never_fall_back(self):
        mc = self.gw.model_config(self.mid, True, "short")
        deferral = "Model load deferred: card VRAM reserve 1536 MiB cannot be kept; free 1 MiB"
        self.assertEqual(self.gw.lite_fallback(self.mid, True, "short", mc, deferral), ("short", mc, deferral))
        mc = self.gw.model_config(self.mid, False, "short")
        other = "Model load deferred: system memory is critical (RAM valve)"
        self.assertEqual(self.gw.lite_fallback(self.mid, False, "short", mc, other), ("short", mc, other))
        self.assertEqual(self.gw.lite_fallback(self.mid, False, "long", mc, deferral), ("long", mc, deferral))
        self.assertEqual(self.gw.lite_fallback(self.mid, False, "short", mc, ""), ("short", mc, ""))

    def test_a_restart_after_a_server_stop_loads_the_lite_layout_instead_of_503(self):
        self.gw.st.vram_free_mib = 13700
        seen = []

        def fake_load(mid, vision, mc, *args, **kwargs):
            seen.append((self.gw.st.layout, mc["nCpuMoe"], mc["moeCacheSlots"]))
            return ""
        with patch.object(self.gw, "load_server_locked", side_effect=fake_load), \
                patch.object(self.gw, "preload_check", return_value=""), \
                patch.object(self.gw, "wait_for_vram_settle", return_value=None):
            self.assertEqual(self.gw.start_llama(self.mid, False, layout="short"), "")
        self.assertEqual(seen, [("lite", 14, 2)])
        self.assertEqual(self.gw.st.layout, "lite")

    # --- the running server: pressure and return --------------------------------------------------------------------------
    def test_pressure_on_the_running_short_layout_switches_to_lite_when_lite_gives_enough_back(self):
        self.run_loaded("short")
        now = time.time()
        for free, expected in ((1000, "lite"), (400, "short"), (2000, "short")):
            with self.subTest(free=free):
                self.gw.st.vram_free_mib = free   # delta 834: 1000+834 >= 1536; 400+834 < 1536; 2000 is above the reserve
                self.assertEqual(self.gw.choose_request_layout(self.mid, False, self.BODY, 0, now), expected)

    def test_the_lite_layout_stays_until_the_return_is_held_for_the_hysteresis_time(self):
        self.run_loaded("lite")
        t0 = time.time()
        self.gw.LITE_STATE.update(since=t0 - 200, relief_since=None, relief_checked=-math.inf)
        self.gw.st.vram_free_mib = 3000          # 3000-834 = 2166 >= 1536+600: the short layout would keep the reserve
        self.assertEqual(self.gw.choose_request_layout(self.mid, False, self.BODY, 0, t0), "lite")          # the first sight
        self.assertEqual(self.gw.choose_request_layout(self.mid, False, self.BODY, 0, t0 + 60), "lite")     # not long enough
        self.gw.st.vram_free_mib = 2000          # the desktop took VRAM again: the clock restarts
        self.assertEqual(self.gw.choose_request_layout(self.mid, False, self.BODY, 0, t0 + 90), "lite")
        self.gw.st.vram_free_mib = 3000
        self.assertEqual(self.gw.choose_request_layout(self.mid, False, self.BODY, 0, t0 + 100), "lite")
        self.assertEqual(self.gw.choose_request_layout(self.mid, False, self.BODY, 0, t0 + 230), "short")

    def test_the_return_waits_for_the_minimum_lite_time_after_the_choice(self):
        self.run_loaded("lite")
        t0 = time.time()
        self.gw.LITE_STATE.update(since=t0 - 30, relief_since=t0 - 500, relief_checked=t0 - 10)
        self.gw.st.vram_free_mib = 3000
        self.assertEqual(self.gw.choose_request_layout(self.mid, False, self.BODY, 0, t0), "lite")

    def test_a_stale_vram_sample_never_moves_the_layout(self):
        self.run_loaded("short")
        self.gw.st.vram_free_mib = 1000
        self.gw.st.vram_sample_at = time.time() - 60
        self.assertEqual(self.gw.choose_request_layout(self.mid, False, self.BODY, 0, time.time()), "short")

    def test_models_without_the_profile_are_never_moved(self):
        self.gw.M["models"][self.mid].pop("liteProfile")
        self.run_loaded("short")
        self.gw.st.vram_free_mib = 1000
        self.assertEqual(self.gw.choose_request_layout(self.mid, False, self.BODY, 0, time.time()), "short")

    # --- the whole desktop-pressure story -----------------------------------------------------------------------------------
    def test_scenario_guard_stop_then_lite_without_503_then_back_to_short(self):
        started = []

        def fake_load(mid, vision, mc, *args, **kwargs):
            started.append((self.gw.st.layout, mc["nCpuMoe"], mc["moeCacheSlots"]))
            proc = Mock()
            proc.poll.return_value = None
            self.gw.st.proc, self.gw.st.status = proc, "READY"
            return ""
        patches = (patch.object(self.gw, "load_server_locked", side_effect=fake_load), patch.object(self.gw, "preload_check", return_value=""),
                   patch.object(self.gw, "wait_for_vram_settle", return_value=None),
                   patch.object(self.gw, "running_vram_mib", return_value=11526))
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        # 1. the server was stopped (the stand's sentinel), the desktop holds ~2.4 GiB: free after the stop is 13700 MiB
        self.gw.st.vram_free_mib = 13700
        layout = self.gw.choose_request_layout(self.mid, False, self.BODY, 0, time.time())
        self.assertEqual(layout, "short")                              # nothing runs yet: the choice is made at admission
        self.assertEqual(self.gw.acquire_model(self.mid, False, layout), "")   # no 503: the lite layout is started at once
        self.assertEqual(started, [("lite", 14, 2)])
        self.assertIn("vram_lite_selected", self.events())
        self.gw.st.busy = 0
        # 2. the next requests keep the lite layout; the desktop still holds the VRAM (free with lite loaded: 2000)
        base = self.gw.LITE_STATE["since"]
        self.gw.st.vram_free_mib = 2000
        for dt in (10, 70):
            self.assertEqual(self.gw.choose_request_layout(self.mid, False, self.BODY, 0, base + dt), "lite")
        # 3. the desktop frees the VRAM: the return needs the margin held for the hysteresis time
        self.gw.st.vram_free_mib = 3000
        self.assertEqual(self.gw.choose_request_layout(self.mid, False, self.BODY, 0, base + 130), "lite")   # the first sight
        back = self.gw.choose_request_layout(self.mid, False, self.BODY, 0, base + 260)
        self.assertEqual(back, "short")
        self.assertEqual(self.gw.acquire_model(self.mid, False, back), "")
        self.assertEqual([item[0] for item in started], ["lite", "short"])
        self.assertEqual(started[1][1:], (12, 20))

    # --- config and status ------------------------------------------------------------------------------------------------
    def test_config_validation(self):
        good = {self.mid: {"nCpuMoe": 12, "liteProfile": {"nCpuMoe": 14, "moeCacheSlots": 2, "vramFreeMinGBToLoad": 11.5}}}
        self.gw.validate_lite_profile_config(good, {"vramCardReserveMiB": 1536})
        self.gw.validate_lite_profile_config({self.mid: {}}, {})      # no profile: nothing required
        bad = {
            "unknown key": ({self.mid: {"liteProfile": {"ctxSize": 1}}}, {"vramCardReserveMiB": 1536}),
            "empty": ({self.mid: {"liteProfile": {}}}, {"vramCardReserveMiB": 1536}),
            "no reserve": (good, {}),
            "fewer experts on the CPU": ({self.mid: {"nCpuMoe": 16, "liteProfile": {"nCpuMoe": 14}}}, {"vramCardReserveMiB": 1536}),
            "negative slots": ({self.mid: {"liteProfile": {"moeCacheSlots": -1}}}, {"vramCardReserveMiB": 1536}),
            "bad baseline": ({self.mid: {"liteProfile": {"vramFreeMinGBToLoad": "x"}}}, {"vramCardReserveMiB": 1536}),
            "bad return time": (good, {"vramCardReserveMiB": 1536, "vramLiteReturnSeconds": 0}),
            "bad margin": (good, {"vramCardReserveMiB": 1536, "vramLiteReturnMarginMiB": -1}),
        }
        for name, (models, config) in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.gw.validate_lite_profile_config(models, config)

    CANDIDATE = Path.home() / "Desktop/AI/reports/layout1/model-launch.candidate.json"

    @unittest.skipUnless(CANDIDATE.exists(), "the candidate config is not on this machine")
    def test_the_candidate_passes_the_gateway_validators_and_keeps_mtp_in_every_layout(self):
        models = json.loads(self.CANDIDATE.read_text(encoding="utf-8"))["models"]
        self.gw.validate_long_context_config(models)
        self.gw.validate_lite_profile_config(models, {"vramCardReserveMiB": 1536})
        # the exact ncmoe values change with every layout re-measure (the lite profile only has to move more experts to the CPU: below)
        self.assertIn("tiel-coder", models)
        for name in models:                                                # every model of the candidate has a lite layout that moves more experts to the CPU
            self.assertGreater(models[name]["liteProfile"]["nCpuMoe"], models[name]["nCpuMoe"], name)
            # VRAMCOR4: and a lite twin of the long layout
            self.assertGreater(models[name]["longLiteProfile"]["nCpuMoe"], models[name]["longContextProfile"]["nCpuMoe"], name)
        mtp_models = [name for name, model in models.items() if model.get("specType") == "draft-mtp"]
        self.assertIn("tiel-coder", mtp_models)
        for name in mtp_models:                                            # whatever the candidate lists (Occamy is gone; Sage joined)
            profile = models[name]["longContextProfile"]
            # MTP always (2026-10-20), with the measured exception of the long profile (LAYOUT1d/1f/1g, SPEEDBACK1: with experts on the CPU
            # the draft context costs more VRAM than it gives back, MTP variants of B did not fit at ncmoe 16-18)
            self.assertEqual(profile["specType"], "none", name)
            self.assertEqual(models[name]["specType"], "draft-mtp", name)

    def test_status_publishes_the_lite_state(self):
        status = self.gw.status_snapshot()
        self.assertTrue(status["lite"]["available"] in (True, False))
        self.run_loaded("lite")
        self.gw.LITE_STATE["since"] = time.time()
        status = self.gw.status_snapshot()
        self.assertTrue(status["lite"]["available"])
        self.assertEqual(status["layout"], "lite")


if __name__ == "__main__":
    unittest.main()
