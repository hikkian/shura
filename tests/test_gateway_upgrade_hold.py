"""GWSTALL1: a request is never held for a layout upgrade the card cannot admit while the same model already runs in a layout
that fits it. SAGE1 (2026-10-26, Occamy): lite had just answered, the governor's relief asked for short 5 s after leaving MIN,
short was 248 MiB short of its admission, and every request got a 503 while lite stood ready. Mocked server, no GPU."""
import json
import math
import time
import unittest
from unittest.mock import Mock, patch

import test_gateway as fixture


class GatewayUpgradeHold(unittest.TestCase):
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
        self.patch_attr("UPGRADE_HOLD", {})
        self.patch_attr("GOV", {**self.gw.GOV, "budget": None, "values": None, "state": None, "too_big": {}})
        self.patch_attr("LITE_STATE", {"since": -math.inf, "relief_since": None, "relief_checked": -math.inf})
        self.gw.G["vramValve"] = False
        self.gw.G["vramPolicy"] = "governor"
        self.gw.G["vramReloadWaitSeconds"] = 0.3
        self.gw.G_RAW["vramCardReserveMiB"] = 1536
        self.gw.G["vramCardReserveMiB"] = 1536
        mc = self.gw.M["models"][self.mid]
        mc.update({"ctxSize": 32768, "moeCacheSlots": 12, "vramFreeMinGBToLoad": 0, "nCpuMoe": 11,
                   "liteProfile": {"nCpuMoe": 16, "moeCacheSlots": 2, "vramFreeMinGBToLoad": 0},
                   "longContextProfile": {"ctxSize": 200000, "nCpuMoe": 15, "moeCacheSlots": 2, "vramFreeMinGBToLoad": 0},
                   "longContextSwitchTokens": 28000})
        self.gw.st.vram_sample_at = time.time()
        self.gw.st.monitor_error = ""
        self.gw.st.vram_free_mib = 2000
        self.gw.st.busy = 0
        self.gw.st.vision = False
        self.patch_attr("running_vram_mib", Mock(return_value=12344.0))

    def events(self):
        return [e["reason"] for e in self.gw.st.events]

    def run_loaded(self, layout):
        self.gw.st.proc, self.gw.st.status, self.gw.st.model_id, self.gw.st.layout = self.proc, "READY", self.mid, layout

    def nothing_loaded(self):
        self.gw.st.proc, self.gw.st.status, self.gw.st.model_id, self.gw.st.layout = None, "UNLOADED", "", "short"

    GOV_SHORT = "Not enough free VRAM for occamy/short under the governor (248 MiB short)"
    BODY = json.dumps({"messages": [{"role": "user", "content": "hello " * 1200}]}).encode()

    # --- refusals carry a code ---------------------------------------------------------------------------------------------
    def test_preload_check_refusals_carry_codes_and_keep_their_text(self):
        mc = self.gw.model_config(self.mid, False, "short")
        self.gw.st.vram_sample_at = time.time() - 100
        stale = self.gw.preload_check(mc)
        self.assertEqual(self.gw.refusal_code(stale), "telemetry")
        self.assertIn("fresh VRAM sample", stale)
        self.gw.st.vram_sample_at, self.gw.st.monitor_error = time.time(), ""
        mc["vramFreeMinGBToLoad"] = 30
        legacy = self.gw.preload_check(mc)
        self.assertEqual(self.gw.refusal_code(legacy), "vram")
        self.assertTrue(legacy.startswith("Not enough free VRAM ("), legacy)
        budget = self.gw.governor_budget()
        budget.fixed_learned[(self.mid, "short")] = 12800
        governed = self.gw.preload_check(mc, reclaim=(0.0, 12344 / 1024), admit=(self.mid, "short"))
        self.assertEqual(self.gw.refusal_code(governed), "vram_governor")
        self.assertIn("under the governor (248 MiB short)", governed)
        with patch.object(self.gw, "mem_available_gb", return_value=0.1):
            self.gw.G["ramFreeMinGBToLoad"] = 4
            self.assertEqual(self.gw.refusal_code(self.gw.preload_check(mc)), "ram")
        self.gw.st.override = "OFF"
        self.assertEqual(self.gw.refusal_code(self.gw.light_admission()), "ai_off")
        self.assertEqual(self.gw.refusal_code("a plain string that mentions VRAM"), None)
        self.assertEqual(json.loads(json.dumps({"e": governed}))["e"], str(governed))   # status JSON sees the plain text

    def test_lite_fallback_reacts_to_the_reserve_code_not_to_the_wording(self):
        self.gw.G["vramPolicy"] = "legacy"
        mc = self.gw.model_config(self.mid, False, "short")
        text = "Model load deferred: card VRAM reserve 1536 MiB cannot be kept; free 13700 MiB"
        with patch.object(self.gw, "choose_startup_slots", return_value=""):
            layout, _mc, reason = self.gw.lite_fallback(self.mid, False, "short", mc, self.gw.Refusal(text, "vram_reserve"))
            self.assertEqual((layout, reason), ("lite", ""))
            layout, _mc, reason = self.gw.lite_fallback(self.mid, False, "short", mc, self.gw.Refusal(text, "config"))
            self.assertEqual((layout, reason), ("short", text))

    # --- the loaded layout answers ---------------------------------------------------------------------------------------------
    def test_a_refused_upgrade_is_served_by_the_loaded_lite_layout_and_held(self):
        self.run_loaded("lite")
        refusal = self.gw.Refusal(self.GOV_SHORT, "vram_governor")
        with patch.object(self.gw, "start_llama", return_value=refusal) as start:
            t0 = time.monotonic()
            self.assertEqual(self.gw.acquire_model(self.mid, False, "short"), "")
            self.assertLess(time.monotonic() - t0, 0.5)
            self.assertEqual((self.gw.st.busy, self.gw.st.layout, self.gw.st.status), (1, "lite", "READY"))
            self.assertEqual(start.call_count, 1)
            self.assertEqual(self.events().count("layout_upgrade_held"), 1)
            # within the hold the next request does not even ask: no admission, no log line, straight to the loaded server
            self.assertEqual(self.gw.acquire_model(self.mid, False, "short"), "")
            self.assertEqual((self.gw.st.busy, start.call_count), (2, 1))
            # after the hold the upgrade is asked again on a request boundary; refused again: served again, one more event
            self.gw.UPGRADE_HOLD[(self.mid, "short")]["until"] = time.time() - 1
            self.assertEqual(self.gw.acquire_model(self.mid, False, "short"), "")
            self.assertEqual((self.gw.st.busy, start.call_count), (3, 2))
            self.assertEqual(self.events().count("layout_upgrade_held"), 2)
        with patch.object(self.gw, "start_llama", return_value="") as start:   # admitted after the hold: the switch happens as before
            self.gw.UPGRADE_HOLD[(self.mid, "short")]["until"] = time.time() - 1
            self.assertEqual(self.gw.acquire_model(self.mid, False, "short"), "")
            self.assertEqual(start.call_args.kwargs.get("layout"), "short")

    def test_a_memory_refusal_before_the_start_is_also_served_by_the_loaded_layout(self):
        self.run_loaded("lite")
        with patch.object(self.gw, "preload_check", return_value=self.gw.Refusal("Not enough free RAM to load the model", "ram")), \
                patch.object(self.gw, "start_llama") as start:
            self.assertEqual(self.gw.acquire_model(self.mid, False, "short"), "")
            start.assert_not_called()
            self.assertEqual(self.gw.st.busy, 1)

    def test_the_hold_does_not_wait_for_a_streaming_answer_on_the_loaded_layout(self):
        self.run_loaded("lite")
        self.gw.UPGRADE_HOLD[(self.mid, "short")] = {"until": time.time() + 60, "reason": self.GOV_SHORT}
        self.gw.st.busy = 1
        t0 = time.monotonic()
        self.assertEqual(self.gw.acquire_model(self.mid, False, "short"), "")
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertEqual(self.gw.st.busy, 2)

    def test_a_smaller_window_does_not_answer_a_long_request_the_error_comes_back_at_once(self):
        self.run_loaded("short")
        refusal = self.gw.Refusal("Not enough free VRAM (14.3 GB free, need 20.00 GB)", "vram")
        with patch.object(self.gw, "start_llama", side_effect=lambda *a, **kw: refusal if kw.get("layout") == "long" else ""):
            t0 = time.monotonic()
            error = self.gw.acquire_model(self.mid, False, "long")
        self.assertTrue(error.startswith("Could not switch to the long context: Not enough free VRAM"), error)
        self.assertEqual(self.gw.refusal_code(error), "vram")      # the wrapped text keeps the code (Retry-After, no serving)
        self.assertLess(time.monotonic() - t0, 0.25)
        self.assertEqual((self.gw.st.busy, self.gw.st.layout), (0, "short"))
        self.assertNotIn("layout_upgrade_held", self.events())

    def test_a_loaded_long_layout_answers_a_short_request_that_cannot_switch(self):
        self.run_loaded("long")
        with patch.object(self.gw, "start_llama", return_value=self.gw.Refusal(self.GOV_SHORT, "vram_governor")):
            self.assertEqual(self.gw.acquire_model(self.mid, False, "short"), "")
        self.assertEqual((self.gw.st.busy, self.gw.st.layout), (1, "long"))

    def test_mode_refusals_and_other_models_keep_the_old_answer(self):
        self.run_loaded("lite")
        for refusal in (self.gw.Refusal("AI is manually disabled (ai-off)", "ai_off"), "Gateway is shutting down"):
            with patch.object(self.gw, "start_llama", return_value=refusal):
                self.assertEqual(self.gw.acquire_model(self.mid, False, "short"), refusal)
        self.assertEqual(self.gw.st.busy, 0)
        self.assertEqual(self.gw.UPGRADE_HOLD, {})

    # --- nothing loaded: the bounded wait stays, decided by the code ------------------------------------------------------------
    def test_without_a_loaded_model_a_vram_refusal_waits_the_bounded_time_and_a_mode_refusal_does_not(self):
        self.nothing_loaded()
        with patch.object(self.gw, "preload_check", return_value=self.gw.Refusal("Not enough free VRAM (1.0 GB free, need 12.00 GB)", "vram")):
            t0 = time.monotonic()
            error = self.gw.acquire_model(self.mid, False, "short")
            waited = time.monotonic() - t0
        self.assertIn("Not enough free VRAM", error)
        self.assertGreaterEqual(waited, 0.3)
        self.assertLess(waited, 2)
        with patch.object(self.gw, "preload_check", return_value=self.gw.Refusal("AI is manually disabled (ai-off)", "ai_off")):
            t0 = time.monotonic()
            self.gw.acquire_model(self.mid, False, "short")
            self.assertLess(time.monotonic() - t0, 0.2)
        with patch.object(self.gw, "preload_check", return_value="the text says VRAM but carries no code"):
            t0 = time.monotonic()
            self.gw.acquire_model(self.mid, False, "short")
            self.assertLess(time.monotonic() - t0, 0.2)   # classification is by code, not by the word

    # --- the relief asks the admission ------------------------------------------------------------------------------------------
    def test_governor_relief_returns_to_short_only_when_its_admission_passes(self):
        self.run_loaded("lite")
        self.gw.GOV["state"] = "FULL"
        budget = self.gw.governor_budget()
        budget.fixed_learned[(self.mid, "short")] = 12800          # SAGE1: fixed 12800 + reserve 1536 + hysteresis 256 = 14592
        self.gw.st.vram_free_mib = 2000                              # + 12344 held by lite = 14344: 248 MiB short
        self.assertFalse(self.gw.governor_lite_relief(self.mid, False, time.time(), "short"))
        self.assertEqual(self.events().count("layout_upgrade_held"), 1)
        self.assertIn("248 MiB short", self.gw.UPGRADE_HOLD[(self.mid, "short")]["reason"])
        self.gw.st.vram_free_mib = 2300                              # 14644 would be admitted, but the hold stands: no recomputation
        self.assertFalse(self.gw.governor_lite_relief(self.mid, False, time.time(), "short"))
        self.assertEqual(self.events().count("layout_upgrade_held"), 1)
        self.gw.UPGRADE_HOLD[(self.mid, "short")]["until"] = time.time() - 1
        self.assertTrue(self.gw.governor_lite_relief(self.mid, False, time.time(), "short"))
        budget.fixed_learned.pop((self.mid, "short"))                # nothing learned yet: nothing to refuse on
        self.gw.st.vram_free_mib = 2000
        self.assertTrue(self.gw.governor_lite_relief(self.mid, False, time.time(), "short"))
        self.gw.GOV["state"] = "MIN"                                 # the governor's own rule still gates the return
        self.assertFalse(self.gw.governor_lite_relief(self.mid, False, time.time(), "short"))

    def test_sage1_replay_the_request_after_lite_answered_stays_on_lite_and_is_admitted_at_once(self):
        self.run_loaded("lite")
        self.gw.GOV["state"] = "FULL"
        self.gw.governor_budget().fixed_learned[(self.mid, "short")] = 12800
        self.gw.st.vram_free_mib = 2000
        layout = self.gw.choose_request_layout(self.mid, False, self.BODY, 0, time.time())
        self.assertEqual(layout, "lite")
        with patch.object(self.gw, "start_llama") as start:
            t0 = time.monotonic()
            self.assertEqual(self.gw.acquire_model(self.mid, False, layout), "")
            self.assertLess(time.monotonic() - t0, 0.5)
            start.assert_not_called()
        self.assertEqual(self.gw.st.busy, 1)
        self.assertIn("layout_upgrade_held", self.events())
        self.gw.st.busy = 0
        self.gw.st.vram_free_mib = 2300                              # the desktop gave 300 MiB back: within the hold lite still answers
        self.assertEqual(self.gw.choose_request_layout(self.mid, False, self.BODY, 0, time.time()), "lite")
        self.gw.UPGRADE_HOLD[(self.mid, "short")]["until"] = time.time() - 1   # the hold is over: the upgrade is asked again
        self.assertEqual(self.gw.choose_request_layout(self.mid, False, self.BODY, 0, time.time()), "short")


    # --- GWSTALL2: one log line per distinct refusal, a count instead of a line per retry ----------------------------------------
    def test_load_blocked_repeats_are_counted_not_written(self):
        self.patch_attr("LOAD_BLOCKED", {})
        for mib in (223, 196, 221, 213, 200, 192):
            self.gw.log_load_blocked(f"Not enough free VRAM for occamy/short under the governor ({mib} MiB short)")
        self.assertEqual(self.gw.log.call_count, 1)
        self.gw.log_load_blocked("Not enough free VRAM for occamy/lite under the governor (10 MiB short)")   # a different reason
        self.assertEqual(self.gw.log.call_count, 2)
        with patch.object(self.gw.time, "time", return_value=time.time() + self.gw.LOAD_BLOCKED_REPEAT_SECONDS + 1):
            self.gw.log_load_blocked("Not enough free VRAM for occamy/short under the governor (205 MiB short)")
        self.assertEqual(self.gw.log.call_count, 3)
        self.assertIn("refused 7 times", self.gw.log.call_args.args[0])
        self.gw.load_blocked_cleared()
        self.assertEqual(self.gw.log.call_count, 4)
        self.assertIn("Load admitted after", self.gw.log.call_args.args[0])
        self.assertEqual(self.gw.LOAD_BLOCKED, {})
        self.gw.load_blocked_cleared()                      # nothing counted: nothing written
        self.assertEqual(self.gw.log.call_count, 4)


if __name__ == "__main__":
    unittest.main()

