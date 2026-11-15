"""Gateway prefill yield: resize to 0 for a large new prefill, restore after, degrade safely. Fake server, no GPU."""
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
import test_gateway as fixture


def resize_ok(slots):
    return 200, {"success": True, "old_slots": 0, "new_slots": slots}


class GatewayPrefillYield(unittest.TestCase):
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
        self.patch_attr("M", {**self.gw.M, "models": {mid: dict(mc) for mid, mc in self.gw.M["models"].items()}})
        self.patch_attr("G_RAW", {**self.gw.G_RAW, "prefillCacheYield": True, "vramValve": False, "slotGears": False})
        self.patch_attr("RUNTIME_SLOTS", {self.mid: 30})
        self.patch_attr("PREFILL_YIELD", self.gw.prefill_yield.YieldState())
        self.gw.G.update(prefillCacheYield=True, prefillCacheYieldMinTokens=4000, vramValve=False, slotGears=False,
                         slotSaveCheckpoints=1)
        self.gw.M["models"][self.mid]["moeCacheSlots"] = 30
        self.gw.M["models"][self.mid]["moeCacheProfile"] = "/tmp/fake-profile.csv"
        self.gw.st.status = "READY"
        self.gw.st.model_id = self.mid
        self.gw.st.vision = False
        self.gw.st.proc = type("P", (), {"poll": lambda self: None, "pid": 4321})()
        self.gw.st.busy = 1  # the request being served is already counted busy, as in handle_generation
        self.gw.st.monitor_error = ""

    def resize_calls(self, mock):
        return [c.args[2]["slots"] for c in mock.call_args_list if c.args[1] == "/moe-cache/slots"]

    def big_body(self):
        return b"x" * 40000  # 10000 estimated tokens, above the 4000-token threshold

    def test_large_new_prefill_shrinks_to_zero_then_restores(self):
        def fake(method, path, body, timeout=5):
            return resize_ok(body["slots"])
        with patch.object(self.gw, "llama_call", side_effect=fake) as call:
            back = self.gw.prefill_yield_begin(self.mid, False, self.big_body())
            self.assertEqual(back, 30)
            self.assertEqual(self.gw.RUNTIME_SLOTS[self.mid], 0)
            self.gw.prefill_yield_end(back, self.mid)
        self.assertEqual(self.resize_calls(call), [0, 30])
        self.assertEqual(self.gw.RUNTIME_SLOTS[self.mid], 30)
        counters = self.gw.PREFILL_YIELD.counters
        self.assertEqual((counters["yielded"], counters["restored"]), (1, 1))
        self.assertIsNone(self.gw.PREFILL_YIELD.pending)

    def test_small_request_does_no_resize(self):
        with patch.object(self.gw, "llama_call") as call:
            self.assertIsNone(self.gw.prefill_yield_begin(self.mid, False, b"x" * 1000))
            self.gw.prefill_yield_end(None, self.mid)
        call.assert_not_called()
        self.assertEqual(self.gw.RUNTIME_SLOTS[self.mid], 30)

    def test_switch_off_keeps_behaviour_and_state(self):
        self.gw.G["prefillCacheYield"] = False
        with patch.object(self.gw, "llama_call") as call:
            self.assertIsNone(self.gw.prefill_yield_begin(self.mid, False, self.big_body()))
        call.assert_not_called()
        self.assertIsNone(self.gw.PREFILL_YIELD.last_len)
        self.assertEqual(self.gw.PREFILL_YIELD.counters["skipped"], 0)

    def test_resize_failure_lets_the_request_through_unchanged(self):
        with patch.object(self.gw, "llama_call", return_value=(500, {"success": False, "message": "allocation failed"})) as call:
            self.assertIsNone(self.gw.prefill_yield_begin(self.mid, False, self.big_body()))
            self.gw.prefill_yield_end(None, self.mid)
        self.assertEqual(self.resize_calls(call), [0])
        self.assertEqual(self.gw.RUNTIME_SLOTS[self.mid], 30)
        self.assertEqual(self.gw.PREFILL_YIELD.counters["resize_failed"], 1)
        self.assertIsNone(self.gw.PREFILL_YIELD.pending)

    def test_resize_exception_also_lets_the_request_through(self):
        with patch.object(self.gw, "llama_call", side_effect=OSError("connection refused")):
            self.assertIsNone(self.gw.prefill_yield_begin(self.mid, False, self.big_body()))
        self.assertEqual(self.gw.RUNTIME_SLOTS[self.mid], 30)

    def test_failed_restore_is_kept_and_retried_when_idle(self):
        script = [resize_ok(0), (500, {"success": False, "message": "busy"}), resize_ok(30)]
        with patch.object(self.gw, "llama_call", side_effect=lambda *a, **k: script.pop(0)) as call:
            back = self.gw.prefill_yield_begin(self.mid, False, self.big_body())
            self.gw.prefill_yield_end(back, self.mid)
            self.assertEqual(self.gw.PREFILL_YIELD.pending, {"model": self.mid, "slots": 30})
            self.gw.st.busy = 0
            self.gw.prefill_yield_tick()
        self.assertEqual(self.resize_calls(call), [0, 30, 30])
        self.assertIsNone(self.gw.PREFILL_YIELD.pending)
        self.assertEqual(self.gw.RUNTIME_SLOTS[self.mid], 30)
        self.assertEqual(self.gw.PREFILL_YIELD.counters["restore_failed"], 1)
        self.assertEqual(self.gw.PREFILL_YIELD.counters["restored"], 1)

    def test_pending_restore_blocks_the_valve_from_moving_the_cache(self):
        self.gw.G["vramValve"] = True
        self.gw.PREFILL_YIELD.pending = {"model": self.mid, "slots": 30}
        with patch.object(self.gw, "llama_call") as call:
            self.gw.vram_valve_tick(time.monotonic())
        call.assert_not_called()

    def test_server_replaced_before_restore_voids_the_debt(self):
        self.gw.PREFILL_YIELD.pending = {"model": self.mid, "slots": 30}
        self.gw.st.model_id = "some-other-model"
        with patch.object(self.gw, "llama_call") as call:
            self.gw.prefill_yield_tick()
        call.assert_not_called()
        self.assertIsNone(self.gw.PREFILL_YIELD.pending)

    def test_other_requests_in_flight_mean_no_resize(self):
        self.gw.st.busy = 2
        with patch.object(self.gw, "llama_call") as call:
            self.assertIsNone(self.gw.prefill_yield_begin(self.mid, False, self.big_body()))
        call.assert_not_called()
        self.assertEqual(self.gw.PREFILL_YIELD.counters["skipped"], 1)

    def test_restore_runs_even_when_the_answer_fails_midway(self):
        # the handler calls prefill_yield_end in its finally, so the restore does not depend on the answer succeeding
        with patch.object(self.gw, "llama_call", side_effect=lambda m, p, b, timeout=5: resize_ok(b["slots"])) as call:
            back = self.gw.prefill_yield_begin(self.mid, False, self.big_body())
            try:
                raise ConnectionResetError("client went away")
            except ConnectionResetError:
                pass
            finally:
                self.gw.prefill_yield_end(back, self.mid)
        self.assertEqual(self.resize_calls(call), [0, 30])
        self.assertEqual(self.gw.RUNTIME_SLOTS[self.mid], 30)

    def test_config_rejects_prefill_yield_with_slot_gears(self):
        with self.assertRaises(ValueError):
            self.gw.validate_prefill_yield_config({"prefillCacheYield": True, "slotGears": True})
        self.gw.validate_prefill_yield_config({"prefillCacheYield": False, "slotGears": True})

    def test_live_resize_flag_follows_the_switch(self):
        self.gw.G["prefillCacheYield"] = True
        on = self.gw.build_args(self.gw.model_config(self.mid, False), False)
        self.assertIn("--moe-cache-live-resize", on)
        self.gw.G["prefillCacheYield"] = False
        off = self.gw.build_args(self.gw.model_config(self.mid, False), False)
        self.assertNotIn("--moe-cache-live-resize", off)

    def test_status_reports_counters_and_switch(self):
        snap = self.gw.status_snapshot()["prefill_yield"]
        self.assertTrue(snap["enabled"])
        self.assertEqual(set(snap["counters"]), set(self.gw.prefill_yield.YieldState.COUNTERS))


if __name__ == "__main__":
    unittest.main()
