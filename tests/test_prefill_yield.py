"""Pure prefill-yield policy: validation, token estimate, decision and counters (no server)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gateway"))
import prefill_yield as py  # noqa: E402


class PrefillYieldPolicy(unittest.TestCase):
    def test_defaults_are_off_and_4000_tokens(self):
        self.assertEqual(py.DEFAULTS, {"prefillCacheYield": False, "prefillCacheYieldMinTokens": 4000})

    def test_validate_rejects_non_bool_switch_and_bad_threshold(self):
        with self.assertRaises(ValueError):
            py.validate({"prefillCacheYield": "yes"})
        with self.assertRaises(ValueError):
            py.validate({"prefillCacheYieldMinTokens": 0})
        with self.assertRaises(ValueError):
            py.validate({"prefillCacheYieldMinTokens": 2.5})
        py.validate({})
        py.validate({"prefillCacheYield": True, "prefillCacheYieldMinTokens": 8000})

    def test_new_tokens_first_request_counts_whole_body(self):
        self.assertEqual(py.new_tokens_estimate(40000, None), 10000)

    def test_new_tokens_grown_conversation_counts_only_the_growth(self):
        self.assertEqual(py.new_tokens_estimate(40400, 40000), 100)

    def test_new_tokens_shorter_body_is_a_new_conversation(self):
        self.assertEqual(py.new_tokens_estimate(20000, 40000), 5000)

    def test_large_new_prefill_is_yielded(self):
        self.assertEqual(py.should_yield(True, 5000, 4000, 30, True, 1, False, False, False), (True, py.REASON_OK))

    def test_small_request_is_not_yielded(self):
        ok, reason = py.should_yield(True, 3999, 4000, 30, True, 1, False, False, False)
        self.assertFalse(ok)
        self.assertEqual(reason, "small request")

    def test_each_guard_blocks_the_yield(self):
        base = dict(enabled=True, new_tokens=9000, minimum=4000, slots_now=30, server_ready=True, busy=1,
                    vision=False, game_active=False, slot_gears=False)
        cases = {
            "disabled": {"enabled": False}, "vision request": {"vision": True}, "game mode active": {"game_active": True},
            "slotGears owns the slot count": {"slot_gears": True}, "server not ready": {"server_ready": False},
            "other requests in flight": {"busy": 2}, "cache already empty": {"slots_now": 0},
        }
        for reason, change in cases.items():
            ok, got = py.should_yield(**{**base, **change})
            self.assertFalse(ok, reason)
            self.assertEqual(got, reason)

    def test_state_counts_and_snapshot(self):
        state = py.YieldState()
        state.count("yielded")
        state.count("restored")
        self.assertIsNone(state.remember_len(100))
        self.assertEqual(state.remember_len(250), 100)
        snap = state.snapshot(True)
        self.assertEqual(snap["counters"]["yielded"], 1)
        self.assertEqual(snap["counters"]["restored"], 1)
        self.assertTrue(snap["enabled"])
        self.assertIsNone(snap["pending"])
        state.pending = {"model": "tiel-coder", "slots": 30}
        self.assertEqual(state.snapshot(False)["pending"], {"model": "tiel-coder", "slots": 30})


if __name__ == "__main__":
    unittest.main()
