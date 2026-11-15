"""Gateway VRAM budget (VRAMLOGIC1g step 3, pure logic): config, admission, lite timer, pause frame. MiB everywhere."""

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gateway"))

from vram_budget import (Budget, BudgetError, check_policy_config, parse_snapshot,  # noqa: E402
                         DEFAULT_RESERVE_MIB, DEFAULT_HYSTERESIS_MIB, DEFAULT_POLICY)


class ConfigTests(unittest.TestCase):
    def test_defaults_are_governor_and_1536_mib_reserve(self):
        policy, values, lite_after = check_policy_config({})
        self.assertEqual(policy, DEFAULT_POLICY)
        self.assertEqual(policy, "governor")
        self.assertEqual(values["reserve_mib"], DEFAULT_RESERVE_MIB)
        self.assertEqual(DEFAULT_RESERVE_MIB, 1536)
        self.assertEqual(values["hysteresis_mib"], DEFAULT_HYSTERESIS_MIB)

    def test_legacy_policy_is_still_selectable(self):
        policy, _values, _lite = check_policy_config({"vramPolicy": "legacy"})
        self.assertEqual(policy, "legacy")

    def test_unknown_policy_is_refused(self):
        with self.assertRaises(BudgetError):
            check_policy_config({"vramPolicy": "valve"})

    def test_gib_value_for_reserve_is_refused(self):
        # 1.5 would be read as MiB and fail the range: the old GB/MiB mix-up cannot pass silently
        with self.assertRaises(BudgetError):
            check_policy_config({"vramReserveMiB": 1.5})

    def test_reserve_range_matches_server(self):
        with self.assertRaises(BudgetError):
            check_policy_config({"vramReserveMiB": 100})     # below 256
        with self.assertRaises(BudgetError):
            check_policy_config({"vramReserveMiB": 5000})    # above 4096
        check_policy_config({"vramReserveMiB": 2048})


class SnapshotTests(unittest.TestCase):
    def snapshot(self, **over):
        snap = {
            "state": "FULL", "reason": "", "ready": True, "pool_frozen": True,
            "card": {"total_mib": 12282, "free_mib": 3212, "min_free_mib_window": 3100, "source": "sysfs"},
            "fixed_mib": {"weights": 7600, "kv": 640, "total": 9070},
            "cache": {}, "policy": {}, "counters": {},
        }
        snap.update(over)
        return snap

    def test_accepts_contract_snapshot(self):
        self.assertEqual(parse_snapshot(self.snapshot())["fixed_mib"]["total"], 9070)

    def test_missing_field_is_refused(self):
        snap = self.snapshot()
        del snap["counters"]
        with self.assertRaises(BudgetError):
            parse_snapshot(snap)

    def test_float_mib_is_refused(self):
        with self.assertRaises(BudgetError):
            parse_snapshot(self.snapshot(card={"total_mib": 12282.0, "free_mib": 3212, "min_free_mib_window": 1, "source": "x"}))


class AdmissionTests(unittest.TestCase):
    def test_learned_fixed_decides_when_present(self):
        b = Budget()
        b.fixed_learned[("tiel", "short")] = 9070
        ok, need, source = b.admission("tiel", "short", free_before_start_mib=10900)
        self.assertTrue(ok)
        self.assertEqual(need, 0)
        self.assertEqual(source, "learned")

    def test_refuses_with_the_mib_shortfall(self):
        b = Budget()
        b.fixed_learned[("tiel", "short")] = 9070                 # needs 9070 + 1536 + 256 = 10862
        ok, need, source = b.admission("tiel", "short", free_before_start_mib=10000)
        self.assertFalse(ok)
        self.assertEqual(need, 862)
        self.assertEqual(source, "learned")

    def test_estimate_is_used_only_without_learned_value(self):
        b = Budget()
        ok, need, source = b.admission("tiel", "lite", free_before_start_mib=11000, estimate_mib=8000)
        self.assertTrue(ok)
        self.assertEqual(source, "estimate")

    def test_no_value_at_all_is_an_error(self):
        with self.assertRaises(BudgetError):
            Budget().admission("tiel", "short", free_before_start_mib=5000)

    def test_margin_is_reserve_plus_hysteresis(self):
        b = Budget(reserve_mib=1536, hysteresis_mib=256)
        self.assertEqual(b.admit_margin_mib, 1792)


class LiteTimerTests(unittest.TestCase):
    def test_not_in_min_never_triggers_lite(self):
        b = Budget(lite_after_s=60)
        self.assertFalse(b.governor_says_lite("FULL", now=0))
        self.assertFalse(b.governor_says_lite("TRIM", now=1000))

    def test_min_timer_resets_when_state_leaves_min(self):
        b = Budget(lite_after_s=60)
        b.governor_says_lite("MIN", now=0)
        b.governor_says_lite("GROW", now=50)              # left MIN: the clock restarts
        self.assertFalse(b.governor_says_lite("MIN", now=55))
        self.assertFalse(b.governor_says_lite("MIN", now=100))
        self.assertTrue(b.governor_says_lite("MIN", now=116))


class ReliefTests(unittest.TestCase):
    def test_relief_waits_t_lite_after_leaving_min(self):
        b = Budget(lite_after_s=60)
        b.observe("MIN", now=0)
        self.assertFalse(b.lite_relief_ok("GROW", now=10))     # left MIN at 10: T_lite not over
        self.assertFalse(b.lite_relief_ok("FULL", now=69))
        self.assertTrue(b.lite_relief_ok("FULL", now=71))      # 61 s after leaving MIN

    def test_never_relieved_while_in_min(self):
        b = Budget(lite_after_s=0)
        self.assertFalse(b.lite_relief_ok("MIN", now=5))

    def test_no_min_ever_means_normal_layout_allowed(self):
        self.assertTrue(Budget(lite_after_s=60).lite_relief_ok("FULL", now=0))


class PauseFrameTests(unittest.TestCase):
    def frame(self, **inner):
        body = {"checkpoint": "vram-pause.bin", "slot": 0, "reason": "pause_floor", "n_past": 12345,
                "generated_tokens": 87, "saved": True}
        body.update(inner)
        return {"error": {"code": 503, "type": "vram_pause", "message": json.dumps(body)}}

    def test_pause_frame_is_recognised_and_normal_chunks_are_not(self):
        b = Budget()
        frame = b.pause_frame(self.frame())
        self.assertEqual(frame["n_past"], 12345)
        self.assertEqual(frame["generated_tokens"], 87)
        self.assertTrue(frame["saved"])
        self.assertEqual(frame["checkpoint"], "vram-pause.bin")
        self.assertIsNone(b.pause_frame({"choices": [{"delta": {"content": "x"}}]}))
        self.assertIsNone(b.pause_frame({"error": {"code": 500, "type": "server_error", "message": "boom"}}))

    def test_malformed_pause_frame_is_refused(self):
        with self.assertRaises(BudgetError):
            Budget().pause_frame({"error": {"code": 503, "type": "vram_pause", "message": "not json"}})
        with self.assertRaises(BudgetError):
            Budget().pause_frame({"error": {"code": 503, "type": "vram_pause", "message": json.dumps({"n_past": "1", "saved": True})}})

    def test_resume_plan_needs_a_saved_checkpoint(self):
        b = Budget()
        ok = b.pause_frame(self.frame())
        self.assertEqual(b.resume_plan(ok, next_layout_admitted=True)["action"], "restart_and_restore")
        self.assertEqual(b.resume_plan(ok, next_layout_admitted=False)["action"], "wait_and_retry")
        failed = b.pause_frame(self.frame(saved=False, checkpoint=""))
        self.assertEqual(b.resume_plan(failed, next_layout_admitted=True)["action"], "fail")


class TokensFileTests(unittest.TestCase):
    def write(self, payload):
        import tempfile
        d = tempfile.mkdtemp(prefix="vl1i-tokens-")
        path = os.path.join(d, "vram-pause.bin.tokens.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return path

    def base(self, **over):
        data = {"slot": 0, "reason": "pause_floor", "n_past": 4, "tokens": [1, 2, 3, 4], "multimodal": False,
                "processing": True, "generated_tokens": 2, "n_predict": 10, "n_predict_left": 8,
                "params": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "seed": 7}}
        data.update(over)
        return data

    def test_valid_file_gives_the_exact_ids_and_what_is_left(self):
        from vram_budget import load_pause_tokens
        got = load_pause_tokens(self.write(self.base()), frame_n_past=4)
        self.assertEqual(got["tokens"], [1, 2, 3, 4])
        self.assertEqual(got["n_predict_left"], 8)
        self.assertEqual(got["params"]["seed"], 7)
        self.assertTrue(got["processing"])

    def test_unlimited_generation_keeps_minus_one(self):
        from vram_budget import load_pause_tokens
        got = load_pause_tokens(self.write(self.base(n_predict=-1, n_predict_left=-1)))
        self.assertEqual(got["n_predict_left"], -1)

    def test_multimodal_slot_has_no_exact_continuation(self):
        from vram_budget import load_pause_tokens
        with self.assertRaises(BudgetError):
            load_pause_tokens(self.write(self.base(multimodal=True)))

    def test_token_count_must_match_n_past_and_the_frame(self):
        from vram_budget import load_pause_tokens
        with self.assertRaises(BudgetError):
            load_pause_tokens(self.write(self.base(n_past=5)))
        with self.assertRaises(BudgetError):
            load_pause_tokens(self.write(self.base()), frame_n_past=99)

    def test_idle_slot_needs_no_params(self):
        from vram_budget import load_pause_tokens
        got = load_pause_tokens(self.write({"slot": 0, "reason": "x", "n_past": 2, "tokens": [9, 8], "multimodal": False,
                                            "processing": False, "generated_tokens": 0}))
        self.assertFalse(got["processing"])
        self.assertIsNone(got["params"])

    def test_pause_frame_names_the_tokens_file(self):
        frame = Budget().pause_frame({"error": {"code": 503, "type": "vram_pause", "message": json.dumps(
            {"checkpoint": "vram-pause.bin", "tokens_file": "vram-pause.bin.tokens.json", "slot": 0, "reason": "pause_floor",
             "n_past": 4, "generated_tokens": 2, "saved": True})}})
        self.assertEqual(frame["tokens_file"], "vram-pause.bin.tokens.json")


if __name__ == "__main__":
    unittest.main()
