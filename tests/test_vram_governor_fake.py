"""Governor scenarios on the fake (VRAMLOGIC1g step 0). Virtual clock, no GPU.

Scenarios from the card: start -> warmup -> READY; desk growth -> trim and MIN; a jump bigger than the cache -> pause
with a checkpoint and continuation; idle; policy validation. Contract: reports/vramlogic1/API.md.
"""

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "gateway"))

from support.fake_vram import FakeVram, validate_policy_patch, PolicyError  # noqa: E402
from vram_budget import Budget, parse_snapshot  # noqa: E402

# A layout whose fixed part leaves about 3.2 GiB free on a 12 GiB card once loaded (the API.md example is larger).
FIXED = {"weights": 7600, "kv": 640, "draft": 120, "pool": 410, "sched": 300, "vision": 0}


def make(**kw):
    kw.setdefault("fixed", FIXED)
    kw.setdefault("total_mib", 12282)
    return FakeVram(**kw)


class StartupTests(unittest.TestCase):
    def test_warmup_then_ready_and_pool_frozen(self):
        v = make(warmup_seconds=4.2)
        self.assertEqual(v.state, "WARMUP")
        self.assertFalse(v.ready)
        v.tick(4.0)
        self.assertFalse(v.ready)
        v.tick(0.4)
        self.assertTrue(v.ready)
        self.assertTrue(v.pool_frozen)
        self.assertEqual(v.state, "FULL")

    def test_snapshot_matches_contract_fields(self):
        v = make()
        v.tick(5)
        snap = parse_snapshot(v.snapshot())
        self.assertEqual(snap["fixed_mib"]["total"], sum(FIXED.values()))
        self.assertEqual(snap["cache"]["mib"], snap["cache"]["chunks"] * snap["cache"]["chunk_mib"])
        self.assertEqual(snap["cache"]["slots"], snap["cache"]["chunks"] * snap["cache"]["chunk_slots"])

    def test_learned_fixed_is_measured_after_warmup_only(self):
        v = make()
        budget = Budget()
        with self.assertRaises(ValueError):
            budget.learn("tiel", "short", v.snapshot())   # still WARMUP: ready is false
        v.tick(5)
        self.assertEqual(budget.learn("tiel", "short", v.snapshot()), sum(FIXED.values()))


class GrowthAndTrimTests(unittest.TestCase):
    def test_cache_grows_in_steps_and_keeps_reserve_plus_hysteresis(self):
        v = make(policy={"grow_seconds": 30})
        v.tick(5)
        self.assertEqual(v.chunks, 0)
        seen = []
        for _ in range(40):
            v.tick(10)
            seen.append(v.chunks)
            self.assertGreaterEqual(v.card_free(), v.policy["reserve_mib"] + 0)
        self.assertGreater(v.chunks, 0)
        self.assertEqual(seen, sorted(seen))                     # never shrinks on its own
        # VRAMLOGIC2: the cache fills up to the working target (reserve + buffer) + hysteresis, not to the floor + hysteresis
        self.assertLessEqual(v.card_free(), v.target_mib() + v.policy["hysteresis_mib"] + v.chunk_mib + 1)
        self.assertGreaterEqual(v.card_free(), v.target_mib() + v.policy["hysteresis_mib"])

    def test_desk_jump_trims_chunks_and_restores_reserve_within_one_second(self):
        v = make(policy={"grow_seconds": 0}, start_chunks=10)
        v.tick(5)
        before = v.chunks
        v.set_desk(1100)
        v.tick(1.0)
        self.assertLess(v.chunks, before)
        self.assertGreaterEqual(v.card_free(), v.policy["reserve_mib"])
        self.assertEqual(v.state, "TRIM")
        self.assertLessEqual(v.counters["free_below_reserve_max_s"], 1.0)

    def test_simulate_hook_takes_desk_for_seconds_then_gives_it_back(self):
        v = make(policy={"grow_seconds": 0}, start_chunks=5)
        v.tick(5)
        base = v.desk_mib
        v.simulate({"free_delta_mib": -1024, "seconds": 5})
        self.assertEqual(v.desk_mib, base + 1024)
        v.tick(6)
        self.assertEqual(v.desk_mib, base)

    def test_max_chunks_zero_for_game_mode_empties_cache_and_blocks_growth(self):
        v = make(policy={"grow_seconds": 0}, start_chunks=6)
        v.tick(5)
        code, body = v.post_policy({"max_chunks": 0})
        self.assertEqual(code, 200)
        v.tick(1)
        self.assertEqual(v.chunks, 0)
        v.tick(60)
        self.assertEqual(v.chunks, 0)


class MinAndPauseTests(unittest.TestCase):
    def test_jump_bigger_than_cache_goes_min_then_gateway_goes_lite_after_t_lite(self):
        v = make(policy={"grow_seconds": 0, "pause_floor_mib": 0}, start_chunks=0)
        v.tick(5)
        v.set_desk(2600)             # free drops below the reserve with no cache to give back
        v.tick(1.0)
        self.assertEqual(v.state, "MIN")
        budget = Budget(lite_after_s=60)
        self.assertFalse(budget.governor_says_lite("MIN", now=0.0))      # MIN just started
        self.assertFalse(budget.governor_says_lite("MIN", now=59.0))
        self.assertTrue(budget.governor_says_lite("MIN", now=61.0))      # longer than T_lite: lite on a request boundary

    def test_pause_when_min_and_free_below_floor_for_two_polls(self):
        v = make(policy={"grow_seconds": 0, "pause_floor_mib": 768}, start_chunks=0)
        v.tick(5)
        v.set_desk(3000)             # free below the floor, no cache to give back
        v.tick(0.6)
        self.assertEqual(v.state, "PAUSED")
        self.assertEqual(v.exit_code, 75)
        self.assertEqual(len(v.frames), 1)
        frame = json.loads(v.frames[0]["error"]["message"])
        self.assertEqual(frame["reason"], "pause_floor")
        self.assertTrue(frame["checkpoint"].endswith(".bin"))
        self.assertTrue(frame["saved"])

    def test_gateway_pause_frame_parses_and_resume_restores_same_response(self):
        v = make(policy={"grow_seconds": 0, "pause_floor_mib": 768}, start_chunks=0)
        v.tick(5)
        v.set_desk(3000)
        v.tick(0.6)
        budget = Budget()
        frame = budget.pause_frame(v.frames[0])
        self.assertIsNotNone(frame)
        plan = budget.resume_plan(frame, next_layout_admitted=True)
        self.assertEqual(plan["action"], "restart_and_restore")
        self.assertEqual(plan["checkpoint"], frame["checkpoint"])
        self.assertEqual(budget.resume_plan(frame, next_layout_admitted=False)["action"], "wait_and_retry")

    def test_post_pause_saves_checkpoint_and_exits_75(self):
        v = make()
        v.tick(5)
        code, body = v.post_pause({"checkpoint": "/tmp/x.bin", "reason": "gateway"})
        self.assertEqual(code, 202)
        self.assertEqual(body, {"state": "PAUSING"})
        self.assertEqual(v.exit_code, 75)


class PolicyTests(unittest.TestCase):
    def test_policy_patch_is_atomic_and_validated(self):
        v = make()
        before = v.get_policy()
        code, body = v.post_policy({"reserve_mib": 1024, "hysteresis_mib": 99999})
        self.assertEqual(code, 400)
        self.assertEqual(v.get_policy(), before)              # nothing applied
        code, body = v.post_policy({"reserve_mib": 1024})
        self.assertEqual(code, 200)
        self.assertEqual(body["reserve_mib"], 1024)

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(PolicyError):
            validate_policy_patch({"lite_pressure": True}, {"reserve_mib": 1536})

    def test_max_chunks_accepts_null_and_range(self):
        self.assertIsNone(validate_policy_patch({"max_chunks": None}, {})["max_chunks"])
        with self.assertRaises(PolicyError):
            validate_policy_patch({"max_chunks": 65}, {})

    def test_policy_reserve_is_in_mib_not_gib(self):
        with self.assertRaises(PolicyError):
            validate_policy_patch({"reserve_mib": 1.5}, {})  # a GiB value must not pass as MiB


if __name__ == "__main__":
    unittest.main()
