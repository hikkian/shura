"""Pure hysteresis and step-selection tests for the opt-in VRAM valve."""
import unittest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))
from vram_valve import SlotCostEstimator, Valve, valid_slot_cost


class VramValvePolicy(unittest.TestCase):
    def valve(self, initial=18, maximum=18, **config):
        return Valve({"vramValveHysteresisMiB": 200, "vramValveUpSeconds": 60,
                      **config}, maximum, initial)

    def test_default_ladder_uses_only_verified_levels(self):
        self.assertEqual(Valve({}, 18).levels, [18, 16, 8])

    def test_custom_levels_remain_configurable(self):
        levels = [18, 16, 12, 8, 4, 0]
        self.assertEqual(Valve({"vramValveLevels": levels}, 18).levels, levels)

    def test_low_vram_chooses_highest_gear_that_restores_reserve(self):
        valve = self.valve()
        self.assertEqual(valve.target(0, 100, 194), 8)  # 16 leaves 172; 8 leaves 460.

    def test_extreme_low_vram_holds_minimum_gear_and_parks(self):
        valve = self.valve(initial=8)
        self.assertIsNone(valve.target(0, 0, 194))
        self.assertFalse(valve.park_due(1.9))
        self.assertTrue(valve.park_due(2.0))
        self.assertIn("minimum 8", valve.reason)

    def test_busy_request_defers_resize_until_boundary(self):
        valve = self.valve()
        self.assertIsNone(valve.target(0, 100, 194, busy=True))
        self.assertEqual(valve.target(0.1, 100, 194), 8)

    def test_upshift_requires_reserve_slot_cost_hysteresis_and_stable_window(self):
        valve = self.valve(initial=0)
        # 682 MiB is exactly reserve + eight-slot cost + hysteresis.
        self.assertIsNone(valve.target(0, 681, 194))
        self.assertIsNone(valve.target(1, 682, 194))
        self.assertEqual(valve.target(61, 682, 194), 8)

    def test_upshift_is_capped_at_configured_maximum(self):
        valve = self.valve(initial=8, maximum=8)
        valve.current = 8
        self.assertIsNone(valve.target(0, 10000, 194))
        self.assertEqual(valve.current, 8)

    def test_no_sample_or_game_state_never_changes_slots(self):
        valve = self.valve()
        self.assertIsNone(valve.target(0, None, 194))
        self.assertIsNone(valve.target(1, 10_000, 194, ready=False))
        self.assertEqual(valve.current, 18)

    def test_failed_resize_uses_backoff_and_keeps_level(self):
        valve = self.valve()
        self.assertEqual(valve.target(0, 100, 194), 8)
        valve.completed(0, 8, False)
        self.assertEqual(valve.current, 18)
        self.assertIsNone(valve.target(1, 100, 194))
        self.assertEqual(valve.target(2.1, 100, 194), 8)

    def test_model_max_bounds_custom_levels(self):
        valve = self.valve(initial=18, maximum=18, vramValveLevels=[30, 24, 18, 8, 0])
        self.assertEqual(valve.levels, [18, 8, 0])
        self.assertEqual(valve.maximum, 18)

    def test_no_implicit_zero_when_configured_ladder_stops_at_twenty(self):
        valve = Valve({"vramValveLevels": [60, 40, 20]}, 60)
        self.assertEqual(valve.levels, [60, 40, 20])
        self.assertEqual(valve.target(0, 0, 194), 40)
        valve.completed(0, 40, True)
        self.assertEqual(valve.target(1, 0, 194), 20)
        valve.completed(1, 20, True)
        self.assertIsNone(valve.target(1.1, 0, 194))
        self.assertEqual(valve.current, 20)
        self.assertTrue(valve.park_due(3.1))

    def test_zero_is_retained_only_when_explicit_or_already_reported(self):
        self.assertEqual(Valve({"vramValveLevels": [18, 16, 8, 0]}, 18).levels, [18, 16, 8, 0])
        self.assertEqual(Valve({"vramValveLevels": [18, 16, 8]}, 18, 0).levels, [18, 16, 8, 0])

    def test_slot_cost_source_priority_config_then_measured_then_default(self):
        costs = SlotCostEstimator()
        before, after, foreign = [500, 500], [1125.3, 1125.3], {11: 100, 22: 80}
        learned = costs.observe_resize("tiel", 60, 40, before, after, foreign, foreign)
        self.assertTrue(learned["accepted"])
        self.assertAlmostEqual(learned["measurement_mib"], 31.265, places=2)
        self.assertEqual(costs.resolve("tiel", {}, 36.156)["source"], "measured")
        self.assertEqual(costs.resolve("tiel", {"moeCacheSlotMiB": 31.3}, 36.156)["source"], "config")
        self.assertEqual(costs.resolve("new-model", {}, 36.156)["source"], "default")
        self.assertAlmostEqual(costs.resolve("new-model", {}, 36.156)["value_mib"], 36.156)

    def test_slot_cost_invalid_config_values_fall_back_without_being_used(self):
        costs = SlotCostEstimator()
        invalid = [None, True, False, "31.3", math.nan, math.inf, -1, 9.99, 80.01]
        for index, value in enumerate(invalid):
            with self.subTest(value=value):
                info = costs.resolve(f"m{index}", {"moeCacheSlotMiB": value}, 36.156)
                self.assertEqual(info["source"], "default")
                self.assertEqual(info["value_mib"], 36.156)
                self.assertIsNotNone(info["error"])
        self.assertIsNone(valid_slot_cost(math.nan))
        self.assertIsNone(valid_slot_cost(-2))
        self.assertIsNone(valid_slot_cost(81))

    def test_slot_cost_measurement_rejects_noise_foreign_load_and_implausible_values(self):
        costs = SlotCostEstimator()
        foreign = {11: 100}
        cases = [
            ([500, 600], [1125, 1125], foreign, foreign),  # probe spread > 64
            ([500, 500], [1125, 1125], foreign, {11: 200}),  # foreign load changed
            ([500, 500], [550, 550], foreign, foreign),  # < 10 MiB / slot
            ([500, 500], [2500, 2500], foreign, foreign),  # > 80 MiB / slot
            ([500], [1125, 1125], foreign, foreign),  # not settled
        ]
        for before, after, foreign_before, foreign_after in cases:
            result = costs.observe_resize("tiel", 60, 40, before, after, foreign_before, foreign_after)
            self.assertFalse(result["accepted"])
        self.assertNotIn("tiel", costs.measured)

    def test_slot_cost_uses_last_three_per_model_and_never_cross_contaminates(self):
        costs = SlotCostEstimator()
        foreign = {11: 100}
        for price in (30, 31, 32, 33):
            result = costs.observe_resize("tiel", 60, 40, [500, 500], [500 + price * 20] * 2,
                                          foreign, foreign)
            self.assertTrue(result["accepted"])
        self.assertEqual(len(costs.measured["tiel"]), 3)
        self.assertAlmostEqual(costs.resolve("tiel", {}, 36)["value_mib"], 32)
        self.assertEqual(costs.resolve("occamy", {}, 36.2)["source"], "default")
        occamy = costs.observe_resize("occamy", 12, 10, [500, 500], [572.4, 572.4], foreign, foreign)
        self.assertTrue(occamy["accepted"])
        self.assertAlmostEqual(costs.resolve("occamy", {}, 36.2)["value_mib"], 36.2, places=2)
        self.assertAlmostEqual(costs.resolve("tiel", {}, 36.2)["value_mib"], 32)

    def test_apex_and_iq4_growth_thresholds_use_their_model_prices(self):
        cases = ((54, 40, 31.3), (24, 18, 36.2))
        for maximum, current, price in cases:
            levels = [maximum, current]
            valve = Valve({"vramValveLevels": levels, "vramValveUpSeconds": 60}, maximum, current)
            required = 194 + (maximum - current) * price + 200
            self.assertIsNone(valve.target(0, required - 1, 194, per_slot_mib=price))
            self.assertIsNone(valve.target(1, required, 194, per_slot_mib=price))
            self.assertIsNone(valve.target(60, required, 194, per_slot_mib=price))
            self.assertEqual(valve.target(61, required, 194, per_slot_mib=price), maximum)

    def test_overestimated_config_price_cannot_consume_the_reserve_on_shrink(self):
        # Even the maximum accepted estimate (80 MiB/slot) selects a single 54->40 step;
        # the known APEX cost still frees more than the 194 MiB reserve.
        valve = Valve({"vramValveLevels": [54, 40], "vramValveHysteresisMiB": 200}, 54, 54)
        target = valve.target(0, 0, 194, per_slot_mib=80.0)
        self.assertEqual(target, 40)
        actual_free_after = (54 - target) * 31.3
        self.assertGreaterEqual(actual_free_after, 194)


if __name__ == "__main__":
    unittest.main()
