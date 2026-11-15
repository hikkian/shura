"""Vulkan speed prediction (PUB5b): the coefficients come from open llama-bench data (catalog/speed-calibration-vulkan.json),
the planner carries the same numbers, and a prediction made from them says so ("оценка, ±20 %" / "оценка, ±50 %, мало данных").
Offline and deterministic. Expectations are about which coefficient is used and what label is shown, not exact tok/s.
Run: python3 -m unittest discover -s tests"""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
from universal import planner, report  # noqa: E402

HW = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "tests/fixtures/hardware").glob("*.json"))}
SYN = json.loads((ROOT / "tests/fixtures/catalog_test.json").read_text())
CAL = json.loads((ROOT / "catalog/speed-calibration-vulkan.json").read_text())
DENSE_LABEL = "оценка, ±20 %"
MOE_LABEL = "оценка, ±50 %, мало данных"


def amd_vulkan(name="rx7900xtx_24g_64g", **kw):
    """Default path on an AMD card: Vulkan, then CPU (native builds not asked for)."""
    return planner.plan(HW[name], SYN, menu=False, **kw)


class CalibrationFile(unittest.TestCase):
    def test_planner_numbers_match_the_calibration_file(self):
        d = planner.DEFAULTS
        self.assertEqual(d["vulkan_efficiency"]["dense"], CAL["dense"]["efficiency"])
        self.assertEqual(d["vulkan_efficiency"]["moe"], CAL["moe"]["efficiency"])
        self.assertEqual(d["vulkan_efficiency"]["intel_dense"], CAL["intel_dense"]["efficiency"])
        self.assertEqual(d["vulkan_label"]["dense"], CAL["dense"]["label"])
        self.assertEqual(d["vulkan_label"]["moe"], CAL["moe"]["label"])
        self.assertEqual(d["vulkan_label"]["intel_dense"], CAL["intel_dense"]["label"])


class VulkanLabel(unittest.TestCase):
    def test_dense_model_on_vulkan_gets_the_twenty_percent_label(self):
        p = amd_vulkan(model="test-dense-8b")
        self.assertTrue(p["ok"], p.get("reasons"))
        self.assertEqual(p["mode"], "gpu", "the check needs the dense model fully on the GPU")
        self.assertEqual(p["speed_estimate"], {"key": "dense", "label": DENSE_LABEL})

    def test_moe_model_on_vulkan_has_no_label_because_its_coefficient_is_not_applied(self):
        # the measured MoE value (0.12, one card) would lower the default quant on AMD: kept out of planning (PUB5b)
        p = amd_vulkan(model="tiel-coder-35b-a3b-mtp")
        self.assertTrue(p["ok"], p.get("reasons"))
        self.assertIsNone(p["speed_estimate"])

    def test_intel_arc_on_vulkan_uses_its_own_dense_coefficient(self):
        hw = copy.deepcopy(HW["rx7900xtx_24g_64g"])
        hw["gpus"][0]["vendor"] = "intel"
        p = planner.plan(hw, SYN, menu=False, model="test-dense-8b")
        self.assertEqual(p["mode"], "gpu")
        self.assertEqual(p["speed_estimate"], {"key": "intel_dense", "label": MOE_LABEL})

    def test_native_cuda_path_has_no_vulkan_label(self):
        p = planner.plan(HW["ref_rtx4070s_12g_32g"], SYN, config={"native_backends": True}, menu=False)
        self.assertTrue(p["ok"], p.get("reasons"))
        self.assertIsNone(p["speed_estimate"])

    def test_the_card_prints_the_label_on_the_expected_line(self):
        p = amd_vulkan(model="test-dense-8b")
        expected = next(line for line in report.card(HW["rx7900xtx_24g_64g"], p).splitlines() if "Expected" in line)
        self.assertIn(DENSE_LABEL, expected)


class CoefficientMovesThePrediction(unittest.TestCase):
    def test_a_lower_dense_efficiency_predicts_a_slower_model(self):
        base = amd_vulkan(model="test-dense-8b")["predicted_tok_s"]["mid"]
        slower = amd_vulkan(model="test-dense-8b", config={"vulkan_efficiency": {"dense": 0.5}})["predicted_tok_s"]["mid"]
        self.assertLess(slower, base)

    def test_the_moe_value_does_not_change_the_plan_yet(self):
        # regression guard for the quality rule: the MoE coefficient is recorded, not applied
        base = amd_vulkan(model="tiel-coder-35b-a3b-mtp")
        other = amd_vulkan(model="tiel-coder-35b-a3b-mtp", config={"vulkan_efficiency": {"moe": 0.30}})
        self.assertEqual(base["quant"], other["quant"])
        self.assertEqual(base["predicted_tok_s"], other["predicted_tok_s"])


if __name__ == "__main__":
    unittest.main()
