"""Universal planner tests. Offline and deterministic: hardware comes from tests/fixtures/hardware/*.json (profiles of
machines we describe, not machines we own) and the model catalog from catalog/models.json plus a synthetic catalog
(tests/fixtures/catalog_test.json) that stands in for big MoE and small dense models.
Expectations are about the SHAPE of a plan (what fits, what class, which backends), never exact tok/s.
Run: python3 -m unittest discover -s tests"""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
from universal import planner, schema  # noqa: E402

HW = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "tests/fixtures/hardware").glob("*.json"))}
REAL = json.loads((ROOT / "catalog/models.json").read_text())
SYN = json.loads((ROOT / "tests/fixtures/catalog_test.json").read_text())
GiB = 1024 ** 3


def plan(name, cat=REAL, **kw):
    return planner.plan(HW[name], cat, **kw)


class Schemas(unittest.TestCase):
    def test_all_fixtures_and_catalogs_are_valid(self):
        for name, hw in HW.items():
            self.assertEqual(schema.validate_hardware(hw), [], name)
        self.assertEqual(schema.validate_catalog(REAL), [])
        self.assertEqual(schema.validate_catalog(SYN), [])

    def test_problems_are_listed_together_and_readably(self):
        bad = copy.deepcopy(HW["ref_rtx4070s_12g_32g"])
        bad["os"]["family"] = "amiga"
        bad["gpus"][0]["vram_total"] = -5
        bad["memory"]["available"] = bad["memory"]["total"] * 2
        problems = schema.validate_hardware(bad)
        self.assertGreaterEqual(len(problems), 3)
        self.assertTrue(any("os.family" in p for p in problems))
        with self.assertRaises(ValueError):
            planner.plan(bad, REAL)

    def test_catalog_rejects_bad_default_quant_and_duplicate_quality(self):
        bad = copy.deepcopy(REAL)
        bad["models"][0]["default_quant"] = "NOPE"
        bad["models"][0]["quants"][1]["quality"] = bad["models"][0]["quants"][0]["quality"]
        problems = schema.validate_catalog(bad)
        self.assertTrue(any("default_quant" in p for p in problems))
        self.assertTrue(any("quality" in p for p in problems))


class ReferenceMachine(unittest.TestCase):
    """The one machine we measured: RTX 4070 SUPER 12 GB, 32 GB DDR4 at ~35 GB/s."""

    def test_gets_the_benchmarked_quant_in_hybrid_mode_with_cuda_first(self):
        p = plan("ref_rtx4070s_12g_32g")
        self.assertTrue(p["ok"])
        self.assertEqual((p["model"], p["quant"], p["mode"]), ("tiel-coder-35b-a3b-mtp", "IQ4_XS", "hybrid"))
        self.assertEqual(p["backend_candidates"][0], "cuda")
        self.assertTrue(20 <= p["settings"]["n_cpu_moe"] <= 34)
        self.assertEqual(p["settings"]["threads"], 6)

    def test_prediction_is_near_what_upstream_style_offload_measured(self):
        # Stock llama.cpp with --n-cpu-moe on this machine measured about 26-30 tok/s at ~187k (docs/BENCHMARKS.md).
        mid = plan("ref_rtx4070s_12g_32g")["predicted_tok_s"]["mid"]
        self.assertTrue(20 <= mid <= 40, mid)

    def test_no_quality_upgrade_without_speed_headroom(self):
        self.assertEqual(plan("ref_rtx4070s_12g_32g")["quant"], "IQ4_XS")

    def test_headless_gpu_gets_more_experts_on_the_gpu_than_a_display_gpu(self):
        headless = copy.deepcopy(HW["ref_rtx4070s_12g_32g"])
        headless["gpus"][0]["display"], headless["gpus"][0]["vram_used"] = False, 0
        self.assertLess(planner.plan(headless, REAL)["settings"]["n_cpu_moe"],
                        plan("ref_rtx4070s_12g_32g")["settings"]["n_cpu_moe"])


class HardwareClasses(unittest.TestCase):
    def test_small_ram_steps_down_or_fits_without_overcommitting(self):
        p = plan("rtx3060_12g_16g")
        self.assertTrue(p["ok"])
        self.assertLessEqual(p["memory"]["ram_need"], p["memory"]["ram_budget"])

    def test_big_gpus_upgrade_quality_and_keep_more_experts_on_the_gpu(self):
        big = plan("rtx4090_24g_64g")
        self.assertEqual(big["quant"], "Q5_K_XL")
        self.assertLess(big["settings"]["n_cpu_moe"], plan("ref_rtx4070s_12g_32g")["settings"]["n_cpu_moe"])

    def test_amd_and_intel_get_several_backends_to_probe(self):
        amd = plan("rx7900xtx_24g_64g")
        self.assertEqual(amd["backend_candidates"][:2], ["rocm", "vulkan"])
        self.assertTrue(amd["needs_probe"])
        intel = plan("arc_a770_16g_32g")
        self.assertEqual(intel["backend_candidates"][:3], ["sycl", "vulkan", "openvino"])
        self.assertTrue(intel["needs_probe"])

    def test_apple_is_unified_memory_and_needs_no_probe(self):
        p = plan("apple_m2max_64g")
        self.assertEqual((p["mode"], p["backend_candidates"][0], p["needs_probe"]), ("unified", "metal", False))
        self.assertIsNone(p["settings"]["n_cpu_moe"])

    def test_cpu_only_is_not_automatically_small(self):
        server, laptop = plan("epyc7551x2_512g_cpu", SYN), plan("laptop_cpu_16g", SYN)
        self.assertEqual(server["mode"], "cpu")
        self.assertGreater(
            next(m for m in SYN["models"] if m["id"] == server["model"])["params_total_b"],
            next(m for m in SYN["models"] if m["id"] == laptop["model"])["params_total_b"])
        self.assertEqual(server["backend_candidates"], ["cpu"])
        self.assertEqual(server["settings"]["numa"], "distribute")
        self.assertEqual(server["settings"]["threads"], 64)
        self.assertTrue(any("NUMA" in w for w in server["warnings"]))

    def test_model_too_slow_for_the_machine_is_not_picked_when_a_faster_one_fits(self):
        # 235B-A22B on dual EPYC would read ~8 GB per token at ~72 GB/s effective: well below 15 tok/s.
        p = plan("epyc7551x2_512g_cpu", SYN)
        self.assertNotEqual(p["model"], "test-moe-235b-a22b")
        self.assertGreaterEqual(p["predicted_tok_s"]["mid"], 15)

    def test_huge_unified_memory_can_take_the_biggest_model_that_is_fast_enough(self):
        self.assertEqual(plan("apple_m3ultra_192g", SYN)["model"], "test-moe-235b-a22b")


class Refusals(unittest.TestCase):
    def test_refuses_with_a_reason_when_nothing_fits(self):
        p = plan("laptop_cpu_16g")
        self.assertFalse(p["ok"])
        self.assertIn("RAM", p["reasons"][0])
        self.assertEqual(p["backend_candidates"], ["cpu"])

    def test_below_minimum_speed_is_a_warning_not_a_silent_pick(self):
        p = plan("laptop_cpu_16g", SYN)
        self.assertTrue(p["ok"])
        self.assertTrue(any("below" in w for w in p["warnings"]))

    def test_unknown_forced_quant_is_reported(self):
        p = plan("ref_rtx4070s_12g_32g", model="tiel-coder-35b-a3b-mtp", quant="Q9")
        self.assertFalse(p["ok"])
        self.assertIn("unknown quant", p["reasons"][0])


class Properties(unittest.TestCase):
    def test_faster_memory_never_lowers_the_prediction(self):
        slow, fast = copy.deepcopy(HW["ref_rtx4070s_12g_32g"]), copy.deepcopy(HW["ref_rtx4070s_12g_32g"])
        slow["memory"]["bandwidth_gbs"], fast["memory"]["bandwidth_gbs"] = 20, 60
        a, b = planner.plan(slow, REAL, quant="IQ4_XS", model="tiel-coder-35b-a3b-mtp"), \
            planner.plan(fast, REAL, quant="IQ4_XS", model="tiel-coder-35b-a3b-mtp")
        self.assertLess(a["predicted_tok_s"]["mid"], b["predicted_tok_s"]["mid"])

    def test_more_vram_never_puts_more_experts_on_the_cpu(self):
        small, large = copy.deepcopy(HW["rtx3060_12g_16g"]), copy.deepcopy(HW["rtx3060_12g_16g"])
        large["gpus"][0]["vram_total"] = 24 * GiB
        kw = dict(model="tiel-coder-35b-a3b-mtp", quant="IQ4_XS")
        self.assertGreaterEqual(planner.plan(small, REAL, **kw)["settings"]["n_cpu_moe"],
                                planner.plan(large, REAL, **kw)["settings"]["n_cpu_moe"])

    def test_dense_model_splits_layers_when_vram_is_short(self):
        hw = copy.deepcopy(HW["ref_rtx4070s_12g_32g"])
        hw["gpus"][0]["vram_total"] = 6 * GiB
        p = planner.plan(hw, SYN, model="test-dense-8b", quant="Q8")
        self.assertTrue(p["ok"])
        self.assertEqual(p["mode"], "hybrid")
        self.assertTrue(0 < p["settings"]["ngl"] < 32)

    def test_default_context_is_capped_per_mode(self):
        caps = planner.DEFAULTS["context_cap"]
        for name in HW:
            for cat in (REAL, SYN):
                p = plan(name, cat)
                if p["ok"]:
                    self.assertLessEqual(p["settings"]["context"], caps[p["mode"]], (name, p["mode"]))
        self.assertLessEqual(plan("epyc7551x2_512g_cpu", SYN)["settings"]["context"], 32768)

    def test_deterministic(self):
        self.assertEqual(plan("rx9070xt_16g_32g"), plan("rx9070xt_16g_32g"))

    def test_unmeasured_bandwidth_is_flagged_and_lowers_confidence(self):
        hw = copy.deepcopy(HW["ref_rtx4070s_12g_32g"])
        del hw["memory"]["bandwidth_gbs"]
        p = planner.plan(hw, REAL)
        self.assertEqual(p["confidence"], "low")
        self.assertTrue(any("not measured" in w for w in p["warnings"]))


if __name__ == "__main__":
    unittest.main()
