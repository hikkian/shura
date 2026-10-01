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

    def test_fork_tier_reproduces_the_measured_curve(self):
        # Measured with the fork, ncmoe 26 + 24 slots + turbo3 + MTP (docs/BENCHMARKS.md): 54 tok/s near empty, 42 at 128k,
        # 38.6-39.2 at 187k. The plan for this machine is that layout's neighbour.
        p = plan("ref_rtx4070s_12g_32g")
        self.assertEqual((p["tier"], p["settings"]["context"], p["settings"]["kv_type"]), ("fork", 200000, "turbo3"))
        by = p["speed_by_fill"]
        self.assertTrue(48 <= by["empty"] <= 58, by)
        self.assertTrue(35 <= by["full"] <= 41, by)
        self.assertTrue(24 >= p["settings"]["moe_cache_slots"] >= 16)
        self.assertEqual(p["speed_rung"], "target")
        self.assertLessEqual(p["memory"]["vram_need"], p["memory"]["vram_budget"])

    def test_standard_tier_stays_in_the_range_stock_llama_cpp_measured(self):
        # Stock llama.cpp with --n-cpu-moe on this machine measured about 26-30 tok/s at ~187k (docs/BENCHMARKS.md).
        p = plan("ref_rtx4070s_12g_32g", config={"fork": {"enabled": False}, "tqp_enabled": False})
        self.assertEqual(p["tier"], "standard")
        self.assertTrue(20 <= p["predicted_tok_s"]["mid"] <= 45, p["predicted_tok_s"])

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

    def test_the_chosen_window_scores_best_among_the_windows_that_fit(self):
        # forcing any other window (same model and quant) never scores higher than what the planner chose
        by_id = {m["id"]: m for cat in (REAL, SYN) for m in cat["models"]}
        for name in HW:
            for cat in (REAL, SYN):
                p = plan(name, cat)
                if not p["ok"]:
                    continue
                self.assertLessEqual(p["memory"]["ram_need"], p["memory"]["ram_budget"])
                for ctx in planner.DEFAULTS["contexts"]:
                    if ctx == p["settings"]["context"] or ctx > by_id[p["model"]]["context_max"]:
                        continue
                    if p["tier"] == "fork" and ctx > 200000:        # beyond the verified window: the margin rule, not the score
                        continue
                    other = planner.plan(HW[name], cat, model=p["model"], quant=p["quant"], menu=False,
                                         config={"contexts": (ctx,)})
                    too_slow = other["ok"] and other["speed_by_fill"]["typical"] < planner.DEFAULTS["min_tok_s"]
                    self.assertTrue(not other["ok"] or too_slow or other["score"] <= p["score"] + 1e-9,
                                    (name, p["model"], p["settings"]["context"], ctx))

    def test_more_vram_or_ram_never_shrinks_the_window(self):
        small, large = copy.deepcopy(HW["rtx3060_12g_16g"]), copy.deepcopy(HW["rtx3060_12g_16g"])
        small["gpus"][0]["vram_total"], large["gpus"][0]["vram_total"] = 8 * GiB, 24 * GiB
        small, large = copy.deepcopy(HW["ref_rtx4070s_12g_32g"]), copy.deepcopy(HW["ref_rtx4070s_12g_32g"])
        small["gpus"][0]["vram_total"], large["gpus"][0]["vram_total"] = 8 * GiB, 24 * GiB
        kw = dict(model="tiel-coder-35b-a3b-mtp", quant="IQ3_XXS")
        self.assertLessEqual(planner.plan(small, REAL, **kw)["settings"]["context"],
                             planner.plan(large, REAL, **kw)["settings"]["context"])
        lean, rich = copy.deepcopy(HW["epyc7551x2_256g_cpu"]), copy.deepcopy(HW["epyc7551x2_256g_cpu"])
        lean["memory"]["total"], lean["memory"]["available"] = 20 * GiB, 16 * GiB
        kw2 = dict(model="tiel-coder-35b-a3b-mtp", quant="Q2_K_XL")      # the smallest quant just fits 20 GB of RAM
        self.assertLess(planner.plan(lean, REAL, **kw2)["settings"]["context"],
                        planner.plan(rich, REAL, **kw2)["settings"]["context"])

    def test_a_big_machine_gets_the_full_window_and_keeps_model_quality(self):
        # dual EPYC, 256 GB in 16 channels: the window is capacity, not a reason to run a smaller quant
        for name in ("epyc7551x2_256g_cpu", "epyc7551x2_512g_cpu"):
            p = plan(name)
            self.assertEqual(p["settings"]["context"], 262144, name)
            self.assertEqual(p["quant"], "IQ4_XS", name)
            self.assertLess(p["memory"]["kv_share"], 0.05)
            self.assertLess(p["speed_by_fill"]["full"], p["speed_by_fill"]["empty"])   # the price of a full window is shown

    def test_speed_falls_as_the_window_fills_and_the_range_is_reported(self):
        by = plan("ref_rtx4070s_12g_32g")["speed_by_fill"]
        self.assertGreater(by["empty"], by["typical"])
        self.assertGreater(by["typical"], by["full"])

    def test_deterministic(self):
        self.assertEqual(plan("rx9070xt_16g_32g"), plan("rx9070xt_16g_32g"))

    def test_unmeasured_bandwidth_is_flagged_and_lowers_confidence(self):
        hw = copy.deepcopy(HW["ref_rtx4070s_12g_32g"])
        del hw["memory"]["bandwidth_gbs"]
        p = planner.plan(hw, REAL)
        self.assertEqual(p["confidence"], "low")
        self.assertTrue(any("not measured" in w for w in p["warnings"]))


class FitToMemory(unittest.TestCase):
    """The user-facing promise: the plan follows the VRAM and RAM of the machine, aims for 35+ tok/s and gives up the window
    before the quality (docs/HARDWARE.md, "Selection policy")."""

    def test_12g_vram_and_32g_ram_gets_200k_at_target_speed(self):
        p = plan("ref_rtx4070s_12g_32g")
        self.assertEqual((p["quant"], p["settings"]["context"], p["speed_rung"]), ("IQ4_XS", 200000, "target"))
        self.assertGreaterEqual(p["speed_by_fill"]["typical"], 35)

    def test_12g_vram_and_16g_ram_keeps_the_window_and_steps_the_quant_down_to_q3(self):
        p = plan("rtx4070s_12g_16g")
        self.assertTrue(p["ok"])
        self.assertLessEqual(p["memory"]["ram_need"], p["memory"]["ram_budget"])      # never overcommits the small RAM
        self.assertIn(p["quant"], ("IQ4_XS", "Q3_K_XL"))                              # never below Q3-class quality
        self.assertGreaterEqual(p["settings"]["context"], 100000)
        self.assertEqual(p["speed_rung"], "target")

    def test_16g_vram_gets_a_bigger_window_than_12g(self):
        big, ref = plan("rtx4070tisuper_16g_32g"), plan("ref_rtx4070s_12g_32g")
        self.assertGreater(big["settings"]["context"], ref["settings"]["context"])
        self.assertEqual(big["speed_rung"], "target")

    def test_8g_vram_keeps_a_useful_window_even_when_35_tok_s_is_out_of_reach(self):
        p = plan("rtx4060_8g_32g")
        self.assertGreaterEqual(p["settings"]["context"], 65536)
        self.assertIn(p["speed_rung"], ("target", "comfort"))
        self.assertGreaterEqual(p["speed_by_fill"]["typical"], 20)

    def test_rx9070_16g_with_48g_ddr4_on_windows(self):
        # the first real person this was written for (Ryzen 7 5700X3D, 48 GB, RX 9070): upstream tier, the default quant
        p = plan("rx9070_16g_48g_windows")
        self.assertEqual((p["tier"], p["quant"], p["speed_rung"]), ("standard", "IQ4_XS", "target"))
        self.assertGreaterEqual(p["settings"]["context"], 200000)          # more VRAM and RAM than the 12 GB reference: not less window
        self.assertEqual(p["settings"]["kv_type"], "q4_0")                  # q8_0 would miss 35 tok/s at that window
        self.assertEqual(p["backend_candidates"][:2], ["rocm", "vulkan"])
        self.assertTrue(p["needs_probe"])
        self.assertGreaterEqual(p["speed_by_fill"]["typical"], 35)

    def test_rx9070_16g_with_48g_ddr4_on_fedora(self):
        # the same machine on Linux (how it will really be installed): upstream engine, a big window, q4_0 KV where q8_0 is too slow
        p = plan("rx9070_16g_48g_fedora")
        self.assertEqual((p["tier"], p["quant"], p["speed_rung"]), ("standard", "IQ4_XS", "target"))
        self.assertGreaterEqual(p["settings"]["context"], 200000)
        self.assertIn(p["settings"]["kv_type"], ("q4_0", "q8_0"))
        self.assertEqual(p["backend_candidates"][:2], ["rocm", "vulkan"])

    def test_the_kv_cache_is_as_precise_as_the_speed_target_allows(self):
        # a machine that reaches 35 tok/s with q8_0 keeps q8_0; one that only reaches it with q4_0 gets q4_0
        roomy = plan("rx9070xt_16g_32g", config={"contexts": (65536,), "tqp_enabled": False})
        self.assertEqual((roomy["settings"]["kv_type"], roomy["speed_rung"]), ("q8_0", "target"))
        tight = plan("rx9070_16g_48g_windows", config={"contexts": (262144,), "tqp_enabled": False})
        self.assertEqual(tight["settings"]["kv_type"], "q4_0")

    def test_never_below_the_catalog_quant_floor_unless_forced(self):
        for name in HW:
            p = plan(name)
            if p["ok"] and p["model"] == "tiel-coder-35b-a3b-mtp":
                self.assertNotIn(p["quant"], ("Q2_K_XL", "IQ3_XXS"), name)
        self.assertEqual(plan("rtx4060_8g_32g", model="tiel-coder-35b-a3b-mtp", quant="IQ3_XXS")["quant"], "IQ3_XXS")

    def test_a_window_nobody_ran_needs_a_speed_margin_and_vram_headroom(self):
        # the fork was verified at 200k: a 12 GB card stays at 200k, a card with room to spare may go beyond
        self.assertEqual(plan("ref_rtx4070s_12g_32g")["settings"]["context"], 200000)
        self.assertEqual(plan("rtx4090_24g_64g")["settings"]["context"], 262144)

    def test_the_vram_plan_leaves_the_desktop_reserve(self):
        p = plan("ref_rtx4070s_12g_32g")
        gpu = HW["ref_rtx4070s_12g_32g"]["gpus"][0]
        self.assertLessEqual(p["memory"]["vram_need"] + gpu["vram_used"], gpu["vram_total"] - 0.7 * GiB)


if __name__ == "__main__":
    unittest.main()
