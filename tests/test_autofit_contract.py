"""INSTALL1c step 3: the contract of the autofit interface (installer/universal/autofit_api.py). Offline, deterministic.
Run: python3 -B -m unittest tests.test_autofit_contract"""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
from universal import autofit_api as A, detect as D  # noqa: E402

FX = ROOT / "tests/fixtures/detect"
HWD = ROOT / "tests/fixtures/hardware"
CAT = json.loads((ROOT / "catalog/models.json").read_text())


def hw(name):
    return json.loads((HWD / f"{name}.json").read_text())


def amd_rocm():
    """AMD card with ROCm 7.2.4 / gfx1101 as the extended detection reads it from tests/fixtures/detect/extra_amd_gfx1101."""
    h = hw("rx9070xt_16g_32g")
    h["gpus"][0]["backends"] = ["vulkan"]       # the static list must not decide: the detected driver does
    h["system"] = {"drivers": {
        "rocm": D.parse_rocm_version(FX / "extra_amd_gfx1101/rocm_version.txt"),
        "rocm_gfx": [D.parse_kfd_gfx(FX / "extra_amd_gfx1101/kfd_properties_node1.txt")]}}
    return h


def amd_no_rocm():
    h = amd_rocm()
    h["system"]["drivers"].update(rocm=None, rocm_gfx=[])
    return h


class BackendOrder(unittest.TestCase):
    def test_a_amd_with_rocm_starts_with_hip(self):
        h = amd_rocm()
        self.assertEqual(h["system"]["drivers"]["rocm"], "7.2.4-45")
        self.assertEqual(h["system"]["drivers"]["rocm_gfx"], ["gfx1101"])
        self.assertEqual(A.backend_order(h), ["hip", "vulkan", "cpu"])

    def test_b_amd_without_rocm_is_vulkan(self):
        self.assertEqual(A.backend_order(amd_no_rocm()), ["vulkan", "cpu"])

    def test_c_nvidia_is_cuda_first(self):
        self.assertEqual(A.backend_order(hw("rtx4070s_12g_16g")), ["cuda", "vulkan", "cpu"])

    def test_d_cpu_machine_is_cpu(self):
        self.assertEqual(A.backend_order(hw("laptop_cpu_16g")), ["cpu"])

    def test_engine_caps_filter_the_order_and_cpu_stays_last(self):
        self.assertEqual(A.backend_order(amd_rocm(), ["vulkan", "cpu"]), ["vulkan", "cpu"])
        self.assertEqual(A.backend_order(amd_rocm(), ["rocm"]), ["hip", "cpu"])
        self.assertEqual(A.backend_order(hw("rtx4070s_12g_16g"), []), ["cpu"])


class Output(unittest.TestCase):
    def test_e_choice_has_all_fields(self):
        c = A.choose(amd_rocm(), CAT, ["hip", "vulkan", "cpu"])
        self.assertTrue(c.ok, c.reasons)
        self.assertEqual(c.backend, "hip")
        for f in ("model_id", "quant", "backend", "mode", "reasons"):
            self.assertTrue(getattr(c, f), f)
        self.assertGreater(c.context, 0)
        self.assertIn(c.limiting, ("vram", "ram", "disk", "speed"))
        self.assertTrue(c.speed_range["low"] <= c.speed_range["mid"] <= c.speed_range["high"])
        self.assertFalse(c.speed_range["measured"])
        if c.alternative is None:
            self.assertTrue(any("альтернативы нет" in n for n in c.notes))
        else:
            for k in ("model_id", "quant", "backend", "context", "tok_s", "why_not"):
                self.assertIn(k, c.alternative)

    def test_alternative_is_next_backend_with_numbers(self):
        c = A.choose(amd_rocm(), CAT, ["hip", "vulkan", "cpu"])
        self.assertIsNotNone(c.alternative)
        self.assertEqual(c.alternative["backend"], "vulkan")
        self.assertIn("tok/s", c.alternative["why_not"])

    def test_cpu_machine_does_not_fail(self):
        c = A.choose(hw("laptop_cpu_16g"), CAT, ["cpu"])
        self.assertIsInstance(c, A.Choice)
        if c.ok:
            self.assertEqual(c.backend, "cpu")
        else:
            self.assertTrue(c.reasons)

    def test_no_suitable_model_is_a_reason_not_an_exception(self):
        tiny = hw("laptop_cpu_16g")
        tiny["memory"]["total"] = tiny["memory"]["available"] = 4 * 1024 ** 3
        c = A.choose(tiny, CAT, ["cpu"])
        self.assertFalse(c.ok)
        self.assertIsNone(c.model_id)
        self.assertTrue(c.reasons and "не подходит" in c.reasons[0])

    def test_no_disk_is_a_reason(self):
        h = hw("rtx4070s_12g_16g")
        h["disk"]["free"] = 5 * 1024 ** 3
        c = A.choose(h, CAT)
        self.assertFalse(c.ok)
        self.assertIn("диск", c.reasons[0])

    def test_bad_input_is_a_reason(self):
        c = A.choose({"schema": 1}, CAT)
        self.assertFalse(c.ok)
        self.assertIn("неверны", c.reasons[0])


class Reference(unittest.TestCase):
    def test_f_reference_is_never_chosen_even_if_it_fits_better(self):
        cat = copy.deepcopy(CAT)
        ref = copy.deepcopy(cat["reference"][0])
        ref["id"], ref["capability"] = "ref-super", 99        # the most capable model by far, and small enough to fit
        cat["reference"].append(ref)
        for h in (hw("rtx4070s_12g_16g"), hw("rx9070xt_16g_32g"), hw("laptop_cpu_16g")):
            c = A.choose(h, cat)
            self.assertNotIn(c.model_id, {r["id"] for r in cat["reference"]})
        # the same id also listed under models is dropped too
        cat["models"].append(copy.deepcopy(ref))
        self.assertNotEqual(A.choose(hw("rtx4070s_12g_16g"), cat).model_id, "ref-super")

    def test_only_reference_means_no_model(self):
        cat = copy.deepcopy(CAT)
        cat["reference"].append(copy.deepcopy(cat["models"][0]))
        cat["reference"][-1]["id"] = "r2"
        cat["models"] = []
        c = A.choose(hw("rtx4070s_12g_16g"), cat)
        self.assertFalse(c.ok)


class Determinism(unittest.TestCase):
    def test_g_same_input_same_output(self):
        for h in (amd_rocm(), hw("rtx4070s_12g_16g"), hw("laptop_cpu_16g")):
            a = A.choose(copy.deepcopy(h), copy.deepcopy(CAT), ["hip", "cuda", "vulkan", "cpu"])
            b = A.choose(copy.deepcopy(h), copy.deepcopy(CAT), ["hip", "cuda", "vulkan", "cpu"])
            self.assertEqual(a.to_dict(), b.to_dict())

    def test_input_is_not_mutated(self):
        h, c = amd_rocm(), copy.deepcopy(CAT)
        h0 = copy.deepcopy(h)
        A.choose(h, c)
        self.assertEqual((h, c), (h0, CAT))


class Calibration(unittest.TestCase):
    def test_h_calibrate_and_recalibrate_say_not_measured(self):
        h = amd_rocm()
        cal = A.calibrate(h)
        self.assertFalse(cal["measured"])
        self.assertEqual(cal["status"], "не измерено")
        c = A.recalibrate(h, CAT, ["hip", "vulkan", "cpu"], profile="fast")
        self.assertTrue(c.ok)
        self.assertFalse(c.speed_range["measured"])
        self.assertIn("не измерено", c.speed_range["status"])

    def test_recalibrate_with_a_new_profile_of_the_machine(self):
        weaker = amd_rocm()
        weaker["memory"]["total"] = 8 * 1024 ** 3
        weaker["memory"]["available"] = 6 * 1024 ** 3
        self.assertIsInstance(A.recalibrate(weaker, CAT), A.Choice)

    def test_measured_scale_is_applied_and_failing_runner_does_not_raise(self):
        h = amd_rocm()
        base = A.choose(h, CAT)
        cal = A.calibrate(h, base, runner=lambda *_: base.speed_range["mid"] / 2)
        self.assertTrue(cal["measured"])
        self.assertAlmostEqual(cal["speed_scale"], 0.5, places=2)
        scaled = A.recalibrate(h, CAT, calibration=cal)
        self.assertTrue(scaled.speed_range["measured"])
        self.assertLess(scaled.speed_range["mid"], base.speed_range["mid"])
        bad = A.calibrate(h, base, runner=lambda *_: 1 / 0)
        self.assertFalse(bad["measured"])
        self.assertEqual(bad["status"], "не измерено")


if __name__ == "__main__":
    unittest.main()
