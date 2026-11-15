"""GEVAL1: the general gate and the combined verdict, on synthetic results (no model).
    cd bench/agentic-eval && python3 -B -m unittest rsi.test_accept_general
"""
import unittest

from rsi.accept_general import decide_with_general, general_gate


def results(arm, scores, families=None, seconds=100, tokens=1000, prefix="gq_fam"):
    out = []
    for i, s in enumerate(scores):
        fam = families[i] if families else f"{prefix}{i % 6}"
        out.append({"arm": arm, "task": f"{fam}-general-{i:02d}", "family": fam, "rep": 1, "score": s, "resolved": s >= 0.75,
                    "seconds": seconds, "tokens": tokens})
    return out


class Gate(unittest.TestCase):
    def test_no_regression_passes_when_equal(self):
        base = results("b", [0.8, 0.6, 1.0, 0.4, 0.9, 0.7, 0.5, 0.8, 0.6, 1.0, 0.3, 0.7])
        cand = results("c", [0.8, 0.6, 1.0, 0.4, 0.9, 0.7, 0.5, 0.8, 0.6, 1.0, 0.3, 0.7])
        g = general_gate(cand, base)
        self.assertTrue(g["ok"])
        self.assertEqual(g["pairs"], 12)

    def test_no_regression_fails_on_a_clear_drop(self):
        base = results("b", [1.0] * 12)
        cand = results("c", [0.5] * 12)
        self.assertFalse(general_gate(cand, base)["ok"])

    def test_improve_needs_a_gain_above_zero(self):
        base = results("b", [0.5] * 12)
        self.assertFalse(general_gate(results("c", [0.5] * 12), base, improve=True)["ok"])
        self.assertTrue(general_gate(results("c", [0.9] * 12), base, improve=True)["ok"])

    def test_unpaired_tasks_are_ignored(self):
        base = results("b", [0.5] * 6)
        cand = results("c", [0.5] * 12)
        self.assertEqual(general_gate(cand, base)["pairs"], 6)


class Combined(unittest.TestCase):
    def test_code_model_needs_code_gain_and_no_general_drop(self):
        code_b = results("b", [0.4] * 12, prefix="py_x")
        code_c = results("c", [0.8] * 12, prefix="py_x")
        gen_b = results("b", [0.7] * 12)
        v = decide_with_general(code_c, code_b, results("c", [0.7] * 12), gen_b, model_kind="code", allow_stub=True)
        self.assertTrue(v["accept"], v)
        v = decide_with_general(code_c, code_b, results("c", [0.3] * 12), gen_b, model_kind="code", allow_stub=True)
        self.assertFalse(v["accept"])
        self.assertTrue(v["primary"]["accept"])
        self.assertFalse(v["guard"]["ok"])

    def test_general_model_swaps_the_roles(self):
        code_b = results("b", [0.6] * 12, prefix="py_x")
        gen_b = results("b", [0.5] * 12)
        v = decide_with_general(results("c", [0.6] * 12, prefix="py_x"), code_b, results("c", [0.9] * 12), gen_b, model_kind="general", allow_stub=True)
        self.assertTrue(v["accept"], v)
        v = decide_with_general(results("c", [0.2] * 12, prefix="py_x"), code_b, results("c", [0.9] * 12), gen_b, model_kind="general", allow_stub=True)
        self.assertFalse(v["accept"])
        self.assertEqual(v["guard"]["role"], "guard (no code regression)")

    def test_stub_noise_is_declared(self):
        g = general_gate(results("c", [0.5] * 6), results("b", [0.5] * 6))
        self.assertIn("STUB", g["noise_source"])


if __name__ == "__main__":
    unittest.main()
