import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "agentic-eval"))
from gate.gate import (analyze_counters, clopper_pearson, fuse_counter,
                       simulate_switch_counter, validate_switches)
import proxy
import run_experiment


class SwitchCounterTests(unittest.TestCase):
    def test_reference_exact_bounds(self):
        self.assertAlmostEqual(clopper_pearson(0, 100)[1], .029513, places=5)
        self.assertAlmostEqual(clopper_pearson(1, 157)[1], .029858, places=5)
        self.assertAlmostEqual(clopper_pearson(2, 208)[1], .029957, places=5)
        self.assertAlmostEqual(clopper_pearson(0, 12)[1], .220922, places=5)
        self.assertAlmostEqual(clopper_pearson(0, 14)[1], .192636, places=5)

    def test_declarations_and_family_union_count(self):
        pre = {"switch": {"id": "brake", "condition": {"type": "probe", "name": "brake"},
                          "slices": [{"kind": "code"}], "unit": "family", "budgets": {"code": .03}},
               "rows": [{"name": "code", "delta": .03}]}
        validate_switches(pre)
        rows = []
        for family in range(12):
            for task in range(4):
                rows.append({"arm": "stack", "family": str(family), "task": f"{family}-{task}",
                             "slice": {"kind": "code"},
                             "switches": {"brake": {"fired": family == 0}}})
        got = analyze_counters(rows, pre)["code"]
        self.assertEqual((got["n_units"], got["fired_units"]), (12, 1))
        self.assertFalse(got["closed"])

    def test_live_counter_format_and_fuse(self):
        recs = [{"switch": "s", "mode": "live", "unit": f"u{i}",
                 "slice": {"lang": "ru", "project": "p", "kind": "chat"},
                 "fired": i < 4, "first_fire_token": 1234 if i < 4 else -1,
                 "requests": 7, "source": "live"} for i in range(20)]
        got = fuse_counter(recs, "s", .03, checks=5, look=1)
        self.assertEqual(got["n_units"], 20)
        self.assertEqual(got["fired_units"], 4)
        self.assertIn(got["action"], ("keep_live", "return_to_shadow"))

    def test_json_and_stream_metadata_are_preserved(self):
        payload = json.dumps({"timings": {"reasoning_tokens": 42,
                         "reasoning_budget": {"brake_fired": True, "budget_hit": False,
                                              "first_fire_token": 42, "shadow": True}},
                              "usage": {"completion_tokens_details": {"reasoning_tokens": 42}},
                              "choices": [{"finish_reason": "stop"}]}).encode()
        found = proxy.scan_usage(payload)
        self.assertEqual(found["reasoning_tokens"], 42)
        self.assertTrue(found["reasoning_budget"]["brake_fired"])
        stream = b"data: " + payload + b"\n\ndata: [DONE]\n\n"
        self.assertEqual(proxy.scan_usage(stream)["reasoning_budget"]["first_fire_token"], 42)

    def test_old_experiment_results_still_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text(json.dumps({"arm": "base", "task": "t", "family": "f", "kind": "bugfix"}) + "\n")
            old = run_experiment.load_done(path)[0]
            self.assertEqual(old["reasoning_tokens"], 0)
            self.assertEqual(old["switches"], {})
            self.assertEqual(old["slice"]["project"], "f")

    def test_simulation_reports_correlated_family_unit_rates(self):
        got = simulate_switch_counter(repeats=1000, families=100, seed=24)
        self.assertEqual(got["tasks_per_family"], 4)
        self.assertEqual(got["within_family_correlation"], 1.0)
        self.assertLessEqual(got["rates"]["0.03"]["row_closed_rate"], .05)
        self.assertEqual(got["rates"]["0.0"]["row_closed_rate"], 1.0)
        self.assertLessEqual(got["rates"]["0.03"]["fuse_rate"], .05)
