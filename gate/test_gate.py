import unittest

from gate.gate import (analyze_row, exact_paired_sign_test, exact_randomization_test,
                       paired_bootstrap, simulate, validate_prereg, analyze_probes,
                       analyze_anchor, merge_waves, train_plan, append_decision,
                       family_resample_study, alpha_for_looks, verdict,
                       simulate_target_gate)
import tempfile
from pathlib import Path


class GateTests(unittest.TestCase):
    def test_paired_bootstrap_reproducible_and_zero_width_for_constants(self):
        self.assertEqual(paired_bootstrap([.2] * 8, reps=200, seed=9), (.2, .2))
        self.assertEqual(paired_bootstrap([1, 2, 3], reps=500, seed=7),
                         paired_bootstrap([1, 2, 3], reps=500, seed=7))

    def test_exact_paired_tests(self):
        self.assertEqual(exact_randomization_test([1, 1, 1]), .25)
        self.assertAlmostEqual(exact_paired_sign_test([1, 1, 1]), .25)
        self.assertEqual(exact_randomization_test([0, 0]), 1.0)

    def test_single_named_candidate_only(self):
        p = {"candidate": "c", "rows": [{"name": "quality", "delta": .1}]}
        self.assertIs(validate_prereg(p), p)
        with self.assertRaises(ValueError):
            validate_prereg({**p, "candidates": ["c", "d"]})

    def test_candidate_verdict_requires_proven_preregistered_target(self):
        rows = [{"name":"quality", "delta":.03, "target":True, "kind":"agent"},
                {"name":"language", "delta":.03, "kind":"agent"}]
        pre = {"candidate":"c", "rows":rows}
        no_target = verdict({"candidate":"c", "rows":[{**rows[0], "target":False}, rows[1]]},
                            [{"name":"quality", "zone":"green"},
                             {"name":"language", "zone":"green"}])
        self.assertEqual(no_target["overall"], "grey")
        unproven = verdict(pre, [{"name":"quality", "zone":"grey", "gain_eligible":True,
                                  "target_better":False, "lower_bound":-.01},
                                 {"name":"language", "zone":"green"}])
        self.assertEqual(unproven["overall"], "grey")
        proven = verdict(pre, [{"name":"quality", "zone":"green", "gain_eligible":True,
                                "target_better":True, "lower_bound":.01},
                               {"name":"language", "zone":"green"}])
        self.assertEqual(proven["overall"], "green")
        self.assertTrue(proven["target_proven"])

    def test_target_gate_operating_characteristics(self):
        got = simulate_target_gate(repeats=1000, n=80, seed=20261113)
        self.assertAlmostEqual(got["rates"]["aa_without_target"], 0)
        self.assertLessEqual(got["rates"]["aa_with_target"], got["alpha"])
        self.assertGreaterEqual(got["rates"]["harmful_red"], .99)
        self.assertGreater(got["rates"]["target_win_green"], .80)

    def test_simulation_repeatable_and_loss_passes_less(self):
        a = simulate(12, 2, .221, .12, trials=500, seed=17)
        b = simulate(12, 2, .221, .12, trials=500, seed=17)
        self.assertEqual(a, b)
        self.assertLess(a["-0.05"], a["+0.00"])

    def test_k_samples_require_common_seeds_and_return_task_flip_table(self):
        pre = {"candidate": "c", "control": "base", "bootstrap_reps": 100,
               "rows": [{"name": "r", "delta": .1, "n": 1, "k": 2,
                         "kind": "agent", "target": True}]}
        records = []
        for arm, score in (("base", .2), ("c", .4)):
            for rep, seed in ((0, 11), (1, 12)):
                records.append({"row":"r", "arm":arm, "task":"t", "rep":rep,
                                "seed":seed, "family":"f", "score":score})
        out = analyze_row(pre["rows"][0], records, pre)
        self.assertEqual(out["n"], 1)
        self.assertEqual(out["k"], 2)
        self.assertEqual(out["flips"][0]["direction"], "candidate_better")
        records[-1]["seed"] = 99
        with self.assertRaises(ValueError):
            analyze_row(pre["rows"][0], records, pre)

    def test_p0_probes_veto_and_early_stop(self):
        spec = {"profile": "deterministic", "checks": {
            "identity": {"metric": "identical", "op": "equals", "value": True},
            "kl": {"metric": "value", "op": "max", "limit": 0.01}}}
        got = analyze_probes(spec, [{"probe":"identity", "identical":True},
                                    {"probe":"kl", "value":0.02}])
        self.assertTrue(got["veto"])
        self.assertTrue(got["early_stop"])
        self.assertEqual(got["failed"], ["kl"])
        live = {"profile":"live", "checks":{"kl":{"limit":1}, "top8_changed":{"limit":1},
                "exposure":{"limit":1}, "expert_cache":{"limit":1}, "mtp_accept":{"limit":1},
                "speed_pairs":{"metric":"delta","noise_metric":"se","op":"min","limit":-.03}}}
        probes = [{"probe":k,"value":0,"delta":-.02,"se":.01} for k in live["checks"]]
        self.assertTrue(analyze_probes(live, probes)["veto"])

    def test_anchor_detects_regression_and_recommends_rollback(self):
        recs = []
        for t in ("a", "b", "c"):
            recs += [{"task":t,"arm":"stack","score":.2},
                     {"task":t,"arm":"tiel-deploy2","score":.5}]
        for r in recs:
            r["row"] = "code"
            r["family"] = "repo-a" if r["task"] in ("a", "b") else "repo-b"
        got = analyze_anchor(recs, "new", delta=.1, reps=100,
                             last_change_by_row={"code":"last-enabled-change"})
        self.assertEqual(got["zone"], "red")
        self.assertTrue(got["auto_rollback"])
        self.assertEqual(got["rows"]["code"]["distinct_families"], 2)
        self.assertEqual(got["rows"]["code"]["last_change_to_rollback"], "last-enabled-change")

    def test_wave_merge_requires_fresh_tasks(self):
        got = merge_waves([[{"task":"a"}], [{"task":"b"}]])
        self.assertEqual([r["wave"] for r in got], [1, 2])
        pre = {"candidate":"c", "control":"base", "looks":3, "look":2,
               "bootstrap_reps":100, "rows":[{"name":"r", "delta":.1, "n":2, "k":1}]}
        pooled = []
        for task in ("a", "b"):
            pooled.extend({"row":"r","task":task,"family":task,"arm":arm,"score":score}
                          for arm,score in (("base",.5),("c",.6)))
        out = analyze_row(pre["rows"][0], merge_waves([pooled[:2], pooled[2:]]), pre)
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["look"], 2)
        with self.assertRaises(ValueError):
            merge_waves([[{"task":"a"}], [{"task":"a"}]])

    def test_train_bisects_only_red_and_never_credits_speed(self):
        got = train_plan({"x":"red", "y":"grey", "z":"green"})
        self.assertEqual(got["red"], ["x"])
        self.assertEqual(got["grey"], ["y"])
        self.assertFalse(got["speed_credit_between_candidates"])
        self.assertEqual(got["grey_action"], "recheck whole train on fresh tasks")

    def test_look_alpha_is_split_in_advance(self):
        self.assertAlmostEqual(alpha_for_looks(3), .05 / 3)
        pre = {"candidate":"c", "control":"base", "looks":3, "look":3,
               "bootstrap_reps":100, "rows":[{"name":"r", "delta":.1, "n":1, "k":1}]}
        recs = [{"row":"r","task":"t","family":"f","arm":a,"score":v}
                for a,v in (("base",.5),("c",.5))]
        self.assertAlmostEqual(analyze_row(pre["rows"][0], recs, pre)["alpha_per_look"], .05/3)

    def test_family_resampling_is_seeded_and_reports_family_support(self):
        recs=[]
        for fam in ("a","b","c"):
            for t in range(4):
                recs.extend([{"arm":"base","task":f"{fam}{t}","family":fam,"rep":1,"score":.5},
                             {"arm":"base","task":f"{fam}{t}","family":fam,"rep":2,"score":.5}])
        args = dict(sizes=(6,), deltas=(.03,), effects=(0,), trials=12, seed=77)
        one = family_resample_study(recs, **args)
        self.assertEqual(one, family_resample_study(recs, **args))
        self.assertEqual(one[0]["families_available"], 3)

    def test_decision_log_append_and_sealed_access_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            append_decision(path, {"date":"2026-01-01", "overall":"grey"}, sealed_access=True, annual_limit=1)
            with self.assertRaises(ValueError):
                append_decision(path, {"date":"2026-05-01", "overall":"green"}, sealed_access=True, annual_limit=1)


if __name__ == "__main__":
    unittest.main()
