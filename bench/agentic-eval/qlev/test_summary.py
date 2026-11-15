import unittest

from qlev.summary import summarize_pair


class SummaryTests(unittest.TestCase):
    def rows(self, arm_values):
        rows = []
        for arm, values in arm_values.items():
            for family_ix in range(5):
                for task_ix in range(2):
                    task = f"f{family_ix}-t{task_ix}"
                    for rep in (1, 2):
                        resolved, score, seconds = values
                        rows.append({"arm": arm, "task": task, "family": f"f{family_ix}", "rep": rep,
                                     "resolved": resolved, "score": score, "seconds": seconds,
                                     "reasoning_tokens": 120 if arm != "base" else 160,
                                     "timed_out": False})
        return rows

    def test_quality_gate_uses_paired_task_and_family_bootstraps(self):
        rows = self.rows({"base": (0, 0.5, 100.0), "S": (1, 1.0, 105.0)})
        result = summarize_pair(rows, "base", "S", bootstrap_samples=500, seed=4, minimum_reps=2)
        self.assertEqual(result["gate"], "pass")
        self.assertEqual(result["paired_tasks"], 10)
        self.assertEqual(result["metrics"]["score"]["task_bootstrap_95"][0], 0.5)
        self.assertIsNotNone(result["metrics"]["resolved"]["family_bootstrap_95"])
        self.assertEqual(result["metrics"]["reasoning_tokens"]["mean_delta"], -40)

    def test_speed_gate_requires_noninferiority_and_less_time(self):
        rows = self.rows({"base": (1, 0.8, 100.0), "D": (1, 0.8, 90.0)})
        result = summarize_pair(rows, "base", "D", bootstrap_samples=300, seed=5, minimum_reps=2)
        self.assertEqual(result["gate"], "pass")
        self.assertEqual(result["metrics"]["seconds"]["mean_delta"], -10)

    def test_gate_stays_inconclusive_without_two_repeats(self):
        rows = [{"arm": arm, "task": "a", "family": "f", "rep": 1, "resolved": True,
                 "score": 1, "seconds": 1, "reasoning_tokens": 1, "timed_out": False}
                for arm in ("base", "C")]
        result = summarize_pair(rows, "base", "C", bootstrap_samples=100, minimum_reps=2)
        self.assertIn("insufficient repeats", result["gate"])

    def test_fractional_baseline_rows_are_not_turned_into_flags(self):
        # the base arm is given as per-task means (0.5 = solved in one of two runs); the candidate as run flags
        rows = []
        for task in ("a", "b", "c", "d"):
            for rep in (1, 2):
                rows.append({"arm": "base", "task": task, "family": task, "rep": rep, "resolved": 0.5, "score": 0.5, "seconds": 100, "timed_out": 0.25})
                rows.append({"arm": "D", "task": task, "family": task, "rep": rep, "resolved": rep == 1, "score": 0.5, "seconds": 90, "timed_out": False})
        result = summarize_pair(rows, "base", "D", bootstrap_samples=200, seed=3, minimum_reps=2)
        self.assertEqual(result["baseline_means"]["resolved"], 0.5)
        self.assertEqual(result["baseline_means"]["timed_out"], 0.25)
        self.assertEqual(result["candidate_means"]["resolved"], 0.5)
        self.assertEqual(result["metrics"]["resolved"]["mean_delta"], 0.0)     # 0.5 against runs that average 0.5
        self.assertEqual(result["metrics"]["timed_out"]["mean_delta"], -0.25)

    def test_string_flags_and_numbers_are_read_as_before(self):
        from qlev.summary import _number
        self.assertEqual(_number({"resolved": True}, "resolved"), 1.0)
        self.assertEqual(_number({"resolved": False}, "resolved"), 0.0)
        self.assertEqual(_number({"timed_out": "true"}, "timed_out"), 1.0)
        self.assertEqual(_number({"resolved": 1}, "resolved"), 1.0)
        self.assertIsNone(_number({"resolved": None}, "resolved"))
        self.assertIsNone(_number({"resolved": "maybe"}, "resolved"))
        self.assertEqual(_number({"score": "0.75"}, "score"), 0.75)


if __name__ == "__main__":
    unittest.main()
