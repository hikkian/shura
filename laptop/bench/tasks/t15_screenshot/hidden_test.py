import json, re, subprocess, sys, unittest

import grades


def run_main():
    return subprocess.run([sys.executable, "main.py"], capture_output=True, text=True, timeout=30)


class H(unittest.TestCase):
    def test_main_runs(self):
        r = run_main()
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertNotIn("Traceback", r.stderr)
        out = r.stdout
        for name in ("Айдана", "Бекзат", "Данияр", "Томирис", "Ерлан"):
            self.assertIn(name, out)

    def test_values(self):
        out = run_main().stdout
        want = {"Айдана": 4.75, "Бекзат": 3.6, "Томирис": 4.67, "Ерлан": 2.67}
        for name, v in want.items():
            line = next((ln for ln in out.splitlines() if name in ln), "")
            nums = [float(x.replace(",", ".")) for x in re.findall(r"\d+[.,]\d+", line)]
            self.assertTrue(any(abs(n - v) < 0.01 for n in nums), f"{name}: {line!r}")

    def test_order(self):
        out = run_main().stdout
        pos = {n: out.find(n) for n in ("Айдана", "Томирис", "Бекзат", "Ерлан")}
        self.assertTrue(all(p >= 0 for p in pos.values()))
        self.assertLess(pos["Айдана"], pos["Томирис"])
        self.assertLess(pos["Томирис"], pos["Бекзат"])
        self.assertLess(pos["Бекзат"], pos["Ерлан"])

    def test_average_unit(self):
        self.assertAlmostEqual(grades.average([5, 4]), 4.5)
        try:
            v = grades.average([])
        except Exception as e:  # noqa
            self.fail("average([]) падает: %r" % e)
        self.assertTrue(v is None or v == 0)

    def test_other_inputs(self):
        rows = grades.build_rows([
            {"name": "A", "grades": []},
            {"name": "B", "grades": [5]},
            {"name": "C", "grades": [3, 3]},
        ])
        self.assertEqual([r[0] for r in rows][:2], ["B", "C"])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][2], "A")

    def test_empty_list(self):
        self.assertEqual(grades.build_rows([]), [])


if __name__ == "__main__":
    unittest.main()
