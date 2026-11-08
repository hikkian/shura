import csv, os, random, subprocess, sys, tempfile, unittest
from decimal import Decimal

import report


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "region", "product", "quantity", "unit_price"])
        w.writerows(rows)


SMALL = [
    ("2023-11-03", "Север", "Лампа, настольная", 2, "1500.50"),
    ("2023-11-20", "Юг", 'Стол "Модерн"', 1, "20000"),
    ("2023-12-01", "Север", "Лампа, настольная", 1, "1500.50"),
    ("2023-12-15", "Юг", "Стул", 4, "3000"),
    ("2023-12-31", "Запад", "Диван", 1, "45000.99"),
    ("2024-01-02", "Север", "Стул", 2, "3000"),
    ("2024-01-10", "Запад", "Книга", 10, "250.25"),
]
# total: 3001 + 20000 + 1500.5 + 12000 + 45000.99 + 6000 + 2502.5 = 90004.99
# regions: Север 3001+1500.5+6000=10501.5 ; Юг 32000 ; Запад 47503.49
# products: Диван 45000.99; Стул 18000; Стол 20000; Лампа 4501.5; Книга 2502.5 -> top3 Диван, Стол, Стул
# months: 2023-11: 23001 ; 2023-12: 58501.49 ; 2024-01: 8502.5 -> 2023-12


class H(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_small_exact(self):
        p = os.path.join(self.d, "s.csv")
        write(p, SMALL)
        r = report.summarize(p)
        self.assertAlmostEqual(float(r["total"]), 90004.99, places=2)
        self.assertEqual(set(r["by_region"]), {"Север", "Юг", "Запад"})
        self.assertAlmostEqual(float(r["by_region"]["Север"]), 10501.5, places=2)
        self.assertAlmostEqual(float(r["by_region"]["Юг"]), 32000.0, places=2)
        self.assertAlmostEqual(float(r["by_region"]["Запад"]), 47503.49, places=2)
        top = [(n, float(v)) for n, v in r["top_products"]]
        self.assertEqual([n for n, _ in top], ["Диван", 'Стол "Модерн"', "Стул"])
        self.assertAlmostEqual(top[0][1], 45000.99, places=2)
        self.assertAlmostEqual(top[2][1], 18000.0, places=2)
        self.assertEqual(r["best_month"], "2023-12")

    def test_random_big(self):
        rnd = random.Random(99)
        regs = ["A", "Б", "C", "Д"]
        prods = [f"Товар {i}, шт" for i in range(9)]
        rows = []
        for _ in range(500):
            rows.append((f"2024-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}", rnd.choice(regs),
                         rnd.choice(prods), rnd.randint(1, 9), f"{rnd.randint(100, 99999) / 100:.2f}"))
        p = os.path.join(self.d, "b.csv")
        write(p, rows)
        tot = Decimal(0)
        reg, pr, mo = {}, {}, {}
        for d, rg, pd, q, u in rows:
            v = Decimal(q) * Decimal(u)
            tot += v
            reg[rg] = reg.get(rg, 0) + v
            pr[pd] = pr.get(pd, 0) + v
            mo[d[:7]] = mo.get(d[:7], 0) + v
        r = report.summarize(p)
        self.assertAlmostEqual(float(r["total"]), float(tot), places=2)
        for k, v in reg.items():
            self.assertAlmostEqual(float(r["by_region"][k]), float(v), places=2)
        want = sorted(pr.items(), key=lambda kv: -kv[1])[:3]
        self.assertEqual([n for n, _ in r["top_products"]], [n for n, _ in want])
        for (n, v), (_, w) in zip(r["top_products"], want):
            self.assertAlmostEqual(float(v), float(w), places=2)
        self.assertEqual(r["best_month"], max(mo, key=mo.get))

    def test_cli(self):
        p = os.path.join(self.d, "s.csv")
        write(p, SMALL)
        out = subprocess.run([sys.executable, "report.py", p], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        txt = out.stdout
        flat = txt.replace(" ", "").replace(" ", "").replace(",", "").replace("_", "")
        self.assertIn("90004", flat)
        self.assertIn("2023-12", txt)
        for name in ("Север", "Юг", "Запад", "Диван", "Стул"):
            self.assertIn(name, txt)

    def test_shipped_file(self):
        # на реальной выгрузке функция работает и результат согласован
        r = report.summarize("sales.csv")
        self.assertAlmostEqual(sum(float(v) for v in r["by_region"].values()), float(r["total"]), places=1)
        self.assertEqual(len(r["top_products"]), 3)
        vals = [float(v) for _, v in r["top_products"]]
        self.assertEqual(vals, sorted(vals, reverse=True))


if __name__ == "__main__":
    unittest.main()
