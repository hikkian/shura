import pathlib
import subprocess
import unittest

PRE = r"""
import assert from "node:assert/strict";
import { allocate, applyDiscount } from "./lib/index.js";

// независимый эталон: веса — десятичные строки без экспоненты, арифметика на BigInt
function oracle(total, ws) {
  const dec = ws.map((s) => (s.split(".")[1] || "").length);
  const k = Math.max(0, ...dec);
  const sc = ws.map((s) => { const [i, f = ""] = s.split("."); return BigInt(i + f.padEnd(k, "0")); });
  const W = sc.reduce((a, b) => a + b, 0n);
  const neg = total < 0;
  const n = BigInt(neg ? -total : total);
  const q = sc.map((x) => (n * x) / W), r = sc.map((x) => (n * x) % W);
  let left = n - q.reduce((a, b) => a + b, 0n);
  const idx = sc.map((_, i) => i).sort((a, b) => (r[a] === r[b] ? a - b : r[a] > r[b] ? -1 : 1));
  for (const i of idx) { if (left > 0n) { q[i]++; left--; } }
  return q.map((x) => (neg && x !== 0n ? -Number(x) : Number(x)));
}
function rng(seed) { // mulberry32
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const sum = (a) => a.reduce((x, y) => x + y, 0);
const eq = (a, b, m) => assert.deepStrictEqual(a, b, m);   // strict: различает 0 и -0
"""


class H(unittest.TestCase):
    def check(self, body):
        pathlib.Path("_hidden.mjs").write_text(PRE + body + "\nconsole.log('OK');\n")
        r = subprocess.run(["node", "_hidden.mjs"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, (r.stderr or r.stdout)[-800:])

    def test_basic_largest_remainder(self):
        self.check("""
        eq(allocate(100, [1, 1, 1]), [34, 33, 33]);
        eq(allocate(10, [1, 2, 3]), [2, 3, 5]);
        eq(allocate(8, [1, 2, 4]), [1, 2, 5]);
        eq(allocate(10, [1, 1, 1, 1, 1, 1, 1]), [2, 2, 2, 1, 1, 1, 1]);
        eq(allocate(100, [50, 30, 20]), [50, 30, 20]);
        eq(allocate(7, [5]), [7]);
        eq(allocate(2, [1, 10]), [0, 2]);
        """)

    def test_ties_go_to_lower_index(self):
        self.check("""
        eq(allocate(5, [1, 1]), [3, 2]);
        eq(allocate(1, [1, 1, 1]), [1, 0, 0]);
        eq(allocate(2, [1, 1, 1]), [1, 1, 0]);
        eq(allocate(3, [2, 2, 2, 2]), [1, 1, 1, 0]);
        eq(allocate(1, [1, 3, 3]), [0, 1, 0]);      // 0.14, 0.43, 0.43 — побеждает больший остаток, при равных меньший индекс
        """)

    def test_zero_weights_get_zero(self):
        self.check("""
        eq(allocate(10, [0, 1, 0, 1]), [0, 5, 0, 5]);
        eq(allocate(1, [0, 0, 1]), [0, 0, 1]);
        eq(allocate(3, [0, 1, 1]), [0, 2, 1]);
        eq(allocate(5, [1, 0]), [5, 0]);
        """)

    def test_negative_total_is_mirror_without_minus_zero(self):
        self.check("""
        eq(allocate(-10, [1, 1, 1]), [-4, -3, -3]);
        eq(allocate(-1, [1, 1]), [-1, 0]);
        eq(allocate(0, [1, 2]), [0, 0]);
        eq(allocate(-5, [0, 1]), [0, -5]);
        eq(allocate(-100, [3, 3, 4]), [-30, -30, -40]);
        eq(allocate(-7, [1, 1, 1]), [-3, -2, -2]);
        eq(allocate(-0, [1, 1]), [0, 0]);
        """)

    def test_decimal_weights_exact(self):
        self.check("""
        eq(allocate(5, [0.1, 0.2, 0.7]), [1, 1, 3]);
        eq(allocate(100, [0.1, 0.1, 0.1]), [34, 33, 33]);
        eq(allocate(1, [0.1, 0.2, 0.3, 0.4]), [0, 0, 0, 1]);
        eq(allocate(3, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), [1, 1, 1, 0, 0, 0]);
        eq(allocate(10, [1e-7, 3e-7]), [3, 7]);
        eq(allocate(1, [0.7, 0.1, 0.2]), [1, 0, 0]);
        eq(allocate(3, [0.1, 0.2, 0.3, 0.4]), oracle(3, ["0.1", "0.2", "0.3", "0.4"]));
        eq(allocate(1000, [0.15, 0.35, 0.5]), [150, 350, 500]);
        eq(allocate(7, [0.3, 0.3, 0.3]), [3, 2, 2]);
        eq(allocate(1, [1e21, 1e21]), [1, 0]);
        """)

    def test_huge_totals_and_weights(self):
        self.check("""
        const M = Number.MAX_SAFE_INTEGER;
        eq(allocate(M, [1, 1, 1]), [3002399751580331, 3002399751580330, 3002399751580330]);
        for (const ws of [[3, 5, 7], [1, 2, 3, 4, 5, 6, 7], [M, M], [M, 1], [123456789, 987654321, 55555555]]) {
          const o = oracle(M, ws.map(String)), a = allocate(M, ws);
          eq(a, o, JSON.stringify(ws)); assert.equal(sum(a), M);
          const b = allocate(-M, ws);
          eq(b, o.map((x) => (x === 0 ? 0 : -x))); assert.equal(sum(b), -M);
        }
        eq(allocate(M, [1e15, 2e15]), oracle(M, ["1000000000000000", "2000000000000000"]));
        """)

    def test_property_integer_weights(self):
        self.check("""
        const R = rng(12345);
        for (let c = 0; c < 3000; c++) {
          const n = 1 + Math.floor(R() * 8);
          const big = R() < 0.3;
          const ws = Array.from({ length: n }, () => (R() < 0.15 ? 0 : 1 + Math.floor(R() * (big ? 1e9 : 20))));
          if (sum(ws) === 0) ws[0] = 1;
          const total = Math.floor((R() - 0.3) * (R() < 0.3 ? 1e15 : 1e5));
          const a = allocate(total, ws);
          assert.equal(a.length, n);
          assert.equal(sum(a), total, `sum ${JSON.stringify([total, ws])} -> ${a}`);
          eq(a, oracle(total, ws.map(String)), JSON.stringify([total, ws]));
          ws.forEach((w, i) => { if (w === 0) assert.ok(a[i] === 0, "zero weight got money"); });
        }
        """)

    def test_property_decimal_weights(self):
        self.check("""
        const R = rng(777);
        for (let c = 0; c < 2000; c++) {
          const n = 1 + Math.floor(R() * 6);
          const strs = Array.from({ length: n }, () => (R() < 0.1 ? "0" : (Math.floor(R() * 2000) / 100).toFixed(R() < 0.5 ? 2 : 1)));
          if (strs.every((s) => Number(s) === 0)) strs[0] = "0.1";
          const ws = strs.map(Number);
          const total = Math.floor((R() - 0.4) * 1e6);
          const a = allocate(total, ws);
          assert.equal(sum(a), total);
          eq(a, oracle(total, strs.map((s) => String(Number(s)))), JSON.stringify([total, strs]));
        }
        """)

    def test_invalid_arguments(self):
        self.check("""
        const M = Number.MAX_SAFE_INTEGER;
        for (const t of [1.5, NaN, Infinity, -Infinity, M + 1, -M - 1, "10", null, undefined, 1e300])
          assert.throws(() => allocate(t, [1, 1]), RangeError, "total=" + String(t));
        for (const w of [[], [0], [0, 0, 0], [-1, 2], [1, NaN], [Infinity, 1], [1, "2"], [1, undefined], null, undefined, "12"])
          assert.throws(() => allocate(10, w), RangeError, "weights=" + String(w));
        assert.doesNotThrow(() => allocate(0, [0, 1]));
        assert.doesNotThrow(() => allocate(M, [1]));
        const w = [1, 2, 3]; const out = allocate(6, w);
        eq(w, [1, 2, 3]); assert.notEqual(out, w);
        """)

    def test_discount_rounds_correctly(self):
        self.check("""
        const lines = [{ name: "a", cents: 3333 }, { name: "b", cents: 3333 }, { name: "c", cents: 3334 }];
        const r = applyDiscount(lines, 1000);
        eq(r.map((x) => x.discount), [333, 333, 334]);
        eq(r.map((x) => x.net), [3000, 3000, 3000]);
        eq(r.map((x) => x.name), ["a", "b", "c"]);
        eq(lines[0], { name: "a", cents: 3333 });                 // вход не изменён
        const r2 = applyDiscount([{ cents: 100 }, { cents: 100 }, { cents: 100 }], 100);
        eq(r2.map((x) => x.discount), [34, 33, 33]);
        eq(r2.map((x) => x.net), [66, 67, 67]);
        """)

    def test_discount_edges(self):
        self.check("""
        const L = [{ name: "x", cents: 500 }, { name: "y", cents: 0 }, { name: "z", cents: 250 }];
        eq(applyDiscount(L, 0).map((x) => x.net), [500, 0, 250]);
        eq(applyDiscount(L, 750).map((x) => x.net), [0, 0, 0]);
        eq(applyDiscount(L, 750).map((x) => x.discount), [500, 0, 250]);
        eq(applyDiscount(L, 1).map((x) => x.discount), [1, 0, 0]);
        assert.throws(() => applyDiscount(L, 751), RangeError);
        assert.throws(() => applyDiscount(L, -1), RangeError);
        assert.throws(() => applyDiscount(L, 1.5), RangeError);
        assert.throws(() => applyDiscount([{ cents: -5 }, { cents: 10 }], 1), RangeError);
        eq(applyDiscount([], 0), []);
        eq(applyDiscount([{ name: "f", cents: 0 }], 0), [{ name: "f", cents: 0, discount: 0, net: 0 }]);
        assert.throws(() => applyDiscount([{ cents: 0 }], 1), RangeError);
        assert.throws(() => applyDiscount([], 5), RangeError);
        """)

    def test_discount_property(self):
        self.check("""
        const R = rng(99);
        for (let c = 0; c < 1500; c++) {
          const n = 1 + Math.floor(R() * 7);
          const lines = Array.from({ length: n }, (_, i) => ({ name: "n" + i, cents: R() < 0.15 ? 0 : Math.floor(R() * (R() < 0.5 ? 100 : 1e7)) }));
          const S = sum(lines.map((l) => l.cents));
          const D = Math.floor(R() * (S + 1));
          const r = applyDiscount(lines, D);
          assert.equal(sum(r.map((x) => x.discount)), D);
          assert.equal(sum(r.map((x) => x.net)), S - D, JSON.stringify([lines, D]));
          r.forEach((x, i) => { assert.ok(x.net >= 0 && x.net <= lines[i].cents); assert.ok(Number.isInteger(x.discount)); });
          if (S > 0) eq(r.map((x) => x.discount), oracle(D, lines.map((l) => String(l.cents))));
        }
        """)
