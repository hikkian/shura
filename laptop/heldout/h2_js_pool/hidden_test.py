import pathlib
import subprocess
import unittest

JS = r"""
import assert from "node:assert/strict";
import { mapLimit } from "./lib/pool.js";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// порядок результатов + индекс вторым аргументом
{
  const delays = [40, 5, 25, 1, 15];
  const seen = [];
  const out = await mapLimit(delays, 2, async (d, i) => { await sleep(d); seen.push(i); return `${i}:${d}`; });
  assert.deepEqual(out, ["0:40", "1:5", "2:25", "3:1", "4:15"]);
  assert.deepEqual([...seen].sort(), [0, 1, 2, 3, 4]);
}
// лимит соблюдается и достигается
{
  let cur = 0, max = 0;
  const out = await mapLimit(Array.from({ length: 9 }, (_, i) => i), 3, async (x) => {
    cur++; max = Math.max(max, cur);
    await sleep(5 + (x % 3) * 4);
    cur--; return x * x;
  });
  assert.deepEqual(out, [0, 1, 4, 9, 16, 25, 36, 49, 64]);
  assert.ok(max <= 3 && max >= 2, "параллелизм: max=" + max);
}
// limit больше числа элементов, пустой массив, синхронные fn
assert.deepEqual(await mapLimit([1, 2], 10, async (x) => x + 1), [2, 3]);
assert.deepEqual(await mapLimit([], 3, async (x) => x), []);
assert.deepEqual(await mapLimit([1, 2, 3], 2, (x) => x * 10), [10, 20, 30]);
// falsy-результаты не теряются
assert.deepEqual(await mapLimit([0, 1, 2], 2, async (x) => (x === 1 ? undefined : x === 2 ? null : 0)), [0, undefined, null]);
// limit = 1 — строго по очереди
{
  const log = [];
  await mapLimit([20, 1, 10], 1, async (d, i) => { log.push("s" + i); await sleep(d); log.push("e" + i); });
  assert.deepEqual(log, ["s0", "e0", "s1", "e1", "s2", "e2"]);
}
// невалидный limit -> отклонённый промис с RangeError
for (const bad of [0, -1, 1.5, NaN, "2", undefined]) {
  await assert.rejects(() => mapLimit([1, 2], bad, async (x) => x), RangeError, "limit=" + String(bad));
}
// ошибка пробрасывается, новые задания не стартуют
{
  const started = [];
  const boom = new Error("boom");
  await assert.rejects(
    () => mapLimit([0, 1, 2, 3, 4, 5, 6, 7], 2, async (x) => {
      started.push(x);
      await sleep(5);
      if (x === 1) throw boom;
      return x;
    }),
    (e) => e === boom,
  );
  const n = started.length;
  await sleep(60);
  assert.equal(started.length, n, "после ошибки запустились новые задания");
  assert.ok(n <= 4, "слишком много стартов: " + n);
}
// ошибка в синхронной части fn тоже отклоняет промис
await assert.rejects(() => mapLimit([1], 1, () => { throw new TypeError("x"); }), TypeError);
console.log("OK");
"""


class H(unittest.TestCase):
    def test_js(self):
        pathlib.Path("_hidden.mjs").write_text(JS)
        r = subprocess.run(["node", "_hidden.mjs"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, (r.stderr or r.stdout)[-600:])
