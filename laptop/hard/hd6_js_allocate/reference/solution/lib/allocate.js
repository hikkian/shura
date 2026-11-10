/**
 * allocate(total, weights) -> number[]
 *
 * Делит сумму `total` (целое число копеек) на части пропорционально `weights`.
 *
 * total    — целое число копеек, |total| <= Number.MAX_SAFE_INTEGER; может быть отрицательным (возврат).
 * weights  — непустой массив неотрицательных конечных чисел (целых или дробных вроде 0.1), хотя бы одно > 0.
 *
 * Метод наибольшего остатка: точная доля части i равна total * weights[i] / sum(weights); каждая часть
 * сначала получает целую часть своей точной доли, а недостающие копейки раздаются по одной частям с
 * наибольшим дробным остатком точной доли; при равных остатках раньше получает часть с меньшим индексом.
 * Для отрицательного total результат зеркальный: allocate(-n, w) == allocate(n, w) со сменой знака.
 *
 * Вес считается по своей десятичной записи (0.1 — ровно одна десятая), а не по двоичному представлению.
 * Результат — массив целых чисел (0, а не -0), сумма которого равна total.
 *
 * Неверные аргументы (нецелый или небезопасный total, пустой или некорректный список весов,
 * нулевая сумма весов) — RangeError.
 */
const bad = (m) => new RangeError(m);

// десятичная запись числа -> { digits: BigInt, exp: int }, value = digits * 10^exp
function dec(w) {
  const m = /^(\d+)(?:\.(\d+))?(?:e([+-]?\d+))?$/i.exec(String(w));
  if (!m) throw bad("bad weight " + String(w));
  const frac = m[2] || "";
  return { digits: BigInt(m[1] + frac), exp: Number(m[3] || 0) - frac.length };
}

export function allocate(total, weights) {
  if (typeof total !== "number" || !Number.isSafeInteger(total)) throw bad("total must be a safe integer");
  if (!Array.isArray(weights) || weights.length === 0) throw bad("weights must be a non-empty array");
  for (const w of weights) if (typeof w !== "number" || !Number.isFinite(w) || w < 0) throw bad("bad weight");
  const ds = weights.map(dec);
  const shift = Math.max(0, ...ds.map((d) => -d.exp));
  const scaled = ds.map((d) => d.digits * 10n ** BigInt(d.exp + shift));
  const W = scaled.reduce((a, b) => a + b, 0n);
  if (W === 0n) throw bad("sum of weights is zero");
  const n = BigInt(Math.abs(total));
  const q = scaled.map((s) => (n * s) / W);
  const r = scaled.map((s) => (n * s) % W);
  let left = n - q.reduce((a, b) => a + b, 0n);
  const order = weights.map((_, i) => i).sort((i, j) => (r[i] === r[j] ? i - j : r[i] > r[j] ? -1 : 1));
  for (const i of order) {
    if (left === 0n) break;
    q[i] += 1n;
    left -= 1n;
  }
  return q.map((x) => (total < 0 && x !== 0n ? -Number(x) : Number(x)));
}
