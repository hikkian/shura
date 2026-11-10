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
export function allocate(total, weights) {
  if (!Number.isInteger(total)) throw new RangeError("total must be an integer");
  if (!Array.isArray(weights) || weights.length === 0) throw new RangeError("no weights");
  const sum = weights.reduce((a, b) => a + b, 0);
  if (!(sum > 0)) throw new RangeError("bad weights");
  const sign = total < 0 ? -1 : 1;
  const t = Math.abs(total);
  const exact = weights.map((w) => (t * w) / sum);
  const parts = exact.map(Math.floor);
  let left = t - parts.reduce((a, b) => a + b, 0);
  const order = exact.map((e, i) => i).sort((i, j) => exact[j] - Math.floor(exact[j]) - (exact[i] - Math.floor(exact[i])) || i - j);
  for (const i of order) {
    if (left <= 0) break;
    parts[i]++;
    left--;
  }
  return parts.map((p) => sign * p);
}
