import { allocate } from "./allocate.js";

/**
 * applyDiscount(lines, discount) -> [{ ...line, discount, net }]
 *
 * lines    — [{ name, cents }], cents — неотрицательное целое.
 * discount — общая скидка в копейках, целое, 0 <= discount <= сумма строк; иначе RangeError.
 *
 * Скидка делится между строками пропорционально их сумме (через allocate), для каждой строки
 * discount — её доля скидки, net = cents - discount. Сумма net всегда равна (сумма строк - скидка).
 * Пустой счёт или счёт на нулевую сумму со скидкой 0 допустимы (доли нулевые).
 */
export function applyDiscount(lines, discount) {
  if (!Number.isSafeInteger(discount) || discount < 0) throw new RangeError("bad discount");
  for (const l of lines) if (!Number.isSafeInteger(l.cents) || l.cents < 0) throw new RangeError("bad cents");
  const total = lines.reduce((s, l) => s + l.cents, 0);
  if (discount > total) throw new RangeError("discount exceeds total");
  if (total === 0) return lines.map((l) => ({ ...l, discount: 0, net: l.cents }));
  const shares = allocate(discount, lines.map((l) => l.cents));
  return lines.map((l, i) => ({ ...l, discount: shares[i], net: l.cents - shares[i] }));
}
