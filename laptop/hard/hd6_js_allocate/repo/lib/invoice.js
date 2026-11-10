import { allocate } from "./allocate.js";

/**
 * applyDiscount(lines, discount) -> [{ ...line, discount, net }]
 *
 * lines    — [{ name, cents }], cents — неотрицательное целое; любое другое значение cents — RangeError.
 * discount — общая скидка в копейках, целое, 0 <= discount <= сумма строк; иначе RangeError.
 *
 * Скидка делится между строками пропорционально их сумме (через allocate), для каждой строки
 * discount — её доля скидки, net = cents - discount. Сумма net всегда равна (сумма строк - скидка).
 * Входные массив и объекты не изменяются, результат — новый массив новых объектов.
 * Пустой счёт или счёт на нулевую сумму со скидкой 0 допустимы (доли нулевые).
 */
export function applyDiscount(lines, discount) {
  const total = lines.reduce((s, l) => s + l.cents, 0);
  return lines.map((l) => {
    const share = Math.round((discount * l.cents) / total);
    return { ...l, discount: share, net: l.cents - share };
  });
}
