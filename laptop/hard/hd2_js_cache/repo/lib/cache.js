import { systemNow } from "./clock.js";

/**
 * Кэш «ключ -> значение» с лимитом по числу записей и временем жизни.
 *
 * new Cache({ max, ttl = 0, now = systemNow, onEvict })
 *   max     — максимум живых записей, положительное целое, иначе RangeError.
 *   ttl     — время жизни записи по умолчанию, мс; 0 — без срока.
 *   now     — функция, возвращающая текущее время в мс.
 *   onEvict — (key, value, reason) => void, вызывается при удалении самим кэшем:
 *             reason "expired" — запись протухла, "capacity" — вытеснена из-за лимита.
 *             При delete, clear и перезаписи ключа не вызывается. Протухшая запись удаляется (и onEvict с
 *             причиной "expired" вызывается) при обращении к ней через get/has/peek/delete, а также при set,
 *             когда нужно освободить место; size и keys тоже убирают протухшие записи.
 *
 * Запись, созданная в момент t с ttl, живёт, пока now() - t < ttl; при now() - t >= ttl она протухла.
 * Протухшие записи не считаются: не видны через get/has/peek/keys, не входят в size.
 *
 * get(key)        — значение или undefined; помечает запись как самую свежеиспользованную,
 *                   но срок жизни не продлевает.
 * peek(key), has(key) — не меняют порядок использования.
 * set(key, value, { ttl }?) — записывает; ttl записи (если задан) заменяет ttl по умолчанию (0 — без срока).
 *                   Перезапись ключа начинает срок жизни заново и делает запись самой свежей.
 *                   «Места нет» — когда физически хранимых записей (включая протухшие, которые ещё не убраны)
 *                   уже max. Тогда сначала удаляются все протухшие записи (для каждой вызывается onEvict с причиной
 *                   "expired"), и только если места всё равно нет — вытесняется давно всего не использованная
 *                   живая запись (причина "capacity").
 *                   Перезапись существующего ключа никого не вытесняет.
 * delete(key)     — true, если живая запись была удалена.
 * clear(), size   — очистка и число живых записей.
 * keys()          — ключи живых записей от давно всего не использованной к самой свежей.
 * getOrLoad(key, loader) — Promise значения: из кэша, если запись жива, иначе вызывает loader()
 *                   (может быть async) и кладёт результат в кэш. Попадание в кэш считается использованием
 *                   записи, как get (обновляет давность). Одновременные вызовы с тем же ключом делят один
 *                   вызов loader. Если loader упал, ошибка уходит всем ожидающим, в кэш ничего не
 *                   попадает, и следующий вызов пробует снова.
 */
export class Cache {
  constructor({ max, ttl = 0, now = systemNow, onEvict } = {}) {
    this.max = max;
    this.ttl = ttl;
    this.now = now;
    this.onEvict = onEvict;
    this.map = new Map(); // key -> { value, expires }
  }

  get(key) {
    const e = this.map.get(key);
    if (!e) return undefined;
    return e.value;
  }

  peek(key) {
    const e = this.map.get(key);
    return e ? e.value : undefined;
  }

  has(key) {
    return this.map.has(key);
  }

  set(key, value, opts = {}) {
    const ttl = opts.ttl ?? this.ttl;
    if (this.map.size >= this.max) {
      const oldest = this.map.keys().next().value;
      this.map.delete(oldest);
    }
    this.map.set(key, { value, expires: ttl ? this.now() + ttl : Infinity });
    return this;
  }

  delete(key) {
    return this.map.delete(key);
  }

  clear() {
    this.map.clear();
  }

  get size() {
    return this.map.size;
  }

  keys() {
    return [...this.map.keys()];
  }

  async getOrLoad(key, loader) {
    throw new Error("not implemented");
  }
}
