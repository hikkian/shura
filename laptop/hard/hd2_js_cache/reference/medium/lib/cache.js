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
 *             При delete, clear и перезаписи ключа не вызывается.
 *
 * Запись, созданная в момент t с ttl, живёт, пока now() - t < ttl; при now() - t >= ttl она протухла.
 * Протухшие записи не считаются: не видны через get/has/peek/keys, не входят в size.
 *
 * get(key)        — значение или undefined; помечает запись как самую свежеиспользованную,
 *                   но срок жизни не продлевает.
 * peek(key), has(key) — не меняют порядок использования.
 * set(key, value, { ttl }?) — записывает; ttl записи (если задан) заменяет ttl по умолчанию (0 — без срока).
 *                   Перезапись ключа начинает срок жизни заново и делает запись самой свежей.
 *                   Когда места нет, сначала удаляются протухшие записи, и только если этого
 *                   не хватило — вытесняется давно всего не использованная живая запись.
 *                   Перезапись существующего ключа никого не вытесняет.
 * delete(key)     — true, если живая запись была удалена.
 * clear(), size   — очистка и число живых записей.
 * keys()          — ключи живых записей от давно всего не использованной к самой свежей.
 * getOrLoad(key, loader) — Promise значения: из кэша, если запись жива, иначе вызывает loader()
 *                   (может быть async) и кладёт результат в кэш. Одновременные вызовы с тем же
 *                   ключом делят один вызов loader. Если loader упал, ошибка уходит всем ожидающим,
 *                   в кэш ничего не попадает, и следующий вызов пробует снова.
 */
export class Cache {
  constructor({ max, ttl = 0, now = systemNow, onEvict } = {}) {
        this.max = max;
    this.ttl = ttl;
    this.now = now;
    this.onEvict = onEvict;
    this.map = new Map(); // key -> { value, expires }; порядок вставки = от давнего к свежему
    this.inflight = new Map();
  }

  _dead(e) {
    return e.expires <= this.now();
  }

  _drop(key, reason) {
    const e = this.map.get(key);
    this.map.delete(key);
    if (reason === "capacity" && this.onEvict) this.onEvict(key, e.value, reason);
  }

  _live(key) {
    const e = this.map.get(key);
    if (!e) return undefined;
    if (this._dead(e)) {
      this._drop(key, "expired");
      return undefined;
    }
    return e;
  }

  _purge() {
    for (const [k, e] of [...this.map]) if (this._dead(e)) this._drop(k, "expired");
  }

  get(key) {
    const e = this._live(key);
    if (!e) return undefined;
    this.map.delete(key);
    this.map.set(key, e);
    return e.value;
  }

  peek(key) {
    const e = this._live(key);
    return e ? e.value : undefined;
  }

  has(key) {
    return this._live(key) !== undefined;
  }

  set(key, value, opts = {}) {
    const ttl = opts.ttl ?? this.ttl;
    this.map.delete(key);
    if (this.map.size >= this.max) this._purge();
    while (this.map.size >= this.max) this._drop(this.map.keys().next().value, "capacity");
    this.map.set(key, { value, expires: ttl ? this.now() + ttl : Infinity });
    return this;
  }

  delete(key) {
    if (!this._live(key)) return false;
    this.map.delete(key);
    return true;
  }

  clear() {
    this.map.clear();
  }

  get size() {
    this._purge();
    return this.map.size;
  }

  keys() {
    this._purge();
    return [...this.map.keys()];
  }

  getOrLoad(key, loader) {
    const v = this.get(key);
    if (v) return Promise.resolve(v);
    if (this.inflight.has(key)) return this.inflight.get(key);
    const p = Promise.resolve()
      .then(() => loader())
      .then((v) => {
        this.set(key, v);
        return v;
      })
      .finally(() => this.inflight.delete(key));
    this.inflight.set(key, p);
    return p;
  }
}
