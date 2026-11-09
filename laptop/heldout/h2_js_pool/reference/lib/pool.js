/**
 * mapLimit(items, limit, fn) -> Promise<Array>
 *
 * Применяет async-функцию fn(item, index) к каждому элементу items,
 * держа одновременно не больше `limit` выполняющихся заданий.
 * Результаты возвращаются в порядке items (а не в порядке завершения).
 * Если какое-то задание упало, возвращаемый промис отклоняется этой же ошибкой,
 * и новые задания больше не запускаются.
 * limit должен быть целым числом >= 1, иначе промис отклоняется с RangeError.
 */
export async function mapLimit(items, limit, fn) {
  if (!Number.isInteger(limit) || limit < 1) throw new RangeError("limit must be an integer >= 1");
  const list = Array.from(items);
  const results = new Array(list.length);
  let next = 0;
  let failed = false;
  async function worker() {
    while (!failed && next < list.length) {
      const i = next++;
      try {
        results[i] = await fn(list[i], i);
      } catch (e) {
        failed = true;
        throw e;
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, list.length) }, worker));
  return results;
}
