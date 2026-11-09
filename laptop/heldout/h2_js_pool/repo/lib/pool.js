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
  const results = [];
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const item = items[next++];
      results.push(await fn(item));
    }
  }
  await Promise.all(Array.from({ length: limit }, worker));
  return results;
}
