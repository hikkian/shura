import { test } from "node:test";
import assert from "node:assert/strict";
import { chunk } from "../lib/chunk.js";
test("chunks", () => {
  assert.deepEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]);
});
