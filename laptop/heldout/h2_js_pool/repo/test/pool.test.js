import { test } from "node:test";
import assert from "node:assert/strict";
import { mapLimit } from "../lib/pool.js";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

test("keeps order", async () => {
  const out = await mapLimit([30, 5, 15], 3, async (ms) => {
    await sleep(ms);
    return ms * 2;
  });
  assert.deepEqual(out, [60, 10, 30]);
});
