import { test } from "node:test";
import assert from "node:assert/strict";
import { allocate, applyDiscount, fmt } from "../lib/index.js";

test("exact split", () => {
  assert.deepEqual(allocate(100, [1, 1]), [50, 50]);
  assert.deepEqual(allocate(90, [1, 2]), [30, 60]);
});

test("discount over lines", () => {
  const r = applyDiscount([{ name: "a", cents: 1000 }, { name: "b", cents: 3000 }], 400);
  assert.deepEqual(r.map((x) => x.net), [900, 2700]);
});

test("fmt", () => {
  assert.equal(fmt(1999), "19.99");
});
