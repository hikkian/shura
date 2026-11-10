import { test } from "node:test";
import assert from "node:assert/strict";
import { Cache } from "../lib/index.js";

test("set/get", () => {
  const c = new Cache({ max: 2 });
  c.set("a", 1).set("b", 2);
  assert.equal(c.get("a"), 1);
  assert.equal(c.size, 2);
});

test("lru eviction when full", () => {
  const c = new Cache({ max: 2 });
  c.set("a", 1).set("b", 2).set("c", 3);
  assert.equal(c.get("a"), undefined);
  assert.equal(c.get("c"), 3);
});

test("ttl", () => {
  let t = 0;
  const c = new Cache({ max: 5, ttl: 100, now: () => t });
  c.set("a", 1);
  t = 50;
  assert.equal(c.get("a"), 1);
});
