import pathlib
import subprocess
import unittest

PRE = r"""
import assert from "node:assert/strict";
import { Cache } from "./lib/cache.js";
let t = 0;
const now = () => t;
const mk = (o = {}) => new Cache({ now, ...o });
"""


def js(body):
    pathlib.Path("_hidden.mjs").write_text(PRE + body + "\nconsole.log('OK');\n")
    r = subprocess.run(["node", "_hidden.mjs"], capture_output=True, text=True, timeout=60)
    return r


class H(unittest.TestCase):
    def check(self, body):
        r = js(body)
        self.assertEqual(r.returncode, 0, (r.stderr or r.stdout)[-700:])

    def test_ttl_boundary(self):
        self.check("""
        const c = mk({ max: 5, ttl: 100 });
        c.set("a", 1);
        t = 99; assert.equal(c.get("a"), 1); assert.equal(c.has("a"), true);
        t = 100; assert.equal(c.get("a"), undefined); assert.equal(c.has("a"), false);
        """)

    def test_get_refreshes_recency_not_ttl(self):
        self.check("""
        const c = mk({ max: 2, ttl: 100 });
        c.set("a", 1); c.set("b", 2);
        t = 50; c.get("a");
        c.set("c", 3);                                   // вытесняется b, не a
        assert.deepEqual(c.keys(), ["a", "c"]);
        t = 100; assert.equal(c.get("a"), undefined);    // get срок не продлил
        assert.equal(c.get("c"), 3);                     // c создан в t=50, жив до 150
        """)

    def test_overwrite_resets_ttl_and_recency_without_eviction(self):
        self.check("""
        const c = mk({ max: 2, ttl: 100 });
        c.set("a", 1); c.set("b", 2);
        t = 60; c.set("a", 11);
        assert.deepEqual(c.keys(), ["b", "a"]);
        assert.equal(c.size, 2);
        t = 120; assert.equal(c.get("b"), undefined); assert.equal(c.get("a"), 11);
        t = 159; assert.equal(c.get("a"), 11);
        t = 160; assert.equal(c.get("a"), undefined);
        """)

    def test_expired_purged_before_lru_eviction(self):
        self.check("""
        const ev = [];
        const c = mk({ max: 3, onEvict: (k, v, r) => ev.push([k, r]) });
        c.set("old", 1, { ttl: 10 }); c.set("a", 2); c.set("b", 3);
        c.get("old");                                    // old — самый свежий по использованию
        t = 10;                                          // но он протух
        c.set("c", 4);                                   // место освобождает протухший, а не a
        assert.deepEqual(c.keys(), ["a", "b", "c"]);
        assert.deepEqual(ev, [["old", "expired"]]);
        c.set("d", 5);                                   // теперь вытесняется a
        assert.deepEqual(c.keys(), ["b", "c", "d"]);
        assert.deepEqual(ev, [["old", "expired"], ["a", "capacity"]]);
        """)

    def test_size_and_keys_count_only_live(self):
        self.check("""
        const c = mk({ max: 5 });
        c.set("a", 1, { ttl: 10 }); c.set("b", 2, { ttl: 20 }); c.set("c", 3);
        t = 15;
        assert.equal(c.size, 2);
        assert.deepEqual(c.keys(), ["b", "c"]);
        t = 25;
        assert.equal(c.size, 1);
        assert.equal(c.delete("b"), false);   // протухшая запись — не «удалена»
        assert.equal(c.delete("c"), true);
        assert.equal(c.size, 0);
        """)

    def test_has_peek_do_not_touch_order(self):
        self.check("""
        const c = mk({ max: 2 });
        c.set("a", 1); c.set("b", 2);
        assert.equal(c.has("a"), true); assert.equal(c.peek("a"), 1);
        c.set("c", 3);
        assert.deepEqual(c.keys(), ["b", "c"]);
        """)

    def test_per_entry_ttl(self):
        self.check("""
        const c = mk({ max: 5, ttl: 100 });
        c.set("short", 1, { ttl: 10 }); c.set("forever", 2, { ttl: 0 }); c.set("def", 3);
        t = 10; assert.equal(c.get("short"), undefined);
        t = 99; assert.equal(c.get("def"), 3);
        t = 100; assert.equal(c.get("def"), undefined);
        t = 1e9; assert.equal(c.get("forever"), 2);
        const d = mk({ max: 2 });                // без ttl по умолчанию — вечные, но можно задать на запись
        d.set("x", 1); d.set("y", 2, { ttl: 5 });
        t = 1e9 + 5; assert.equal(d.get("x"), 1); assert.equal(d.get("y"), undefined);
        """)

    def test_onevict_semantics(self):
        self.check("""
        const ev = [];
        const c = mk({ max: 2, ttl: 100, onEvict: (k, v, r) => ev.push([k, v, r]) });
        c.set("a", 1); c.set("a", 2);          // перезапись
        c.set("b", 3); c.delete("b"); c.set("b", 4);
        c.set("c", 5);                          // вытесняет a
        assert.deepEqual(ev, [["a", 2, "capacity"]]);
        t = 100;
        assert.equal(c.get("b"), undefined);
        assert.deepEqual(ev, [["a", 2, "capacity"], ["b", 4, "expired"]]);
        c.clear();
        assert.equal(ev.length, 2);
        """)

    def test_falsy_values_are_cached(self):
        self.check("""
        const c = mk({ max: 5 });
        for (const [k, v] of [["z", 0], ["e", ""], ["f", false], ["n", null]]) c.set(k, v);
        assert.equal(c.has("z"), true); assert.equal(c.has("n"), true);
        assert.equal(c.get("z"), 0); assert.equal(c.get("e"), ""); assert.equal(c.get("f"), false); assert.equal(c.get("n"), null);
        let calls = 0;
        for (let i = 0; i < 3; i++) assert.equal(await c.getOrLoad("q", async () => { calls++; return 0; }), 0);
        assert.equal(calls, 1);
        """)

    def test_getorload_dedupes_concurrent(self):
        self.check("""
        const c = mk({ max: 5, ttl: 100 });
        let calls = 0;
        const loader = async () => { calls++; await new Promise((r) => setTimeout(r, 20)); return "v"; };
        const rs = await Promise.all([c.getOrLoad("k", loader), c.getOrLoad("k", loader), c.getOrLoad("k", loader)]);
        assert.deepEqual(rs, ["v", "v", "v"]);
        assert.equal(calls, 1);
        assert.equal(await c.getOrLoad("k", loader), "v"); assert.equal(calls, 1);
        t = 100;
        assert.equal(await c.getOrLoad("k", loader), "v"); assert.equal(calls, 2);   // протухло — грузим заново
        // разные ключи не мешают друг другу
        const other = await Promise.all([c.getOrLoad("x", async () => 1), c.getOrLoad("y", async () => 2)]);
        assert.deepEqual(other, [1, 2]);
        """)

    def test_getorload_failure(self):
        self.check("""
        const c = mk({ max: 5 });
        let calls = 0;
        const bad = async () => { calls++; await new Promise((r) => setTimeout(r, 10)); throw new Error("boom"); };
        const rs = await Promise.allSettled([c.getOrLoad("k", bad), c.getOrLoad("k", bad)]);
        assert.deepEqual(rs.map((r) => r.status), ["rejected", "rejected"]);
        assert.equal(rs[0].reason.message, "boom");
        assert.equal(calls, 1);
        assert.equal(c.has("k"), false);
        assert.equal(await c.getOrLoad("k", async () => { calls++; return "ok"; }), "ok");
        assert.equal(calls, 2);
        // синхронное исключение в loader — тоже отклонённый промис, не зависший in-flight
        await assert.rejects(() => c.getOrLoad("s", () => { throw new TypeError("sync"); }), TypeError);
        assert.equal(await c.getOrLoad("s", async () => 7), 7);
        """)

    def test_getorload_counts_as_use(self):
        self.check("""
        const c = mk({ max: 2 });
        c.set("a", 1); c.set("b", 2);
        await c.getOrLoad("a", async () => 99);
        await c.getOrLoad("n", async () => 3);       // вытесняет b
        assert.deepEqual(c.keys(), ["a", "n"]);
        """)

    def test_invalid_max(self):
        self.check("""
        for (const bad of [0, -1, 1.5, NaN, "2", undefined])
          assert.throws(() => new Cache({ max: bad }), RangeError, String(bad));
        assert.doesNotThrow(() => new Cache({ max: 1 }));
        """)
