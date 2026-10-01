FAMILY = {
    "name": "f03_lrucache", "lang": "python",
    "implement": {
        "prompt": """Write the module `ttlcache.py` with the class `TTLCache` (standard library only).

TTLCache(capacity, ttl, clock=time.monotonic)
  capacity: max number of live entries (int >= 1, otherwise ValueError); ttl: seconds an entry lives after it was LAST WRITTEN
  (number > 0, otherwise ValueError). `clock` is a function returning the current time as a float; always use it (never time.time).

  put(key, value)    store the value; the entry is now the most recently used and its expiry is clock() + ttl. If the cache is full
                     after removing expired entries, evict the least recently used one. Re-putting an existing key replaces
                     the value, refreshes the expiry and does not evict anything.
  get(key, default=None)  the value if the key exists and has not expired, else `default`. A hit makes the entry the most
                     recently used (it does NOT extend its expiry). An expired entry is removed when it is found.
  delete(key) -> bool    remove the key; True if it was present and live, else False.
  __len__            the number of live (not expired) entries.
  __contains__(key)  True if the key is live. It does not change the recency order.
  keys()             the live keys from least to most recently used.
  stats()            a dict {"hits", "misses", "evictions", "expirations"}: get() counts a hit or a miss (a get of an expired entry
                     is a miss and counts one expiration); each LRU eviction counts one eviction; every expired entry removed by
                     any operation counts one expiration (count each entry once).
An entry is expired when clock() >= its expiry time.
""",
        "start": {"ttlcache.py": '"""A cache with a size limit and a time to live (to be written)."""\nimport time\n\n\nclass TTLCache:\n    def __init__(self, capacity, ttl, clock=time.monotonic):\n        raise NotImplementedError\n'},
        "solution": {"ttlcache.py": '''"""A cache with a size limit and a time to live."""
import time
from collections import OrderedDict


class TTLCache:
    def __init__(self, capacity, ttl, clock=time.monotonic):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be an int >= 1")
        if not isinstance(ttl, (int, float)) or ttl <= 0:
            raise ValueError("ttl must be > 0")
        self.capacity, self.ttl, self.clock = capacity, ttl, clock
        self._data = OrderedDict()          # key -> (value, expires_at); order: least -> most recently used
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "expirations": 0}

    def _purge(self):
        now = self.clock()
        for key in [k for k, (_, exp) in self._data.items() if now >= exp]:
            del self._data[key]
            self._stats["expirations"] += 1

    def put(self, key, value):
        self._purge()
        if key in self._data:
            del self._data[key]
        elif len(self._data) >= self.capacity:
            self._data.popitem(last=False)
            self._stats["evictions"] += 1
        self._data[key] = (value, self.clock() + self.ttl)

    def get(self, key, default=None):
        entry = self._data.get(key)
        if entry is None:
            self._stats["misses"] += 1
            return default
        if self.clock() >= entry[1]:
            del self._data[key]
            self._stats["expirations"] += 1
            self._stats["misses"] += 1
            return default
        self._data.move_to_end(key)
        self._stats["hits"] += 1
        return entry[0]

    def delete(self, key):
        self._purge()
        return self._data.pop(key, None) is not None

    def __len__(self):
        self._purge()
        return len(self._data)

    def __contains__(self, key):
        entry = self._data.get(key)
        if entry is None:
            return False
        if self.clock() >= entry[1]:
            del self._data[key]
            self._stats["expirations"] += 1
            return False
        return True

    def keys(self):
        self._purge()
        return list(self._data)

    def stats(self):
        return dict(self._stats)
'''},
    },
    "hidden": {"test_ttlcache.py": '''import unittest
from ttlcache import TTLCache


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class Basics(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.c = TTLCache(3, 10, clock=self.clock)

    def test_validation(self):
        for cap in (0, -1, 1.5, "3", None):
            with self.assertRaises(ValueError):
                TTLCache(cap, 1)
        for ttl in (0, -5, None, "1"):
            with self.assertRaises(ValueError):
                TTLCache(1, ttl)

    def test_put_get_default(self):
        self.c.put("a", 1)
        self.assertEqual(self.c.get("a"), 1)
        self.assertIsNone(self.c.get("zz"))
        self.assertEqual(self.c.get("zz", 7), 7)
        self.assertEqual(len(self.c), 1)

    def test_lru_eviction_order(self):
        for k in "abc":
            self.c.put(k, k)
        self.c.get("a")                       # a is now most recent; b is least recent
        self.c.put("d", "d")
        self.assertNotIn("b", self.c)
        self.assertEqual(self.c.keys(), ["c", "a", "d"])
        self.assertEqual(self.c.stats()["evictions"], 1)

    def test_reput_does_not_evict_and_replaces(self):
        for k in "abc":
            self.c.put(k, k)
        self.c.put("a", "A")
        self.assertEqual(len(self.c), 3)
        self.assertEqual(self.c.get("a"), "A")
        self.assertEqual(self.c.stats()["evictions"], 0)
        self.assertEqual(self.c.keys(), ["b", "c", "a"])

    def test_contains_does_not_touch_recency(self):
        for k in "abc":
            self.c.put(k, k)
        self.assertIn("a", self.c)
        self.assertEqual(self.c.keys(), ["a", "b", "c"])

    def test_delete(self):
        self.c.put("a", 1)
        self.assertTrue(self.c.delete("a"))
        self.assertFalse(self.c.delete("a"))
        self.assertEqual(len(self.c), 0)


class Expiry(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.c = TTLCache(3, 10, clock=self.clock)

    def test_expires_at_exactly_ttl(self):
        self.c.put("a", 1)
        self.clock.t += 9.999
        self.assertEqual(self.c.get("a"), 1)
        self.clock.t += 0.001
        self.assertIsNone(self.c.get("a"))

    def test_get_does_not_extend_life_but_put_does(self):
        self.c.put("a", 1)
        self.clock.t += 6
        self.c.get("a")
        self.clock.t += 5
        self.assertIsNone(self.c.get("a"))
        self.c.put("b", 2)
        self.clock.t += 6
        self.c.put("b", 3)
        self.clock.t += 6
        self.assertEqual(self.c.get("b"), 3)

    def test_expired_entries_do_not_count_towards_capacity(self):
        self.c.put("a", 1)
        self.c.put("b", 2)
        self.clock.t += 10
        self.c.put("c", 3)
        self.c.put("d", 4)
        self.c.put("e", 5)
        self.assertEqual(self.c.stats()["evictions"], 0)
        self.assertEqual(sorted(self.c.keys()), ["c", "d", "e"])

    def test_len_and_keys_ignore_expired(self):
        self.c.put("a", 1)
        self.clock.t += 5
        self.c.put("b", 2)
        self.clock.t += 6
        self.assertEqual(len(self.c), 1)
        self.assertEqual(self.c.keys(), ["b"])

    def test_stats(self):
        self.c.put("a", 1)
        self.c.get("a")
        self.c.get("x")
        self.clock.t += 10
        self.c.get("a")                       # expired: a miss and an expiration
        self.assertEqual(self.c.stats(), {"hits": 1, "misses": 2, "evictions": 0, "expirations": 1})
        self.c.put("b", 1)
        self.clock.t += 10
        len(self.c)
        len(self.c)                           # the entry is counted once
        self.assertEqual(self.c.stats()["expirations"], 2)

    def test_delete_of_expired_is_false(self):
        self.c.put("a", 1)
        self.clock.t += 11
        self.assertFalse(self.c.delete("a"))
'''},
    "bugfix": {
        "symptom": "A small cache (ttlcache.py) evicts the wrong entry: after reading key \"a\" it should become the most recently used one, but when the cache is full the next put() still removes \"a\" first.",
        "bug": [("ttlcache.py", '''        self._data.move_to_end(key)
        self._stats["hits"] += 1''', '''        self._stats["hits"] += 1''')],
        "repro": {"test_repro.py": 'import unittest\nfrom ttlcache import TTLCache\n\n\nclass Repro(unittest.TestCase):\n    def test_get_refreshes_recency(self):\n        c = TTLCache(2, 100, clock=lambda: 0.0)\n        c.put("a", 1)\n        c.put("b", 2)\n        c.get("a")\n        c.put("c", 3)\n        self.assertIn("a", c)\n        self.assertNotIn("b", c)\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
