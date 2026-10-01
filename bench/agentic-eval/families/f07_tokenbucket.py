FAMILY = {
    "name": "f07_tokenbucket", "lang": "python",
    "implement": {
        "prompt": """Write the module `ratelimit.py` with the class `RateLimiter` (standard library only), a token-bucket limiter with one bucket per key.

RateLimiter(rate, burst, clock=time.monotonic)
  rate: tokens added per second (number > 0); burst: bucket capacity (number >= 1); otherwise ValueError. `clock` returns the
  current time in seconds as a float; always use it. A bucket starts FULL the first time a key is seen, and is refilled lazily:
  tokens = min(burst, tokens + rate * (now - last_seen)). If the clock goes backwards, treat the elapsed time as 0.

  allow(key, cost=1) -> bool   take `cost` tokens if the bucket has at least that many (return True), otherwise take nothing and
                               return False. cost must be a number > 0 and <= burst, otherwise ValueError.
  retry_after(key, cost=1) -> float   seconds until `cost` tokens will be available: 0.0 if available now, otherwise
                               (cost - tokens) / rate. Does not take tokens. A key never seen counts as a full bucket.
  tokens(key) -> float         the current number of tokens (after refilling) of the key; burst for an unknown key.
  reset(key=None)              forget one key (it becomes full again) or, with no argument, all keys.
  keys() -> list               the keys currently tracked, in the order they were first seen.
Fractions of a token are kept (do not round).
""",
        "start": {"ratelimit.py": '"""Token-bucket rate limiter (to be written)."""\nimport time\n\n\nclass RateLimiter:\n    def __init__(self, rate, burst, clock=time.monotonic):\n        raise NotImplementedError\n'},
        "solution": {"ratelimit.py": '''"""Token-bucket rate limiter."""
import time


class RateLimiter:
    def __init__(self, rate, burst, clock=time.monotonic):
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
            raise ValueError("rate must be > 0")
        if isinstance(burst, bool) or not isinstance(burst, (int, float)) or burst < 1:
            raise ValueError("burst must be >= 1")
        self.rate, self.burst, self.clock = rate, burst, clock
        self._b = {}                    # key -> [tokens, last_seen]

    def _bucket(self, key):
        now = self.clock()
        b = self._b.get(key)
        if b is None:
            b = self._b[key] = [float(self.burst), now]
        else:
            elapsed = max(0.0, now - b[1])
            b[0] = min(float(self.burst), b[0] + self.rate * elapsed)
            b[1] = max(b[1], now)
        return b

    def _check_cost(self, cost):
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost <= 0 or cost > self.burst:
            raise ValueError("cost must be > 0 and <= burst")

    def allow(self, key, cost=1):
        self._check_cost(cost)
        b = self._bucket(key)
        if b[0] >= cost:
            b[0] -= cost
            return True
        return False

    def retry_after(self, key, cost=1):
        self._check_cost(cost)
        if key not in self._b:
            return 0.0
        b = self._bucket(key)
        return 0.0 if b[0] >= cost else (cost - b[0]) / self.rate

    def tokens(self, key):
        if key not in self._b:
            return float(self.burst)
        return self._bucket(key)[0]

    def reset(self, key=None):
        if key is None:
            self._b.clear()
        else:
            self._b.pop(key, None)

    def keys(self):
        return list(self._b)
'''},
    },
    "hidden": {"test_ratelimit.py": '''import unittest
from ratelimit import RateLimiter


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Limiter(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.rl = RateLimiter(rate=2, burst=5, clock=self.clock)

    def test_validation(self):
        for rate in (0, -1, None, "1"):
            with self.assertRaises(ValueError):
                RateLimiter(rate, 5)
        for burst in (0, 0.5, None):
            with self.assertRaises(ValueError):
                RateLimiter(1, burst)

    def test_starts_full_then_runs_dry(self):
        self.assertEqual([self.rl.allow("a") for _ in range(7)], [True] * 5 + [False] * 2)

    def test_refill_over_time(self):
        for _ in range(5):
            self.rl.allow("a")
        self.assertFalse(self.rl.allow("a"))
        self.clock.t += 1.0                      # 2 tokens
        self.assertTrue(self.rl.allow("a"))
        self.assertTrue(self.rl.allow("a"))
        self.assertFalse(self.rl.allow("a"))

    def test_refill_is_capped_at_burst(self):
        self.rl.allow("a")
        self.clock.t += 1000
        self.assertEqual(self.rl.tokens("a"), 5.0)

    def test_fractions_are_kept(self):
        for _ in range(5):
            self.rl.allow("a")
        self.clock.t += 0.25                     # 0.5 token
        self.assertAlmostEqual(self.rl.tokens("a"), 0.5)
        self.assertFalse(self.rl.allow("a"))
        self.clock.t += 0.25
        self.assertTrue(self.rl.allow("a"))

    def test_keys_are_independent(self):
        for _ in range(5):
            self.rl.allow("a")
        self.assertFalse(self.rl.allow("a"))
        self.assertTrue(self.rl.allow("b"))
        self.assertEqual(self.rl.keys(), ["a", "b"])

    def test_cost(self):
        self.assertTrue(self.rl.allow("a", cost=3))
        self.assertFalse(self.rl.allow("a", cost=3))
        self.assertEqual(self.rl.tokens("a"), 2.0)       # a refused request takes nothing
        self.assertTrue(self.rl.allow("a", cost=2))
        for bad in (0, -1, 6, "1", None):
            with self.assertRaises(ValueError):
                self.rl.allow("a", cost=bad)

    def test_retry_after(self):
        self.assertEqual(self.rl.retry_after("never-seen"), 0.0)
        self.assertEqual(self.rl.retry_after("never-seen"), 0.0)
        self.assertEqual(self.rl.keys(), [])             # asking does not create the key
        for _ in range(5):
            self.rl.allow("a")
        self.assertAlmostEqual(self.rl.retry_after("a"), 0.5)
        self.assertAlmostEqual(self.rl.retry_after("a", cost=3), 1.5)
        self.clock.t += 0.25
        self.assertAlmostEqual(self.rl.retry_after("a"), 0.25)
        self.assertEqual(self.rl.tokens("a"), 0.5)       # retry_after took nothing

    def test_clock_going_backwards(self):
        for _ in range(5):
            self.rl.allow("a")
        self.clock.t -= 50
        self.assertFalse(self.rl.allow("a"))
        self.clock.t += 50.5                               # back at 'last_seen' + 0.5 s
        self.assertTrue(self.rl.allow("a"))

    def test_reset(self):
        for _ in range(5):
            self.rl.allow("a")
            self.rl.allow("b")
        self.rl.reset("a")
        self.assertEqual(self.rl.tokens("a"), 5.0)
        self.assertEqual(self.rl.tokens("b"), 0.0)
        self.rl.reset()
        self.assertEqual(self.rl.keys(), [])
        self.assertTrue(self.rl.allow("b"))
'''},
    "bugfix": {
        "symptom": "A rate limiter (ratelimit.py) lets a client through too often: with rate=2 and burst=5, after the bucket is empty a wait of 0.25 s (half a token) must still refuse a request, but allow() accepts it.",
        "bug": [("ratelimit.py", "        if b[0] >= cost:\n            b[0] -= cost\n            return True\n        return False", "        if b[0] + 0.5 >= cost:\n            b[0] -= cost\n            return True\n        return False")],
        "repro": {"test_repro.py": 'import unittest\nfrom ratelimit import RateLimiter\n\n\nclass Repro(unittest.TestCase):\n    def test_half_a_token_is_not_enough(self):\n        t = [0.0]\n        rl = RateLimiter(rate=2, burst=5, clock=lambda: t[0])\n        for _ in range(5):\n            rl.allow("k")\n        t[0] += 0.25\n        self.assertFalse(rl.allow("k"))\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
