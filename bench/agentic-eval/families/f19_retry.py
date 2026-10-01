FAMILY = {
    "name": "f19_retry", "lang": "python",
    "implement": {
        "prompt": """Write the module `retrying.py` (standard library only) with the decorator factory `retry` and the exception `RetryError`.

retry(attempts=3, base_delay=1.0, factor=2.0, max_delay=60.0, jitter=0.0, retry_on=(Exception,), sleep=time.sleep, rng=random.random,
      on_retry=None, give_up_on=())
  Returns a decorator. The decorated function is called; if it raises an exception that is an instance of one of `retry_on` and NOT of
  one of `give_up_on`, it is called again, up to `attempts` calls in total. Any other exception propagates at once, unchanged.
  Before the k-th retry (k = 1, 2, ...) call sleep(delay) with
      delay = min(max_delay, base_delay * factor ** (k - 1)) * (1 + jitter * (2 * rng() - 1))
  (so jitter=0.5 spreads the delay between 50% and 150%); delay is never negative. No sleep after the last failed attempt.
  on_retry(attempt, exc, delay), if given, is called just before each sleep with attempt = the number of the call that just failed
  (1-based), the exception, and the delay about to be slept.
  When all attempts failed raise RetryError (a subclass of Exception) with attributes `attempts` (the number of calls made) and
  `last` (the last exception); the last exception is also its __cause__.
  The decorated function keeps its name and docstring (functools.wraps) and returns the function's result on success.
  Validation at decoration time (ValueError): attempts must be an int >= 1; base_delay >= 0; factor >= 1; max_delay >= 0; 0 <= jitter <= 1.
""",
        "start": {"retrying.py": '"""Retry with exponential backoff (to be written)."""\nimport random\nimport time\n\n\nclass RetryError(Exception):\n    pass\n\n\ndef retry(attempts=3, base_delay=1.0, factor=2.0, max_delay=60.0, jitter=0.0, retry_on=(Exception,), sleep=time.sleep,\n          rng=random.random, on_retry=None, give_up_on=()):\n    raise NotImplementedError\n'},
        "solution": {"retrying.py": '''"""Retry with exponential backoff."""
import functools
import random
import time


class RetryError(Exception):
    def __init__(self, attempts, last):
        super().__init__(f"gave up after {attempts} attempts: {last!r}")
        self.attempts, self.last = attempts, last


def retry(attempts=3, base_delay=1.0, factor=2.0, max_delay=60.0, jitter=0.0, retry_on=(Exception,), sleep=time.sleep,
          rng=random.random, on_retry=None, give_up_on=()):
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise ValueError("attempts must be an int >= 1")
    if base_delay < 0 or factor < 1 or max_delay < 0 or not (0 <= jitter <= 1):
        raise ValueError("bad delay parameters")
    retry_on, give_up_on = tuple(retry_on), tuple(give_up_on)

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last = None
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except BaseException as exc:
                    if not isinstance(exc, retry_on) or (give_up_on and isinstance(exc, give_up_on)):
                        raise
                    last = exc
                if attempt == attempts:
                    break
                delay = min(max_delay, base_delay * factor ** (attempt - 1)) * (1 + jitter * (2 * rng() - 1))
                delay = max(0.0, delay)
                if on_retry:
                    on_retry(attempt, last, delay)
                sleep(delay)
            raise RetryError(attempts, last) from last
        return wrapper
    return decorator
'''},
    },
    "hidden": {"test_retrying.py": '''import unittest
from retrying import RetryError, retry


class Flaky:
    def __init__(self, fail_times, exc=ValueError):
        self.calls, self.fail_times, self.exc = 0, fail_times, exc

    def __call__(self, x=0):
        """doc"""
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc(f"fail {self.calls}")
        return x + 100


class Retry(unittest.TestCase):
    def setUp(self):
        self.sleeps = []

    def deco(self, **kw):
        return retry(sleep=self.sleeps.append, **kw)

    def test_success_first_time(self):
        f = Flaky(0)
        self.assertEqual(self.deco()(f)(1), 101)
        self.assertEqual(self.sleeps, [])

    def test_backoff_sequence(self):
        f = Flaky(3)
        self.assertEqual(self.deco(attempts=5, base_delay=1, factor=2)(f)(), 100)
        self.assertEqual(self.sleeps, [1, 2, 4])
        self.assertEqual(f.calls, 4)

    def test_max_delay_caps(self):
        f = Flaky(5)
        self.deco(attempts=6, base_delay=10, factor=3, max_delay=50)(f)()
        self.assertEqual(self.sleeps, [10, 30, 50, 50, 50])

    def test_gives_up_with_retryerror(self):
        f = Flaky(10)
        with self.assertRaises(RetryError) as cm:
            self.deco(attempts=3)(f)()
        self.assertEqual(cm.exception.attempts, 3)
        self.assertIsInstance(cm.exception.last, ValueError)
        self.assertIs(cm.exception.__cause__, cm.exception.last)
        self.assertEqual(f.calls, 3)
        self.assertEqual(len(self.sleeps), 2)                        # no sleep after the last attempt

    def test_single_attempt_never_sleeps(self):
        with self.assertRaises(RetryError):
            self.deco(attempts=1)(Flaky(5))()
        self.assertEqual(self.sleeps, [])

    def test_retry_on_and_give_up_on(self):
        f = Flaky(2, exc=KeyError)
        with self.assertRaises(KeyError):                            # not in retry_on: propagates unchanged, after one call
            self.deco(retry_on=(ValueError,))(f)()
        self.assertEqual(f.calls, 1)
        g = Flaky(2, exc=PermissionError)
        with self.assertRaises(PermissionError):
            self.deco(retry_on=(OSError,), give_up_on=(PermissionError,))(g)()
        self.assertEqual(g.calls, 1)
        h = Flaky(2, exc=FileNotFoundError)
        self.assertEqual(self.deco(retry_on=(OSError,), give_up_on=(PermissionError,))(h)(), 100)

    def test_jitter(self):
        rolls = iter([0.0, 1.0, 0.5])
        f = Flaky(3)
        retry(attempts=4, base_delay=10, factor=1, jitter=0.5, sleep=self.sleeps.append, rng=lambda: next(rolls))(f)()
        self.assertEqual(self.sleeps, [5.0, 15.0, 10.0])

    def test_on_retry_callback(self):
        seen = []
        f = Flaky(2)
        self.deco(attempts=3, base_delay=1, on_retry=lambda n, e, d: seen.append((n, str(e), d)))(f)()
        self.assertEqual(seen, [(1, "fail 1", 1), (2, "fail 2", 2)])

    def test_wraps(self):
        f = Flaky(0)

        @self.deco()
        def named(x):
            """the doc"""
            return x
        self.assertEqual(named.__name__, "named")
        self.assertEqual(named.__doc__, "the doc")
        self.assertEqual(named(3), 3)

    def test_validation(self):
        for kw in ({"attempts": 0}, {"attempts": 1.5}, {"attempts": True}, {"base_delay": -1}, {"factor": 0.5}, {"max_delay": -1},
                   {"jitter": 1.5}, {"jitter": -0.1}):
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError):
                    retry(**kw)

    def test_zero_delays_are_fine(self):
        self.deco(attempts=3, base_delay=0)(Flaky(2))()
        self.assertEqual(self.sleeps, [0, 0])
'''},
    "bugfix": {
        "symptom": "A retry decorator (retrying.py) sleeps once too often: after the last failed attempt it still sleeps before giving up, but it must raise RetryError immediately (attempts=3 means two sleeps).",
        "bug": [("retrying.py", "                if attempt == attempts:\n                    break\n", "")],
        "repro": {"test_repro.py": 'import unittest\nfrom retrying import RetryError, retry\n\n\nclass Repro(unittest.TestCase):\n    def test_no_sleep_after_the_last_attempt(self):\n        sleeps = []\n\n        @retry(attempts=3, sleep=sleeps.append)\n        def boom():\n            raise ValueError("x")\n        with self.assertRaises(RetryError):\n            boom()\n        self.assertEqual(len(sleeps), 2)\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
