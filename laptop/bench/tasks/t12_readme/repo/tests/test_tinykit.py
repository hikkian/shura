import unittest

from tinykit import RateLimitExceeded, TokenBucket, chunked, human_size, retry, slugify


class T(unittest.TestCase):
    def test_bucket(self):
        t = [0.0]
        b = TokenBucket(rate=1, capacity=2, clock=lambda: t[0])
        self.assertTrue(b.try_acquire())
        self.assertTrue(b.try_acquire())
        self.assertFalse(b.try_acquire())
        t[0] = 1.0
        self.assertTrue(b.try_acquire())
        with self.assertRaises(RateLimitExceeded):
            b.acquire()

    def test_retry(self):
        calls = []

        @retry(times=3, sleep=lambda s: None)
        def f():
            calls.append(1)
            if len(calls) < 3:
                raise OSError
            return "ok"
        self.assertEqual(f(), "ok")

    def test_text(self):
        self.assertEqual(slugify("Привет, Мир!"), "privet-mir")
        self.assertEqual(human_size(1500), "1.5 KB")

    def test_chunked(self):
        self.assertEqual(list(chunked(range(5), 2)), [[0, 1], [2, 3], [4]])


if __name__ == "__main__":
    unittest.main()
