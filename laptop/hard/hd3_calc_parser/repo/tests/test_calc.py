import unittest

from calc import evaluate


class T(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(evaluate("1 + 2 * 3"), 7)
        self.assertEqual(evaluate("(1 + 2) * 3"), 9)
        self.assertEqual(evaluate("2 + 3 * x", {"x": 4}), 14)
        self.assertEqual(evaluate("-4 + 1"), -3)
        self.assertEqual(evaluate("7 / 2"), 3.5)


if __name__ == "__main__":
    unittest.main()
