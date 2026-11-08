import unittest
from timeutil import parse_duration as p
class H(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(p("1h30m"), 5400)
        self.assertEqual(p("45s"), 45)
        self.assertEqual(p("2h 5s"), 7205)
        self.assertEqual(p("1d2h"), 93600)
        self.assertEqual(p("90"), 90)
        self.assertEqual(p("1H30M"), 5400)
        self.assertEqual(p(" 3m "), 180)
    def test_bad(self):
        for s in ["", "   ", "5x", "1m1h", "1h1h", "-5s", "h", "1h-2m"]:
            with self.assertRaises(ValueError, msg=s): p(s)
