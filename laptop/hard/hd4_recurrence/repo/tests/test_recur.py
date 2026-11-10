import unittest
from datetime import date

from recur import Rule, expand


class T(unittest.TestCase):
    def test_daily(self):
        r = Rule("DAILY", interval=2, count=3)
        self.assertEqual(expand(date(2025, 1, 1), r, date(2025, 1, 1), date(2025, 12, 31)),
                         [date(2025, 1, 1), date(2025, 1, 3), date(2025, 1, 5)])

    def test_monthly_simple(self):
        r = Rule("MONTHLY")
        self.assertEqual(expand(date(2025, 1, 15), r, date(2025, 1, 1), date(2025, 3, 31)),
                         [date(2025, 1, 15), date(2025, 2, 15), date(2025, 3, 15)])

    def test_weekly_until(self):
        r = Rule("WEEKLY", until=date(2025, 1, 15))
        self.assertEqual(expand(date(2025, 1, 1), r, date(2025, 1, 1), date(2025, 2, 1)),
                         [date(2025, 1, 1), date(2025, 1, 8), date(2025, 1, 15)])


if __name__ == "__main__":
    unittest.main()
