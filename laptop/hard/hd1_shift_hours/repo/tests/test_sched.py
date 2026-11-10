import unittest
from datetime import date, datetime, timezone

from sched import Schedule, open_minutes, is_open

S = Schedule("Europe/Berlin", {0: [("09:00", "17:30")], 1: [("09:00", "12:00"), ("13:00", "18:00")]})


class T(unittest.TestCase):
    def test_plain_day(self):
        self.assertEqual(open_minutes(S, date(2025, 6, 2)), 510)
        self.assertEqual(open_minutes(S, date(2025, 6, 3)), 480)
        self.assertEqual(open_minutes(S, date(2025, 6, 4)), 0)

    def test_is_open(self):
        self.assertTrue(is_open(S, datetime(2025, 6, 2, 8, 0, tzinfo=timezone.utc)))
        self.assertFalse(is_open(S, datetime(2025, 6, 2, 18, 0, tzinfo=timezone.utc)))


if __name__ == "__main__":
    unittest.main()
