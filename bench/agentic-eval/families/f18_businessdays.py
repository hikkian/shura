FAMILY = {
    "name": "f18_businessdays", "lang": "python",
    "implement": {
        "prompt": """Write the module `bizdays.py` (standard library only) with the class `Calendar` working on datetime.date values.

Calendar(holidays=(), weekend=(5, 6))
  holidays: an iterable of dates that are not business days; weekend: the weekday numbers (Monday = 0 ... Sunday = 6) that are not
  business days. weekend may be empty (every day except holidays is a business day) but must not contain all of 0..6 (ValueError).
  Values in `weekend` outside 0..6 raise ValueError. Holidays that fall on weekends are fine. Only datetime.date (not datetime.datetime)
  is accepted anywhere: a datetime raises TypeError.
  is_business_day(d) -> bool
  next_business_day(d) -> date   the first business day strictly AFTER d.
  prev_business_day(d) -> date   the last business day strictly BEFORE d.
  add_business_days(d, n) -> date   move n business days: n > 0 forward, n < 0 backward, n == 0 returns d if it is a business day, else the
                                    NEXT business day. Starting day d may itself be a non-business day: add_business_days(Saturday, 1)
                                    is the first business day after that Saturday, the same as next_business_day(d).
  count_business_days(start, end) -> int   the number of business days in the half-open range [start, end) (start included, end not);
                                    if end < start return the NEGATIVE of count(end, start).
  business_days_between(start, end) -> list   the business days in [start, end) in order (empty if end <= start).
  month_end(year, month) -> date   the last BUSINESS day of the month.
""",
        "start": {"bizdays.py": '"""A business-day calendar (to be written)."""\nimport datetime as dt\n\n\nclass Calendar:\n    def __init__(self, holidays=(), weekend=(5, 6)):\n        raise NotImplementedError\n'},
        "solution": {"bizdays.py": '''"""A business-day calendar."""
import calendar as _cal
import datetime as dt


def _date(d):
    if isinstance(d, dt.datetime) or not isinstance(d, dt.date):
        raise TypeError("expected datetime.date")
    return d


class Calendar:
    def __init__(self, holidays=(), weekend=(5, 6)):
        weekend = tuple(weekend)
        if any((not isinstance(w, int)) or w < 0 or w > 6 for w in weekend) or set(weekend) >= set(range(7)):
            raise ValueError("bad weekend")
        self._weekend = set(weekend)
        self._holidays = {_date(h) for h in holidays}

    def is_business_day(self, d):
        d = _date(d)
        return d.weekday() not in self._weekend and d not in self._holidays

    def next_business_day(self, d):
        d = _date(d)
        while True:
            d += dt.timedelta(days=1)
            if self.is_business_day(d):
                return d

    def prev_business_day(self, d):
        d = _date(d)
        while True:
            d -= dt.timedelta(days=1)
            if self.is_business_day(d):
                return d

    def add_business_days(self, d, n):
        d = _date(d)
        if n == 0:
            return d if self.is_business_day(d) else self.next_business_day(d)
        step = self.next_business_day if n > 0 else self.prev_business_day
        for _ in range(abs(n)):
            d = step(d)
        return d

    def count_business_days(self, start, end):
        start, end = _date(start), _date(end)
        if end < start:
            return -self.count_business_days(end, start)
        return len(self.business_days_between(start, end))

    def business_days_between(self, start, end):
        start, end = _date(start), _date(end)
        out, d = [], start
        while d < end:
            if self.is_business_day(d):
                out.append(d)
            d += dt.timedelta(days=1)
        return out

    def month_end(self, year, month):
        last = dt.date(year, month, _cal.monthrange(year, month)[1])
        return last if self.is_business_day(last) else self.prev_business_day(last)
'''},
    },
    "hidden": {"test_bizdays.py": '''import datetime as dt
import unittest
from bizdays import Calendar

D = dt.date
# 2026-03-02 is a Monday
CAL = Calendar(holidays=[D(2026, 3, 4), D(2026, 3, 8)])


class Days(unittest.TestCase):
    def test_is_business_day(self):
        self.assertTrue(CAL.is_business_day(D(2026, 3, 2)))
        self.assertFalse(CAL.is_business_day(D(2026, 3, 4)))          # holiday
        self.assertFalse(CAL.is_business_day(D(2026, 3, 7)))          # Saturday
        self.assertFalse(CAL.is_business_day(D(2026, 3, 8)))          # Sunday and holiday

    def test_next_prev(self):
        self.assertEqual(CAL.next_business_day(D(2026, 3, 3)), D(2026, 3, 5))
        self.assertEqual(CAL.next_business_day(D(2026, 3, 6)), D(2026, 3, 9))
        self.assertEqual(CAL.prev_business_day(D(2026, 3, 5)), D(2026, 3, 3))
        self.assertEqual(CAL.prev_business_day(D(2026, 3, 9)), D(2026, 3, 6))
        self.assertEqual(CAL.next_business_day(D(2026, 3, 7)), D(2026, 3, 9))

    def test_types(self):
        with self.assertRaises(TypeError):
            CAL.is_business_day(dt.datetime(2026, 3, 2, 12, 0))
        with self.assertRaises(TypeError):
            CAL.next_business_day("2026-03-02")
        with self.assertRaises(TypeError):
            Calendar(holidays=[dt.datetime(2026, 1, 1)])

    def test_weekend_validation(self):
        for bad in [(0, 1, 2, 3, 4, 5, 6), (7,), (-1,), ("5",)]:
            with self.assertRaises(ValueError):
                Calendar(weekend=bad)
        c = Calendar(weekend=())
        self.assertTrue(c.is_business_day(D(2026, 3, 7)))
        c2 = Calendar(weekend=(4, 5))                                   # Friday and Saturday
        self.assertTrue(c2.is_business_day(D(2026, 3, 8)))
        self.assertFalse(c2.is_business_day(D(2026, 3, 6)))


class Arithmetic(unittest.TestCase):
    def test_add(self):
        self.assertEqual(CAL.add_business_days(D(2026, 3, 2), 1), D(2026, 3, 3))
        self.assertEqual(CAL.add_business_days(D(2026, 3, 2), 2), D(2026, 3, 5))      # skips the holiday on the 4th
        self.assertEqual(CAL.add_business_days(D(2026, 3, 2), 5), D(2026, 3, 10))
        self.assertEqual(CAL.add_business_days(D(2026, 3, 10), -3), D(2026, 3, 5))
        self.assertEqual(CAL.add_business_days(D(2026, 3, 9), -1), D(2026, 3, 6))

    def test_add_zero_and_from_non_business_day(self):
        self.assertEqual(CAL.add_business_days(D(2026, 3, 3), 0), D(2026, 3, 3))
        self.assertEqual(CAL.add_business_days(D(2026, 3, 7), 0), D(2026, 3, 9))
        self.assertEqual(CAL.add_business_days(D(2026, 3, 7), 1), D(2026, 3, 9))
        self.assertEqual(CAL.add_business_days(D(2026, 3, 7), 2), D(2026, 3, 10))
        self.assertEqual(CAL.add_business_days(D(2026, 3, 7), -1), D(2026, 3, 6))

    def test_count_half_open(self):
        self.assertEqual(CAL.count_business_days(D(2026, 3, 2), D(2026, 3, 2)), 0)
        self.assertEqual(CAL.count_business_days(D(2026, 3, 2), D(2026, 3, 3)), 1)
        self.assertEqual(CAL.count_business_days(D(2026, 3, 2), D(2026, 3, 9)), 4)       # 2,3,5,6
        self.assertEqual(CAL.count_business_days(D(2026, 3, 9), D(2026, 3, 2)), -4)
        self.assertEqual(CAL.count_business_days(D(2026, 3, 7), D(2026, 3, 9)), 0)

    def test_between(self):
        self.assertEqual(CAL.business_days_between(D(2026, 3, 3), D(2026, 3, 9)), [D(2026, 3, 3), D(2026, 3, 5), D(2026, 3, 6)])
        self.assertEqual(CAL.business_days_between(D(2026, 3, 9), D(2026, 3, 3)), [])

    def test_month_end(self):
        self.assertEqual(CAL.month_end(2026, 2), D(2026, 2, 27))          # 28th is a Saturday
        self.assertEqual(CAL.month_end(2026, 1), D(2026, 1, 30))          # 31st is a Saturday
        self.assertEqual(CAL.month_end(2026, 3), D(2026, 3, 31))
        c = Calendar(holidays=[D(2026, 4, 30)])
        self.assertEqual(c.month_end(2026, 4), D(2026, 4, 29))
'''},
    "bugfix": {
        "symptom": "A business-day calendar (bizdays.py) counts the wrong number of days: count_business_days(date(2026,3,2), date(2026,3,3)) should be 1 (the end date is excluded) but returns 2.",
        "bug": [("bizdays.py", "        while d < end:\n", "        while d <= end:\n")],
        "repro": {"test_repro.py": 'import datetime as dt\nimport unittest\nfrom bizdays import Calendar\n\n\nclass Repro(unittest.TestCase):\n    def test_end_is_excluded(self):\n        c = Calendar()\n        self.assertEqual(c.count_business_days(dt.date(2026, 3, 2), dt.date(2026, 3, 3)), 1)\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
