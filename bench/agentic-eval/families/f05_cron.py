FAMILY = {
    "name": "f05_cron", "lang": "python",
    "implement": {
        "prompt": """Write the module `cron.py` (standard library only) with the function `next_run(expr, after)`.

expr is a classic 5-field cron expression: "minute hour day-of-month month day-of-week", fields separated by whitespace.
after is a naive datetime.datetime. next_run returns the first datetime STRICTLY AFTER `after` (seconds and microseconds of the
result are 0) that matches the expression. If none exists within 8 years after `after`, raise ValueError.

Field syntax (each field is a comma-separated list of items; the field matches if any item matches):
  *            every value of the field          */n   every n-th value starting at the field's minimum
  a            one value                         a-b   an inclusive range           a-b/n  every n-th value inside the range
Ranges: minute 0-59, hour 0-23, day-of-month 1-31, month 1-12, day-of-week 0-6 where 0 is Sunday (7 is NOT accepted).
Names are not supported. Invalid syntax, values out of range, a reversed range (5-2) and a step of 0 raise ValueError.
Day rule (like Vixie cron): if BOTH day-of-month and day-of-week are restricted (neither field is exactly "*"), a day matches when
EITHER of them matches; if only one is restricted, that one decides; if both are "*", every day matches. A day-of-month that does not
exist in a month (31 in April) simply never matches in that month.
""",
        "start": {"cron.py": '"""Cron schedules (to be written)."""\n\n\ndef next_run(expr, after):\n    raise NotImplementedError\n'},
        "solution": {"cron.py": '''"""Cron schedules."""
import datetime as dt

_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))


def _field(text, lo, hi):
    values = set()
    for item in text.split(","):
        if not item:
            raise ValueError("empty item")
        step = 1
        if "/" in item:
            item, s = item.split("/", 1)
            if not s.isdigit() or int(s) == 0:
                raise ValueError("bad step")
            step = int(s)
        if item == "*":
            a, b = lo, hi
        elif "-" in item:
            x, y = item.split("-", 1)
            if not (x.isdigit() and y.isdigit()):
                raise ValueError("bad range")
            a, b = int(x), int(y)
            if a > b:
                raise ValueError("reversed range")
        elif item.isdigit():
            a = int(item)
            b = hi if "/" in text.split(",")[0] and False else a
            if step != 1:
                b = hi
        else:
            raise ValueError("bad item")
        if a < lo or b > hi:
            raise ValueError("out of range")
        values.update(range(a, b + 1, step))
    return values


def _parse(expr):
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError("expected 5 fields")
    fields = [_field(p, lo, hi) for p, (lo, hi) in zip(parts, _RANGES)]
    return fields, parts[2] != "*", parts[4] != "*"


def next_run(expr, after):
    (minutes, hours, doms, months, dows), dom_r, dow_r = _parse(expr)
    t = after.replace(second=0, microsecond=0) + dt.timedelta(minutes=1)
    limit = after + dt.timedelta(days=366 * 8 + 2)
    while t <= limit:
        if t.month not in months:
            t = (t.replace(day=1, hour=0, minute=0) + dt.timedelta(days=32)).replace(day=1)
            continue
        dow = (t.weekday() + 1) % 7
        if dom_r and dow_r:
            day_ok = t.day in doms or dow in dows
        elif dom_r:
            day_ok = t.day in doms
        elif dow_r:
            day_ok = dow in dows
        else:
            day_ok = True
        if not day_ok:
            t = t.replace(hour=0, minute=0) + dt.timedelta(days=1)
            continue
        if t.hour not in hours:
            t = t.replace(minute=0) + dt.timedelta(hours=1)
            continue
        if t.minute not in minutes:
            t += dt.timedelta(minutes=1)
            continue
        return t
    raise ValueError("no matching time within 8 years")
'''},
    },
    "hidden": {"test_cron.py": '''import datetime as dt
import unittest
from cron import next_run

D = dt.datetime


class Basics(unittest.TestCase):
    def test_every_minute_is_strictly_after(self):
        self.assertEqual(next_run("* * * * *", D(2026, 3, 1, 10, 30, 0)), D(2026, 3, 1, 10, 31))
        self.assertEqual(next_run("* * * * *", D(2026, 3, 1, 10, 30, 45, 5)), D(2026, 3, 1, 10, 31))

    def test_fixed_time_today_or_tomorrow(self):
        self.assertEqual(next_run("30 14 * * *", D(2026, 3, 1, 10, 0)), D(2026, 3, 1, 14, 30))
        self.assertEqual(next_run("30 14 * * *", D(2026, 3, 1, 14, 30)), D(2026, 3, 2, 14, 30))
        self.assertEqual(next_run("0 0 * * *", D(2026, 12, 31, 23, 59, 59)), D(2027, 1, 1, 0, 0))

    def test_steps_and_lists_and_ranges(self):
        self.assertEqual(next_run("*/15 * * * *", D(2026, 3, 1, 10, 16)), D(2026, 3, 1, 10, 30))
        self.assertEqual(next_run("5,35 * * * *", D(2026, 3, 1, 10, 6)), D(2026, 3, 1, 10, 35))
        self.assertEqual(next_run("0 9-17/4 * * *", D(2026, 3, 1, 10, 0)), D(2026, 3, 1, 13, 0))
        self.assertEqual(next_run("10/20 * * * *", D(2026, 3, 1, 10, 0)), D(2026, 3, 1, 10, 10))
        self.assertEqual(next_run("10/20 * * * *", D(2026, 3, 1, 10, 10)), D(2026, 3, 1, 10, 30))
        self.assertEqual(next_run("0 0 1-3 * *", D(2026, 3, 3, 0, 0)), D(2026, 4, 1, 0, 0))

    def test_months_roll_over(self):
        self.assertEqual(next_run("0 12 15 6 *", D(2026, 7, 1)), D(2027, 6, 15, 12, 0))
        self.assertEqual(next_run("0 0 1 */3 *", D(2026, 2, 10)), D(2026, 4, 1, 0, 0))


class Days(unittest.TestCase):
    def test_day_of_week_only(self):
        # 2026-03-01 is a Sunday
        self.assertEqual(next_run("0 8 * * 1", D(2026, 3, 1, 9, 0)), D(2026, 3, 2, 8, 0))
        self.assertEqual(next_run("0 8 * * 0", D(2026, 3, 1, 9, 0)), D(2026, 3, 8, 8, 0))
        self.assertEqual(next_run("0 8 * * 1-5", D(2026, 3, 6, 9, 0)), D(2026, 3, 9, 8, 0))

    def test_both_restricted_means_either(self):
        # the 15th OR any Monday
        self.assertEqual(next_run("0 0 15 * 1", D(2026, 3, 10, 0, 0)), D(2026, 3, 15, 0, 0))
        self.assertEqual(next_run("0 0 15 * 1", D(2026, 3, 1, 0, 0)), D(2026, 3, 2, 0, 0))

    def test_nonexistent_day_is_skipped(self):
        self.assertEqual(next_run("0 0 31 * *", D(2026, 4, 1)), D(2026, 5, 31))

    def test_leap_day(self):
        self.assertEqual(next_run("0 0 29 2 *", D(2026, 1, 1)), D(2028, 2, 29))

    def test_impossible_raises(self):
        with self.assertRaises(ValueError):
            next_run("0 0 30 2 *", D(2026, 1, 1))


class Invalid(unittest.TestCase):
    def test_bad_expressions(self):
        for bad in ["", "* * * *", "* * * * * *", "60 * * * *", "* 24 * * *", "* * 0 * *", "* * 32 * *", "* * * 0 *", "* * * 13 *",
                    "* * * * 7", "*/0 * * * *", "5-2 * * * *", "a * * * *", "1- * * * *", "*/x * * * *", ",5 * * * *", "* * * JAN *", "-1 * * * *"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    next_run(bad, D(2026, 1, 1))
'''},
    "bugfix": {
        "symptom": "A cron helper (cron.py) returns the wrong day for expressions that restrict both day-of-month and day-of-week: \"0 0 15 * 1\" (the 15th OR any Monday) from 2026-03-01 should give Monday 2026-03-02 but returns the 15th.",
        "bug": [("cron.py", "            day_ok = t.day in doms or dow in dows", "            day_ok = t.day in doms and dow in dows")],
        "repro": {"test_repro.py": 'import datetime as dt\nimport unittest\nfrom cron import next_run\n\n\nclass Repro(unittest.TestCase):\n    def test_either_day_rule(self):\n        self.assertEqual(next_run("0 0 15 * 1", dt.datetime(2026, 3, 1)), dt.datetime(2026, 3, 2))\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
