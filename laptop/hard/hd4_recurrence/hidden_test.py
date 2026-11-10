import unittest
from datetime import date

from recur import Rule, expand

D = date


def ex(start, rule, begin, end):
    return expand(start, rule, begin, end)


class H(unittest.TestCase):
    def test_monthly_skips_missing_days(self):
        r = Rule("MONTHLY")
        self.assertEqual(ex(D(2025, 1, 31), r, D(2025, 1, 1), D(2025, 12, 31)),
                         [D(2025, 1, 31), D(2025, 3, 31), D(2025, 5, 31), D(2025, 7, 31),
                          D(2025, 8, 31), D(2025, 10, 31), D(2025, 12, 31)])
        # пропущенные месяцы в count не входят
        self.assertEqual(ex(D(2025, 1, 31), Rule("MONTHLY", count=4), D(2025, 1, 1), D(2030, 1, 1)),
                         [D(2025, 1, 31), D(2025, 3, 31), D(2025, 5, 31), D(2025, 7, 31)])
        # 29 число и високосный год
        self.assertEqual(ex(D(2024, 1, 29), Rule("MONTHLY", bymonthday=[29]), D(2024, 1, 1), D(2025, 3, 31)),
                         [D(2024, m, 29) for m in range(1, 13)] + [D(2025, 1, 29), D(2025, 3, 29)])

    def test_monthly_interval_anchored_to_start_month(self):
        self.assertEqual(ex(D(2025, 1, 31), Rule("MONTHLY", interval=2), D(2025, 1, 1), D(2025, 12, 31)),
                         [D(2025, 1, 31), D(2025, 3, 31), D(2025, 5, 31), D(2025, 7, 31)])
        self.assertEqual(ex(D(2025, 2, 10), Rule("MONTHLY", interval=3, bymonthday=[-1]), D(2025, 1, 1), D(2025, 12, 31)),
                         [D(2025, 2, 28), D(2025, 5, 31), D(2025, 8, 31), D(2025, 11, 30)])

    def test_negative_monthdays(self):
        self.assertEqual(ex(D(2025, 1, 10), Rule("MONTHLY", bymonthday=[-1]), D(2025, 1, 1), D(2025, 4, 30)),
                         [D(2025, 1, 31), D(2025, 2, 28), D(2025, 3, 31), D(2025, 4, 30)])
        self.assertEqual(ex(D(2024, 1, 10), Rule("MONTHLY", bymonthday=[-1]), D(2024, 2, 1), D(2024, 2, 29)),
                         [D(2024, 2, 29)])
        self.assertEqual(ex(D(2025, 1, 1), Rule("MONTHLY", bymonthday=[-31]), D(2025, 1, 1), D(2025, 4, 30)),
                         [D(2025, 1, 1), D(2025, 3, 1)])   # -31 есть только в 31-дневных месяцах
        self.assertEqual(ex(D(2025, 1, 10), Rule("MONTHLY", bymonthday=[1, -1]), D(2025, 1, 1), D(2025, 3, 31)),
                         [D(2025, 1, 31), D(2025, 2, 1), D(2025, 2, 28), D(2025, 3, 1), D(2025, 3, 31)])
        # совпадающие числа не дублируются
        self.assertEqual(ex(D(2025, 4, 1), Rule("MONTHLY", bymonthday=[-1, 30]), D(2025, 4, 1), D(2025, 4, 30)),
                         [D(2025, 4, 30)])

    def test_start_is_occurrence_only_if_it_matches(self):
        self.assertEqual(ex(D(2025, 1, 15), Rule("MONTHLY", bymonthday=[1], count=2), D(2025, 1, 1), D(2025, 12, 31)),
                         [D(2025, 2, 1), D(2025, 3, 1)])
        self.assertEqual(ex(D(2025, 1, 8), Rule("WEEKLY", byweekday=[0], count=2), D(2025, 1, 1), D(2025, 12, 31)),
                         [D(2025, 1, 13), D(2025, 1, 20)])
        self.assertEqual(ex(D(2025, 1, 8), Rule("WEEKLY", byweekday=[2, 4], count=3), D(2025, 1, 1), D(2025, 12, 31)),
                         [D(2025, 1, 8), D(2025, 1, 10), D(2025, 1, 15)])

    def test_weekly_interval_with_weekdays(self):
        # старт в среду 8 января; неделя начинается с понедельника
        r = Rule("WEEKLY", interval=2, byweekday=[2, 0])
        self.assertEqual(ex(D(2025, 1, 8), r, D(2025, 1, 1), D(2025, 2, 9)),
                         [D(2025, 1, 8), D(2025, 1, 20), D(2025, 1, 22), D(2025, 2, 3), D(2025, 2, 5)])
        # воскресенье принадлежит неделе, которая началась в понедельник до него
        r = Rule("WEEKLY", interval=2, byweekday=[6, 0])
        self.assertEqual(ex(D(2025, 1, 4), r, D(2025, 1, 1), D(2025, 2, 2)),
                         [D(2025, 1, 5), D(2025, 1, 13), D(2025, 1, 19), D(2025, 1, 27), D(2025, 2, 2)])
        # без byweekday — день недели start
        self.assertEqual(ex(D(2025, 1, 8), Rule("WEEKLY", interval=3), D(2025, 1, 1), D(2025, 3, 1)),
                         [D(2025, 1, 8), D(2025, 1, 29), D(2025, 2, 19)])

    def test_yearly_feb29(self):
        want = [D(y, 2, 29) for y in range(2024, 2105, 4) if y != 2100]
        self.assertEqual(ex(D(2024, 2, 29), Rule("YEARLY"), D(2000, 1, 1), D(2105, 1, 1)), want)
        self.assertEqual(ex(D(2024, 2, 29), Rule("YEARLY", count=3), D(2000, 1, 1), D(2200, 1, 1)),
                         [D(2024, 2, 29), D(2028, 2, 29), D(2032, 2, 29)])
        self.assertEqual(ex(D(2025, 3, 15), Rule("YEARLY", interval=2), D(2025, 1, 1), D(2030, 1, 1)),
                         [D(2025, 3, 15), D(2027, 3, 15), D(2029, 3, 15)])

    def test_count_counts_from_start_and_includes_exdates(self):
        r = Rule("DAILY", count=5)
        self.assertEqual(ex(D(2025, 1, 1), r, D(2025, 1, 3), D(2025, 1, 31)),
                         [D(2025, 1, 3), D(2025, 1, 4), D(2025, 1, 5)])
        r = Rule("DAILY", count=5, exdates=[D(2025, 1, 2), D(2025, 1, 4)])
        self.assertEqual(ex(D(2025, 1, 1), r, D(2025, 1, 1), D(2025, 1, 31)),
                         [D(2025, 1, 1), D(2025, 1, 3), D(2025, 1, 5)])
        r = Rule("MONTHLY", count=3, exdates=(D(2025, 2, 15),))
        self.assertEqual(ex(D(2025, 1, 15), r, D(2025, 1, 1), D(2025, 12, 31)), [D(2025, 1, 15), D(2025, 3, 15)])

    def test_until_inclusive_and_with_count(self):
        self.assertEqual(ex(D(2025, 1, 1), Rule("DAILY", until=D(2025, 1, 5)), D(2025, 1, 1), D(2025, 12, 31)),
                         [D(2025, 1, i) for i in range(1, 6)])
        self.assertEqual(ex(D(2025, 1, 1), Rule("DAILY", until=D(2025, 1, 5), count=3), D(2025, 1, 1), D(2025, 12, 31)),
                         [D(2025, 1, i) for i in range(1, 4)])
        self.assertEqual(ex(D(2025, 1, 1), Rule("DAILY", until=D(2025, 1, 5), count=30), D(2025, 1, 1), D(2025, 12, 31)),
                         [D(2025, 1, i) for i in range(1, 6)])
        self.assertEqual(ex(D(2025, 1, 15), Rule("MONTHLY", until=D(2025, 3, 15)), D(2025, 1, 1), D(2025, 12, 31)),
                         [D(2025, 1, 15), D(2025, 2, 15), D(2025, 3, 15)])
        self.assertEqual(ex(D(2025, 1, 15), Rule("MONTHLY", until=D(2025, 3, 14)), D(2025, 1, 1), D(2025, 12, 31)),
                         [D(2025, 1, 15), D(2025, 2, 15)])

    def test_range_bounds(self):
        r = Rule("DAILY")
        self.assertEqual(ex(D(2025, 1, 10), r, D(2025, 1, 1), D(2025, 1, 12)), [D(2025, 1, 10), D(2025, 1, 11), D(2025, 1, 12)])
        self.assertEqual(ex(D(2025, 1, 1), r, D(2025, 1, 5), D(2025, 1, 5)), [D(2025, 1, 5)])
        self.assertEqual(ex(D(2025, 1, 1), r, D(2025, 1, 6), D(2025, 1, 5)), [])
        self.assertEqual(ex(D(2025, 1, 10), r, D(2025, 1, 1), D(2025, 1, 9)), [])
        # окно далеко от старта
        self.assertEqual(ex(D(2020, 1, 1), Rule("WEEKLY"), D(2025, 1, 1), D(2025, 1, 15)),
                         [D(2025, 1, 1), D(2025, 1, 8), D(2025, 1, 15)])
        self.assertEqual(ex(D(2020, 1, 31), Rule("MONTHLY", count=10), D(2025, 1, 1), D(2030, 1, 1)), [])

    def test_validation(self):
        bad = [dict(freq="HOURLY"), dict(freq="DAILY", interval=0), dict(freq="WEEKLY", byweekday=[7]),
               dict(freq="WEEKLY", byweekday=[-1]), dict(freq="MONTHLY", byweekday=[1]),
               dict(freq="MONTHLY", bymonthday=[0]), dict(freq="MONTHLY", bymonthday=[32]),
               dict(freq="MONTHLY", bymonthday=[-32]), dict(freq="WEEKLY", bymonthday=[1]),
               dict(freq="DAILY", count=0), dict(freq="YEARLY", bymonthday=[1])]
        for kw in bad:
            with self.assertRaises(ValueError, msg=str(kw)):
                Rule(**kw)
        Rule("MONTHLY", bymonthday=[-31, 31, 1, -1])
        Rule("WEEKLY", byweekday=[0, 6])

    def test_date_max(self):
        self.assertEqual(ex(D(9999, 12, 30), Rule("DAILY"), D(9999, 1, 1), D.max), [D(9999, 12, 30), D(9999, 12, 31)])
        self.assertEqual(ex(D(9999, 11, 15), Rule("MONTHLY"), D(9999, 1, 1), D.max), [D(9999, 11, 15), D(9999, 12, 15)])
        self.assertEqual(ex(D(9999, 12, 20), Rule("WEEKLY"), D(9999, 1, 1), D.max), [D(9999, 12, 20), D(9999, 12, 27)])
        self.assertEqual(ex(D(9998, 6, 1), Rule("YEARLY"), D(9998, 1, 1), D.max), [D(9998, 6, 1), D(9999, 6, 1)])
        self.assertEqual(ex(D(9999, 12, 31), Rule("MONTHLY", bymonthday=[-1]), D(9999, 1, 1), D.max), [D(9999, 12, 31)])
