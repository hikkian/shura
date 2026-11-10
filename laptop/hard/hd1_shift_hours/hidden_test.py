import unittest
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from sched import Schedule, open_minutes, is_open, next_change

UTC = timezone.utc
BER = "Europe/Berlin"


def one(tz, day, *wins):
    """Расписание с окнами только в день недели даты `day`."""
    return Schedule(tz, {day.weekday(): list(wins)})


def utc(*a):
    return datetime(*a, tzinfo=UTC)


class H(unittest.TestCase):
    def test_plain_and_overnight_split(self):
        s = Schedule(BER, {0: [("09:00", "17:30")], 4: [("22:00", "02:00")]})
        self.assertEqual(open_minutes(s, date(2025, 6, 2)), 510)
        self.assertEqual(open_minutes(s, date(2025, 6, 6)), 120)   # пятница, до полуночи
        self.assertEqual(open_minutes(s, date(2025, 6, 7)), 120)   # суббота, хвост ночного окна
        self.assertEqual(open_minutes(s, date(2025, 6, 8)), 0)

    def test_spring_forward_day(self):
        d = date(2025, 3, 30)
        self.assertEqual(open_minutes(one(BER, d, ("00:00", "06:00")), d), 300)
        self.assertEqual(open_minutes(one(BER, d, ("09:00", "17:00")), d), 480)

    def test_fall_back_day(self):
        d = date(2025, 10, 26)
        self.assertEqual(open_minutes(one(BER, d, ("00:00", "06:00")), d), 420)
        self.assertEqual(open_minutes(one(BER, d, ("09:00", "17:00")), d), 480)

    def test_overnight_across_transition(self):
        sat = date(2025, 3, 29)
        s = one(BER, sat, ("22:00", "06:00"))
        self.assertEqual(open_minutes(s, sat), 120)
        self.assertEqual(open_minutes(s, date(2025, 3, 30)), 300)
        sat2 = date(2025, 10, 25)
        s2 = one(BER, sat2, ("22:00", "06:00"))
        self.assertEqual(open_minutes(s2, sat2), 120)
        self.assertEqual(open_minutes(s2, date(2025, 10, 26)), 420)

    def test_window_starting_in_gap(self):
        d = date(2025, 3, 30)
        self.assertEqual(open_minutes(one(BER, d, ("02:30", "05:00")), d), 90)

    def test_overlapping_windows_counted_once(self):
        d = date(2025, 6, 2)
        self.assertEqual(open_minutes(one(BER, d, ("09:00", "13:00"), ("12:00", "15:00")), d), 360)
        self.assertEqual(open_minutes(one(BER, d, ("09:00", "17:00"), ("10:00", "11:00")), d), 480)
        # ночное окно предыдущего дня пересекается с утренним окном
        s = Schedule(BER, {0: [("22:00", "03:00")], 1: [("01:00", "05:00")]})
        self.assertEqual(open_minutes(s, date(2025, 6, 3)), 300)

    def test_full_day_windows(self):
        for d, expect in [(date(2025, 6, 1), 1440), (date(2025, 3, 30), 1380), (date(2025, 10, 26), 1500)]:
            s = one(BER, d, ("00:00", "00:00"))
            self.assertEqual(open_minutes(s, d), expect, d)
            self.assertEqual(open_minutes(s, d + timedelta(days=1)), 0, d)

    def test_half_hour_dst_zone(self):
        d = date(2025, 4, 6)   # Lord Howe: сдвиг на 30 минут
        self.assertEqual(open_minutes(one("Australia/Lord_Howe", d, ("00:00", "06:00")), d), 390)

    def test_is_open_boundaries_and_spill(self):
        s = Schedule(BER, {0: [("09:00", "17:30")], 4: [("22:00", "02:00")]})
        z = ZoneInfo(BER)
        L = lambda *a: datetime(*a, tzinfo=z)
        self.assertTrue(is_open(s, L(2025, 6, 2, 9, 0)))
        self.assertFalse(is_open(s, L(2025, 6, 2, 8, 59, 59)))
        self.assertTrue(is_open(s, L(2025, 6, 2, 17, 29, 59)))
        self.assertFalse(is_open(s, L(2025, 6, 2, 17, 30)))
        self.assertTrue(is_open(s, L(2025, 6, 6, 23, 59, 59)))
        self.assertTrue(is_open(s, L(2025, 6, 7, 1, 59, 59)))     # суббота ночью
        self.assertFalse(is_open(s, L(2025, 6, 7, 2, 0)))
        self.assertFalse(is_open(s, L(2025, 6, 8, 1, 0)))
        # момент в другом поясе
        self.assertTrue(is_open(s, utc(2025, 6, 2, 7, 0)))        # 09:00 в Берлине
        self.assertFalse(is_open(s, utc(2025, 6, 2, 6, 59)))
        self.assertTrue(is_open(s, datetime(2025, 6, 2, 4, 0, tzinfo=ZoneInfo("America/New_York"))))

    def test_is_open_around_transitions(self):
        d = date(2025, 10, 26)
        s = one(BER, d, ("01:00", "02:30"))   # конец 02:30 — первое вхождение, т.е. 00:30 UTC
        self.assertTrue(is_open(s, utc(2025, 10, 25, 23, 0)))
        self.assertTrue(is_open(s, utc(2025, 10, 26, 0, 29, 59)))
        self.assertFalse(is_open(s, utc(2025, 10, 26, 0, 30)))
        self.assertFalse(is_open(s, utc(2025, 10, 26, 1, 0)))     # повторные 02:00-02:59
        self.assertFalse(is_open(s, utc(2025, 10, 26, 1, 30)))
        g = date(2025, 3, 30)
        s2 = one(BER, g, ("02:30", "05:00"))
        self.assertFalse(is_open(s2, utc(2025, 3, 30, 1, 29)))
        self.assertTrue(is_open(s2, utc(2025, 3, 30, 1, 30)))     # 03:30 CEST
        self.assertFalse(is_open(s2, utc(2025, 3, 30, 3, 0)))

    def test_next_change_basic(self):
        s = Schedule(BER, {0: [("09:00", "12:00"), ("12:00", "15:00")], 2: [("10:00", "11:00")]})
        # понедельник 10:00 по Берлину, окна примыкают -> смена в 15:00
        self.assertEqual(next_change(s, utc(2025, 6, 2, 8, 0)), utc(2025, 6, 2, 13, 0))
        # из закрытого состояния — ближайшее открытие
        self.assertEqual(next_change(s, utc(2025, 6, 2, 13, 0)), utc(2025, 6, 4, 8, 0))
        # строго после instant
        self.assertEqual(next_change(s, utc(2025, 6, 2, 7, 0)), utc(2025, 6, 2, 13, 0))
        self.assertEqual(next_change(s, utc(2025, 6, 4, 8, 0)), utc(2025, 6, 4, 9, 0))
        # через неделю
        self.assertEqual(next_change(s, utc(2025, 6, 4, 9, 0)), utc(2025, 6, 9, 7, 0))

    def test_next_change_overnight_and_dst(self):
        s = Schedule(BER, {5: [("22:00", "06:00")]})
        # открыто с субботы 22:00 до воскресенья 06:00; ночь перехода на лето 29->30 марта
        self.assertEqual(next_change(s, utc(2025, 3, 29, 12, 0)), utc(2025, 3, 29, 21, 0))
        self.assertEqual(next_change(s, utc(2025, 3, 29, 21, 0)), utc(2025, 3, 30, 4, 0))   # 06:00 CEST
        s2 = Schedule(BER, {5: [("22:00", "06:00")]})
        self.assertEqual(next_change(s2, utc(2025, 10, 25, 21, 0)), utc(2025, 10, 26, 5, 0))  # 06:00 CET

    def test_next_change_none(self):
        self.assertIsNone(next_change(Schedule(BER, {}), utc(2025, 6, 2, 8, 0)))
        always = Schedule(BER, {i: [("00:00", "00:00")] for i in range(7)})
        self.assertIsNone(next_change(always, utc(2025, 6, 2, 8, 0)))
        self.assertIsNone(next_change(always, utc(2025, 3, 30, 0, 30)))
        chained = Schedule(BER, {i: [("00:00", "12:00"), ("12:00", "00:00")] for i in range(7)})
        self.assertIsNone(next_change(chained, utc(2025, 10, 26, 0, 30)))


if __name__ == "__main__":
    unittest.main()
