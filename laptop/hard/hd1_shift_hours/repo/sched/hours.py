from datetime import datetime, timedelta


def open_minutes(schedule, day):
    """Сколько минут магазин открыт в день `day` (datetime.date)."""
    total = 0
    for start, end in schedule.day_windows(day.weekday()):
        a = datetime.combine(day, start, tzinfo=schedule.zone)
        b = datetime.combine(day, end, tzinfo=schedule.zone)
        if b < a:
            b += timedelta(days=1)
        total += int((b - a).total_seconds() // 60)
    return total


def is_open(schedule, instant):
    """Открыт ли магазин в момент `instant`."""
    local = instant.astimezone(schedule.zone)
    for start, end in schedule.day_windows(local.weekday()):
        if start <= local.time() < end:
            return True
    return False


def next_change(schedule, instant):
    raise NotImplementedError
