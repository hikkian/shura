from datetime import datetime, timedelta, time


def _wall(schedule, day):
    for start, end in schedule.day_windows(day.weekday()):
        a = datetime.combine(day, start)
        b = datetime.combine(day, end)
        if b <= a:
            b += timedelta(days=1)
        yield a, b


def open_minutes(schedule, day):
    total = 0
    for a, b in _wall(schedule, day):
        total += int((b - a).total_seconds() // 60)
    return total


def is_open(schedule, instant):
    local = instant.astimezone(schedule.zone).replace(tzinfo=None)
    for d in (local.date() - timedelta(days=1), local.date()):
        for a, b in _wall(schedule, d):
            if a <= local < b:
                return True
    return False


def next_change(schedule, instant):
    local = instant.astimezone(schedule.zone).replace(tzinfo=None)
    state = is_open(schedule, instant)
    t = local.replace(second=0, microsecond=0)
    for _ in range(8 * 24 * 60):
        t += timedelta(minutes=1)
        if is_open(schedule, t.replace(tzinfo=schedule.zone)) != state:
            return t.replace(tzinfo=schedule.zone)
    return None
