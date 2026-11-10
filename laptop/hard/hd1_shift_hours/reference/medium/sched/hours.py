from datetime import datetime, timedelta, timezone

UTC = timezone.utc


def _win(schedule, d):
    zone = schedule.zone
    for s, e in schedule.day_windows(d.weekday()):
        ed = d + timedelta(days=1) if e <= s else d
        yield (datetime.combine(d, s, tzinfo=zone).astimezone(UTC),
               datetime.combine(ed, e, tzinfo=zone).astimezone(UTC))


def open_minutes(schedule, day):
    # окна, начавшиеся в этот день (ночное окно целиком относится ко дню начала)
    return sum(int((b - a).total_seconds() // 60) for a, b in _win(schedule, day))


def is_open(schedule, instant):
    d = instant.astimezone(schedule.zone).date()
    return any(a <= instant < b for x in (d - timedelta(days=1), d) for a, b in _win(schedule, x))


def next_change(schedule, instant):
    d = instant.astimezone(schedule.zone).date()
    pts = sorted({p for i in range(-1, 9) for a, b in _win(schedule, d + timedelta(days=i)) for p in (a, b)})
    for p in pts:
        if p > instant:
            return p
    return None
