from datetime import datetime, timedelta, timezone

UTC = timezone.utc


def _to_utc(day, t, zone):
    # fold=0: пропуск -> смещение «до перехода» (02:30 -> 03:30), дубль -> первое вхождение
    return datetime.combine(day, t, tzinfo=zone).astimezone(UTC)


def _intervals(schedule, first_day, last_day):
    """Объединённые интервалы (UTC) окон, начинающихся в днях first_day..last_day включительно."""
    zone = schedule.zone
    raw = []
    d = first_day
    while d <= last_day:
        for s, e in schedule.day_windows(d.weekday()):
            ed = d + timedelta(days=1) if e <= s else d
            a, b = _to_utc(d, s, zone), _to_utc(ed, e, zone)
            if b > a:
                raw.append((a, b))
        d += timedelta(days=1)
    raw.sort()
    out = []
    for a, b in raw:
        if out and a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def open_minutes(schedule, day):
    zone = schedule.zone
    lo = _to_utc(day, datetime.min.time(), zone)
    hi = _to_utc(day + timedelta(days=1), datetime.min.time(), zone)
    secs = 0
    for a, b in _intervals(schedule, day - timedelta(days=1), day):
        a, b = max(a, lo), min(b, hi)
        if b > a:
            secs += (b - a).total_seconds()
    return int(secs // 60)


def is_open(schedule, instant):
    local = instant.astimezone(schedule.zone).date()
    return any(a <= instant < b for a, b in
               _intervals(schedule, local - timedelta(days=1), local + timedelta(days=1)))


def next_change(schedule, instant):
    local = instant.astimezone(schedule.zone).date()
    horizon = instant + timedelta(days=8)
    best = None
    for a, b in _intervals(schedule, local - timedelta(days=1), local + timedelta(days=10)):
        for p in (a, b):
            if instant < p <= horizon and (best is None or p < best):
                best = p
    return best
