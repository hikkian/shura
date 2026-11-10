from datetime import date, timedelta

from .calendar_utils import days_in_month, monday_of


def _periods(start, rule):
    """Бесконечный генератор: для каждого периода — отсортированный список дат-кандидатов (без отбора по start)."""
    k = 0
    iv = rule.interval
    if rule.freq == "DAILY":
        while True:
            try:
                yield [start + timedelta(days=k * iv)]
            except OverflowError:
                return
            k += 1
    elif rule.freq == "WEEKLY":
        days = sorted(set(rule.byweekday if rule.byweekday is not None else [start.weekday()]))
        mon = monday_of(start)
        while True:
            try:
                m = mon + timedelta(weeks=k * iv)
            except OverflowError:
                return
            row = []
            for w in days:
                try:
                    row.append(m + timedelta(days=w))
                except OverflowError:
                    pass
            yield row
            k += 1
    elif rule.freq == "MONTHLY":
        base = start.year * 12 + start.month - 1
        spec = rule.bymonthday if rule.bymonthday is not None else [start.day]
        while True:
            y, m = divmod(base + k * iv, 12)
            m += 1
            if y > 9999:
                return
            n = days_in_month(y, m)
            days = sorted({x if x > 0 else n + 1 + x for x in spec if 1 <= (x if x > 0 else n + 1 + x) <= n})
            yield [date(y, m, x) for x in days]
            k += 1
    else:
        while True:
            y = start.year + k * iv
            if y > 9999:
                return
            yield [date(y, start.month, min(start.day, days_in_month(y, start.month)))]
            k += 1


def expand(start, rule, begin, end):
    limit = min(end, rule.until) if rule.until else end
    ex = set(rule.exdates)
    out = []
    n = 0
    for cands in _periods(start, rule):
        if cands and cands[0] > limit and all(c > limit for c in cands):
            break
        if not cands:
            continue
        for d in cands:
            if d < start:
                continue
            if d > limit:
                return out
            if d not in ex:
                n += 1
            if d >= begin and d not in ex:
                out.append(d)
            if rule.count and n >= rule.count:
                return out
    return out
