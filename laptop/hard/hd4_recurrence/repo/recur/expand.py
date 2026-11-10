from datetime import date, timedelta

from .calendar_utils import add_months


def expand(start, rule, begin, end):
    out = []
    n = 0
    k = 0
    while True:
        if rule.freq == "DAILY":
            d = start + timedelta(days=k * rule.interval)
        elif rule.freq == "WEEKLY":
            d = start + timedelta(weeks=k * rule.interval)
        elif rule.freq == "MONTHLY":
            d = add_months(start, k * rule.interval)
        else:
            try:
                d = start.replace(year=start.year + k * rule.interval)
            except ValueError:
                d = start.replace(year=start.year + k * rule.interval, day=28)
        k += 1
        if d > end or (rule.until and d > rule.until):
            break
        if d >= begin:
            out.append(d)
            n += 1
            if rule.count and n >= rule.count:
                break
    return out
