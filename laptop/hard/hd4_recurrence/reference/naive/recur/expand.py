from datetime import date, timedelta

from .calendar_utils import add_months


def expand(start, rule, begin, end):
    out = []
    n = 0
    k = 0
    ex = set(rule.exdates)
    while True:
        if rule.freq == "DAILY":
            cands = [start + timedelta(days=k * rule.interval)]
        elif rule.freq == "WEEKLY":
            base = start + timedelta(weeks=k * rule.interval)
            days = rule.byweekday if rule.byweekday is not None else [start.weekday()]
            cands = [base - timedelta(days=base.weekday()) + timedelta(days=w) for w in sorted(days)]
            cands = [c for c in cands if c >= start]
        elif rule.freq == "MONTHLY":
            first = add_months(start.replace(day=1), k * rule.interval)
            days = rule.bymonthday if rule.bymonthday is not None else [start.day]
            cands = [first.replace(day=min(x, 28)) if x > 28 else first.replace(day=x) for x in days if x > 0]
            cands = [c for c in cands if c >= start]
        else:
            cands = [date(start.year + k * rule.interval, start.month, min(start.day, 28))]
        k += 1
        if all(c > end for c in cands) or all(rule.until and c > rule.until for c in cands) and cands:
            break
        for d in cands:
            if d > end or (rule.until and d > rule.until):
                return out
            if d >= begin and d not in ex:
                out.append(d)
                n += 1
                if rule.count and n >= rule.count:
                    return out
    return out
