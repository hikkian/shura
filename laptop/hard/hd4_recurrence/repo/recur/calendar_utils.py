import calendar
from datetime import date, timedelta


def days_in_month(year, month):
    return calendar.monthrange(year, month)[1]


def add_months(d, n):
    """Сдвиг на n месяцев; число месяца прижимается к концу месяца (31 янв + 1 мес = 28/29 фев)."""
    y, m = divmod(d.year * 12 + d.month - 1 + n, 12)
    return date(y, m + 1, min(d.day, days_in_month(y, m + 1)))


def monday_of(d):
    return d - timedelta(days=d.weekday())
