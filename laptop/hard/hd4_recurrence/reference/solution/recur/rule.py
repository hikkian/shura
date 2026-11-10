from dataclasses import dataclass
from typing import Optional, Sequence
import datetime

FREQS = ("DAILY", "WEEKLY", "MONTHLY", "YEARLY")


@dataclass
class Rule:
    freq: str
    interval: int = 1
    byweekday: Optional[Sequence[int]] = None
    bymonthday: Optional[Sequence[int]] = None
    count: Optional[int] = None
    until: Optional[datetime.date] = None
    exdates: Sequence[datetime.date] = ()

    def __post_init__(self):
        if self.freq not in FREQS:
            raise ValueError("bad freq %r" % (self.freq,))
        if not isinstance(self.interval, int) or isinstance(self.interval, bool) or self.interval < 1:
            raise ValueError("interval must be an integer >= 1")
        if self.byweekday is not None:
            if self.freq != "WEEKLY":
                raise ValueError("byweekday is only for WEEKLY")
            if not self.byweekday or any(not isinstance(x, int) or not 0 <= x <= 6 for x in self.byweekday):
                raise ValueError("bad byweekday")
        if self.bymonthday is not None:
            if self.freq != "MONTHLY":
                raise ValueError("bymonthday is only for MONTHLY")
            if not self.bymonthday or any(not isinstance(x, int) or x == 0 or not -31 <= x <= 31 for x in self.bymonthday):
                raise ValueError("bad bymonthday")
        if self.count is not None and (not isinstance(self.count, int) or self.count < 1):
            raise ValueError("count must be >= 1")
