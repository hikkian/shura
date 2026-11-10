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
        if self.interval < 1:
            raise ValueError("interval must be >= 1")
