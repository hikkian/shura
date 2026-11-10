from dataclasses import dataclass, field
from datetime import time
from zoneinfo import ZoneInfo


def parse_hm(s):
    """'09:30' -> datetime.time(9, 30)"""
    h, m = s.split(":")
    return time(int(h), int(m))


@dataclass
class Schedule:
    tz: str
    windows: dict = field(default_factory=dict)  # weekday (0=Mon) -> [("HH:MM", "HH:MM"), ...]

    @property
    def zone(self):
        return ZoneInfo(self.tz)

    def day_windows(self, weekday):
        """Окна, начинающиеся в этот день недели, как пары datetime.time."""
        return [(parse_hm(a), parse_hm(b)) for a, b in self.windows.get(weekday, [])]
