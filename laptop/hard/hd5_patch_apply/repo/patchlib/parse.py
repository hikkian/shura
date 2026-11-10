import re
from dataclasses import dataclass, field

from .errors import PatchError

HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list = field(default_factory=list)  # (op, text) где op в " ", "-", "+"


def parse_patch(text):
    hunks = []
    cur = None
    for line in text.splitlines():
        if line.startswith(("---", "+++", "diff ", "index ")):
            continue
        m = HEADER.match(line)
        if m:
            a, b, c, d = m.groups()
            cur = Hunk(int(a), int(b or 1), int(c), int(d or 1))
            hunks.append(cur)
        elif cur is not None and line[:1] in (" ", "-", "+"):
            cur.lines.append((line[0], line[1:]))
    return hunks
