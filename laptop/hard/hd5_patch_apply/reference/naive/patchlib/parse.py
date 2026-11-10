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
    lines: list = field(default_factory=list)


def parse_patch(text):
    hunks = []
    cur = None
    for line in text.splitlines():
        if line.startswith(("--- ", "+++ ", "diff ", "index ")):
            continue
        m = HEADER.match(line)
        if m:
            a, b, c, d = m.groups()
            cur = Hunk(int(a), int(1 if b is None else b), int(c), int(1 if d is None else d))
            hunks.append(cur)
        elif cur is not None and line.startswith("\\"):
            continue
        elif cur is not None:
            cur.lines.append((line[:1] or " ", line[1:]))
    if not hunks:
        raise PatchError("no hunks")
    return hunks
