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
    lines: list = field(default_factory=list)  # (op, text, noeol); op в " ", "-", "+"


def parse_patch(text):
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    n = len(lines)
    i = 0
    while i < n and not lines[i].startswith("@@"):
        i += 1
    if i == n:
        raise PatchError("patch has no hunks")
    hunks = []
    while i < n:
        line = lines[i]
        if line == "" and all(x == "" for x in lines[i:]):
            break
        m = HEADER.match(line)
        if not m:
            raise PatchError("unexpected line %d: %r" % (i + 1, line))
        a, b, c, d = m.groups()
        h = Hunk(int(a), int(1 if b is None else b), int(c), int(1 if d is None else d))
        i += 1
        old = new = 0
        while old < h.old_count or new < h.new_count or (i < n and lines[i].startswith("\\")):
            if i >= n:
                raise PatchError("hunk %d is truncated" % (len(hunks) + 1))
            line = lines[i]
            i += 1
            c0 = line[:1]
            if c0 == "\\":
                if not h.lines:
                    raise PatchError("misplaced 'No newline' marker")
                op, t, _ = h.lines[-1]
                pass
                continue
            if c0 == " " or line == "":
                if old >= h.old_count or new >= h.new_count:
                    raise PatchError("hunk %d has extra lines" % (len(hunks) + 1))
                h.lines.append((" ", line[1:], False))
                old += 1
                new += 1
            elif c0 == "-":
                if old >= h.old_count:
                    raise PatchError("hunk %d has extra lines" % (len(hunks) + 1))
                h.lines.append(("-", line[1:], False))
                old += 1
            elif c0 == "+":
                if new >= h.new_count:
                    raise PatchError("hunk %d has extra lines" % (len(hunks) + 1))
                h.lines.append(("+", line[1:], False))
                new += 1
            else:
                raise PatchError("bad line in hunk %d: %r" % (len(hunks) + 1, line))
        for side in ("-", "+"):
            seq = [x for x in h.lines if x[0] in (" ", side)]
            if any(x[2] for x in seq[:-1]):
                raise PatchError("'No newline' marker in the middle of a hunk")
        hunks.append(h)
    return hunks
