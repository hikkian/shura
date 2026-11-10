from .parse import parse_patch
from .errors import ConflictError


def apply_patch(text, patch, reverse=False):
    lines = text.splitlines()
    out = []
    pos = 0
    for n, h in enumerate(parse_patch(patch), 1):
        ops = h.lines
        if reverse:
            ops = [({"-": "+", "+": "-"}.get(op, op), s) for op, s in ops]
        start = h.new_start if reverse else h.old_start
        start = max(start - 1, 0) if (h.new_count if reverse else h.old_count) else start
        out.extend(lines[pos:start])
        pos = start
        for op, s in ops:
            if op in (" ", "-"):
                if pos >= len(lines) or lines[pos] != s:
                    raise ConflictError("hunk %d does not apply" % n, n)
                pos += 1
            if op in (" ", "+"):
                out.append(s)
    out.extend(lines[pos:])
    return "\n".join(out) + "\n"
