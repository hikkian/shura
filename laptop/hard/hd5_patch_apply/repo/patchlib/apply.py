from .parse import parse_patch
from .errors import ConflictError


def apply_patch(text, patch, reverse=False):
    lines = text.splitlines(keepends=True)
    out = []
    pos = 0
    for n, h in enumerate(parse_patch(patch), 1):
        start = h.old_start - 1
        out.extend(lines[pos:start])
        pos = start
        for op, s in h.lines:
            if op in (" ", "-"):
                if pos >= len(lines) or lines[pos].rstrip("\n") != s:
                    raise ConflictError("hunk %d does not apply" % n, n)
                pos += 1
            if op in (" ", "+"):
                out.append(s + "\n")
    out.extend(lines[pos:])
    return "".join(out)
