from .errors import ConflictError
from .parse import parse_patch


def _split(text):
    parts = text.split("\n")
    out = [p + "\n" for p in parts[:-1]]
    if parts[-1] != "":
        out.append(parts[-1])
    return out


def _side(h, drop, keep_new):
    """Строки одной стороны ханка вместе с признаком конца строки."""
    res = []
    for op, t, noeol in h.lines:
        if op == " " or op == keep_new:
            res.append(t + ("" if noeol else "\n"))
    return res


def apply_patch(text, patch, reverse=False):
    orig = _split(text)
    out = []
    cursor = 0
    drift = 0
    for n, h in enumerate(parse_patch(patch), 1):
        if reverse:
            old, new = _side(h, None, "+"), _side(h, None, "-")
            start, cnt = h.new_start, h.new_count
        else:
            old, new = _side(h, None, "-"), _side(h, None, "+")
            start, cnt = h.old_start, h.old_count
        nominal = start - 1 if cnt > 0 else start
        want = nominal + drift
        if not old:
            p = min(max(want, cursor), len(orig))
        else:
            best = None
            for q in range(cursor, len(orig) - len(old) + 1):
                if orig[q:q + len(old)] == old:
                    dist = abs(q - want)
                    if best is None or dist < best[0]:
                        best = (dist, q)
            if best is None:
                raise ConflictError("hunk %d does not apply" % n, n)
            p = best[1]
        out.extend(orig[cursor:p])
        out.extend(new)
        cursor = p + len(old)
        drift = p - nominal
    out.extend(orig[cursor:])
    return "".join(out)
