import re

from .errors import ParseError

TOKEN_RE = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?)|(?P<name>[A-Za-z_]\w*)|(?P<op>[-+*/()]))")


def tokenize(text):
    """-> список (kind, value, pos); kind: 'num' | 'name' | 'op'"""
    out = []
    i = 0
    while i < len(text):
        if text[i:].strip() == "":
            break
        m = TOKEN_RE.match(text, i)
        if not m:
            raise ValueError("bad character %r" % text[i])
        kind = m.lastgroup
        out.append((kind, m.group(kind), m.start(kind)))
        i = m.end()
    return out
