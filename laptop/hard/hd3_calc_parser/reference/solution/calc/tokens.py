import re

from .errors import ParseError

TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|(?P<name>[A-Za-z_]\w*)|(?P<op>\*\*|[-+*/%(),]))")


def tokenize(text):
    """-> список (kind, value, pos); kind: 'num' | 'name' | 'op'"""
    out = []
    i = 0
    while i < len(text):
        if text[i:].strip() == "":
            break
        m = TOKEN_RE.match(text, i)
        if not m:
            j = i
            while text[j].isspace():
                j += 1
            raise ParseError("unexpected character %r" % text[j], j)
        kind = m.lastgroup
        out.append((kind, m.group(kind), m.start(kind)))
        i = m.end()
    return out
