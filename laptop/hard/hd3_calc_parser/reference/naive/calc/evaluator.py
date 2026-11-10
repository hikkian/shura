from .errors import EvalError
from .parser import parse


def _ev(node, vars):
    k = node[0]
    if k == "num":
        return node[1]
    if k == "var":
        if node[1] not in vars:
            raise EvalError("unknown variable %s" % node[1])
        return vars[node[1]]
    if k == "neg":
        return -_ev(node[1], vars)
    if k == "pos":
        return _ev(node[1], vars)
    if k == "call":
        args = [_ev(a, vars) for a in node[2]]
        f = {"abs": abs, "min": min, "max": max}.get(node[1])
        if f is None:
            raise EvalError("unknown function %s" % node[1])
        return f(*args)
    _, op, l, r, _pos = node
    a, b = _ev(l, vars), _ev(r, vars)
    return {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "/": lambda: a / b,
            "%": lambda: a % b, "**": lambda: a ** b}[op]()


def evaluate(text, variables=None):
    return _ev(parse(text), variables or {})
