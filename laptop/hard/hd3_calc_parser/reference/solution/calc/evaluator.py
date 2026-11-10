from .errors import EvalError
from .parser import parse


def _arity(name, args, pos, lo, hi=None):
    if len(args) < lo or (hi is not None and len(args) > hi):
        raise EvalError("wrong number of arguments for %s: %d" % (name, len(args)), pos)


FUNCS = {
    "abs": lambda a, p: (_arity("abs", a, p, 1, 1), abs(a[0]))[1],
    "min": lambda a, p: (_arity("min", a, p, 1), min(a))[1],
    "max": lambda a, p: (_arity("max", a, p, 1), max(a))[1],
}


def _ev(node, vars):
    k = node[0]
    if k == "num":
        return node[1]
    if k == "var":
        if node[1] not in vars:
            raise EvalError("unknown variable %r" % node[1], node[2])
        return vars[node[1]]
    if k == "neg":
        return -_ev(node[1], vars)
    if k == "pos":
        return +_ev(node[1], vars)
    if k == "call":
        f = FUNCS.get(node[1])
        if f is None:
            raise EvalError("unknown function %r" % node[1], node[3])
        return f([_ev(a, vars) for a in node[2]], node[3])
    _, op, l, r, pos = node
    a, b = _ev(l, vars), _ev(r, vars)
    try:
        if op == "+":
            res = a + b
        elif op == "-":
            res = a - b
        elif op == "*":
            res = a * b
        elif op == "/":
            res = a / b
        elif op == "%":
            res = a % b
        else:
            res = a ** b
    except ZeroDivisionError:
        raise EvalError("division by zero", pos) from None
    except OverflowError:
        raise EvalError("numeric overflow", pos) from None
    if isinstance(res, float) and res in (float("inf"), float("-inf")) and abs(a) != float("inf") and abs(b) != float("inf"):
        raise EvalError("numeric overflow", pos)
    if isinstance(res, complex):
        raise EvalError("result is not a real number", pos)
    return res


def evaluate(text, variables=None):
    return _ev(parse(text), variables or {})
