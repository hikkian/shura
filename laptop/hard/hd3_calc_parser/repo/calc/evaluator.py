from .parser import parse


def _ev(node, vars):
    k = node[0]
    if k == "num":
        return node[1]
    if k == "var":
        return vars[node[1]]
    if k == "neg":
        return -_ev(node[1], vars)
    _, op, l, r, _pos = node
    a, b = _ev(l, vars), _ev(r, vars)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return a / b


def evaluate(text, variables=None):
    return _ev(parse(text), variables or {})
