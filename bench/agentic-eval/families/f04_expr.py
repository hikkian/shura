FAMILY = {
    "name": "f04_expr", "lang": "python",
    "implement": {
        "prompt": """Write the module `calc.py` with the function `evaluate(text, variables=None)` (standard library only; do NOT use eval/exec/ast.literal_eval).

It evaluates an arithmetic expression and returns a float (or an int when every number involved is an int and no division
happened; division always returns a float: 7/2 -> 3.5, 6/3 -> 2.0).
Grammar and rules:
  * numbers: 12, 3.5, .5, 2e3 (decimal digits, optional fraction, optional exponent); variables: names [A-Za-z_][A-Za-z0-9_]*
    looked up in `variables` (a dict); an unknown name raises NameError.
  * binary operators, from lowest to highest precedence: + -  |  * / %  |  ^ (power, RIGHT associative: 2^3^2 = 2^9 = 512).
    % is the remainder with the sign of the divisor (Python's %). Division or % by zero raises ZeroDivisionError.
  * unary minus and plus bind tighter than * / % but LOOSER than ^: -2^2 = -4, and 2^-1 = 0.5 (a unary sign is allowed right
    after ^). Parentheses group.
  * functions: abs(x), min(a, b, ...), max(a, b, ...) (at least one argument), sqrt(x), round(x) (round half away from zero,
    returns an int). Wrong argument count or sqrt of a negative number raises ValueError.
  * whitespace is ignored anywhere between tokens. Anything else (unbalanced parentheses, a missing operand, trailing garbage,
    an unknown function, an empty string) raises SyntaxError.
""",
        "start": {"calc.py": '"""Expression evaluator (to be written)."""\n\n\ndef evaluate(text, variables=None):\n    raise NotImplementedError\n'},
        "solution": {"calc.py": '''"""Expression evaluator."""
import math
import re

_TOKEN = re.compile(r"\\s*(?:(\\d+\\.?\\d*(?:[eE][+-]?\\d+)?|\\.\\d+(?:[eE][+-]?\\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(.))")


def _tokens(text):
    out, pos = [], 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            break
        pos = m.end()
        num, name, op = m.groups()
        if num is not None:
            out.append(("num", float(num) if any(c in num for c in ".eE") else int(num)))
        elif name is not None:
            out.append(("name", name))
        elif op is not None and not op.isspace():
            out.append(("op", op))
    return out


def _round_half_away(x):
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


class _Parser:
    def __init__(self, tokens, variables):
        self.t, self.i, self.vars = tokens, 0, variables or {}

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expect(self, ch):
        if self.take() != ("op", ch):
            raise SyntaxError(f"expected {ch!r}")

    def expr(self):
        v = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            r = self.term()
            v = v + r if op == "+" else v - r
        return v

    def term(self):
        v = self.unary()
        while self.peek() in (("op", "*"), ("op", "/"), ("op", "%")):
            op = self.take()[1]
            r = self.unary()
            if op == "*":
                v = v * r
            elif r == 0:
                raise ZeroDivisionError("division by zero")
            elif op == "/":
                v = v / r
            else:
                v = v % r
        return v

    def unary(self):
        if self.peek() in (("op", "-"), ("op", "+")):
            op = self.take()[1]
            v = self.unary()
            return -v if op == "-" else v
        return self.power()

    def power(self):
        base = self.atom()
        if self.peek() == ("op", "^"):
            self.take()
            exp = self.signed_power_operand()
            return base ** exp
        return base

    def signed_power_operand(self):
        if self.peek() in (("op", "-"), ("op", "+")):
            op = self.take()[1]
            v = self.signed_power_operand()
            return -v if op == "-" else v
        return self.power()

    def atom(self):
        kind, val = self.take()
        if kind == "num":
            return val
        if kind == "op" and val == "(":
            v = self.expr()
            self.expect(")")
            return v
        if kind == "name":
            if self.peek() == ("op", "("):
                self.take()
                args = []
                if self.peek() != ("op", ")"):
                    args.append(self.expr())
                    while self.peek() == ("op", ","):
                        self.take()
                        args.append(self.expr())
                self.expect(")")
                return self.call(val, args)
            if val not in self.vars:
                raise NameError(val)
            return self.vars[val]
        raise SyntaxError("unexpected token")

    def call(self, name, args):
        if name in ("min", "max"):
            if not args:
                raise ValueError(f"{name} needs an argument")
            return (min if name == "min" else max)(args)
        if name not in ("abs", "sqrt", "round"):
            raise SyntaxError(f"unknown function {name}")
        if len(args) != 1:
            raise ValueError(f"{name} takes one argument")
        x = args[0]
        if name == "abs":
            return abs(x)
        if name == "round":
            return _round_half_away(x)
        if x < 0:
            raise ValueError("sqrt of a negative number")
        return math.sqrt(x)


def evaluate(text, variables=None):
    tokens = _tokens(text)
    if not tokens:
        raise SyntaxError("empty expression")
    p = _Parser(tokens, variables)
    v = p.expr()
    if p.i != len(tokens):
        raise SyntaxError("trailing input")
    return v
'''},
    },
    "hidden": {"test_calc.py": '''import unittest
from calc import evaluate


class Arithmetic(unittest.TestCase):
    def test_precedence_and_types(self):
        self.assertEqual(evaluate("1 + 2 * 3"), 7)
        self.assertIsInstance(evaluate("1 + 2 * 3"), int)
        self.assertEqual(evaluate("(1 + 2) * 3"), 9)
        self.assertEqual(evaluate("7/2"), 3.5)
        self.assertIsInstance(evaluate("6/3"), float)
        self.assertEqual(evaluate("10 - 4 - 3"), 3)
        self.assertEqual(evaluate("2 * 3 % 4"), 2)

    def test_numbers(self):
        self.assertEqual(evaluate(".5 + 1.5"), 2.0)
        self.assertEqual(evaluate("2e3"), 2000.0)
        self.assertEqual(evaluate("1.5e-1 * 2"), 0.3)

    def test_power_is_right_associative_and_binds_tighter_than_unary_minus(self):
        self.assertEqual(evaluate("2^3^2"), 512)
        self.assertEqual(evaluate("-2^2"), -4)
        self.assertEqual(evaluate("2^-1"), 0.5)
        self.assertEqual(evaluate("(-2)^2"), 4)
        self.assertEqual(evaluate("2 * -3"), -6)
        self.assertEqual(evaluate("--3"), 3)
        self.assertEqual(evaluate("+4"), 4)
        self.assertEqual(evaluate("-3 % 5"), 2)

    def test_modulo_sign_of_divisor(self):
        self.assertEqual(evaluate("7 % -3"), -2)
        self.assertEqual(evaluate("-7 % 3"), 2)

    def test_zero_division(self):
        for e in ["1/0", "5 % 0", "1/(2-2)"]:
            with self.assertRaises(ZeroDivisionError):
                evaluate(e)


class Variables(unittest.TestCase):
    def test_lookup(self):
        self.assertEqual(evaluate("x * y + 1", {"x": 3, "y": 4}), 13)
        self.assertEqual(evaluate("_a1 ^ 2", {"_a1": 5}), 25)

    def test_unknown_name(self):
        with self.assertRaises(NameError):
            evaluate("x + 1", {})
        with self.assertRaises(NameError):
            evaluate("x")


class Functions(unittest.TestCase):
    def test_functions(self):
        self.assertEqual(evaluate("abs(-3)"), 3)
        self.assertEqual(evaluate("min(3, 1, 2)"), 1)
        self.assertEqual(evaluate("max(3, 1+5, 2)"), 6)
        self.assertEqual(evaluate("max(7)"), 7)
        self.assertEqual(evaluate("sqrt(16)"), 4.0)
        self.assertEqual(evaluate("2 * max(1, 2) ^ 2"), 8)

    def test_round_half_away_from_zero(self):
        self.assertEqual(evaluate("round(2.5)"), 3)
        self.assertEqual(evaluate("round(-2.5)"), -3)
        self.assertEqual(evaluate("round(2.4)"), 2)
        self.assertEqual(evaluate("round(-0.4)"), 0)
        self.assertIsInstance(evaluate("round(2.5)"), int)

    def test_bad_calls(self):
        for e in ["abs()", "abs(1, 2)", "sqrt(-1)", "max()", "round(1, 2)"]:
            with self.assertRaises(ValueError):
                evaluate(e)
        with self.assertRaises(SyntaxError):
            evaluate("foo(1)")


class Syntax(unittest.TestCase):
    def test_errors(self):
        for e in ["", "   ", "1 +", "* 2", "(1 + 2", "1 + 2)", "1 2", "1 +* 2", "()", "1 $ 2", "min(1,)", "min(,1)", "2^"]:
            with self.subTest(e=e):
                with self.assertRaises(SyntaxError):
                    evaluate(e)

    def test_whitespace_everywhere(self):
        self.assertEqual(evaluate("  1\\t+\\n2 * ( 3 ) "), 7)
'''},
    "bugfix": {
        "symptom": "An expression evaluator (calc.py) gets the power operator wrong: 2^3^2 should be 512 (power is right associative) but evaluate returns 64.",
        "bug": [("calc.py", '''        base = self.atom()
        if self.peek() == ("op", "^"):
            self.take()
            exp = self.signed_power_operand()
            return base ** exp
        return base''', '''        base = self.atom()
        while self.peek() == ("op", "^"):
            self.take()
            base = base ** self.atom()
        return base''')],
        "repro": {"test_repro.py": 'import unittest\nfrom calc import evaluate\n\n\nclass Repro(unittest.TestCase):\n    def test_power_chain(self):\n        self.assertEqual(evaluate("2^3^2"), 512)\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
