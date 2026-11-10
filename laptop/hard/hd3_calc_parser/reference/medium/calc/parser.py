from .errors import ParseError
from .tokens import tokenize

# AST: ("num", v, pos) | ("var", name, pos) | ("neg", node, pos) | ("pos", node, pos)
#      | ("bin", op, left, right, pos) | ("call", name, [args], pos)


class Parser:
    def __init__(self, text):
        self.text = text
        self.toks = tokenize(text)
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def is_op(self, *ops):
        t = self.peek()
        return t is not None and t[0] == "op" and t[1] in ops

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def fail(self, t=None):
        t = t or self.peek()
        if t is None:
            raise ParseError("unexpected end of expression", len(self.text))
        raise ParseError("unexpected %r" % t[1], t[2])

    def expect(self, op):
        if not self.is_op(op):
            self.fail()
        return self.next()

    def parse(self):
        node = self.expr()
        if self.peek() is not None:
            self.fail()
        return node

    def expr(self):
        left = self.term()
        while self.is_op("+", "-"):
            op = self.next()
            left = ("bin", op[1], left, self.term(), op[2])
        return left

    def term(self):
        left = self.unary()
        while self.is_op("*", "/", "%"):
            op = self.next()
            left = ("bin", op[1], left, self.unary(), op[2])
        return left

    def unary(self):
        if self.is_op("-", "+"):
            t = self.next()
            operand = self.atom()
            node = ("neg" if t[1] == "-" else "pos", operand, t[2])
            if self.is_op("**"):
                op = self.next()
                node = ("bin", "**", node, self.unary(), op[2])
            return node
        return self.power()

    def atom_pow(self):
        base = self.power()
        return base

    def power(self):
        # унарный минус привязан к степени «снизу»: -2**2 == 4, зато 2**-2 работает
        base = self.atom()
        if self.is_op("**"):
            op = self.next()
            return ("bin", "**", base, self.unary(), op[2])
        return base

    def atom(self):
        t = self.next()
        if t is None:
            self.fail(None)
        if t[0] == "num":
            s = t[1]
            return ("num", int(s) if s.isdigit() else float(s), t[2])
        if t[0] == "name":
            if self.is_op("("):
                self.next()
                args = []
                if not self.is_op(")"):
                    args.append(self.expr())
                    while self.is_op(","):
                        self.next()
                        args.append(self.expr())
                self.expect(")")
                return ("call", t[1], args, t[2])
            return ("var", t[1], t[2])
        if t[1] == "(":
            node = self.expr()
            self.expect(")")
            return node
        self.i -= 1
        self.fail(t)


def parse(text):
    return Parser(text).parse()
