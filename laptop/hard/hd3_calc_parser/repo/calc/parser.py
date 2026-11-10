from .tokens import tokenize

# AST: ("num", v, pos) | ("var", name, pos) | ("neg", node, pos) | ("bin", op, left, right, pos)


class Parser:
    def __init__(self, text):
        self.text = text
        self.toks = tokenize(text)
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self):
        node = self.expr()
        if self.peek() is not None:
            raise ValueError("unexpected token")
        return node

    def expr(self):
        left = self.term()
        while self.peek() and self.peek()[1] in "+-":
            op = self.next()
            left = ("bin", op[1], left, self.term(), op[2])
        return left

    def term(self):
        left = self.unary()
        while self.peek() and self.peek()[1] in "*/":
            op = self.next()
            left = ("bin", op[1], left, self.unary(), op[2])
        return left

    def unary(self):
        t = self.peek()
        if t and t[1] == "-":
            self.next()
            return ("neg", self.unary(), t[2])
        return self.atom()

    def atom(self):
        t = self.next()
        if t is None:
            raise ValueError("unexpected end")
        if t[0] == "num":
            return ("num", float(t[1]) if "." in t[1] else int(t[1]), t[2])
        if t[0] == "name":
            return ("var", t[1], t[2])
        if t[1] == "(":
            node = self.expr()
            self.next()
            return node
        raise ValueError("unexpected token")


def parse(text):
    return Parser(text).parse()
