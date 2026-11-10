class CalcError(Exception):
    def __init__(self, message, pos=None):
        super().__init__(message)
        self.pos = pos


class ParseError(CalcError):
    pass


class EvalError(CalcError):
    pass
