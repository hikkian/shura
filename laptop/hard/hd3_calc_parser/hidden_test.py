import unittest

from calc import evaluate, CalcError, ParseError, EvalError


def perr(text, pos):
    try:
        evaluate(text)
    except ParseError as e:
        return e.pos == pos or "pos=%r" % (e.pos,)
    except Exception as e:  # noqa
        return "got %s: %s" % (type(e).__name__, e)
    return "no error"


class H(unittest.TestCase):
    def test_precedence_and_left_assoc(self):
        for s, v in [("1 + 2 * 3", 7), ("10 - 2 * 3", 4), ("2 - 3 - 4", -5), ("8 / 4 / 2", 1.0),
                     ("2 * 3 % 4", 2), ("20 % 6 * 2", 4), ("(1 + 2) * (3 + 4)", 21), ("100 / 10 * 5", 50.0),
                     ("1 + 2 ** 3 * 2", 17), ("2 * 3 ** 2", 18), ("10 % 4 ** 2", 10)]:
            self.assertEqual(evaluate(s), v, s)

    def test_power_right_assoc(self):
        for s, v in [("2 ** 3 ** 2", 512), ("(2 ** 3) ** 2", 64), ("2 ** 2 ** 3", 256), ("3 ** 2 ** 0", 3)]:
            self.assertEqual(evaluate(s), v, s)

    def test_unary_minus_vs_power(self):
        for s, v in [("-2 ** 2", -4), ("(-2) ** 2", 4), ("2 ** -2", 0.25), ("-2 ** -2", -0.25),
                     ("2 ** -1 ** 2", 0.5), ("-3 ** 2 + 1", -8)]:
            self.assertEqual(evaluate(s), v, s)

    def test_unary_chains_and_mixed(self):
        for s, v in [("--3", 3), ("- - 3", 3), ("-+-3", 3), ("+3", 3), ("2 - -3", 5), ("2 * -3", -6),
                     ("-(1 + 2) * 2", -6), ("3 - +2", 1), ("-2 * 3", -6), ("-7 % 3", 2), ("7 % -3", -2)]:
            self.assertEqual(evaluate(s), v, s)

    def test_numbers(self):
        for s, v in [(".5 + 1", 1.5), ("1. + 1", 2.0), ("1e3", 1000.0), ("2.5E-2", 0.025), ("1e2 * 2", 200.0),
                     ("0.1 + 0.2", 0.1 + 0.2), ("007", 7)]:
            self.assertEqual(evaluate(s), v, s)
        self.assertIsInstance(evaluate("2 ** 3"), int)
        self.assertEqual(evaluate("2 ** 100"), 1267650600228229401496703205376)
        self.assertEqual(evaluate("10 ** 20 + 1"), 100000000000000000001)
        self.assertEqual(evaluate("7 / 2"), 3.5)

    def test_variables_and_functions(self):
        self.assertEqual(evaluate("x * y - z", {"x": 3, "y": 4, "z": 2}), 10)
        self.assertEqual(evaluate("_a1 + b_2", {"_a1": 1, "b_2": 2}), 3)
        self.assertEqual(evaluate("max(1, 2 ** 3, 5)"), 8)
        self.assertEqual(evaluate("min(3)"), 3)
        self.assertEqual(evaluate("abs(-3 - 4)"), 7)
        self.assertEqual(evaluate("max(min(4, 9), abs(-2)) * 2"), 8)
        self.assertEqual(evaluate("  max ( 1 ,\t2 )  "), 2)
        self.assertEqual(evaluate("-max(1, 2) ** 2"), -4)

    def test_parse_error_positions(self):
        cases = [("", 0), ("   ", 3), ("1 +", 3), ("* 2", 0), ("(1 + 2", 6), ("1 + 2)", 5), ("2 3", 2),
                 ("2(3)", 1), ("2 x", 2), ("1 + * 2", 4), ("min(1,)", 6), ("min(1 2)", 6), ("1 $ 2", 2),
                 ("  $", 2), ("()", 1), ("1 ** ", 5), ("1e", 1), ("a b", 2), ("(1)(2)", 3), ("1 +\n", 4)]
        for text, pos in cases:
            self.assertIs(perr(text, pos), True, "%r expected pos %d: %s" % (text, pos, perr(text, pos)))

    def test_parse_errors_are_parse_errors(self):
        for text in ["", "1 +", "(", ")", "1 2", "1 $", "f(", "f(1,", "1 + + ", "*", "** 2", "2 **"]:
            with self.assertRaises(ParseError, msg=repr(text)):
                evaluate(text)
            with self.assertRaises(CalcError, msg=repr(text)):
                evaluate(text)

    def test_division_by_zero(self):
        for text, pos in [("1 / 0", 2), ("1 + 4 / 0", 6), ("5 % 0", 2), ("0 ** -1", 2), ("1 / (2 - 2)", 2),
                          ("0.0 ** -2", 4), ("1 / 0.0", 2)]:
            with self.assertRaises(EvalError, msg=text) as cm:
                evaluate(text)
            self.assertEqual(cm.exception.pos, pos, text)

    def test_unknown_names_and_arity(self):
        with self.assertRaises(EvalError) as cm:
            evaluate("1 + foo * 2")
        self.assertEqual(cm.exception.pos, 4)
        self.assertIn("foo", str(cm.exception))
        with self.assertRaises(EvalError) as cm:
            evaluate("sqrt(4)")
        self.assertEqual(cm.exception.pos, 0)
        self.assertIn("sqrt", str(cm.exception))
        with self.assertRaises(EvalError):
            evaluate("x", {"y": 1})
        for text in ["abs()", "abs(1, 2)", "min()", "max()"]:
            with self.assertRaises(EvalError, msg=text):
                evaluate(text)

    def test_non_real_and_overflow(self):
        for text in ["(-8) ** (1 / 3)", "(-1) ** 0.5", "(-2.5) ** 1.5", "10.0 ** 400", "2.0 ** 5000", "1e308 * 10"]:
            with self.assertRaises(EvalError, msg=text):
                evaluate(text)
        self.assertEqual(evaluate("(-8) ** 2"), 64)
        self.assertEqual(evaluate("(-2) ** 3"), -8)
        self.assertEqual(evaluate("(-2.0) ** 2"), 4.0)
        self.assertEqual(evaluate("4 ** 0.5"), 2.0)

    def test_only_calc_errors_escape(self):
        for text in ["1 / 0", "foo", "1 +", "(-1) ** 0.5", "$", "abs(1, 2)", "10.0 ** 400", "10 ** 400 / 3"]:
            try:
                evaluate(text)
            except CalcError:
                pass
            except Exception as e:  # noqa
                self.fail("%r -> %s: %s" % (text, type(e).__name__, e))
        self.assertEqual(evaluate("10 ** 400 / 10 ** 399"), 10.0)
