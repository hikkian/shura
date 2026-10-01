FAMILY = {
    "name": "f11_money", "lang": "python",
    "implement": {
        "prompt": """Write the module `money.py` (standard library only; use decimal.Decimal, never float arithmetic) with the class `Money`.

Money(amount, currency): amount is an int, a str like "12.30" or a Decimal (a float raises TypeError); currency is a 3-letter upper-case
  code (otherwise ValueError). The amount is stored rounded to the currency's minor unit with ROUND_HALF_EVEN (banker's rounding):
  2 decimals for all currencies except JPY (0 decimals) and KWD (3 decimals).
Attributes: amount (Decimal, already rounded) and currency (str). Instances are immutable, hashable and compare equal when amount
  and currency are equal (Money("1.10","USD") == Money("1.1","USD")); comparing different currencies with == is False, with < etc. raises ValueError.
Arithmetic: + and - with another Money of the SAME currency (otherwise ValueError) return a Money; * and / with an int or Decimal
  return a Money (rounded as above; / by zero raises ZeroDivisionError); unary - works; a Money times a float raises TypeError.
allocate(ratios) -> list of Money: split the amount in proportion to the non-negative integer `ratios` (at least one must be > 0) so that
  the parts add up EXACTLY to the original amount: every part is first rounded DOWN to the minor unit, then the remaining minor
  units are given one each to the parts with the largest fractional remainder (ties: the earlier part first). A part with ratio 0 gets
  0 and never receives a leftover unit. Negative amounts allocate the same way on the absolute value and negate.
str(): "12.30 USD" (always the full number of minor-unit digits). repr(): Money('12.30', 'USD').
""",
        "start": {"money.py": '"""Money (to be written)."""\n\n\nclass Money:\n    def __init__(self, amount, currency):\n        raise NotImplementedError\n'},
        "solution": {"money.py": '''"""Money."""
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal

_DIGITS = {"JPY": 0, "KWD": 3}


class Money:
    __slots__ = ("amount", "currency")

    def __init__(self, amount, currency):
        if not (isinstance(currency, str) and len(currency) == 3 and currency.isalpha() and currency.isupper() and currency.isascii()):
            raise ValueError("currency must be a 3-letter upper-case code")
        if isinstance(amount, (float, bool)):
            raise TypeError("amount must be int, str or Decimal")
        if isinstance(amount, (int, str, Decimal)):
            d = Decimal(amount)
        else:
            raise TypeError("amount must be int, str or Decimal")
        if not d.is_finite():
            raise ValueError("amount must be finite")
        q = Decimal(1).scaleb(-_DIGITS.get(currency, 2))
        object.__setattr__(self, "amount", d.quantize(q, rounding=ROUND_HALF_EVEN))
        object.__setattr__(self, "currency", currency)

    def __setattr__(self, name, value):
        raise AttributeError("Money is immutable")

    def _same(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        if other.currency != self.currency:
            raise ValueError("currency mismatch")
        return other

    def __add__(self, other):
        other = self._same(other)
        return other if other is NotImplemented else Money(self.amount + other.amount, self.currency)

    def __sub__(self, other):
        other = self._same(other)
        return other if other is NotImplemented else Money(self.amount - other.amount, self.currency)

    def __neg__(self):
        return Money(-self.amount, self.currency)

    def __mul__(self, k):
        if isinstance(k, bool) or not isinstance(k, (int, Decimal)):
            raise TypeError("can only multiply by int or Decimal")
        return Money(self.amount * k, self.currency)

    __rmul__ = __mul__

    def __truediv__(self, k):
        if isinstance(k, bool) or not isinstance(k, (int, Decimal)):
            raise TypeError("can only divide by int or Decimal")
        if k == 0:
            raise ZeroDivisionError("division by zero")
        return Money(self.amount / Decimal(k), self.currency)

    def __eq__(self, other):
        return isinstance(other, Money) and (self.amount, self.currency) == (other.amount, other.currency)

    def __hash__(self):
        return hash((self.amount, self.currency))

    def _cmp(self, other):
        other = self._same(other)
        return other

    def __lt__(self, other):
        return self.amount < self._cmp(other).amount

    def __le__(self, other):
        return self.amount <= self._cmp(other).amount

    def __gt__(self, other):
        return self.amount > self._cmp(other).amount

    def __ge__(self, other):
        return self.amount >= self._cmp(other).amount

    def allocate(self, ratios):
        ratios = list(ratios)
        if not ratios or any((not isinstance(r, int)) or isinstance(r, bool) or r < 0 for r in ratios) or sum(ratios) == 0:
            raise ValueError("ratios must be non-negative ints with a positive sum")
        sign = -1 if self.amount < 0 else 1
        unit = Decimal(1).scaleb(-_DIGITS.get(self.currency, 2))
        total = abs(self.amount)
        s = sum(ratios)
        exact = [total * r / s for r in ratios]
        parts = [e.quantize(unit, rounding=ROUND_DOWN) for e in exact]
        left = int((total - sum(parts)) / unit)
        order = sorted((i for i, r in enumerate(ratios) if r > 0), key=lambda i: (-(exact[i] - parts[i]), i))
        for i in order[:left]:
            parts[i] += unit
        return [Money(sign * p, self.currency) for p in parts]

    def __str__(self):
        return f"{self.amount} {self.currency}"

    def __repr__(self):
        return f"Money('{self.amount}', '{self.currency}')"
'''},
    },
    "hidden": {"test_money.py": '''import unittest
from decimal import Decimal
from money import Money


class Construction(unittest.TestCase):
    def test_rounding_half_even(self):
        self.assertEqual(Money("0.125", "USD").amount, Decimal("0.12"))
        self.assertEqual(Money("0.135", "USD").amount, Decimal("0.14"))
        self.assertEqual(Money("2.5", "JPY").amount, Decimal("2"))
        self.assertEqual(Money("3.5", "JPY").amount, Decimal("4"))
        self.assertEqual(Money("1.0005", "KWD").amount, Decimal("1.000"))
        self.assertEqual(Money(5, "USD").amount, Decimal("5.00"))
        self.assertEqual(Money(Decimal("1.999"), "USD").amount, Decimal("2.00"))

    def test_types_and_codes(self):
        with self.assertRaises(TypeError):
            Money(1.5, "USD")
        for cur in ("usd", "US", "USDX", "", "U$D", None, 5):
            with self.assertRaises(ValueError):
                Money(1, cur)

    def test_equality_hash_and_immutability(self):
        self.assertEqual(Money("1.10", "USD"), Money("1.1", "USD"))
        self.assertNotEqual(Money("1", "USD"), Money("1", "EUR"))
        self.assertEqual(len({Money("1.0", "USD"), Money(1, "USD")}), 1)
        with self.assertRaises(AttributeError):
            Money(1, "USD").amount = Decimal(2)

    def test_str_repr(self):
        self.assertEqual(str(Money("12.3", "USD")), "12.30 USD")
        self.assertEqual(str(Money(5, "JPY")), "5 JPY")
        self.assertEqual(str(Money("1", "KWD")), "1.000 KWD")
        self.assertEqual(repr(Money("12.3", "USD")), "Money('12.30', 'USD')")


class Arithmetic(unittest.TestCase):
    def test_add_sub_neg(self):
        self.assertEqual(Money("1.10", "USD") + Money("2.25", "USD"), Money("3.35", "USD"))
        self.assertEqual(Money("1.10", "USD") - Money("2.25", "USD"), Money("-1.15", "USD"))
        self.assertEqual(-Money("1.10", "USD"), Money("-1.10", "USD"))
        with self.assertRaises(ValueError):
            Money(1, "USD") + Money(1, "EUR")
        with self.assertRaises(TypeError):
            Money(1, "USD") + 1

    def test_mul_div(self):
        self.assertEqual(Money("10.00", "USD") * 3, Money("30.00", "USD"))
        self.assertEqual(3 * Money("10.00", "USD"), Money("30.00", "USD"))
        self.assertEqual(Money("10.00", "USD") * Decimal("0.075"), Money("0.75", "USD"))      # 0.75 exactly
        self.assertEqual(Money("0.05", "USD") * Decimal("0.5"), Money("0.02", "USD"))         # 0.025 -> half even -> 0.02
        self.assertEqual(Money("10.00", "USD") / 3, Money("3.33", "USD"))
        self.assertEqual(Money("100", "JPY") / 8, Money("12", "JPY"))                         # 12.5 -> 12
        with self.assertRaises(TypeError):
            Money(1, "USD") * 1.5
        with self.assertRaises(ZeroDivisionError):
            Money(1, "USD") / 0

    def test_ordering(self):
        self.assertTrue(Money(1, "USD") < Money(2, "USD"))
        self.assertTrue(Money(2, "USD") >= Money(2, "USD"))
        with self.assertRaises(ValueError):
            Money(1, "USD") < Money(2, "EUR")


class Allocate(unittest.TestCase):
    def test_sums_exactly(self):
        parts = Money("100.00", "USD").allocate([1, 1, 1])
        self.assertEqual(parts, [Money("33.34", "USD"), Money("33.33", "USD"), Money("33.33", "USD")])
        self.assertEqual(sum((p.amount for p in parts), Decimal(0)), Decimal("100.00"))

    def test_proportions(self):
        self.assertEqual(Money("0.05", "USD").allocate([3, 7]), [Money("0.02", "USD"), Money("0.03", "USD")])
        self.assertEqual(Money("10.00", "USD").allocate([70, 30]), [Money("7.00", "USD"), Money("3.00", "USD")])

    def test_largest_remainder_and_ties(self):
        self.assertEqual(Money("0.02", "USD").allocate([1, 1, 1]), [Money("0.01", "USD"), Money("0.01", "USD"), Money("0.00", "USD")])
        self.assertEqual(Money("1.00", "USD").allocate([1, 2]), [Money("0.33", "USD"), Money("0.67", "USD")])
        self.assertEqual(Money("5", "JPY").allocate([1, 1]), [Money("3", "JPY"), Money("2", "JPY")])

    def test_zero_ratio_never_gets_a_leftover(self):
        self.assertEqual(Money("0.01", "USD").allocate([0, 1, 1]), [Money("0.00", "USD"), Money("0.01", "USD"), Money("0.00", "USD")])

    def test_negative_amount(self):
        self.assertEqual(Money("-100.00", "USD").allocate([1, 1, 1]),
                         [Money("-33.34", "USD"), Money("-33.33", "USD"), Money("-33.33", "USD")])

    def test_bad_ratios(self):
        for bad in ([], [0, 0], [-1, 2], [1.5, 1], [True, 1]):
            with self.assertRaises(ValueError):
                Money(1, "USD").allocate(bad)
'''},
    "bugfix": {
        "symptom": "A Money class (money.py) rounds half-way amounts the wrong way: Money(\"0.125\", \"USD\") should round half to even (0.12) but gives 0.13.",
        "bug": [("money.py", "d.quantize(q, rounding=ROUND_HALF_EVEN)", "d.quantize(q, rounding=ROUND_HALF_UP)"), ("money.py", "from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal", "from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal")],
        "repro": {"test_repro.py": 'import unittest\nfrom decimal import Decimal\nfrom money import Money\n\n\nclass Repro(unittest.TestCase):\n    def test_bankers_rounding(self):\n        self.assertEqual(Money("0.125", "USD").amount, Decimal("0.12"))\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
