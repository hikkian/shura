FAMILY = {
    "name": "f22_roman", "lang": "python",
    "implement": {
        "prompt": """Write the module `numerals.py` (standard library only) with four functions.

to_roman(n) -> str : an int 1..3999 as a Roman numeral in standard subtractive form (4 = IV, 9 = IX, 40 = XL, 90 = XC, 400 = CD, 900 = CM; 1994 = MCMXCIV).
  Anything else (0, 4000, negative, non-int, bool) raises ValueError.
from_roman(text) -> int : the inverse, accepting ONLY canonical numerals (the exact strings that to_roman produces), upper case only.
  "IIII", "VX", "IC", "MMMM", "iv", "" and anything else non-canonical raise ValueError.
to_words(n) -> str : an int 0..999999999 in English: "zero", "one", ..., "twenty-one", "one hundred", "one hundred and five" (British "and" after
  hundreds when something follows), "one thousand two hundred and thirty-four", "two million", "one million and one" (an "and" also
  before a final part below 100 when the number above it ends in a unit of thousand/million without hundreds: 1001 = "one thousand and one",
  1100 = "one thousand one hundred", 1000000 = "one million", 1000001 = "one million and one", 2020 = "two thousand and twenty").
  Tens 20-99 are hyphenated ("forty-two", "ninety"). Negative or non-int raises ValueError.
from_words(text) -> int : the inverse of to_words for every value it produces; case-insensitive, extra spaces ignored; invalid text raises ValueError.
For every valid n: from_roman(to_roman(n)) == n (1..3999) and from_words(to_words(n)) == n.
""",
        "start": {"numerals.py": '"""Roman numerals and English number words (to be written)."""\n\n\ndef to_roman(n):\n    raise NotImplementedError\n\n\ndef from_roman(text):\n    raise NotImplementedError\n\n\ndef to_words(n):\n    raise NotImplementedError\n\n\ndef from_words(text):\n    raise NotImplementedError\n'},
        "solution": {"numerals.py": '''"""Roman numerals and English number words."""
_ROMAN = (("M", 1000), ("CM", 900), ("D", 500), ("CD", 400), ("C", 100), ("XC", 90), ("L", 50), ("XL", 40), ("X", 10), ("IX", 9),
          ("V", 5), ("IV", 4), ("I", 1))
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
         "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _int(n, lo, hi):
    if isinstance(n, bool) or not isinstance(n, int) or not lo <= n <= hi:
        raise ValueError(f"expected an int in {lo}..{hi}")


def to_roman(n):
    _int(n, 1, 3999)
    out = []
    for sym, val in _ROMAN:
        while n >= val:
            out.append(sym)
            n -= val
    return "".join(out)


def from_roman(text):
    if not isinstance(text, str) or not text:
        raise ValueError("not a roman numeral")
    total, i = 0, 0
    for sym, val in _ROMAN:
        while text.startswith(sym, i):
            total += val
            i += len(sym)
    if i != len(text) or total < 1 or total > 3999 or to_roman(total) != text:
        raise ValueError("not a canonical roman numeral")
    return total


def _below_thousand(n):
    parts = []
    if n >= 100:
        parts.append(_ONES[n // 100] + " hundred")
        n %= 100
        if n:
            parts.append("and")
    if n >= 20:
        t, o = divmod(n, 10)
        parts.append(_TENS[t] + (f"-{_ONES[o]}" if o else ""))
    elif n or not parts:
        parts.append(_ONES[n])
    return " ".join(parts)


def to_words(n):
    _int(n, 0, 999_999_999)
    if n == 0:
        return "zero"
    groups = [(n // 1_000_000, "million"), (n // 1000 % 1000, "thousand"), (n % 1000, "")]
    words, seen_big = [], False
    for count, unit in groups:
        if not count:
            continue
        if unit:
            words.append(_below_thousand(count) + " " + unit)
            seen_big = True
        else:
            text = _below_thousand(count)
            if seen_big and count < 100:
                text = "and " + text
            words.append(text)
    return " ".join(words)


_WORD_VALUE = {w: i for i, w in enumerate(_ONES)}
_WORD_VALUE.update({t: 10 * i for i, t in enumerate(_TENS) if t})


def from_words(text):
    if not isinstance(text, str):
        raise ValueError("expected text")
    tokens = text.lower().replace("-", " ").split()
    if not tokens:
        raise ValueError("empty")
    total = current = 0
    seen_any = False
    for tok in tokens:
        if tok == "and":
            continue
        seen_any = True
        if tok in _WORD_VALUE:
            current += _WORD_VALUE[tok]
        elif tok == "hundred":
            if current == 0:
                raise ValueError("hundred without a number")
            current *= 100
        elif tok in ("thousand", "million"):
            if current == 0:
                raise ValueError("scale without a number")
            total += current * (1000 if tok == "thousand" else 1_000_000)
            current = 0
        else:
            raise ValueError(f"unknown word {tok!r}")
    if not seen_any:
        raise ValueError("empty")
    result = total + current
    if result > 999_999_999 or to_words(result) != " ".join(t for t in text.lower().replace("-", "-").split()):
        # accept only what to_words produces (modulo case and spaces)
        if to_words(result).replace("-", " ").split() != [t for t in tokens]:
            raise ValueError("not a number as written by to_words")
    return result
'''},
    },
    "hidden": {"test_numerals.py": '''import unittest
from numerals import from_roman, from_words, to_roman, to_words


class Roman(unittest.TestCase):
    def test_known(self):
        for n, s in [(1, "I"), (4, "IV"), (9, "IX"), (14, "XIV"), (40, "XL"), (90, "XC"), (400, "CD"), (900, "CM"), (1994, "MCMXCIV"),
                     (2024, "MMXXIV"), (3999, "MMMCMXCIX"), (3888, "MMMDCCCLXXXVIII")]:
            self.assertEqual(to_roman(n), s)
            self.assertEqual(from_roman(s), n)

    def test_round_trip_all(self):
        for n in range(1, 4000):
            self.assertEqual(from_roman(to_roman(n)), n)

    def test_invalid_input(self):
        for bad in [0, 4000, -1, 1.5, "5", None, True]:
            with self.assertRaises(ValueError):
                to_roman(bad)
        for bad in ["", "IIII", "VX", "IC", "MMMM", "iv", "XXXX", "VV", "IL", "XM", "MCMC", "IIX", "A", " I", "I ", None, 5, "IXI"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    from_roman(bad)


class Words(unittest.TestCase):
    def test_known(self):
        cases = [(0, "zero"), (7, "seven"), (13, "thirteen"), (20, "twenty"), (21, "twenty-one"), (99, "ninety-nine"), (100, "one hundred"),
                 (105, "one hundred and five"), (110, "one hundred and ten"), (342, "three hundred and forty-two"),
                 (1000, "one thousand"), (1001, "one thousand and one"), (1100, "one thousand one hundred"),
                 (1234, "one thousand two hundred and thirty-four"), (2020, "two thousand and twenty"), (20000, "twenty thousand"),
                 (100000, "one hundred thousand"), (100001, "one hundred thousand and one"), (999999, "nine hundred and ninety-nine thousand nine hundred and ninety-nine"),
                 (1000000, "one million"), (1000001, "one million and one"), (2500000, "two million five hundred thousand"),
                 (1000100, "one million one hundred"), (999999999, "nine hundred and ninety-nine million nine hundred and ninety-nine thousand nine hundred and ninety-nine"),
                 (1100000, "one million one hundred thousand"), (1000010, "one million and ten")]
        for n, s in cases:
            with self.subTest(n=n):
                self.assertEqual(to_words(n), s)
                self.assertEqual(from_words(s), n)

    def test_round_trip_sample(self):
        import random
        rng = random.Random(1)
        for n in list(range(0, 2200)) + [rng.randint(0, 999_999_999) for _ in range(3000)]:
            self.assertEqual(from_words(to_words(n)), n)

    def test_from_words_forgiving(self):
        self.assertEqual(from_words("  Twenty-One  "), 21)
        self.assertEqual(from_words("ONE HUNDRED AND FIVE"), 105)
        self.assertEqual(from_words("twenty one"), 21)

    def test_invalid(self):
        for bad in [-1, 1_000_000_000, 1.5, "1", None, True]:
            with self.assertRaises(ValueError):
                to_words(bad)
        for bad in ["", "   ", "banana", "hundred", "thousand", "one banana", "twenty twenty", "one hundred hundred", None, 5, "and"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    from_words(bad)
'''},
    "bugfix": {
        "symptom": "A numerals library (numerals.py) writes some numbers wrongly: to_words(105) must be \"one hundred and five\" (British \"and\" after the hundreds) but it returns \"one hundred five\".",
        "bug": [("numerals.py", '''        n %= 100
        if n:
            parts.append("and")''', '''        n %= 100''')],
        "repro": {"test_repro.py": 'import unittest\nfrom numerals import to_words\n\n\nclass Repro(unittest.TestCase):\n    def test_and_after_hundreds(self):\n        self.assertEqual(to_words(105), "one hundred and five")\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
