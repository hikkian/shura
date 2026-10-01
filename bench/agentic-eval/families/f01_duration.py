FAMILY = {
    "name": "f01_duration", "lang": "python",
    "implement": {
        "prompt": """Write the module `durations.py` in the project folder. It has two functions.

parse_duration(text) -> int : turns a duration string into a whole number of SECONDS.
  * The string is one or more parts, each a non-negative integer followed by a unit: w (7 days), d (24 hours), h, m, s.
    Parts must appear in that order from large to small (w, d, h, m, s), each unit at most once. "1h30m", "2w3d", "45s", "1d12h".
  * Whitespace between parts is allowed ("1h 30m"); leading and trailing whitespace is ignored. Units are lower case only.
  * "0s" is valid and gives 0. A bare number without a unit ("90") is NOT valid.
  * Anything invalid (empty string, unknown unit, wrong order, repeated unit, negative or decimal numbers, other characters)
    raises ValueError.

format_duration(seconds) -> str : the inverse. Uses the largest units that apply, all parts non-zero, in the order w d h m s,
  joined by a single space: 3661 -> "1h 1m 1s", 90061 -> "1d 1h 1m 1s", 0 -> "0s", 604800 -> "1w". Negative input raises ValueError.
  For every non-negative integer n: parse_duration(format_duration(n)) == n.
""",
        "start": {"durations.py": '"""Duration parsing and formatting (to be written)."""\n\n\ndef parse_duration(text):\n    raise NotImplementedError\n\n\ndef format_duration(seconds):\n    raise NotImplementedError\n'},
        "solution": {"durations.py": '''"""Duration parsing and formatting."""
import re

_UNITS = (("w", 604800), ("d", 86400), ("h", 3600), ("m", 60), ("s", 1))
_PATTERN = re.compile(r"^(?:(\\d+)w)?\\s*(?:(\\d+)d)?\\s*(?:(\\d+)h)?\\s*(?:(\\d+)m)?\\s*(?:(\\d+)s)?$")


def parse_duration(text):
    if not isinstance(text, str):
        raise ValueError("duration must be a string")
    text = text.strip()
    if not text:
        raise ValueError("empty duration")
    m = _PATTERN.match(text)
    if not m or all(g is None for g in m.groups()):
        raise ValueError(f"invalid duration: {text!r}")
    return sum(int(g) * size for g, (_, size) in zip(m.groups(), _UNITS) if g is not None)


def format_duration(seconds):
    if seconds < 0:
        raise ValueError("negative duration")
    if seconds == 0:
        return "0s"
    parts = []
    for name, size in _UNITS:
        n, seconds = divmod(seconds, size)
        if n:
            parts.append(f"{n}{name}")
    return " ".join(parts)
'''},
    },
    "hidden": {"test_durations.py": '''import unittest
from durations import format_duration, parse_duration


class Parse(unittest.TestCase):
    def test_single_and_combined(self):
        self.assertEqual(parse_duration("45s"), 45)
        self.assertEqual(parse_duration("1h30m"), 5400)
        self.assertEqual(parse_duration("2w3d"), 2 * 604800 + 3 * 86400)
        self.assertEqual(parse_duration("1d12h"), 129600)
        self.assertEqual(parse_duration("1w1d1h1m1s"), 604800 + 86400 + 3600 + 60 + 1)

    def test_spaces_and_zero(self):
        self.assertEqual(parse_duration("  1h 30m "), 5400)
        self.assertEqual(parse_duration("0s"), 0)
        self.assertEqual(parse_duration("0h0m"), 0)

    def test_units_may_exceed_their_range(self):
        self.assertEqual(parse_duration("90m"), 5400)
        self.assertEqual(parse_duration("100s"), 100)

    def test_invalid(self):
        for bad in ["", "   ", "90", "1x", "1H", "30m1h", "1h1h", "-5s", "1.5h", "h", "1h 2", "1 h", "abc", "1h,30m", "1hh"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_duration(bad)

    def test_non_string(self):
        for bad in [None, 5, 1.5]:
            with self.assertRaises(ValueError):
                parse_duration(bad)


class Format(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(format_duration(3661), "1h 1m 1s")
        self.assertEqual(format_duration(90061), "1d 1h 1m 1s")
        self.assertEqual(format_duration(0), "0s")
        self.assertEqual(format_duration(604800), "1w")
        self.assertEqual(format_duration(59), "59s")
        self.assertEqual(format_duration(3600), "1h")
        self.assertEqual(format_duration(2 * 604800 + 5), "2w 5s")

    def test_negative(self):
        with self.assertRaises(ValueError):
            format_duration(-1)

    def test_round_trip(self):
        for n in [0, 1, 59, 60, 61, 3599, 3600, 86399, 86400, 604799, 604800, 9999999, 123456789]:
            self.assertEqual(parse_duration(format_duration(n)), n)
'''},
    "bugfix": {
        "symptom": "A small library for duration strings (durations.py) gives wrong results: the string \"1d12h\" is expected to be 129600 seconds but the function returns something else.",
        "bug": [("durations.py", '(("w", 604800), ("d", 86400), ("h", 3600), ("m", 60), ("s", 1))', '(("w", 604800), ("d", 8640), ("h", 3600), ("m", 60), ("s", 1))')],
        "repro": {"test_repro.py": 'import unittest\nfrom durations import parse_duration\n\n\nclass Repro(unittest.TestCase):\n    def test_day_and_hours(self):\n        self.assertEqual(parse_duration("1d12h"), 129600)\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
