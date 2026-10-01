FAMILY = {
    "name": "f02_semver", "lang": "python",
    "implement": {
        "prompt": """Write the module `semver.py` in the project folder (standard library only).

parse(text) -> tuple : parses "MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]" (semantic versioning 2.0.0) and returns
  (major, minor, patch, prerelease, build) where major/minor/patch are ints, prerelease is a tuple of identifiers (ints for
  all-digit identifiers, strings otherwise; empty tuple if none) and build is a tuple of strings (empty if none).
  Numeric parts have no leading zeros ("01.2.3" is invalid; "0.0.0" is valid). A prerelease identifier is non-empty, made of
  [0-9A-Za-z-], and an all-digit identifier has no leading zero. Build identifiers are non-empty [0-9A-Za-z-]. Invalid input
  (including a leading "v" and extra spaces) raises ValueError.

compare(a, b) -> -1, 0 or 1 : compares two version strings by semver PRECEDENCE: major, minor, patch numerically; a version
  with a prerelease is lower than the same version without; prerelease identifiers are compared left to right, numeric ones
  numerically and lower than alphanumeric ones, alphanumeric ones in ASCII order, and a shorter list is lower if all earlier
  identifiers are equal. Build metadata is ignored ("1.0.0+a" and "1.0.0+b" compare as 0).

satisfies(version, spec) -> bool : spec is one or more alternatives separated by "||"; an alternative is one or more
  space-separated comparators that must ALL hold. A comparator is one of:
    ">=X", ">X", "<=X", "<X", "=X" or plain "X"  (X is a full version; plain/"=" means equal by precedence)
    "^X": at or above X and below the next change of the left-most non-zero part of X
          (^1.2.3 -> >=1.2.3 <2.0.0, ^0.2.3 -> >=0.2.3 <0.3.0, ^0.0.3 -> >=0.0.3 <0.0.4)
    "~X": at or above X and below the next minor (~1.2.3 -> >=1.2.3 <1.3.0)
  Prerelease versions are compared by precedence like any other (no special exclusion rule). An invalid spec raises ValueError.
""",
        "start": {"semver.py": '"""Semantic versions (to be written)."""\n\n\ndef parse(text):\n    raise NotImplementedError\n\n\ndef compare(a, b):\n    raise NotImplementedError\n\n\ndef satisfies(version, spec):\n    raise NotImplementedError\n'},
        "solution": {"semver.py": '''"""Semantic versions."""
import re

_VER = re.compile(r"^(0|[1-9]\\d*)\\.(0|[1-9]\\d*)\\.(0|[1-9]\\d*)(?:-([0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*))?(?:\\+([0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*))?$")


def parse(text):
    if not isinstance(text, str):
        raise ValueError("version must be a string")
    m = _VER.match(text)
    if not m:
        raise ValueError(f"invalid version: {text!r}")
    major, minor, patch, pre, build = m.groups()
    ids = []
    for ident in (pre.split(".") if pre else []):
        if ident.isdigit():
            if len(ident) > 1 and ident[0] == "0":
                raise ValueError(f"leading zero in prerelease identifier: {ident!r}")
            ids.append(int(ident))
        else:
            ids.append(ident)
    return int(major), int(minor), int(patch), tuple(ids), tuple(build.split(".")) if build else ()


def _key_cmp(a, b):
    (a1, a2, a3, ap, _), (b1, b2, b3, bp, _) = parse(a), parse(b)
    if (a1, a2, a3) != (b1, b2, b3):
        return -1 if (a1, a2, a3) < (b1, b2, b3) else 1
    if not ap and not bp:
        return 0
    if not ap:
        return 1
    if not bp:
        return -1
    for x, y in zip(ap, bp):
        if x == y:
            continue
        if isinstance(x, int) and isinstance(y, int):
            return -1 if x < y else 1
        if isinstance(x, int):
            return -1
        if isinstance(y, int):
            return 1
        return -1 if x < y else 1
    return (len(ap) > len(bp)) - (len(ap) < len(bp))


def compare(a, b):
    return _key_cmp(a, b)


def _bump(version, which):
    major, minor, patch = parse(version)[:3]
    if which == "major":
        return f"{major + 1}.0.0"
    if which == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def _comparator(version, token):
    for op in (">=", "<=", ">", "<", "="):
        if token.startswith(op):
            x = token[len(op):]
            c = compare(version, x)
            return {">=": c >= 0, "<=": c <= 0, ">": c > 0, "<": c < 0, "=": c == 0}[op]
    if token.startswith("^"):
        x = token[1:]
        major, minor, patch = parse(x)[:3]
        upper = _bump(x, "major") if major else _bump(x, "minor") if minor else _bump(x, "patch")
        return compare(version, x) >= 0 and compare(version, upper) < 0
    if token.startswith("~"):
        x = token[1:]
        return compare(version, x) >= 0 and compare(version, _bump(x, "minor")) < 0
    return compare(version, token) == 0


def _strip(token):
    for op in (">=", "<=", ">", "<", "=", "^", "~"):
        if token.startswith(op):
            return token[len(op):]
    return token


def satisfies(version, spec):
    parse(version)
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("empty spec")
    alternatives = [a.split() for a in spec.split("||")]
    for tokens in alternatives:                  # the whole spec must be valid, whichever alternative matches
        if not tokens:
            raise ValueError("empty alternative")
        for t in tokens:
            parse(_strip(t))
    return any(all(_comparator(version, t) for t in tokens) for tokens in alternatives)
'''},
    },
    "hidden": {"test_semver.py": '''import unittest
from semver import compare, parse, satisfies


class Parse(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse("1.2.3"), (1, 2, 3, (), ()))
        self.assertEqual(parse("0.0.0"), (0, 0, 0, (), ()))
        self.assertEqual(parse("1.0.0-alpha.1"), (1, 0, 0, ("alpha", 1), ()))
        self.assertEqual(parse("1.0.0-0.3.7+exp.sha.5114f85"), (1, 0, 0, (0, 3, 7), ("exp", "sha", "5114f85")))
        self.assertEqual(parse("1.0.0+20130313144700"), (1, 0, 0, (), ("20130313144700",)))
        self.assertEqual(parse("1.0.0-x-y-z.--"), (1, 0, 0, ("x-y-z", "--"), ()))

    def test_invalid(self):
        for bad in ["", "1", "1.2", "1.2.3.4", "01.2.3", "1.02.3", "1.2.03", "v1.2.3", " 1.2.3", "1.2.3 ", "1.2.3-", "1.2.3-01",
                    "1.2.3-a..b", "1.2.3+", "1.2.3+a..b", "1.2.3-a_b", "-1.2.3", "1.2.-3", "a.b.c"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse(bad)


class Compare(unittest.TestCase):
    def test_order_from_the_semver_spec(self):
        ordered = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
        for i, a in enumerate(ordered):
            for j, b in enumerate(ordered):
                self.assertEqual(compare(a, b), (i > j) - (i < j), (a, b))

    def test_core_numbers_are_numeric(self):
        self.assertEqual(compare("1.10.0", "1.9.0"), 1)
        self.assertEqual(compare("2.0.0", "10.0.0"), -1)
        self.assertEqual(compare("1.2.3", "1.2.3"), 0)

    def test_build_is_ignored(self):
        self.assertEqual(compare("1.0.0+a", "1.0.0+b"), 0)
        self.assertEqual(compare("1.0.0-rc.1+x", "1.0.0-rc.1"), 0)

    def test_numeric_prerelease_identifier_is_lower_than_alphanumeric(self):
        self.assertEqual(compare("1.0.0-1", "1.0.0-a"), -1)
        self.assertEqual(compare("1.0.0-2", "1.0.0-10"), -1)
        self.assertEqual(compare("1.0.0-a1", "1.0.0-a10"), -1)


class Satisfies(unittest.TestCase):
    def test_plain_and_relational(self):
        self.assertTrue(satisfies("1.2.3", "1.2.3"))
        self.assertTrue(satisfies("1.2.3", "=1.2.3"))
        self.assertFalse(satisfies("1.2.4", "1.2.3"))
        self.assertTrue(satisfies("1.2.3", ">=1.2.3"))
        self.assertFalse(satisfies("1.2.3", ">1.2.3"))
        self.assertTrue(satisfies("1.2.3", "<=1.2.3"))
        self.assertFalse(satisfies("1.2.3", "<1.2.3"))

    def test_and_or(self):
        self.assertTrue(satisfies("1.5.0", ">=1.2.0 <2.0.0"))
        self.assertFalse(satisfies("2.0.0", ">=1.2.0 <2.0.0"))
        self.assertTrue(satisfies("3.1.0", "<1.0.0 || >=3.0.0"))
        self.assertTrue(satisfies("0.5.0", "<1.0.0 || >=3.0.0"))
        self.assertFalse(satisfies("2.0.0", "<1.0.0 || >=3.0.0"))

    def test_caret(self):
        self.assertTrue(satisfies("1.9.9", "^1.2.3"))
        self.assertFalse(satisfies("2.0.0", "^1.2.3"))
        self.assertFalse(satisfies("1.2.2", "^1.2.3"))
        self.assertTrue(satisfies("0.2.9", "^0.2.3"))
        self.assertFalse(satisfies("0.3.0", "^0.2.3"))
        self.assertTrue(satisfies("0.0.3", "^0.0.3"))
        self.assertFalse(satisfies("0.0.4", "^0.0.3"))

    def test_tilde(self):
        self.assertTrue(satisfies("1.2.9", "~1.2.3"))
        self.assertFalse(satisfies("1.3.0", "~1.2.3"))
        self.assertFalse(satisfies("1.2.2", "~1.2.3"))

    def test_prerelease_by_precedence(self):
        self.assertTrue(satisfies("1.0.0-rc.1", "<1.0.0"))
        self.assertTrue(satisfies("1.0.0-rc.2", ">=1.0.0-rc.1 <1.0.0"))

    def test_invalid(self):
        for bad in ["", "  ", ">=", "^", "||", "1.0.0 ||", ">=1.x", "foo"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    satisfies("1.0.0", bad)
        with self.assertRaises(ValueError):
            satisfies("not-a-version", ">=1.0.0")
'''},
    "bugfix": {
        "symptom": "A semantic-version library (semver.py) orders prerelease versions wrongly: \"1.0.0-beta.2\" should be lower than \"1.0.0-beta.11\" (numeric identifiers compare as numbers) but compare() says the opposite.",
        "bug": [("semver.py", '''        if isinstance(x, int) and isinstance(y, int):
            return -1 if x < y else 1
''', '''        if isinstance(x, int) and isinstance(y, int):
            return -1 if str(x) < str(y) else 1
''')],
        "repro": {"test_repro.py": 'import unittest\nfrom semver import compare\n\n\nclass Repro(unittest.TestCase):\n    def test_numeric_prerelease_identifiers(self):\n        self.assertEqual(compare("1.0.0-beta.2", "1.0.0-beta.11"), -1)\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
