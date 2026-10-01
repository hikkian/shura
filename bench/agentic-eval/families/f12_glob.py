FAMILY = {
    "name": "f12_glob", "lang": "python",
    "implement": {
        "prompt": """Write the module `globmatch.py` (standard library only; do NOT use fnmatch, glob or pathlib matching) with `match(pattern, path)` -> bool
and `filter_paths(patterns, paths)` -> list.

Paths use "/" as the separator. match is anchored: the pattern must match the WHOLE path. Pattern syntax:
  ?       exactly one character other than "/"
  *       zero or more characters other than "/"
  **      zero or more characters INCLUDING "/". As a whole path segment it also lets the neighbouring slashes collapse:
          "a/**/b" matches "a/b", "a/x/b" and "a/x/y/b"; "**/b" matches "b" and "x/y/b". A "**" inside a segment ("a**b") is a
          "*" that may cross "/".
  [abc]   one character from the set; ranges "a-f"; a leading "!" or "^" negates ([!a] matches any character except "a" and except "/")
  \\x      the character x literally (also for the special characters \\* \\? \\[ \\\\)
A "/" in a set never matches a slash (sets cannot match "/"). An unclosed "[" is a literal "[". A trailing lone backslash is literal.
"a/**" matches "a/x", "a/x/y" and also "a" itself. Matching is case-sensitive.
filter_paths(patterns, paths): the paths (in their original order) that match at least one of the patterns, with gitignore-like negation:
  patterns are applied in order, a pattern starting with "!" (after which the rest is an ordinary pattern) removes matches again,
  so ["*.py", "!test_*.py"] keeps "a.py" but not "test_a.py"; the last matching pattern decides; unmatched paths are dropped.
""",
        "start": {"globmatch.py": '"""Glob matching (to be written)."""\n\n\ndef match(pattern, path):\n    raise NotImplementedError\n\n\ndef filter_paths(patterns, paths):\n    raise NotImplementedError\n'},
        "solution": {"globmatch.py": '''"""Glob matching."""
import re
from functools import lru_cache


@lru_cache(maxsize=512)
def _compile(pattern):
    i, n, out = 0, len(pattern), []
    segs = pattern.split("/")
    # whole-segment "**" handled on the pattern level
    def seg_regex(seg):
        j, res = 0, []
        while j < len(seg):
            c = seg[j]
            if c == "\\\\":
                if j + 1 < len(seg):
                    res.append(re.escape(seg[j + 1]))
                    j += 2
                else:
                    res.append(re.escape("\\\\"))
                    j += 1
            elif c == "*":
                k = j
                while k < len(seg) and seg[k] == "*":
                    k += 1
                res.append("[^/]*" if k - j == 1 else ".*")
                j = k
            elif c == "?":
                res.append("[^/]")
                j += 1
            elif c == "[":
                k = j + 1
                if k < len(seg) and seg[k] in "!^":
                    k += 1
                if k < len(seg) and seg[k] == "]":
                    k += 1
                while k < len(seg) and seg[k] != "]":
                    k += 1
                if k >= len(seg):
                    res.append(re.escape("["))
                    j += 1
                else:
                    body = seg[j + 1:k]
                    neg = body[:1] in ("!", "^")
                    if neg:
                        body = body[1:]
                    body = body.replace("\\\\", "\\\\\\\\").replace("]", "\\\\]").replace("^", "\\\\^")
                    res.append(("[^/" if neg else "[") + body + "]" if True else "")
                    j = k + 1
            else:
                res.append(re.escape(c))
                j += 1
        return "".join(res)

    parts = []
    for idx, seg in enumerate(segs):
        last = idx == len(segs) - 1
        if seg == "**":
            parts.append(("globstar", last))
        else:
            parts.append(("seg", seg_regex(seg)))
    rx = ""
    for idx, (kind, val) in enumerate(parts):
        last = idx == len(parts) - 1
        if kind == "globstar":
            if last:
                rx += "(?:.*)?" if idx == 0 else "(?:/.*)?"
            else:
                rx += "(?:.*/)?"
        else:
            rx += val
            if not last:
                nxt = parts[idx + 1]
                if not (nxt[0] == "globstar" and idx + 1 == len(parts) - 1):
                    rx += "/"
    return re.compile("^" + rx + "$", re.S)


def match(pattern, path):
    return _compile(pattern).match(path) is not None


def filter_paths(patterns, paths):
    kept = []
    for p in paths:
        decision = False
        for pat in patterns:
            neg = pat.startswith("!")
            if match(pat[1:] if neg else pat, p):
                decision = not neg
        if decision:
            kept.append(p)
    return kept
'''},
    },
    "hidden": {"test_globmatch.py": '''import unittest
from globmatch import filter_paths, match


class Wildcards(unittest.TestCase):
    def test_question_and_star(self):
        self.assertTrue(match("a?c", "abc"))
        self.assertFalse(match("a?c", "ac"))
        self.assertFalse(match("a?c", "a/c"))
        self.assertTrue(match("*.py", "x.py"))
        self.assertTrue(match("*.py", ".py"))
        self.assertFalse(match("*.py", "a/x.py"))
        self.assertFalse(match("*.py", "x.pyc"))
        self.assertTrue(match("a*", "a"))
        self.assertTrue(match("src/*/main.c", "src/lib/main.c"))
        self.assertFalse(match("src/*/main.c", "src/a/b/main.c"))

    def test_anchored(self):
        self.assertFalse(match("a", "ab"))
        self.assertFalse(match("b", "ab"))
        self.assertTrue(match("", ""))
        self.assertFalse(match("", "a"))

    def test_double_star(self):
        self.assertTrue(match("a/**/b", "a/b"))
        self.assertTrue(match("a/**/b", "a/x/b"))
        self.assertTrue(match("a/**/b", "a/x/y/b"))
        self.assertFalse(match("a/**/b", "ab"))
        self.assertTrue(match("**/b", "b"))
        self.assertTrue(match("**/b", "x/y/b"))
        self.assertFalse(match("**/b", "x/yb"))
        self.assertTrue(match("a/**", "a/x"))
        self.assertTrue(match("a/**", "a/x/y"))
        self.assertTrue(match("a/**", "a"))
        self.assertFalse(match("a/**", "b/x"))
        self.assertTrue(match("**", "anything/at/all"))
        self.assertTrue(match("a**b", "a/x/b"))
        self.assertTrue(match("a**b", "ab"))
        self.assertTrue(match("**/*.py", "x/y/z.py"))
        self.assertTrue(match("**/*.py", "z.py"))


class Sets(unittest.TestCase):
    def test_sets(self):
        self.assertTrue(match("[abc]x", "bx"))
        self.assertFalse(match("[abc]x", "dx"))
        self.assertTrue(match("[a-f]1", "c1"))
        self.assertFalse(match("[a-f]1", "g1"))
        self.assertTrue(match("[!a]x", "bx"))
        self.assertFalse(match("[!a]x", "ax"))
        self.assertTrue(match("[^a]x", "bx"))
        self.assertFalse(match("[!a]x", "/x"))
        self.assertFalse(match("a[/]b", "a/b"))

    def test_unclosed_and_literal(self):
        self.assertTrue(match("a[b", "a[b"))
        self.assertTrue(match("\\\\*", "*"))
        self.assertFalse(match("\\\\*", "x"))
        self.assertTrue(match("a\\\\?b", "a?b"))
        self.assertTrue(match("\\\\[x]", "[x]"))
        self.assertTrue(match("a\\\\\\\\b", "a\\\\b"))
        self.assertTrue(match("a\\\\", "a\\\\"))
        self.assertTrue(match("a.b", "a.b"))
        self.assertFalse(match("a.b", "axb"))
        self.assertTrue(match("(x)+", "(x)+"))

    def test_case_sensitive(self):
        self.assertFalse(match("A", "a"))


class Filter(unittest.TestCase):
    def test_negation_and_order(self):
        paths = ["a.py", "test_a.py", "b.txt", "lib/c.py"]
        self.assertEqual(filter_paths(["*.py", "!test_*.py"], paths), ["a.py"])
        self.assertEqual(filter_paths(["**/*.py", "!test_*.py"], paths), ["a.py", "lib/c.py"])
        self.assertEqual(filter_paths(["!*.py", "*.py"], paths), ["a.py", "test_a.py"])
        self.assertEqual(filter_paths([], paths), [])
        self.assertEqual(filter_paths(["*"], paths), ["a.py", "test_a.py", "b.txt"])
        self.assertEqual(filter_paths(["*.py", "!test_*.py", "test_a.py"], paths), ["a.py", "test_a.py"])
'''},
    "bugfix": {
        "symptom": "A glob matcher (globmatch.py) treats a single star like a double star: match(\"*.py\", \"src/x.py\") must be False (a single * never crosses a slash) but returns True.",
        "bug": [("globmatch.py", '                res.append("[^/]*" if k - j == 1 else ".*")', '                res.append(".*")')],
        "repro": {"test_repro.py": 'import unittest\nfrom globmatch import match\n\n\nclass Repro(unittest.TestCase):\n    def test_star_does_not_cross_slash(self):\n        self.assertFalse(match("*.py", "src/x.py"))\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
