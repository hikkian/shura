FAMILY = {
    "name": "f14_ini", "lang": "python",
    "implement": {
        "prompt": """Write the module `miniconf.py` (standard library only; no configparser, tomllib or json) with `loads(text)`: a parser for a small INI/TOML-like format.

Format:
  * `# comment` and `; comment` lines, and blank lines, are ignored. A comment may also follow a value on the same line, preceded by
    whitespace and `#` ("port = 80  # http"), but a `#` inside a quoted string is data.
  * `[section]` starts a section; `[a.b]` is a nested section (the result nests dicts: {"a": {"b": {...}}}). Section names are
    [A-Za-z0-9_-] parts separated by dots. Keys before the first section go to the top level. Reopening a section that already
    exists adds to it; defining the same key twice in one section is an error.
  * `key = value`: key is [A-Za-z0-9_-]+; spaces around `=` and around the value are ignored.
  * values: integers (`42`, `-7`, `1_000` underscores allowed between digits), floats (`3.14`, `-0.5`, `1e3`), booleans `true`/`false`,
    double-quoted strings with the escapes \\" \\\\ \\n \\t, single-quoted strings (no escapes), arrays `[1, 2, "x"]` (nested arrays and
    trailing commas allowed, may NOT span lines), and bare words are errors.
Result: nested dicts with Python values (int, float, bool, str, list).
Every error raises ConfigError (a subclass of ValueError defined in the module) whose message contains "line N" (1-based).
Errors: unknown value syntax, unterminated string, missing `=`, invalid key or section name, duplicate key, text after a value, a
section name that conflicts with an existing non-section value.
""",
        "start": {"miniconf.py": '"""A small configuration parser (to be written)."""\n\n\nclass ConfigError(ValueError):\n    pass\n\n\ndef loads(text):\n    raise NotImplementedError\n'},
        "solution": {"miniconf.py": '''"""A small configuration parser."""
import re


class ConfigError(ValueError):
    pass


_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_INT = re.compile(r"^[+-]?\\d+(?:_\\d+)*$")


def _err(n, msg):
    return ConfigError(f"line {n}: {msg}")


def _strip_comment(s):
    in_s = in_d = False
    i = 0
    while i < len(s):
        c = s[i]
        if in_d:
            if c == "\\\\":
                i += 1
            elif c == '"':
                in_d = False
        elif in_s:
            if c == "'":
                in_s = False
        elif c == '"':
            in_d = True
        elif c == "'":
            in_s = True
        elif c in "#;" and (i == 0 or s[i - 1].isspace()):
            return s[:i]
        i += 1
    return s


def _value(s, n):
    s = s.strip()
    if not s:
        raise _err(n, "missing value")
    v, rest = _parse_value(s, 0, n)
    if rest.strip():
        raise _err(n, "unexpected text after value")
    return v


def _parse_value(s, i, n):
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        raise _err(n, "missing value")
    c = s[i]
    if c == '"':
        out, i = [], i + 1
        while i < len(s):
            ch = s[i]
            if ch == "\\\\":
                if i + 1 >= len(s) or s[i + 1] not in '"\\\\nt':
                    raise _err(n, "bad escape")
                out.append({"n": "\\n", "t": "\\t"}.get(s[i + 1], s[i + 1]))
                i += 2
            elif ch == '"':
                return "".join(out), s[i + 1:]
            else:
                out.append(ch)
                i += 1
        raise _err(n, "unterminated string")
    if c == "'":
        j = s.find("'", i + 1)
        if j < 0:
            raise _err(n, "unterminated string")
        return s[i + 1:j], s[j + 1:]
    if c == "[":
        items, i = [], i + 1
        while True:
            while i < len(s) and s[i] == " ":
                i += 1
            if i >= len(s):
                raise _err(n, "unterminated array")
            if s[i] == "]":
                return items, s[i + 1:]
            v, rest = _parse_value(s, i, n)
            items.append(v)
            i = len(s) - len(rest)
            while i < len(s) and s[i] == " ":
                i += 1
            if i < len(s) and s[i] == ",":
                i += 1
            elif i < len(s) and s[i] == "]":
                continue
            else:
                raise _err(n, "expected , or ] in array")
    m = re.match(r"[^\\s,\\]]+", s[i:])
    if not m:
        raise _err(n, "bad value")
    tok, rest = m.group(0), s[i + m.end():]
    if tok == "true":
        return True, rest
    if tok == "false":
        return False, rest
    if _INT.match(tok):
        return int(tok.replace("_", "")), rest
    if re.match(r"^[+-]?\\d+(?:_\\d+)*(?:\\.\\d+(?:_\\d+)*)?(?:[eE][+-]?\\d+)?$", tok):
        return float(tok.replace("_", "")), rest
    raise _err(n, f"unknown value {tok!r}")


def loads(text):
    root, cur, seen_sections = {}, None, set()
    cur = root
    for n, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw).strip()
        if not line:
            continue
        if line.startswith("["):
            if not line.endswith("]"):
                raise _err(n, "bad section header")
            name = line[1:-1].strip()
            parts = name.split(".")
            if not name or not all(_NAME.match(p) for p in parts):
                raise _err(n, f"invalid section name {name!r}")
            cur = root
            for p in parts:
                nxt = cur.setdefault(p, {})
                if not isinstance(nxt, dict):
                    raise _err(n, f"section {name!r} conflicts with a value")
                cur = nxt
            continue
        if "=" not in line:
            raise _err(n, "expected key = value")
        key, _, val = line.partition("=")
        key = key.strip()
        if not _NAME.match(key):
            raise _err(n, f"invalid key {key!r}")
        if key in cur:
            raise _err(n, f"duplicate key {key!r}")
        cur[key] = _value(val, n)
    return root
'''},
    },
    "hidden": {"test_miniconf.py": '''import unittest
from miniconf import ConfigError, loads


class Values(unittest.TestCase):
    def test_scalars(self):
        self.assertEqual(loads("a = 42\\nb = -7\\nc = 1_000\\nd = 3.14\\ne = -0.5\\nf = 1e3\\ng = true\\nh = false"),
                         {"a": 42, "b": -7, "c": 1000, "d": 3.14, "e": -0.5, "f": 1000.0, "g": True, "h": False})
        self.assertIsInstance(loads("a = 1e3")["a"], float)
        self.assertIsInstance(loads("a = 1")["a"], int)

    def test_strings(self):
        self.assertEqual(loads('a = "x y"')["a"], "x y")
        self.assertEqual(loads('a = "q\\\\"q\\\\\\\\n\\\\n\\\\t"')["a"], 'q"q\\\\n\\n\\t')
        self.assertEqual(loads("a = 'raw \\\\n'")["a"], "raw \\\\n")
        self.assertEqual(loads('a = ""')["a"], "")
        self.assertEqual(loads('a = "has # hash" # real comment')["a"], "has # hash")
        self.assertEqual(loads("a = 'it;s'")["a"], "it;s")

    def test_arrays(self):
        self.assertEqual(loads("a = [1, 2, 3]")["a"], [1, 2, 3])
        self.assertEqual(loads('a = [1, "x", true, 2.5]')["a"], [1, "x", True, 2.5])
        self.assertEqual(loads("a = [[1, 2], [3]]")["a"], [[1, 2], [3]])
        self.assertEqual(loads("a = []")["a"], [])
        self.assertEqual(loads("a = [1, 2,]")["a"], [1, 2])
        self.assertEqual(loads('a = ["a, b", "]"]')["a"], ["a, b", "]"])

    def test_spaces_and_comments(self):
        self.assertEqual(loads("# top\\n; also\\n\\n  a   =   5   # tail\\nb=6 ; tail2\\n"), {"a": 5, "b": 6})


class Sections(unittest.TestCase):
    def test_nesting(self):
        text = "top = 1\\n[a]\\nx = 1\\n[a.b]\\ny = 2\\n[c]\\nz = 3\\n"
        self.assertEqual(loads(text), {"top": 1, "a": {"x": 1, "b": {"y": 2}}, "c": {"z": 3}})

    def test_reopening_adds(self):
        self.assertEqual(loads("[a]\\nx = 1\\n[b]\\ny = 2\\n[a]\\nz = 3\\n"), {"a": {"x": 1, "z": 3}, "b": {"y": 2}})
        self.assertEqual(loads("[a.b]\\nx = 1\\n[a]\\ny = 2\\n"), {"a": {"b": {"x": 1}, "y": 2}})

    def test_names(self):
        self.assertEqual(loads("[my-sec_1]\\nk-1 = 1"), {"my-sec_1": {"k-1": 1}})


class Errors(unittest.TestCase):
    def test_all_errors_name_the_line(self):
        cases = [("a = 1\\nb\\n", 2), ("a = 1\\na = 2\\n", 2), ("\\n\\na = foo\\n", 3), ('a = "open\\n', 1), ("a = 'open\\n", 1),
                 ("a = 1 2\\n", 1), ("[bad name]\\n", 1), ("[a\\n", 1), ("[a.]\\n", 1), ("bad key = 1\\n", 1), ("a = [1, 2\\n", 1),
                 ("a = [1 2]\\n", 1), ("a = 1\\n[a]\\n", 2), ("a = \\n", 1), ('a = "bad\\\\q"\\n', 1), ("[x]\\ny = 1\\n[x]\\ny = 2\\n", 4),
                 ("= 1\\n", 1), ("a = 1_\\n", 1), ("a = 0x10\\n", 1)]
        for text, line in cases:
            with self.subTest(text=text):
                with self.assertRaises(ConfigError) as cm:
                    loads(text)
                self.assertIn(f"line {line}", str(cm.exception))

    def test_is_a_value_error(self):
        self.assertTrue(issubclass(ConfigError, ValueError))
        self.assertEqual(loads(""), {})
'''},
    "bugfix": {
        "symptom": "A config parser (miniconf.py) cuts values short: a quoted value that contains a semicolon, like name = \"a;b\", loses everything after the semicolon (it is treated as a comment), but a comment marker inside quotes is data.",
        "bug": [("miniconf.py", '''        elif c in "#;" and (i == 0 or s[i - 1].isspace()):
            return s[:i]''', '''        elif c in "#;":
            return s[:i]'''), ("miniconf.py", '''        if in_d:
            if c == "\\\\":
                i += 1
            elif c == '"':
                in_d = False
        elif in_s:''', '''        if False:
            pass
        elif in_s:''')],
        "repro": {"test_repro.py": 'import unittest\nfrom miniconf import loads\n\n\nclass Repro(unittest.TestCase):\n    def test_comment_marker_in_quotes(self):\n        self.assertEqual(loads(\'name = "a;b"\'), {"name": "a;b"})\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
