FAMILY = {
    "name": "f13_csv", "lang": "python",
    "implement": {
        "prompt": """Write the module `minicsv.py` (standard library only; do NOT use the csv module) with `parse(text, delimiter=",")` and `to_dicts(text, delimiter=",")`.

parse(text, delimiter) -> list of rows (each a list of str) following RFC 4180 rules:
  * records end with "\\n" or "\\r\\n"; a final line terminator does not create an extra empty record; an empty input gives [].
  * a field may be wrapped in double quotes; inside quotes the delimiter, line breaks and "" (two quotes = one literal quote) are data.
  * a quote in the middle of an unquoted field is an ordinary character; text after the closing quote of a quoted field (other than
    the delimiter or a line end) is an error.
  * a blank line in the middle of the input is a record with one empty field [""] (but not the final empty line).
  * delimiter is exactly one character other than a quote or a line break, otherwise ValueError.
  * errors raise ValueError with a message that contains "line N" where N (1-based) is the physical line on which the problem is
    detected: an unterminated quote is reported at the line where that quoted field STARTED; garbage after a closing quote at the line
    where it occurs.
to_dicts(text, delimiter) -> list of dicts using the first record as header names. Rows shorter than the header are padded with
  "" and rows longer than the header raise ValueError ("line N" of the offending record's first line). Duplicate header names raise
  ValueError. Empty input gives [].
""",
        "start": {"minicsv.py": '"""A small CSV parser (to be written)."""\n\n\ndef parse(text, delimiter=","):\n    raise NotImplementedError\n\n\ndef to_dicts(text, delimiter=","):\n    raise NotImplementedError\n'},
        "solution": {"minicsv.py": '''"""A small CSV parser."""


def parse(text, delimiter=","):
    if not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in '"\\r\\n':
        raise ValueError("delimiter must be one character other than a quote or a line break")
    rows, row, field = [], [], []
    i, n, line = 0, len(text), 1
    record_line = 1
    in_quotes, quote_line, after_quote = False, 1, False
    started = False                      # the current record has any content (a field, a delimiter or a quote)
    while i < n:
        c = text[i]
        if in_quotes:
            if c == '"':
                if i + 1 < n and text[i + 1] == '"':
                    field.append('"')
                    i += 2
                    continue
                in_quotes, after_quote = False, True
            else:
                if c == "\\n":
                    line += 1
                field.append(c)
            i += 1
            continue
        if c == "\\r" and i + 1 < n and text[i + 1] == "\\n":
            i += 1
            continue
        if c == "\\n":
            row.append("".join(field))
            rows.append(row)
            row, field, started, after_quote = [], [], False, False
            line += 1
            record_line = line
            i += 1
            continue
        if after_quote and c != delimiter:
            raise ValueError(f"line {line}: unexpected character after closing quote")
        if c == delimiter:
            row.append("".join(field))
            field, after_quote, started = [], False, True
        elif c == '"' and not field and not after_quote:
            in_quotes, quote_line, started = True, line, True
        else:
            field.append(c)
            started = True
        i += 1
    if in_quotes:
        raise ValueError(f"line {quote_line}: unterminated quoted field")
    if started or field or row or after_quote:
        row.append("".join(field))
        rows.append(row)
    return rows


def to_dicts(text, delimiter=","):
    rows = parse(text, delimiter)
    if not rows:
        return []
    header = rows[0]
    if len(set(header)) != len(header):
        raise ValueError("line 1: duplicate header names")
    out = []
    # recompute physical first lines of records for error messages
    line = 1
    starts = []
    pos_rows = parse(text, delimiter)
    for r in pos_rows:
        starts.append(line)
        line += 1 + sum(f.count("\\n") for f in r)
    for idx, r in enumerate(rows[1:], 1):
        if len(r) > len(header):
            raise ValueError(f"line {starts[idx]}: too many fields")
        out.append({h: (r[j] if j < len(r) else "") for j, h in enumerate(header)})
    return out
'''},
    },
    "hidden": {"test_minicsv.py": '''import unittest
from minicsv import parse, to_dicts


class Parse(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(parse("a,b,c\\n1,2,3\\n"), [["a", "b", "c"], ["1", "2", "3"]])
        self.assertEqual(parse("a,b\\r\\n1,2"), [["a", "b"], ["1", "2"]])
        self.assertEqual(parse(""), [])
        self.assertEqual(parse("\\n"), [[""]])
        self.assertEqual(parse("x"), [["x"]])

    def test_empty_fields(self):
        self.assertEqual(parse(",,\\n"), [["", "", ""]])
        self.assertEqual(parse("a,\\n,b"), [["a", ""], ["", "b"]])

    def test_quotes(self):
        self.assertEqual(parse('"a,b",c\\n'), [["a,b", "c"]])
        self.assertEqual(parse('"he said ""hi""",x'), [['he said "hi"', "x"]])
        self.assertEqual(parse('"line1\\nline2",z\\n'), [["line1\\nline2", "z"]])
        self.assertEqual(parse('"",a'), [["", "a"]])
        self.assertEqual(parse('a"b,c'), [['a"b', "c"]])
        self.assertEqual(parse('""""'), [['"']])
        self.assertEqual(parse('"a\\r\\nb"'), [["a\\r\\nb"]])

    def test_blank_line_in_the_middle(self):
        self.assertEqual(parse("a\\n\\nb\\n"), [["a"], [""], ["b"]])

    def test_delimiter(self):
        self.assertEqual(parse("a;b;c\\n", ";"), [["a", "b", "c"]])
        self.assertEqual(parse("a\\tb", "\\t"), [["a", "b"]])
        self.assertEqual(parse('"a;b";c', ";"), [["a;b", "c"]])
        for bad in ("", ",,", '"', "\\n", "\\r"):
            with self.assertRaises(ValueError):
                parse("a", bad)

    def test_errors_carry_line_numbers(self):
        with self.assertRaises(ValueError) as cm:
            parse('a,b\\n"c,d\\ne,f\\n')
        self.assertIn("line 2", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            parse('a,b\\n"c"x,d\\n')
        self.assertIn("line 2", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            parse('"unterminated')
        self.assertIn("line 1", str(cm.exception))
        with self.assertRaises(ValueError):
            parse('"a" ,b')


class Dicts(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(to_dicts("name,age\\nAnn,30\\nBo,25\\n"), [{"name": "Ann", "age": "30"}, {"name": "Bo", "age": "25"}])
        self.assertEqual(to_dicts(""), [])
        self.assertEqual(to_dicts("a,b\\n"), [])

    def test_short_rows_are_padded_long_rows_fail(self):
        self.assertEqual(to_dicts("a,b,c\\n1\\n"), [{"a": "1", "b": "", "c": ""}])
        with self.assertRaises(ValueError) as cm:
            to_dicts("a,b\\n1,2\\n3,4,5\\n")
        self.assertIn("line 3", str(cm.exception))

    def test_line_number_after_multiline_field(self):
        with self.assertRaises(ValueError) as cm:
            to_dicts('a,b\\n"x\\ny",1\\n1,2,3\\n')
        self.assertIn("line 4", str(cm.exception))

    def test_duplicate_headers(self):
        with self.assertRaises(ValueError):
            to_dicts("a,a\\n1,2")

    def test_delimiter(self):
        self.assertEqual(to_dicts("a;b\\n1;2", ";"), [{"a": "1", "b": "2"}])
'''},
    "bugfix": {
        "symptom": "A CSV parser (minicsv.py) loses data when a quoted field contains an escaped quote: parse('\"he said \"\"hi\"\"\",x') should give [['he said \"hi\"', 'x']] but the quote characters are dropped.",
        "bug": [("minicsv.py", '''                if i + 1 < n and text[i + 1] == '"':
                    field.append('"')
                    i += 2
                    continue''', '''                if i + 1 < n and text[i + 1] == '"':
                    i += 2
                    continue''')],
        "repro": {"test_repro.py": 'import unittest\nfrom minicsv import parse\n\n\nclass Repro(unittest.TestCase):\n    def test_escaped_quote(self):\n        self.assertEqual(parse(\'"he said ""hi""",x\'), [[\'he said "hi"\', "x"]])\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
