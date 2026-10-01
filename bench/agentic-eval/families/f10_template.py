FAMILY = {
    "name": "f10_template", "lang": "python",
    "implement": {
        "prompt": """Write the module `tmpl.py` (standard library only; no eval/exec/str.format on the template) with `render(template, context)`.

Syntax:
  {{ expr }}         output the value; expr is a name or a dotted path (user.name, items.0); a dotted part looks up a dict key,
                     then (for lists) an integer index. A missing name/key/index renders as the EMPTY string. Whitespace inside
                     the braces is ignored. Values are converted with str(); None renders as "". The output is HTML-escaped
                     (& < > " ') unless the expression ends with "|safe": {{ html|safe }}.
  {% if expr %}...{% endif %}  and  {% if expr %}...{% else %}...{% endif %}   the condition is true when the value is "truthy" in Python
                     (a missing name is falsy). "not" in front negates: {% if not items %}.
  {% for x in expr %}...{% endfor %}   repeats the body for every item of a list (a missing or non-list value repeats 0 times); inside, `x`
                     is the item, and `loop.index` (starting at 1) and `loop.last` (True on the last pass) are available. Loops and ifs nest.
  Anything else is output unchanged. {{{{ and }}}} are NOT special. Text and tags may span lines; whitespace is preserved exactly.
Errors (raise ValueError): an unclosed or unexpected tag ({% endif %} with no if, {% else %} outside an if), a malformed tag
(for without `in`), and an unclosed {{ .
The context is not modified.
""",
        "start": {"tmpl.py": '"""A tiny template engine (to be written)."""\n\n\ndef render(template, context):\n    raise NotImplementedError\n'},
        "solution": {"tmpl.py": '''"""A tiny template engine."""
import html
import re

_TOKEN = re.compile(r"(\\{\\{.*?\\}\\}|\\{%.*?%\\})", re.S)


def _lookup(path, scope):
    parts = path.split(".")
    if parts[0] not in scope:
        return None
    cur = scope[parts[0]]
    for p in parts[1:]:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list) and p.isdigit() and int(p) < len(cur):
            cur = cur[int(p)]
        elif isinstance(cur, dict) and p.isdigit() and int(p) in cur:
            cur = cur[int(p)]
        else:
            return None
    return cur


def _parse(tokens):
    root, stack = [], []
    cur = root
    for tok in tokens:
        if tok.startswith("{{"):
            if not tok.endswith("}}"):
                raise ValueError("unclosed {{")
            cur.append(("var", tok[2:-2].strip()))
        elif tok.startswith("{%"):
            words = tok[2:-2].split()
            if not words:
                raise ValueError("empty tag")
            kind = words[0]
            if kind == "if":
                neg = len(words) > 1 and words[1] == "not"
                expr = " ".join(words[2:] if neg else words[1:])
                if not expr:
                    raise ValueError("if without condition")
                node = ["if", neg, expr, [], []]
                cur.append(node)
                stack.append((node, cur))
                cur = node[3]
            elif kind == "else":
                if not stack or stack[-1][0][0] != "if" or cur is stack[-1][0][4]:
                    raise ValueError("unexpected else")
                cur = stack[-1][0][4]
            elif kind == "endif":
                if not stack or stack[-1][0][0] != "if":
                    raise ValueError("unexpected endif")
                cur = stack.pop()[1]
            elif kind == "for":
                if len(words) != 4 or words[2] != "in":
                    raise ValueError("malformed for")
                node = ["for", words[1], words[3], []]
                cur.append(node)
                stack.append((node, cur))
                cur = node[3]
            elif kind == "endfor":
                if not stack or stack[-1][0][0] != "for":
                    raise ValueError("unexpected endfor")
                cur = stack.pop()[1]
            else:
                raise ValueError(f"unknown tag {kind}")
        else:
            cur.append(("text", tok))
    if stack:
        raise ValueError("unclosed block")
    return root


def _emit(nodes, scope, out):
    for n in nodes:
        if n[0] == "text":
            out.append(n[1])
        elif n[0] == "var":
            expr = n[1]
            safe = False
            if "|" in expr:
                name, _, flt = expr.rpartition("|")
                if flt.strip() == "safe":
                    safe, expr = True, name.strip()
            v = _lookup(expr, scope)
            s = "" if v is None else str(v)
            out.append(s if safe else html.escape(s, quote=True))
        elif n[0] == "if":
            _, neg, expr, yes, no = n
            truth = bool(_lookup(expr, scope))
            _emit(yes if truth != neg else no, scope, out)
        else:
            _, name, expr, body = n
            items = _lookup(expr, scope)
            if isinstance(items, list):
                for i, item in enumerate(items):
                    _emit(body, {**scope, name: item, "loop": {"index": i + 1, "last": i == len(items) - 1}}, out)


def render(template, context):
    tokens = [t for t in _TOKEN.split(template) if t != ""]
    stripped = _TOKEN.sub("", template)
    if "{{" in stripped:
        raise ValueError("unclosed {{")
    out = []
    _emit(_parse(tokens), dict(context), out)
    return "".join(out)
'''},
    },
    "hidden": {"test_tmpl.py": '''import unittest
from tmpl import render


class Vars(unittest.TestCase):
    def test_basic_and_whitespace(self):
        self.assertEqual(render("Hello, {{ name }}!", {"name": "Ann"}), "Hello, Ann!")
        self.assertEqual(render("{{name}}-{{   name   }}", {"name": "x"}), "x-x")
        self.assertEqual(render("no tags", {}), "no tags")
        self.assertEqual(render("", {}), "")

    def test_paths_and_missing(self):
        ctx = {"user": {"name": "Bo", "tags": ["a", "b"]}, "items": [10, 20]}
        self.assertEqual(render("{{ user.name }} {{ user.tags.1 }} {{ items.0 }}", ctx), "Bo b 10")
        self.assertEqual(render("[{{ nope }}][{{ user.age }}][{{ items.5 }}][{{ user.name.x }}]", ctx), "[][][][]")
        self.assertEqual(render("[{{ n }}]", {"n": None}), "[]")
        self.assertEqual(render("{{ n }} {{ z }} {{ t }}", {"n": 0, "z": 1.5, "t": True}), "0 1.5 True")

    def test_escaping(self):
        self.assertEqual(render("{{ x }}", {"x": "<a href=\\"q\\">&'</a>"}), "&lt;a href=&quot;q&quot;&gt;&amp;&#x27;&lt;/a&gt;")
        self.assertEqual(render("{{ x|safe }}", {"x": "<b>"}), "<b>")
        self.assertEqual(render("{{ x | safe }}", {"x": "<b>"}), "<b>")

    def test_braces_not_special(self):
        self.assertEqual(render("{ {{ x }} }", {"x": "v"}), "{ v }")
        self.assertEqual(render("{ not a tag } 100%", {}), "{ not a tag } 100%")


class Blocks(unittest.TestCase):
    def test_if_else(self):
        t = "{% if on %}yes{% else %}no{% endif %}"
        self.assertEqual(render(t, {"on": True}), "yes")
        self.assertEqual(render(t, {"on": 0}), "no")
        self.assertEqual(render(t, {}), "no")
        self.assertEqual(render("{% if x %}a{% endif %}b", {"x": 1}), "ab")
        self.assertEqual(render("{% if x %}a{% endif %}b", {"x": 0}), "b")
        self.assertEqual(render("{% if not items %}empty{% endif %}", {"items": []}), "empty")
        self.assertEqual(render("{% if not items %}empty{% else %}some{% endif %}", {"items": [1]}), "some")
        self.assertEqual(render("{% if u.admin %}A{% endif %}", {"u": {"admin": True}}), "A")

    def test_for_and_loop_variables(self):
        t = "{% for x in xs %}{{ loop.index }}:{{ x }}{% if not loop.last %}, {% endif %}{% endfor %}"
        self.assertEqual(render(t, {"xs": ["a", "b", "c"]}), "1:a, 2:b, 3:c")
        self.assertEqual(render(t, {"xs": []}), "")
        self.assertEqual(render(t, {}), "")
        self.assertEqual(render(t, {"xs": "abc"}), "")

    def test_nesting_and_scope(self):
        t = "{% for r in rows %}[{% for c in r %}{{ c }}{% endfor %}]{% endfor %}"
        self.assertEqual(render(t, {"rows": [[1, 2], [], [3]]}), "[12][][3]")
        t2 = "{{ x }}{% for x in xs %}{{ x }}{% endfor %}{{ x }}"
        self.assertEqual(render(t2, {"x": "o", "xs": [1, 2]}), "o12o")
        t3 = "{% for u in users %}{% if u.ok %}{{ u.n }}{% else %}-{% endif %}{% endfor %}"
        self.assertEqual(render(t3, {"users": [{"n": "a", "ok": 1}, {"n": "b", "ok": 0}]}), "a-")

    def test_whitespace_and_lines_preserved(self):
        t = "a\\n{% for x in xs %}\\n- {{ x }}\\n{% endfor %}\\nz"
        self.assertEqual(render(t, {"xs": [1, 2]}), "a\\n\\n- 1\\n\\n- 2\\n\\nz")

    def test_context_unchanged(self):
        ctx = {"xs": [1], "x": "keep"}
        render("{% for x in xs %}{{ x }}{% endfor %}", ctx)
        self.assertEqual(ctx, {"xs": [1], "x": "keep"})


class Errors(unittest.TestCase):
    def test_errors(self):
        for bad in ["{% if x %}", "{% endif %}", "{% else %}", "{% for x in xs %}", "{% endfor %}", "{% for x xs %}", "{{ x",
                    "{% if x %}{% for y in z %}{% endif %}{% endfor %}", "{% if x %}a{% else %}b{% else %}c{% endif %}", "{% foo %}",
                    "text {{ unclosed"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    render(bad, {"x": 1, "xs": [1], "z": []})
'''},
    "bugfix": {
        "symptom": "A template engine (tmpl.py) forgets to escape one character: a value containing a single quote (') is rendered raw, but it must be HTML-escaped to &#x27; like the other special characters.",
        "bug": [("tmpl.py", "html.escape(s, quote=True)", "html.escape(s, quote=False).replace('\"', \"&quot;\")")],
        "repro": {"test_repro.py": 'import unittest\nfrom tmpl import render\n\n\nclass Repro(unittest.TestCase):\n    def test_quote_is_escaped(self):\n        self.assertEqual(render("{{ x }}", {"x": "it\'s"}), "it&#x27;s")\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
