import hashlib
import os
import re
import subprocess
import sys
import unittest

HASHES = {
 "pyproject.toml": "59252f4adfcb13fcaedfa7a6bc829fd0b2d674dde60aa6c87fc82a56d67ee117",
 "tests/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
 "tests/test_envconf.py": "8c5acd94fd73aa63d9fecc1c3dcafcb29bfa61407bf6ae6d153c7bd75242009d",
 "envconf/__init__.py": "f62741a3c15c6797aa651cfef4edbf9835a3b9d072d63ca1f373212f0069b084",
 "envconf/errors.py": "0fb97ab86fb60e00e37c06be4ee405e5a53f0b56fea0e1d043189a94d8259374",
 "envconf/fields.py": "c178c67381d25b39aa3c1a650e1cb60f480209f1abf89a18cab6cfe1a2bcc738",
 "envconf/loader.py": "00a3d6edc3c77cbf2bb70a8353a97ed15aadb57b0d3e839c9820eda71e6048f8",
}

PUBLIC = ["load", "Field", "Config", "ConfigError", "parse_bool"]


def read_readme():
    for n in ("README.md", "readme.md", "Readme.md"):
        if os.path.exists(n):
            return open(n, encoding="utf-8").read()
    return None


def blocks(txt):
    out = []
    for m in re.finditer(r"```[ \t]*(python3?|py)[ \t]*\n(.*?)```", txt, re.S | re.I):
        code = m.group(2)
        if re.search(r"^\s*>>> ", code, re.M):  # doctest-стиль
            lines = []
            for ln in code.splitlines():
                if ln.startswith((">>> ", "... ")):
                    lines.append(ln[4:])
                elif ln.strip() in (">>>", "..."):
                    lines.append("")
            code = "\n".join(lines)
        out.append(code)
    return out


def run(code):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("APP_", "DB_", "MYAPP_", "SVC_"))}
    env["PYTHONPATH"] = os.getcwd()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, env=env)
    return r.returncode == 0, r.stderr[-300:]


class H(unittest.TestCase):
    def test_sources_unchanged(self):
        for rel, h in HASHES.items():
            self.assertTrue(os.path.exists(rel), "удалён " + rel)
            self.assertEqual(hashlib.sha256(open(rel, "rb").read()).hexdigest(), h, "изменён " + rel)

    def test_exists_and_size(self):
        t = read_readme()
        self.assertIsNotNone(t, "нет README.md")
        self.assertGreater(len(t), 1200)

    def test_mentions_every_public_name(self):
        t = read_readme() or ""
        bl = "\n".join(blocks(t))
        for n in PUBLIC:
            self.assertIn(n, t, n + " не упомянут")
            self.assertIn(n, bl, "нет примера для " + n)

    def test_code_blocks_run(self):
        bl = blocks(read_readme() or "")
        self.assertGreaterEqual(len(bl), 3)
        prefix = ""
        failed = []
        for i, b in enumerate(bl):
            ok, err = run(b)
            if not ok:
                ok, err = run(prefix + "\n" + b)  # блок может опираться на предыдущие
            if not ok:
                failed.append((i, err))
            prefix += "\n" + b
        self.assertFalse(failed, "блоки падают: %s" % failed)

    def test_examples_show_required_details(self):
        bl = "\n".join(blocks(read_readme() or ""))
        self.assertRegex(bl, r"prefix\s*=", "в примерах нет prefix=")
        self.assertRegex(bl, r"except\s+(envconf\.)?ConfigError", "нет перехвата ConfigError")
        self.assertRegex(bl, r"\.errors", "не показан атрибут errors")
        self.assertRegex(bl, r"as_dict", "нет as_dict")

    def test_russian_prose(self):
        t = read_readme() or ""
        prose = re.sub(r"```.*?```|`[^`]*`", "", t, flags=re.S)
        letters = re.findall(r"[A-Za-zА-Яа-яЁё]", prose)
        self.assertGreater(len(letters), 400)
        cyr = sum(1 for c in letters if ord(c) > 127) / len(letters)
        self.assertGreater(cyr, 0.6, "текст не на русском (доля кириллицы %.2f)" % cyr)


if __name__ == "__main__":
    unittest.main()
