import os, re, subprocess, sys, unittest

HASHES = {
 "pyproject.toml": "ccea253514313c59e6dcd80dcb5be1acb707f480709495942f1f7bacceeb3d67",
 "tests/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
 "tests/test_tinykit.py": "9caeeeda844f1cdf13c93c6d196d3c2a048471eb4b7a95697aa34f73ce7a5bd7",
 "tinykit/__init__.py": "a6a45b45032bf18a5cfefbff274b697e09740d07f3b1cd4e94c1cf044ebad0b2",
 "tinykit/ratelimit.py": "b31fbde7213c771babbe59860a1d70bd7894d1460a564a2625a04a33a135566b",
 "tinykit/retry.py": "aaa50b7949b35cb706400ee01d42429bf011213efba4255bb226cafe45be43cb",
 "tinykit/seq.py": "25e1719eb8da35ede467ee31038106300e6eddc92b8475e29b72c195373ce723",
 "tinykit/text.py": "b6b7012a582d31b7a58fd656b99b48c22fcda054919e00b08addd6fb9e1cec5f"
}

import hashlib
import tinykit

PUBLIC = list(tinykit.__all__)


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
    env = dict(os.environ, PYTHONPATH=os.getcwd())
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

    def test_mentions_every_public_name_with_example(self):
        t = read_readme() or ""
        bl = "\n".join(blocks(t))
        prose = re.sub(r"```.*?```", "", t, flags=re.S)
        for n in PUBLIC:
            self.assertIn(n, t, n + " не упомянут")
        # у каждой функции/класса (кроме исключения) есть пример кода
        for n in PUBLIC:
            if n == "RateLimitExceeded":
                continue
            self.assertIn(n, bl, "нет примера для " + n)
        self.assertIn("RateLimitExceeded", t)

    def test_code_blocks_run(self):
        t = read_readme() or ""
        bl = blocks(t)
        self.assertGreaterEqual(len(bl), 2)
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

    def test_russian_prose(self):
        t = read_readme() or ""
        prose = re.sub(r"```.*?```|`[^`]*`", "", t, flags=re.S)
        letters = re.findall(r"[A-Za-zА-Яа-яЁё]", prose)
        self.assertGreater(len(letters), 400)
        cyr = sum(1 for c in letters if ord(c) > 127) / len(letters)
        self.assertGreater(cyr, 0.6, "текст не на русском (доля кириллицы %.2f)" % cyr)


if __name__ == "__main__":
    unittest.main()
