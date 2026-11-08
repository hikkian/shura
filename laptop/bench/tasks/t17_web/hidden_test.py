import json, os, re, unittest, urllib.request

ANS = open(os.environ["ANSWER_FILE"], encoding="utf-8", errors="replace").read()


def latest():
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/httpx/json", timeout=20) as r:
            return json.load(r)["info"]["version"]
    except Exception:
        return None


class H(unittest.TestCase):
    def test_pinned_latest(self):
        v = latest()
        if v is None:
            self.skipTest("нет сети: не удалось получить версию с PyPI")
        lines = [l.strip() for l in open("requirements.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]
        httpx_lines = [l for l in lines if re.match(r"httpx\b", l, re.I)]
        self.assertEqual(len(httpx_lines), 1, lines)
        m = re.match(r"httpx(?:\[[^\]]*\])?\s*==\s*([\w.]+)\s*$", httpx_lines[0], re.I)
        self.assertIsNotNone(m, "httpx должен быть закреплён через ==: " + httpx_lines[0])
        self.assertEqual(m.group(1), v)

    def test_other_deps_untouched(self):
        lines = [l.strip() for l in open("requirements.txt", encoding="utf-8") if l.strip()]
        self.assertIn("requests==2.31.0", lines)
        self.assertIn("pytest>=7.0", lines)

    def test_answer_has_source(self):
        self.assertRegex(ANS, r"https?://(www\.)?(pypi\.org|github\.com)/\S+")

    def test_answer_mentions_version(self):
        v = latest()
        if v is None:
            self.skipTest("нет сети")
        self.assertIn(v, ANS)


if __name__ == "__main__":
    unittest.main()
