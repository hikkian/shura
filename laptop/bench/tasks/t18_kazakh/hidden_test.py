import hashlib, os, re, unittest

ANS = open(os.environ["ANSWER_FILE"], encoding="utf-8", errors="replace").read()
HASHES = {
 "README.md": "009bee6f5728faccabad04a4eea749b48a5f90b319b661c08bde9d910aa38440",
 "jobqueue.py": "c4ea727cbab9a1928063880af7b412de1345640509317f2333b0807ff2bc939f"
}
KZ = "әғқңөұүһіӘҒҚҢӨҰҮҺІ"


def prose():
    return re.sub(r"```.*?```|`[^`]*`", "", ANS, flags=re.S)


class H(unittest.TestCase):
    def test_no_edits(self):
        cur = {}
        for root, ds, fs in os.walk("."):
            ds[:] = [d for d in ds if d not in ("__pycache__", ".git")]
            for f in fs:
                if f == "hidden_test.py" or f.endswith(".pyc"):
                    continue
                p = os.path.join(root, f)
                cur[os.path.relpath(p, ".")] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        self.assertEqual(cur, HASHES)

    def test_kazakh_letters(self):
        t = prose()
        n = sum(1 for c in t if c in KZ)
        letters = sum(1 for c in t if c.isalpha()) or 1
        self.assertGreaterEqual(n, 15, "казахских букв: %d" % n)
        self.assertGreater(n / letters, 0.03, "доля казахских букв мала: %.3f" % (n / letters))

    def test_not_english(self):
        t = prose()
        lat = sum(1 for c in t if "a" <= c.lower() <= "z")
        cyr = sum(1 for c in t if "\u0400" <= c <= "\u04ff")
        self.assertGreater(cyr, lat)

    def test_key_names(self):
        self.assertIn("backoff_delay", ANS)
        self.assertRegex(ANS, r"run_next|JobQueue|dead_letter")

    def test_explains_backoff_and_failures(self):
        self.assertRegex(ANS, r"экспонен|base\s*\*\*|2\s*\*\*|\*\*|60|cap|шектеу|ұлғай|арт|көбей")
        self.assertRegex(ANS, r"dead_letter|dead|өлі|сәтсіз|қате|қайта")
