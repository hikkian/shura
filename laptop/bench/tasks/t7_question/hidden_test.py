import os, re, unittest
ANS = open(os.environ["ANSWER_FILE"]).read()
class H(unittest.TestCase):
    def test_answer(self):
        self.assertIn("25", ANS)
        self.assertRegex(ANS, r"client\.py")
    def test_not_legacy(self):
        self.assertFalse(re.search(r"(таймаут|timeout)[^\n]{0,40}\b10\b", ANS, re.I) and "25" not in ANS)
