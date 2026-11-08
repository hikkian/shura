import hashlib, os, re, unittest

ANS = open(os.environ["ANSWER_FILE"], encoding="utf-8", errors="replace").read()
HASHES = {
 "README.md": "cc59227d8306e1cd4e9886160090e4918ec62003a357a4714750405e8436b75c",
 "app.log": "ccbcfa88cbdca51a1fb17135ac4e7ac06f64b7051bcf46df80c8a5104d267f0e"
}


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

    def test_root_cause_component(self):
        self.assertRegex(ANS, r"orders[-_ ]?service|сервис[а-я]* заказов")
        self.assertRegex(ANS, r"pool|пул", )

    def test_root_cause_time(self):
        # первая проблема: 13:41 (смена конфига), 13:43 (первое замедление), 13:47 (первая ошибка пула)
        self.assertRegex(ANS, r"13[:.]4[137]")

    def test_why_config_change(self):
        self.assertRegex(ANS, r"max_size|CHG-4412|конфиг|config|настройк|размер")
        self.assertRegex(ANS, r"(?<![\d:.])5(?![\d:.%])")  # новый размер пула = 5

    def test_not_symptom_as_cause(self):
        m = re.search(r"(первопричин|корнев|коренн|root cause|причин[аы]\s*[:—-])[^\n]{0,80}", ANS, re.I)
        if m:
            seg = m.group(0)
            if re.search(r"gateway|nginx|payment|502", seg, re.I):
                self.assertRegex(seg, r"pool|пул|orders", "причина названа по симптому")
