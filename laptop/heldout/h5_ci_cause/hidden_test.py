import hashlib
import os
import re
import unittest

ANS = open(os.environ["ANSWER_FILE"], encoding="utf-8", errors="replace").read()

HASHES = {
 "README.md": "24a28db8811f8535402343ff04bc11cd80958c2042cd2f3c7c5c3327e651e28c",
 "ci_runs.csv": "48803cf53b0ea5139df05f1e6c639454c49d98cb730b2856d57b301b0dfe2b6b",
 "failures.log": "4933bf40e1488d1ac6487266e72b93c9f71ecaa242706b5426973bb6dae9d0d6",
 "git_log.txt": "b1c65f360c27a92777c80062b8260305375dcc8456ddfeff4eddb1e78101e109",
}


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", t) if s.strip()]


class H(unittest.TestCase):
    def test_no_edits(self):  # любой лишний файл в проекте тоже провал
        cur = {}
        for root, ds, fs in os.walk("."):
            ds[:] = [d for d in ds if d not in ("__pycache__", ".git")]
            for f in fs:
                if f == "hidden_test.py" or f.endswith(".pyc"):
                    continue
                p = os.path.join(root, f)
                cur[os.path.relpath(p, ".")] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        self.assertEqual(cur, HASHES)

    def test_root_commit(self):
        self.assertRegex(ANS, r"d4e8b17|(?i:пересбор\w*|nightly|ночн\w* (событ|сборк))", "не назван коммит d4e8b17 и не названо ночное событие")

    def test_root_mechanism(self):
        self.assertRegex(ANS, r"(?i)tzdata", "не упомянут tzdata")
        # пин сняли / версия обновилась
        self.assertRegex(ANS, r"(?i)(пин|закреп|зафиксир|pin|2023\.3|обнов|upgrad|bump|unpin|ослаб|relax|убра|убир|удал|снят|сняли|снял|привяз|отвяз|отпуст|не\s+зафиксир)")
        self.assertRegex(ANS, r"(?i)(Asia/Almaty|Алматы|Казахстан|utcoffset|\+0?5|\+0?6|UTC\+|часов\w* пояс|time ?zone|таймзон)")

    def test_why_not_failing_earlier(self):
        # почему Wed-прогон d4e8b17 был зелёным: кешированный образ, пересборка ночью
        self.assertRegex(ANS, r"(?i)(кеш|кэш|cache|cached|пересбор|rebuil|no-cache|nightly|ночн)")

    def test_first_failure_time(self):
        self.assertTrue(re.search(r"(?<![\d:])0?2[:.]\s?(00|04|14|19)(?![\d])", ANS)
                        or re.search(r"(?i)(двух часов ночи|два часа ночи|в начале третьего)", ANS),
                        "не указано время начала (ночь 13.03, около 02:00-02:19)")
        self.assertRegex(ANS, r"(13[ .]03|13 марта|четверг|2025-03-13|ночью|ночн)")

    def test_not_symptom_as_cause(self):
        # патч-предложение: отрицание должно стоять рядом с ключевым словом (окно -40/+60), а не где-то в предложении
        for s in sentences(ANS):
            for m in re.finditer(r"(?i)(причин|корн|коренн|root cause|виноват|сломал)", s):
                if not re.search(r"f0c9a2e|refactor|рефактор|Decimal|округл|registry|реестр|таймаут|timeout", s, re.I):
                    continue
                if re.search(r"d4e8b17|tzdata", s, re.I):
                    continue
                win = s[max(0, m.start() - 40): m.end() + 60]
                ok = re.search(r"(?i)(не |нет |вовсе|раньше|до того|симптом|шум|ложн|побочн|лишь|только)", win)
                self.assertTrue(ok, "ложная причина рядом с ключевым словом: " + win)

    def test_no_invented_facts(self):
        # в данных нет таких вещей: выдумывать нельзя
        for pat in [r"(?i)python\s*3\.\d+\s*(обнов|upgrad)", r"(?i)docker\s*daemon\s*(упал|crash)"]:
            self.assertIsNone(re.search(pat, ANS), "выдуман факт: " + pat)

    def test_russian(self):
        prose = re.sub(r"```.*?```|`[^`]*`", "", ANS, flags=re.S)
        prose = re.sub(r"\b[0-9a-f]{7}\b|\S*[/_.]\S*", "", prose)
        letters = re.findall(r"[A-Za-zА-Яа-яЁё]", prose)
        self.assertGreater(len(letters), 150)
        cyr = sum(1 for c in letters if ord(c) > 127) / len(letters)
        self.assertGreater(cyr, 0.7, "ответ не на русском (%.2f)" % cyr)

    def test_not_empty_short(self):
        self.assertGreater(len(ANS.strip()), 200)


if __name__ == "__main__":
    unittest.main()
