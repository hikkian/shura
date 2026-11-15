"""LL2: the memory-search measurement (bench/memory-search). Checks the fixtures (no secrets, no real-looking personal data, queries point at
live records), the metric and threshold functions, the decision rules of PREREG.md on synthetic summaries, and that every arm behaves
(the u61 arm is the production search, deleted and old versions are never found, the trigram arm tolerates typos). No model, no GPU."""
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench" / "memory-search"
sys.path.insert(0, str(ROOT / "gateway"))
import lifelong_store as L  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = load("ll2_run", BENCH / "run_ll2.py")
MQ = load("ll2_make_queries", BENCH / "make_queries.py")
FIX = BENCH / "fixtures"
GROUP_COUNTS = {"plain_ru": 30, "plain_en": 24, "code": 24, "mixed": 24, "case": 26, "paraphrase": 30, "version": 12, "dispute": 6, "crosslang": 10,
                "negative": 30, "negative_deleted": 4}
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")


class FixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="ll2-test-")
        cls.store = R.build_store(Path(cls.tmp.name) / "f.sqlite")
        R.add_trigram(cls.store)
        cls.data = json.loads((FIX / "records.json").read_text(encoding="utf-8"))
        cls.queries = json.loads((FIX / "queries.json").read_text(encoding="utf-8"))["queries"]

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        cls.tmp.cleanup()

    def live_keys(self):
        return {r["key"] for r in self.store.db.execute("SELECT key FROM memory WHERE deleted_at IS NULL")}


class Fixtures(FixtureCase):
    def test_records_are_unique_and_have_no_secrets_or_personal_looking_data(self):
        keys = [r["key"] for r in self.data["records"]] + [c["chain"] for c in self.data["chains"]] + [d["key"] for d in self.data["disputes"]]
        self.assertEqual(len(keys), len(set(keys)))
        blob = json.dumps(self.data, ensure_ascii=False)
        self.assertIsNone(EMAIL.search(blob), "an e-mail address in the fixture")
        self.assertIsNone(PHONE.search(blob.replace("\\n", " ")), "a phone-number-looking string in the fixture")
        for pattern, what in L.SECRET_PATTERNS:
            self.assertIsNone(re.search(pattern, blob), f"{what} in the fixture")

    def test_the_fixture_loads_through_the_production_api(self):
        live = self.live_keys()
        self.assertGreaterEqual(len(live), 120)
        self.assertNotIn("deleted-vulkan-fast", live)                     # forgotten records are not live
        self.assertIn("profile-a-window", live)                           # a chain keeps its key on the newest version
        self.assertIn("disp-ram-a", live)
        self.assertIn("disp-ram-a-alt", live)                             # the other reading of a dispute
        old = [k for (k,) in self.store.db.execute("SELECT key FROM memory WHERE deleted_at IS NOT NULL")]
        self.assertTrue(any("~v" in k for k in old))

    def test_the_queries_have_the_planned_shape(self):
        groups = {}
        for q in self.queries:
            groups[q["group"]] = groups.get(q["group"], 0) + 1
        for g, n in GROUP_COUNTS.items():
            self.assertEqual(groups.get(g), n, g)
        self.assertGreaterEqual(groups.get("typo", 0), 30)
        ids = [q["id"] for q in self.queries]
        self.assertEqual(len(ids), len(set(ids)))
        for q in self.queries:
            self.assertLessEqual(len(q["text"]), 220, q["id"])
            self.assertIsNone(EMAIL.search(q["text"]))

    def test_every_expected_key_is_a_live_record_and_negatives_expect_nothing(self):
        live = self.live_keys()
        for q in self.queries:
            if q["group"] in R.NEGATIVE_GROUPS:
                self.assertEqual(q["expected"], [], q["id"])
            else:
                self.assertTrue(q["expected"], q["id"])
                for k in q["expected"]:
                    self.assertIn(k, live, f"{q['id']}: {k}")
        distinct = {k for q in self.queries if q["group"] not in R.NEGATIVE_GROUPS for k in q["expected"]}
        self.assertGreaterEqual(len(distinct), 70)

    def test_typos_are_generated_with_a_fixed_seed_and_change_a_word(self):
        typos = [q for q in self.queries if q["group"] == "typo"]
        base = {q["id"]: q for q in self.queries}
        for q in typos:
            self.assertNotEqual(q["text"], base[q["from"]]["text"])
            self.assertEqual(q["expected"], base[q["from"]]["expected"])


class Arms(FixtureCase):
    def test_u61_is_the_production_search(self):
        for text in ("как оформлять коммиты", "load-mode dio", "memoryEnabled flag"):
            prod = [(r["key"], r["score"]) for r in self.store.search(text, limit=5)]
            self.assertEqual(R.search_arm(self.store, "u61", text), prod)

    def test_no_arm_returns_a_deleted_record_or_an_old_version(self):
        for arm in R.ARMS:
            for q in self.queries:
                for key, _ in R.search_arm(self.store, arm, q["text"]):
                    self.assertNotIn("~v", key, (arm, q["id"]))
                    self.assertFalse(key.startswith("deleted-"), (arm, q["id"]))

    def test_trigram_finds_a_word_with_a_typo_that_unicode61_misses(self):
        res = R.search_arm(self.store, "tri", "Mizzan")
        self.assertEqual(res[0][0], "mizan-benchmark")
        self.assertEqual(R.search_arm(self.store, "triw", "Mizzan"), [])      # one quoted substring cannot survive a slip
        self.assertEqual(R.search_arm(self.store, "u61", "Mizzan"), [])

    def test_trigram_is_case_insensitive_for_cyrillic(self):
        res = R.search_arm(self.store, "tri", "АВТОР коммитов")
        self.assertIn("commit-author", [k for k, _ in res])

    def test_both_scores_are_the_sum_of_the_two_arms(self):
        text = "андервольт видеокарты"
        a, b = dict(R.search_arm(self.store, "u61", text, limit=20)), dict(R.search_arm(self.store, "tri", text, limit=20))
        for key, score in R.search_arm(self.store, "both", text, limit=20):
            twice = R.CONFIRMED_BONUS if key in a and key in b and self.store._row(key)["trust"] == "confirmed" else 0.0
            self.assertAlmostEqual(score, a.get(key, 0.0) + b.get(key, 0.0) - twice, places=9)     # the trust bonus counts once
        self.assertTrue(any(k in a and k in b for k in dict(R.search_arm(self.store, "both", text, limit=20))))

    def test_the_stem_arm_matches_other_word_forms(self):
        res = R.search_arm(self.store, "stem", "расскажи про коммитов автора")
        self.assertIn("commit-author", [k for k, _ in res])

    def test_a_query_of_stop_words_returns_nothing_instead_of_everything(self):
        for arm in R.ARMS:
            self.assertEqual(R.search_arm(self.store, arm, "и в на"), [])


class Metrics(unittest.TestCase):
    def test_estimated_tokens(self):
        self.assertEqual(R.est_tokens("a" * 9), 3)
        self.assertEqual(R.est_tokens("a" * 10), 4)

    def test_first_hit_rank(self):
        self.assertEqual(R.first_hit_rank(["a", "b", "c"], ["c"]), 3)
        self.assertIsNone(R.first_hit_rank(["a"], ["z"]))

    def test_threshold_keeps_at_most_ten_percent_of_negatives_above_it(self):
        scores = [float(i) for i in range(1, 21)]            # 1..20
        t = R.tune_threshold(scores)
        self.assertLessEqual(sum(1 for s in scores if s > t) / len(scores), 0.10)
        self.assertEqual(sum(1 for s in scores if s > t), 2)  # the lowest such threshold: exactly the top two stay above
        self.assertEqual(R.tune_threshold([0.0, 0.0]), 0.0)
        self.assertEqual(R.tune_threshold([]), 0.0)

    def test_two_fold_false_rate_is_measured_on_the_half_the_threshold_was_not_tuned_on(self):
        rows = [{"top1": float(i)} for i in range(20)]
        self.assertGreaterEqual(R.two_fold_false_rate(rows), 0.0)
        flat = [{"top1": 5.0} for _ in range(20)]
        self.assertEqual(R.two_fold_false_rate(flat), 0.0)    # the threshold lands at 5.0 and nothing scores above it

    def test_bootstrap_is_deterministic_and_signed(self):
        a = [0] * 50 + [1] * 50
        b = [1] * 60 + [0] * 40
        d1 = R.bootstrap_diff(a, b, resamples=500)
        d2 = R.bootstrap_diff(a, b, resamples=500)
        self.assertEqual(d1, d2)
        self.assertAlmostEqual(d1[0], 0.10, places=9)
        self.assertLess(d1[1], d1[0])
        self.assertGreater(d1[2], d1[0])
        same = R.bootstrap_diff(a, a, resamples=200)
        self.assertEqual(same[0], 0.0)


def fake_summary(h3, groups=None, extra=1.0, vec=None):
    g = {name: {"n": 10, "hit@3": 0.9, "hit@1": 0.8, "hit@3_thr": 0.9} for name in R.GUARD_GROUPS}
    g.update(groups or {})
    return {"hit@3": h3, "extra_tokens_pos": extra, "groups": g, "vectors": {"hit@3": vec if vec is not None else [1] * round(h3 * 100) + [0] * (100 - round(h3 * 100))},
            "hit@3_realistic_thr": h3, "false_injection_rate_2fold": 0.05}


class Decisions(unittest.TestCase):
    LAT = {"tri": {"p95_ms": 3.0}, "both": {"p95_ms": 4.0}, "u61": {"p95_ms": 1.0}}

    def test_unmeasured_latency_blocks_a_win(self):
        s = {"u61": fake_summary(0.60), "tri": fake_summary(0.75), "both": fake_summary(0.60)}
        s["u61"]["vectors"]["hit@3"] = [1] * 60 + [0] * 40
        s["tri"]["vectors"]["hit@3"] = [1] * 75 + [0] * 25
        s["both"]["vectors"]["hit@3"] = [1] * 60 + [0] * 40
        verdict, detail = R.decide_trigram(s, {})
        self.assertEqual(verdict, "u61")
        self.assertFalse(detail["tri"]["conditions"]["4_p95_le_20ms_at_1000"])

    def summaries(self, base=0.70, tri=0.80, both=0.82, **kw):
        return {"u61": fake_summary(base), "tri": fake_summary(tri, **kw.get("tri", {})), "both": fake_summary(both, **kw.get("both", {}))}

    def test_a_clear_gain_wins_and_the_simpler_arm_is_taken_when_both_is_not_much_better(self):
        # u61 0.70; tri 0.80 (+10 pp); both 0.81: tri is within 2 pp of both, so tri is chosen
        s = {"u61": fake_summary(0.70, vec=[1] * 70 + [0] * 30), "tri": fake_summary(0.80, vec=[1] * 80 + [0] * 20), "both": fake_summary(0.81, vec=[1] * 81 + [0] * 19)}
        s["tri"]["vectors"]["hit@3"] = [1] * 70 + [1] * 10 + [0] * 20           # a paired improvement on the same queries
        s["both"]["vectors"]["hit@3"] = [1] * 70 + [1] * 11 + [0] * 19
        s["u61"]["vectors"]["hit@3"] = [1] * 70 + [0] * 30
        verdict, detail = R.decide_trigram(s, self.LAT)
        self.assertTrue(detail["tri"]["wins"])
        self.assertEqual(verdict, "tri")

    def test_both_is_chosen_when_it_beats_tri_by_two_points(self):
        s = {"u61": fake_summary(0.60), "tri": fake_summary(0.70), "both": fake_summary(0.75)}
        s["u61"]["vectors"]["hit@3"] = [1] * 60 + [0] * 40
        s["tri"]["vectors"]["hit@3"] = [1] * 70 + [0] * 30
        s["both"]["vectors"]["hit@3"] = [1] * 75 + [0] * 25
        verdict, _ = R.decide_trigram(s, self.LAT)
        self.assertEqual(verdict, "both")

    def test_a_small_gain_does_not_win(self):
        s = {"u61": fake_summary(0.70), "tri": fake_summary(0.73), "both": fake_summary(0.74)}
        s["u61"]["vectors"]["hit@3"] = [1] * 70 + [0] * 30
        s["tri"]["vectors"]["hit@3"] = [1] * 73 + [0] * 27
        s["both"]["vectors"]["hit@3"] = [1] * 74 + [0] * 26
        self.assertEqual(R.decide_trigram(s, self.LAT)[0], "u61")

    def test_each_guard_condition_can_block_a_win(self):
        def build(**over):
            s = {"u61": fake_summary(0.60), "tri": fake_summary(0.75, **over.get("tri", {})), "both": fake_summary(0.60)}
            s["u61"]["vectors"]["hit@3"] = [1] * 60 + [0] * 40
            s["tri"]["vectors"]["hit@3"] = [1] * 75 + [0] * 25
            s["both"]["vectors"]["hit@3"] = [1] * 60 + [0] * 40
            return s
        ok, d = R.decide_trigram(build(), self.LAT)
        self.assertEqual(ok, "tri")
        loses_a_group = build(tri={"groups": {"plain_en": {"n": 10, "hit@3": 0.85, "hit@1": 0.8, "hit@3_thr": 0.85}}})   # -5 pp against 0.9
        self.assertEqual(R.decide_trigram(loses_a_group, self.LAT)[0], "u61")
        more_tokens = build(tri={"extra": 50.0})
        self.assertEqual(R.decide_trigram(more_tokens, self.LAT)[0], "u61")
        slow = dict(self.LAT, tri={"p95_ms": 25.0})
        self.assertEqual(R.decide_trigram(build(), slow)[0], "u61")

    def test_auto_help_verdicts(self):
        summ = {"u61": fake_summary(0.9)}
        idx_overflow = {"100": {"fits_budget": False}}
        idx_fits = {"100": {"fits_budget": True}}
        self.assertEqual(R.decide_inject(summ, "u61", idx_overflow)[0], "B_default_candidate")
        self.assertEqual(R.decide_inject(summ, "u61", idx_fits)[0], "B_optional")
        weak = {"u61": fake_summary(0.7)}
        self.assertEqual(R.decide_inject(weak, "u61", idx_overflow)[0], "A_only")
        noisy = {"u61": dict(fake_summary(0.9), false_injection_rate_2fold=0.25)}
        self.assertEqual(R.decide_inject(noisy, "u61", idx_overflow)[0], "A_only")

    def test_embeddings_verdict(self):
        good = {"u61": fake_summary(0.9, groups={"paraphrase": {"n": 30, "hit@3": 0.2, "hit@1": 0.1, "hit@3_thr": 0.2}})}
        self.assertEqual(R.decide_embeddings(good, "u61")[0], "not_needed_now")
        bad = {"u61": fake_summary(0.9, groups={"typo": {"n": 30, "hit@3": 0.6, "hit@1": 0.5, "hit@3_thr": 0.6}})}
        self.assertEqual(R.decide_embeddings(bad, "u61")[0], "needed_later")


class QueryMaker(unittest.TestCase):
    def test_identifiers_flags_and_paths_get_no_typos(self):
        text = "что делает `--load-mode` в tests/run_suite.py и переменная SHURA_TEST_HANG_SECONDS и memoryEnabled"
        words = [m.group(0) for m in MQ.eligible(text)]
        self.assertNotIn("load", words)
        self.assertNotIn("suite", words)
        self.assertNotIn("SHURA", words)
        self.assertIn("переменная", words)

    def test_a_slip_changes_the_word_but_not_the_first_letter(self):
        import random
        rng = random.Random(1)
        for word in ("коммитов", "уведомления", "profile", "documentation"):
            for _ in range(20):
                new = MQ.slip(word, rng)
                self.assertNotEqual(new, word)
                self.assertEqual(new[0], word[0])


if __name__ == "__main__":
    unittest.main()
