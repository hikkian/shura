"""SHADOW1 part B: the night layer (bench/agentic-eval/shadow.py). No GPU: synthetic runs in the format run_experiment writes."""
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench" / "agentic-eval"))
sys.path.insert(0, str(ROOT / "gateway"))

import experience_ledger  # noqa: E402
import shadow  # noqa: E402


class FakeSource:
    def __init__(self, root, n=30):
        self.rows = []
        for i in range(n):
            fam, kind = f"fam{i // 2}", ("bugfix" if i % 2 else "implement")
            d = Path(root) / "src" / fam / kind
            d.mkdir(parents=True, exist_ok=True)
            self.rows.append({"id": f"{fam}-{kind}", "family": fam, "kind": kind, "dir": str(d), "source": "open"})

    def tasks(self):
        return list(self.rows)


def fake_night(run_dir, plan_res, arms, effect, rng, drop=None):
    """results.jsonl as run_experiment writes it (harness records) for the planned tasks."""
    run_dir = Path(run_dir)
    with open(run_dir / "results.jsonl", "w") as f:
        for tid in plan_res["tasks"]:
            fam = tid.split("-")[0]
            base = rng.random()
            for arm in arms:
                if drop and (tid, arm) == drop:
                    continue
                score = base if arm == "stack" else min(1.0, max(0.0, base + effect.get(arm, 0.0) + rng.gauss(0, 0.05)))
                f.write(json.dumps({"arm": arm, "task": tid, "family": fam, "kind": tid.split("-")[1], "rep": 0,
                                    "score": score, "resolved": score == 1.0, "seconds": 100.0, "reasoning_tokens": 0}) + "\n")


class NightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="shadow-", dir="/dev/shm" if os.path.isdir("/dev/shm") else None))
        self.store = self.tmp / "store"
        self.src = FakeSource(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def plan(self, names, night=None, n=10):
        out = self.tmp / f"run-{'-'.join(names)}-{night}"
        return out, shadow.plan(names, {"exe": "llama-server", "args": []}, out, n, {"reasoning_brake_shadow": True},
                                night=night, source=self.src, store=self.store, tasks_root=self.tmp / f"root-{night}")

    def test_prereg_rules_and_freeze(self):
        with self.assertRaises(ValueError):
            shadow.init("c", "x", {"a": 1}, looks=4, store=self.store)
        with self.assertRaises(ValueError):
            shadow.init("c", "x", {"a": 1}, life_days=29, store=self.store)
        with self.assertRaises(ValueError):
            shadow.init("c", "x", None, store=self.store)                 # it must say what it changes
        pre = shadow.init("c", "trims answers", {"max_tokens": 48}, store=self.store)
        self.assertEqual((pre["looks"], pre["rows"][0]["delta"], pre["shadow"]["life_days"]), (3, 0.03, 28.0))
        self.assertTrue(pre["shadow"]["whole_tasks_only"])
        (self.store / "c" / "results.jsonl").write_text(json.dumps({"row": "agent-dev"}) + "\n")
        with self.assertRaises(ValueError):
            shadow.init("c", "trims answers", {"max_tokens": 64}, store=self.store)     # rules frozen after the first result
        p = self.store / "c" / "prereg.json"
        d = json.loads(p.read_text())
        d["rows"][0]["delta"] = 0.2
        p.write_text(json.dumps(d))
        with self.assertRaises(ValueError):
            shadow.load_prereg("c", self.store)                       # a moved δ is detected

    def test_plan_two_candidates_share_the_stack(self):
        shadow.init("aa", "nothing", {}, store=self.store)
        shadow.init("harm", "max_tokens 48", {"max_tokens": 48}, store=self.store)
        out, res = self.plan(["aa", "harm"])
        cfg = json.loads(Path(res["arms"]).read_text())
        self.assertEqual([a["name"] for a in cfg["arms"]], ["stack", "aa", "harm"])
        self.assertEqual(cfg["arms"][0]["extra"], {"reasoning_brake_shadow": True})
        self.assertEqual(cfg["arms"][2]["extra"], {"reasoning_brake_shadow": True, "max_tokens": 48})
        self.assertEqual(len(res["tasks"]), 10)
        self.assertEqual(len(list((self.tmp / "root-None").glob("*/*"))), 10)
        self.assertIn("run_experiment.py", res["command"][1])
        self.assertIn("--tasks-root", res["command"])

    def test_ingest_verdict_and_fresh_tasks(self):
        shadow.init("harm", "max_tokens 48", {"max_tokens": 48}, store=self.store)
        shadow.init("aa", "nothing", {}, store=self.store)
        rng = random.Random(3)
        out, res = self.plan(["aa", "harm"], night=1)
        fake_night(out, res, ["stack", "aa", "harm"], {"harm": -0.6}, rng, drop=(res["tasks"][0], "harm"))
        added = shadow.ingest(out, ["aa", "harm"], self.store)
        self.assertEqual(added, {"aa": 20, "harm": 18})               # the unpaired task is left out, not imputed
        self.assertEqual(shadow.ingest(out, ["aa", "harm"], self.store), {"aa": 0, "harm": 0})   # idempotent
        v = shadow.verdict("harm", self.store)
        self.assertEqual((v["overall"], v["status"], v["look"]), ("red", "removed: worse", 1))
        va = shadow.verdict("aa", self.store)
        self.assertEqual(va["status"], "in shadow")
        self.assertTrue((self.store / "harm" / "verdict-look1.json").exists())
        out2, res2 = self.plan(["aa"], night=2)
        self.assertFalse(set(res2["tasks"]) & set(res["tasks"]))      # a re-check uses new tasks
        recs = [json.loads(x) for x in (self.store / "aa" / "results.jsonl").read_text().splitlines()]
        self.assertEqual(set(recs[0]) >= {"row", "arm", "task", "family", "rep", "seed", "score"}, True)

    def test_life_and_looks_end_the_shadow(self):
        shadow.init("aa", "nothing", {}, looks=1, store=self.store)
        out, res = self.plan(["aa"], night=1)
        fake_night(out, res, ["stack", "aa"], {}, random.Random(5))
        shadow.ingest(out, ["aa"], self.store)
        v = shadow.verdict("aa", self.store)
        self.assertIn(v["status"], ("removed: not proven", "proven: next train"))   # the only look is spent
        shadow.init("bb", "nothing", {}, store=self.store)
        out, res = self.plan(["bb"], night=1)
        fake_night(out, res, ["stack", "bb"], {}, random.Random(6))
        shadow.ingest(out, ["bb"], self.store)
        pre = shadow.load_prereg("bb", self.store)
        v = shadow.verdict("bb", self.store, now=pre["shadow"]["expires_ts"] + 1)
        if v["overall"] == "grey":
            self.assertEqual(v["status"], "removed: not proven")
        with self.assertRaises(ValueError):
            import time
            orig = shadow._now
            shadow._now = lambda: pre["shadow"]["expires_ts"] + 1
            try:
                self.plan(["bb"], night=2)
            finally:
                shadow._now = orig

    def test_summary_line(self):
        self.assertEqual(shadow.summary(self.store), "в тени никого нет")
        shadow.init("aa", "nothing", {}, store=self.store)
        shadow.init("harm", "max_tokens 48", {"max_tokens": 48}, store=self.store)
        out, res = self.plan(["aa", "harm"], night=1)
        fake_night(out, res, ["stack", "aa", "harm"], {"harm": -0.6}, random.Random(9))
        shadow.ingest(out, ["aa", "harm"], self.store)
        line = shadow.summary(self.store)
        self.assertIn("на проверке в тени: 10 задач", line)
        self.assertIn("aa — пока ничья", line)
        self.assertIn("harm — хуже стека — снят", line)


class ReviewFixNightTests(NightTests):
    def test_empty_night_spends_no_look_and_missing_score_is_loud(self):
        shadow.init("aa", "nothing", {}, store=self.store)
        out, res = self.plan(["aa"], night=1)
        (out / "results.jsonl").write_text("")
        self.assertEqual(shadow.ingest(out, ["aa"], self.store), {"aa": 0})
        self.assertEqual(shadow._nights("aa", self.store), [])
        with open(out / "results.jsonl", "w") as f:
            for arm in ("stack", "aa"):
                f.write(json.dumps({"arm": arm, "task": res["tasks"][0], "family": "f", "rep": 0}) + "\n")
        with self.assertRaises(ValueError):
            shadow.ingest(out, ["aa"], self.store)

    def test_expired_without_results_is_removed(self):
        shadow.init("cc", "nothing", {}, store=self.store)
        pre = shadow.load_prereg("cc", self.store)
        v = shadow.verdict("cc", self.store, now=pre["shadow"]["expires_ts"] + 1)
        self.assertEqual(v["status"], "removed: not proven")
        self.assertIn("снят как не доказанный", shadow.summary(self.store))

    def test_decided_candidate_takes_no_more_nights(self):
        shadow.init("harm", "max_tokens 48", {"max_tokens": 48}, store=self.store)
        out, res = self.plan(["harm"], night=1)
        fake_night(out, res, ["stack", "harm"], {"harm": -0.6}, random.Random(3))
        shadow.ingest(out, ["harm"], self.store)
        self.assertEqual(shadow.verdict("harm", self.store)["status"], "removed: worse")
        with self.assertRaises(ValueError):
            self.plan(["harm"], night=2)
        with self.assertRaises(ValueError):
            shadow.ingest(out, ["harm"], self.store)
        self.assertEqual(shadow.verdict("harm", self.store)["status"], "removed: worse")


class JournalSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="shadow-led-", dir="/dev/shm" if os.path.isdir("/dev/shm") else None))
        self.led = experience_ledger.Ledger(self.tmp / "exp.sqlite")

    def tearDown(self):
        self.led.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_only_consented_exam_tasks_with_tests_and_snapshot(self):
        self.led.set_consent("capture", True)
        made = {}
        for i in range(40):
            block = f"repo{i}"
            r = self.led.add_task(f"задача {i}", repo_snapshot=f"snap{i}" if i % 3 else None, tests="pytest -q" if i % 4 else None,
                                  repo=block, privacy="personal" if i % 7 == 0 else "work")
            made[r["id"]] = (r["split"], i)
        src = shadow.JournalTaskSource(self.led)
        self.assertEqual(src.tasks(), [])                              # no exam consent: nothing is opened
        self.led.set_consent("exam", True)
        got = src.tasks()
        want = [eid for eid, (split, i) in made.items() if split == "exam" and i % 3 and i % 4 and i % 7]
        self.assertEqual(sorted(t["event_id"] for t in got), sorted(want))
        self.assertTrue(all(t["source"] == "journal" and t["tests"] and t["repo_snapshot"] for t in got))
        self.assertFalse(any(made[t["event_id"]][0] in ("train", "sealed_exam") for t in got))
        picked = shadow.pick_tasks(src, 3, 1)
        self.assertLessEqual(len(picked), 3)


class SimulationTests(unittest.TestCase):
    def test_fast_statistics_agree_with_gate(self):
        res = shadow.check_fast_gate(cases=6, pairs=30, seed=11)
        self.assertGreaterEqual(res["agree"], 5)

    def test_harmful_goes_red_and_aa_stays_quiet(self):
        harm = shadow.simulate(-0.15, reps=30, bootstrap_reps=2000)
        self.assertGreaterEqual(harm["share"]["red"], 0.9)
        aa = shadow.simulate(0.0, reps=30, bootstrap_reps=2000)
        self.assertLessEqual(aa["share"]["red"], 0.1)


if __name__ == "__main__":
    unittest.main()
