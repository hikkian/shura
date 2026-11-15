"""RSI1: the general-v1 tasks in the improvement cycle: the sets, the judge hand-off, the screen of an edit, the checkpoint rule and the
final test, all with fake runners (no model, no judge call).
    cd bench/agentic-eval && python3 -B -m unittest rsi.test_general_pilot
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rsi import accept_general, driver, executor, general, pilot  # noqa: E402
from rsi.test_pilot import FakeRunner, PilotCase, task  # noqa: E402


def gtask(tid, family, category="questions", judged=False):
    t = task(tid, family, kind="general-01")
    t.update({"set": "general-v1", "category": category, "judge": judged})
    return t


class GeneralRunner(FakeRunner):
    """Code tasks as in FakeRunner; general tasks score `general_good` under the marked harness and `general_bad` under the plain one."""

    def __init__(self, general_marked=1.0, general_plain=1.0):
        super().__init__()
        self.general_marked, self.general_plain = general_marked, general_plain

    def run(self, task, root, model, arm, rep, out):
        rec = super().run(task, root, model, arm, rep, out)
        if task.get("set") == "general-v1":
            marked = "FAKE-MARK-7" in (Path(root) / model / "harness.md").read_text(encoding="utf-8")
            score = self.general_marked if marked else self.general_plain
            rec.update(score=score, resolved=score >= 0.75)
        return rec


class Sets(unittest.TestCase):
    def test_the_fast_set_is_the_pre_registered_one_task_per_category(self):
        sel = general.select_general()
        pre = json.loads(general.FAST_FILE.read_text(encoding="utf-8"))
        self.assertEqual([t["id"] for t in sel["fast"]], sorted(pre["tasks"]))
        self.assertEqual(len(sel["fast"]), pre["expected_tasks"])
        self.assertEqual(len({t["category"] for t in sel["fast"]}), len(sel["fast"]))          # one per category
        self.assertEqual(general.fast_families(), pre["families"])
        dev_ids = {t["id"] for t in sel["dev"]}
        self.assertTrue({t["id"] for t in sel["fast"]} <= dev_ids)

    def test_the_parts_do_not_overlap_and_the_web_tasks_are_left_out(self):
        sel = general.select_general()
        self.assertFalse({t["id"] for t in sel["dev"]} & {t["id"] for t in sel["test"]})
        self.assertFalse(any(t.get("needs") for part in sel.values() for t in part))
        fam = json.loads(general.SPLITS.read_text(encoding="utf-8"))["families"]
        self.assertTrue(all(fam[t["family"]] == "dev" for t in sel["dev"]))
        self.assertTrue(all(fam[t["family"]] == "test" for t in sel["test"]))

    def test_the_ask_tasks_of_simuser1_are_in_the_general_dev_or_test_part(self):
        ids = {t["id"] for part in general.select_general().values() for t in part}
        self.assertTrue(any(i.startswith("ga_") for i in ids))


class JudgeHandOff(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gen-judge-"))
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.rec = {"task": "gw_x-general-01", "arm": "a", "rep": 1, "score": 0.0, "resolved": False, "test_tail": "judge_pending"}
        self.task = {"dir": self.tmp / "task"}

    def answer(self, pre_ok=True):
        tag = "gw_x-general-01-a-1-abc123"
        (self.tmp / f"{tag}.md").write_text("answer", encoding="utf-8")
        (self.tmp / f"{tag}.json").write_text(json.dumps({"task_id": "gw_x-general-01", "pre": [["lang", pre_ok]]}), encoding="utf-8")

    def test_a_judged_answer_gives_the_record_its_score(self):
        self.answer()
        out = general.finish_record(self.rec, self.task, self.tmp, judge=lambda d, text, image=None: {"score": 0.875, "criteria": [2, 1]})
        self.assertEqual((out["score"], out["resolved"], out["judged"]), (0.875, True, True))

    def test_a_failed_precheck_halves_the_score(self):
        self.answer(pre_ok=False)
        out = general.finish_record(self.rec, self.task, self.tmp, judge=lambda d, text, image=None: {"score": 0.875})
        self.assertEqual((out["score"], out["resolved"]), (0.4375, False))

    def test_a_missing_answer_or_a_judge_without_score_is_marked_not_counted_as_the_models_zero(self):
        out = general.finish_record(self.rec, self.task, self.tmp, judge=lambda *a, **k: {"score": 1.0})
        self.assertEqual(out["judge_failed"], "no answer handed off")
        self.answer()
        out2 = general.finish_record(self.rec, self.task, self.tmp, judge=lambda *a, **k: {"score": None, "error": "x"})
        self.assertTrue(out2["judge_failed"])
        out3 = general.finish_record(self.rec, self.task, self.tmp, judge=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cli down")))
        self.assertIn("cli down", out3["judge_failed"])
        self.assertEqual(general.usable([out, out2, {"score": 1.0}]), [{"score": 1.0}])

    def test_a_run_that_needs_no_judge_is_returned_as_it_is(self):
        rec = {"task": "t", "score": 1.0}
        self.assertIs(general.finish_record(rec, self.task, self.tmp), rec)


class FastGate(unittest.TestCase):
    def rows(self, cand, base):
        mk = lambda s, arm: [{"task": f"t{i}", "rep": r, "family": f"f{i}", "score": s, "arm": arm} for i in range(4) for r in (1, 2)]
        return mk(cand, "c"), mk(base, "b")

    def test_a_gross_drop_refuses_and_a_neutral_edit_passes(self):
        self.assertFalse(general.fast_gate(*self.rows(0.5, 0.8))["ok"])
        self.assertTrue(general.fast_gate(*self.rows(0.8, 0.8))["ok"])
        self.assertTrue(general.fast_gate(*self.rows(0.78, 0.8))["ok"])           # -0.02 is inside the noise of 0.05

    def test_no_pairs_is_not_a_pass(self):
        self.assertFalse(general.fast_gate([], [])["ok"])


class GeneralInTheCycle(PilotCase):
    def general_tasks(self, fast=True):
        dev = [gtask("gq_a-general-01", "gq_a"), gtask("gr_b-general-01", "gr_b", "reasoning")]
        return {"fast": dev[:1] if fast else [], "dev": dev, "test": [gtask("gt_c-general-01", "gt_c", "review_text")]}

    def make_general(self, runner, **kw):
        return self.make(runner, general=self.general_tasks(kw.pop("fast", True)), **kw)

    def test_an_edit_that_does_not_hurt_the_general_tasks_is_screened_versioned_and_gated_at_the_checkpoint(self):
        p = self.make_general(GeneralRunner(1.0, 1.0))
        self.assertEqual(p.run_cycle(), "done")
        steps = self.steps()
        self.assertTrue(steps["t00-js_a-bugfix.general"]["value"]["ok"])
        self.assertTrue(steps["t00-js_a-bugfix.version"]["value"]["applied"])
        cp = steps["cp.001"]["value"]
        self.assertTrue(cp["accepted"])
        self.assertTrue(cp["general"]["accept"] and cp["general"]["guard"]["ok"])
        self.assertEqual(cp["general"]["guard"]["pairs"], 4)                       # 2 general dev tasks x 2 repeats
        self.assertIn("general dev:", (self.library / "EXPERIENCE.md").read_text(encoding="utf-8"))
        self.assertTrue(any(k.startswith("gen.fast.") for k in steps) and any(k.startswith("gen.dev.") for k in steps))

    def test_an_edit_that_drops_the_general_score_is_refused_at_the_screen_before_it_becomes_a_version(self):
        p = self.make_general(GeneralRunner(general_marked=0.4, general_plain=0.9))
        p.run_cycle()
        steps = self.steps()
        self.assertFalse(steps["t00-js_a-bugfix.general"]["value"]["ok"])
        self.assertNotIn("t00-js_a-bugfix.version", steps)
        self.assertNotIn("cp.001", steps)
        self.assertEqual(p.h_file.read_text(encoding="utf-8"), "Base rule.\n")       # the working harness is unchanged
        self.assertFalse(any(k.startswith("gen.dev.") for k in steps), "the whole general dev part is run only for a checkpoint")
        self.assertIn("KEEP_BASE", (self.library / "EXPERIENCE.md").read_text(encoding="utf-8"))
        self.assertIn("general regression", (self.library / "EXPERIENCE.md").read_text(encoding="utf-8"))

    def test_a_checkpoint_the_code_dev_accepts_is_refused_when_the_general_dev_falls_beyond_noise(self):
        p = self.make_general(GeneralRunner(general_marked=0.4, general_plain=0.9), fast=False)   # no screen: the edit reaches the checkpoint
        p.run_cycle()
        cp = self.steps()["cp.001"]["value"]
        self.assertFalse(cp["accepted"])
        self.assertTrue(cp["verdict"]["code_accept"])
        self.assertFalse(cp["general"]["guard"]["ok"])
        self.assertIn("(general tasks)", cp["outcome"])
        self.assertEqual((p.acc_dir / "harness.md").read_text(encoding="utf-8"), "Base rule.\n")

    def test_the_final_test_also_reports_the_general_test_part(self):
        self.test = [task("js_t-bugfix", "js_t")]
        p = self.make_general(GeneralRunner(1.0, 1.0))
        p.run_cycle()
        fin = self.steps()["final.test"]["value"]
        self.assertEqual(fin["general_tasks"], ["gt_c-general-01"])
        self.assertEqual(fin["general"]["guard"]["pairs"], 2)

    def test_without_general_the_cycle_is_what_it_was(self):
        p = self.make(GeneralRunner())
        p.run_cycle()
        steps = self.steps()
        self.assertFalse(any(k.startswith("gen.") or k.endswith(".general") for k in steps))
        self.assertIsNone(steps["cp.001"]["value"].get("general"))

    def test_the_morning_report_lists_the_general_gates(self):
        p = self.make_general(GeneralRunner(1.0, 1.0))
        p.run_cycle()
        text = pilot.morning("p1")
        self.assertIn("General-задачи", text)
        self.assertIn("весь general dev", text)
        self.assertIn("STUB", text)


class GeneralTasksInTheRealRunner(unittest.TestCase):
    def test_run_task_wires_the_question_policy_and_the_request_log_like_run_one(self):
        src = (HERE / "executor.py").read_text(encoding="utf-8")
        for needle in ("harness.question_policy(task, out_dir)", "policy.stats()", "agent_tools_binds()", "requests.jsonl", "task=task"):
            self.assertIn(needle, src)


if __name__ == "__main__":
    unittest.main()
