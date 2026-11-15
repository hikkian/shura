"""LL4b: the gate of the personal layer (rsi/lifelong.py) on a fake runner, offline. Run: cd bench/agentic-eval && python3 -B -m unittest rsi.test_lifelong
A queued rule is measured against the current layer on the control tasks with room; the pre-registered rule decides (decide2 + safety +
cost); a gate_run row, the rule's status and notes, a journal line, EXPERIENCE.md and a notice come out; a rejected rule stays out; the
person's enable in ShuraCode is not touched here; ablation records "no benefit"; a crash resumes without repeating runs; a run whose
harness did not reach the model stops the night; the base layer text is unchanged by the candidate."""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rsi import driver, lifelong, pilot  # noqa: E402

NOTE = lambda key, model="tiel", **kw: json.dumps({"key": key, "model": model, "when": "w", "do": "d", "why": "y", "cost_words": 3, **kw})  # noqa: E731


class GateCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ll4b-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.saved = (driver.RUNS, pilot.HARNESS_CFG)
        self.addCleanup(self._restore)
        driver.RUNS = self.tmp / "runs"
        cfg = self.tmp / "cfg" / "tiel"
        cfg.mkdir(parents=True)
        (cfg / "harness.md").write_text("BASE HARNESS of the product.\n", encoding="utf-8")
        pilot.HARNESS_CFG = self.tmp / "cfg"
        self.store = lifelong.RuleStore(self.tmp / "lifelong.sqlite")
        self.addCleanup(self.store.close)
        self.skills = self.tmp / "skills"
        # 8 control tasks: 6 with room (three families of two), 2 solved before (safety)
        for i in range(8):
            fam = "fam_a" if i < 2 else ("fam_b" if i < 4 else ("fam_c" if i < 6 else "fam_safe"))
            self.store.db.execute("INSERT INTO control_task(family, created_at, source, prompt, repo, snapshot_path, check_kind, check_ref, timeout_s, split, solved_before, base_mean) "
                                  "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (fam, "2026-11-08", "test", f"task {i}", "repo", "/nonexistent", "test_cmd", "pytest", 600, "work",
                                                                    1 if i >= 6 else 0, 1.0 if i >= 6 else 0.75))
        self.store.db.execute("INSERT INTO control_task(family, prompt, split) VALUES ('fam_h', 'holdout task', 'holdout')")

    def _restore(self):
        driver.RUNS, pilot.HARNESS_CFG = self.saved

    def rule(self, key, status="queued", text=None, **notes):
        text = text or f"When w: do FAKE-RULE {key}."
        cur = self.store.db.execute("INSERT INTO rule(text, kind, status, born_from, created_at, notes) VALUES (?,?,?,?,?,?)",
                                    (text, "rule", status, "user_command", "2026-11-08T00:00:00+00:00", NOTE(key, **notes)))
        return cur.lastrowid

    def make(self, runner, name="g1", **kw):
        return lifelong.LifelongGate(name, "tiel", runner, store=self.store, skills_dir=self.skills, repeats=2, fake=True, **kw)

    def row(self, key):
        for r in self.store.db.execute("SELECT * FROM rule"):
            n = json.loads(r["notes"])
            if n["key"] == key:
                return dict(r), n
        return None, None


class Night(GateCase):
    def test_a_helpful_rule_passes_and_everything_is_recorded(self):
        self.rule("tests-first")
        runner = lifelong.FakeRuleRunner()
        g = self.make(runner)
        status = g.run_cycle()
        self.assertEqual(status, "done: candidate 'tests-first' passed", status)
        r, n = self.row("tests-first")
        self.assertEqual(r["status"], "passed")
        self.assertEqual(n["gate"]["decision"], "passed")
        self.assertGreater(n["gate"]["delta"], 0.2)
        runs = self.store.db.execute("SELECT * FROM gate_run").fetchall()
        self.assertEqual(len(runs), 1)
        self.assertEqual((runs[0]["decision"], runs[0]["n_tasks"], runs[0]["repeats"], runs[0]["model"]), ("passed", 6, 2, "tiel"))
        self.assertAlmostEqual(runs[0]["base_mean"], 0.75)
        self.assertAlmostEqual(runs[0]["cand_mean"], 1.0)
        events = [j[0] for j in self.store.db.execute("SELECT event FROM journal")]
        self.assertIn("rule_passed", events)
        exp = (self.skills / "tiel" / "EXPERIENCE.md").read_text()
        self.assertIn("gate run 1: rule 'tests-first' passed", exp)
        steps = driver.load_state("g1")["steps"]
        self.assertIn("Включить?", steps["record"]["value"]["notice"])
        self.assertEqual(steps["sets"]["value"]["growth"], [f"ct{i}" for i in range(1, 7)])
        self.assertEqual(steps["sets"]["value"]["safety"], ["ct7", "ct8"])
        arms = {a.split("-")[0] for _, a, _, _ in runner.calls}
        self.assertEqual(arms, {"base", "cand"})
        self.assertFalse(any(t == "ct9" for t, _, _, _ in runner.calls), "the holdout task is never run by the gate")
        # the candidate harness = base + layer with the rule; the base harness text itself is unchanged
        base_text = (driver.RUNS / "g1" / "trials" / "base" / "tiel" / "harness.md").read_text()
        cand_text = (driver.RUNS / "g1" / "trials" / "cand" / "tiel" / "harness.md").read_text()
        self.assertEqual(base_text, "BASE HARNESS of the product.\n")
        self.assertTrue(cand_text.startswith("BASE HARNESS of the product.\n\nPersonal rules"))
        self.assertIn("FAKE-RULE tests-first", cand_text)
        self.assertEqual((pilot.HARNESS_CFG / "tiel" / "harness.md").read_text(), "BASE HARNESS of the product.\n", "the product's base layer is never written")

    def test_a_rule_that_does_not_help_is_rejected_with_the_numbers(self):
        self.rule("noise")
        g = self.make(lifelong.FakeRuleRunner(help_score=0.75), name="g2")
        self.assertEqual(g.run_cycle(), "done: candidate 'noise' rejected")
        r, n = self.row("noise")
        self.assertEqual(r["status"], "rejected")
        self.assertIn("score bound not above zero", n["gate"].get("reason", "") or driver.load_state("g2")["steps"]["verdict"]["value"]["reason"])
        self.assertEqual(self.store.db.execute("SELECT decision FROM gate_run").fetchone()[0], "rejected")
        self.assertIsNone(driver.load_state("g2")["steps"]["record"]["value"]["notice"])

    def test_a_cost_blowup_is_decisive_for_the_version(self):
        self.rule("slow")
        runner = lifelong.FakeRuleRunner()
        orig = runner.run

        def run(task, root, model, arm, rep, out):
            r = orig(task, root, model, arm, rep, out)
            return dict(r, seconds=30.0 if arm.startswith("cand") else 10.0)
        runner.run = run
        self.assertEqual(self.make(runner, name="g3").run_cycle(), "done: candidate 'slow' rejected")
        self.assertIn("cost over the limit", driver.load_state("g3")["steps"]["verdict"]["value"]["reason"])

    def test_a_safety_drop_refuses_a_gain(self):
        self.rule("breaks-safe")
        runner = lifelong.FakeRuleRunner()
        orig = runner.run

        def run(task, root, model, arm, rep, out):
            r = orig(task, root, model, arm, rep, out)
            if arm.startswith("cand") and task.get("solved_before") == 1:
                return dict(r, score=0.5, resolved=False)
            return r
        runner.run = run
        self.assertEqual(self.make(runner, name="g4").run_cycle(), "done: candidate 'breaks-safe' rejected")
        v = driver.load_state("g4")["steps"]["verdict"]["value"]
        self.assertFalse(v["safety"]["ok"])
        self.assertEqual(sorted(v["safety"]["broke"]), ["ct7", "ct8"])

    def test_the_current_layer_is_the_base_and_a_merged_rule_replaces_its_sources_in_the_candidate(self):
        self.rule("old-a", status="enabled", text="When a: do A.")
        self.rule("old-b", status="enabled", text="When b: do B.")
        self.rule("ab", text="When ab: do FAKE-RULE AB.", merged_from=["old-a", "old-b"])
        g = self.make(lifelong.FakeRuleRunner(), name="g5")
        self.assertEqual(g.run_cycle(), "done: candidate 'ab' passed")
        base_text = (driver.RUNS / "g5" / "trials" / "base" / "tiel" / "harness.md").read_text()
        cand_text = (driver.RUNS / "g5" / "trials" / "cand" / "tiel" / "harness.md").read_text()
        self.assertIn("do A.", base_text)
        self.assertIn("do B.", base_text)
        self.assertNotIn("do A.", cand_text)
        self.assertIn("do FAKE-RULE AB.", cand_text)
        for key in ("old-a", "old-b"):
            self.assertEqual(self.row(key)[0]["status"], "enabled", "the gate retires nothing; enabling in ShuraCode does")

    def test_ablation_records_no_benefit_and_retires_nothing(self):
        self.rule("useless", status="enabled", text="When u: do nothing special.")
        self.rule("useful", status="enabled", text="When w: do FAKE-RULE.")
        g = self.make(lifelong.FakeRuleRunner(), name="g6", ablate="useless")
        self.assertEqual(g.run_cycle(), "done: ablation 'useless' no_benefit")
        r, n = self.row("useless")
        self.assertEqual(r["status"], "enabled")
        self.assertEqual(n["ablation"]["decision"], "no_benefit")
        g2 = self.make(lifelong.FakeRuleRunner(), name="g7", ablate="useful")
        self.assertEqual(g2.run_cycle(), "done: ablation 'useful' has_benefit")
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM gate_run").fetchone()[0], 2)

    def test_a_crash_resumes_without_repeating_runs_and_a_stop_releases_the_candidate(self):
        self.rule("resume")
        runner = lifelong.FakeRuleRunner()
        with self.assertRaises(RuntimeError):
            self.make(runner, name="g8", crash_after="verdict").run_cycle()
        self.assertEqual(self.row("resume")[0]["status"], "testing")
        n = len(runner.calls)
        self.assertEqual(self.make(runner, name="g8").run_cycle(), "done: candidate 'resume' passed")
        self.assertEqual(len(runner.calls), n, "every run came from the saved state")
        # a night that stops early puts the candidate back into the queue
        self.rule("later")
        self.store.db.execute("DELETE FROM control_task")
        status = self.make(lifelong.FakeRuleRunner(), name="g9").run_cycle()
        self.assertTrue(status.startswith("stopped: no control tasks"), status)
        self.assertEqual(self.row("later")[0]["status"], "queued")

    def test_a_run_without_delivery_stops_the_night(self):
        self.rule("undelivered")
        runner = lifelong.FakeRuleRunner()
        orig = runner.run

        def run(task, root, model, arm, rep, out):
            return dict(orig(task, root, model, arm, rep, out), harness_delivery={"checked": True, "delivered": False})
        runner.run = run
        status = self.make(runner, name="g10").run_cycle()
        self.assertTrue(status.startswith("stopped: harness not delivered"), status)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM gate_run").fetchone()[0], 0)
        self.assertEqual(self.row("undelivered")[0]["status"], "queued")

    def test_too_few_tasks_with_room_means_no_measurement(self):
        self.store.db.execute("UPDATE control_task SET base_mean=1.0 WHERE id <= 4")
        self.rule("x")
        status = self.make(lifelong.FakeRuleRunner(), name="g11").run_cycle()
        self.assertTrue(status.startswith("stopped: only 2 control tasks with room"), status)

    def test_an_unknown_base_mean_is_screened_once(self):
        self.store.db.execute("UPDATE control_task SET base_mean=NULL")
        self.rule("x")
        runner = lifelong.FakeRuleRunner()
        self.assertEqual(self.make(runner, name="g12").run_cycle(), "done: candidate 'x' passed")
        screens = [c for c in runner.calls if c[1].startswith("screen")]
        self.assertEqual(len(screens), 8, "one screen run per task without a known base mean")

    def test_prior_base_runs_are_reused_after_the_drift_check(self):
        self.rule("first")
        r1 = lifelong.FakeRuleRunner()
        self.assertEqual(self.make(r1, name="n1").run_cycle(), "done: candidate 'first' passed")
        base_runs_1 = sum(1 for c in r1.calls if c[1].startswith("base"))
        self.assertEqual(base_runs_1, 16)
        self.rule("second")
        r2 = lifelong.FakeRuleRunner()
        self.assertEqual(self.make(r2, name="n2", prior_runs=["n1"]).run_cycle(), "done: candidate 'second' passed")
        base_runs_2 = sum(1 for c in r2.calls if c[1].startswith("base"))
        self.assertEqual(base_runs_2, 4, "only the drift check's fresh runs; the rest of the base comes from the earlier night")
        self.assertTrue(driver.load_state("n2")["steps"]["drift"]["value"]["ok"])


class ReviewFindings(GateCase):
    def test_the_safety_base_is_completed_when_an_earlier_night_had_none(self):
        self.store.db.execute("UPDATE control_task SET solved_before=0 WHERE id IN (7, 8)")
        self.rule("m1")
        self.assertEqual(self.make(lifelong.FakeRuleRunner(), name="m1").run_cycle(), "done: candidate 'm1' passed")
        self.store.db.execute("UPDATE control_task SET solved_before=1, base_mean=1.0 WHERE id IN (7, 8)")
        self.rule("m2")
        runner = lifelong.FakeRuleRunner()
        orig = runner.run

        def run(task, root, model, arm, rep, out):
            r = orig(task, root, model, arm, rep, out)
            return dict(r, score=0.0, resolved=False) if arm.startswith("cand") and task.get("solved_before") == 1 else r
        runner.run = run
        status = self.make(runner, name="m2", prior_runs=["m1"]).run_cycle()
        self.assertEqual(status, "done: candidate 'm2' rejected", status)
        v = driver.load_state("m2")["steps"]["verdict"]["value"]
        self.assertFalse(v["safety"]["ok"])
        self.assertEqual(v["safety"]["pairs"], 4, "two safety tasks x two repeats, run fresh because the earlier night had none")

    def test_the_persons_decision_during_the_night_wins_over_the_gate(self):
        rid = self.rule("w1")
        runner = lifelong.FakeRuleRunner()
        with self.assertRaises(RuntimeError):
            self.make(runner, name="w1", crash_after="verdict").run_cycle()
        # meanwhile the person withdrew it (mcp/rules.py would do exactly this)
        notes = json.loads(self.store.db.execute("SELECT notes FROM rule WHERE id=?", (rid,)).fetchone()[0])
        notes["declined"] = True
        self.store.db.execute("UPDATE rule SET status='rejected', notes=? WHERE id=?", (json.dumps(notes), rid))
        status = self.make(runner, name="w1").run_cycle()
        self.assertTrue(status.startswith("stopped: rule 'w1' is no longer queued"), status)
        self.assertEqual(self.row("w1")[0]["status"], "rejected")
        # and a change between the verdict and the record is noticed too
        rid2 = self.rule("w2")
        gate = self.make(lifelong.FakeRuleRunner(), name="w2")
        real_finish = self.store.finish

        def finish(rule_id, decision, g, fields, actor="code"):
            self.store.db.execute("UPDATE rule SET status='enabled' WHERE id=?", (rule_id,))
            return real_finish(rule_id, decision, g, fields, actor)
        self.store.finish = finish
        gate.run_cycle()
        self.assertEqual(self.row("w2")[0]["status"], "enabled", "the gate never overwrites what the person did")
        self.assertIn("rule_gate_ignored", [j[0] for j in self.store.db.execute("SELECT event FROM journal")])
        self.assertIsNone(driver.load_state("w2")["steps"]["record"]["value"]["notice"])

    def test_a_candidate_that_could_not_be_enabled_is_skipped_not_measured(self):
        for i in range(lifelong.MAX_ENABLED):
            self.rule(f"e{i}", status="enabled", text=f"When e{i}: do e{i}.")
        self.rule("one-more")
        self.rule("ab", merged_from=["e0", "e1"])
        runner = lifelong.FakeRuleRunner()
        status = self.make(runner, name="s1").run_cycle()
        self.assertEqual(status, "done: candidate 'ab' passed", status)
        _, n = self.row("one-more")
        self.assertIn("limit 12", n["gate_skipped"]["reason"])
        self.assertEqual(self.row("one-more")[0]["status"], "queued")
        self.store.db.execute("UPDATE rule SET status='retired' WHERE status='enabled'")   # room again by count, not by tokens
        self.rule("too-long", text="When " + "x " * 1000 + ": do y.")
        status = self.make(lifelong.FakeRuleRunner(), name="s2", candidate="too-long").run_cycle()
        self.assertTrue(status.startswith("stopped: no measurable queued rule"), status)
        self.assertIn("tokens", self.row("too-long")[1]["gate_skipped"]["reason"])

    def test_gate_run_means_come_from_the_paired_records_even_with_a_reused_base(self):
        self.rule("first")
        runner = lifelong.FakeRuleRunner(base_score=0.5)
        self.assertEqual(self.make(runner, name="r1").run_cycle(), "done: candidate 'first' passed")
        self.rule("second")
        r2 = lifelong.FakeRuleRunner(base_score=0.5)
        self.assertEqual(self.make(r2, name="r2", prior_runs=["r1"]).run_cycle(), "done: candidate 'second' passed")
        runs = self.store.db.execute("SELECT base_mean, cand_mean, n_tasks, reason FROM gate_run ORDER BY id").fetchall()
        self.assertAlmostEqual(runs[1]["base_mean"], 0.5)
        self.assertAlmostEqual(runs[1]["cand_mean"], 1.0)
        self.assertIn("rule second", runs[1]["reason"])

    def test_too_few_families_means_no_measurement(self):
        self.store.db.execute("UPDATE control_task SET family='one' WHERE split='work' AND solved_before=0")
        self.rule("x")
        status = self.make(lifelong.FakeRuleRunner(), name="f1").run_cycle()
        self.assertTrue(status.startswith("stopped: the tasks with room come from 1 families"), status)

    def test_the_database_the_gate_creates_is_private_and_kind_skill_rows_are_not_candidates(self):
        self.assertEqual(oct(os.stat(self.tmp / "lifelong.sqlite").st_mode & 0o777), "0o600")
        self.store.db.execute("INSERT INTO rule(text, kind, status, notes) VALUES ('x', 'skill', 'queued', ?)", (NOTE("sk"),))
        self.assertEqual(self.store.rules("tiel", ("queued",)), [])


class ThirdRound(GateCase):
    def test_a_passed_rule_drops_its_skip_mark_and_a_changed_layer_stops_a_resumed_night(self):
        for i in range(lifelong.MAX_ENABLED):
            self.rule(f"e{i}", status="enabled", text=f"When e{i}: do e{i}.")
        rid = self.rule("later")
        self.make(lifelong.FakeRuleRunner(), name="k1").run_cycle()
        self.assertIn("gate_skipped", self.row("later")[1])
        self.store.db.execute("UPDATE rule SET status='retired' WHERE status='enabled'")
        self.assertEqual(self.make(lifelong.FakeRuleRunner(), name="k2").run_cycle(), "done: candidate 'later' passed")
        self.assertNotIn("gate_skipped", self.row("later")[1])
        # resume after a crash with a layer the person changed meanwhile
        self.rule("again")
        runner = lifelong.FakeRuleRunner()
        with self.assertRaises(RuntimeError):
            self.make(runner, name="k3", crash_after="verdict").run_cycle()
        self.rule("new-enabled", status="enabled", text="When n: do n.")
        status = self.make(runner, name="k3").run_cycle()
        self.assertTrue(status.startswith("stopped: the personal layer changed"), status)
        self.assertEqual(self.row("again")[0]["status"], "queued")


class Store(GateCase):
    def test_the_store_shares_the_schema_with_the_gateway_and_the_mcp(self):
        tables = {r[0] for r in self.store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for t in ("rule", "rule_version", "gate_run", "control_task", "journal", "rule_case", "memory"):
            self.assertIn(t, tables)
        self.rule("k")
        self.assertEqual([r["notes"]["key"] for r in self.store.rules("tiel", ("queued",))], ["k"])
        self.assertEqual(self.store.rules("sage", ("queued",)), [])
        tasks = self.store.control_tasks("work")
        self.assertEqual(len(tasks), 8)
        self.assertEqual(tasks[0]["id"], "ct1")

    def test_the_driver_knows_the_step(self):
        import io
        from contextlib import redirect_stdout
        self.rule("cli")
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = driver.main(["--run", "cli1", "lifelong", "--fake", "--db", str(self.tmp / "lifelong.sqlite"), "--skills-dir", str(self.skills)])
        self.assertEqual(code, 0, buf.getvalue())
        self.assertIn("passed", buf.getvalue())
        self.assertIn("Включить?", buf.getvalue())
        with self.assertRaises(SystemExit):
            driver.main(["--run", "cli2", "lifelong", "--db", str(self.tmp / "lifelong.sqlite")])


if __name__ == "__main__":
    unittest.main()
