"""RSI2D tests (offline: no engine, no server, no model). Run: cd bench/agentic-eval && nice -n 19 python3 -B -m unittest rsi.test_rsi2
Cause tagging from logs and test tails, the growth and safety sets, the pre-registered rule with the Bonferroni bound, the simulation, and
the cycle2 night on a fake runner (the stub editors): one cause -> one candidate -> measured on the growth set -> adopted; nothing adopted
when the candidate does not help; resume after a crash; the old pilot untouched."""
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rsi import accept, driver, editors, executor, pilot, rsi2  # noqa: E402


def task(tid, family, kind="implement"):
    return {"id": tid, "family": family, "kind": kind, "lang": "js", "dir": "/nonexistent", "set": "v3"}


def rec(tid, family, rep, score, tail="", timed_out=False, seconds=100.0, tokens=1000, resolved=None):
    return {"arm": "b", "task": tid, "family": family, "kind": "implement", "rep": rep, "score": score, "test_tail": tail,
            "timed_out": timed_out, "resolved": (score >= 1) if resolved is None else resolved, "seconds": seconds, "tokens": tokens}


def write_log(path, reasoning=100, compactions=0, nudges=0, misspelled=0, length_steps=0):
    lines = [{"type": "user", "text": "task"}]
    for _ in range(misspelled):
        lines.append({"type": "tool_use", "part": {"tool": "read", "state": {"status": "error", "input": {"filePath": "/home/al/x/work/a.js"},
                                                                             "error": "The user has specified a rule which prevents you"}}})
    lines.append({"type": "tool_use", "part": {"tool": "read", "state": {"status": "completed", "input": {"filePath": "/tmp/run/work/a.js"}, "output": "x"}}})
    lines.append({"type": "reasoning", "part": {"text": "x" * reasoning}})
    lines.append({"type": "step_finish", "part": {"reason": "tool-calls", "tokens": {"output": 50}}})
    for _ in range(length_steps):
        lines.append({"type": "step_finish", "part": {"reason": "length", "tokens": {"output": 16384}}})
    for _ in range(compactions):
        lines.append({"type": "reasoning", "part": {"text": rsi2.COMPACTION + " and summarize"}})
        lines.append({"type": "user", "text": rsi2.NUDGE + ", or stop."}) if nudges else None
    lines.append({"type": "tool_use", "part": {"tool": "write", "state": {"status": "completed", "input": {"filePath": "/tmp/run/work/a.js"}}}})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(l) for l in lines if l) + "\n")


class Causes(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rsi2-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_log_features_count_overflow_compactions_and_mistyped_paths(self):
        log = self.tmp / "a.agent.log"
        write_log(log, reasoning=30000, compactions=2, nudges=1, misspelled=3, length_steps=1)
        f = rsi2.log_features(log)
        self.assertEqual((f["compactions"], f["nudges"], f["misspelled"], f["length_steps"]), (2, 2, 3, 1))
        self.assertGreaterEqual(f["max_reason"], 30000)
        self.assertEqual(f["edits"], 1)

    def test_tags_follow_the_hidden_test_tail_and_the_timeout_shape(self):
        quiet = dict(max_reason=100, compactions=0, nudges=0, refusals=0, misspelled=0, length_steps=0, steps=3, first_write_step=1, out_tokens=1, test_runs=0, edits=1)
        self.assertEqual(rsi2.tag_run(rec("a", "f", 1, 0.86, "AssertionError: ValueError not raised"), quiet), ["error_contract"])
        self.assertEqual(rsi2.tag_run(rec("a", "f", 1, 0.86, "assert.ok(fs.existsSync('x/index.js'))"), quiet), ["deliverables"])
        self.assertEqual(rsi2.tag_run(rec("a", "f", 1, 0.5, "Lists differ: [7, 1, 2] != [7, 2]"), quiet), ["logic"])
        self.assertEqual(rsi2.tag_run(rec("a", "f", 1, 0.14, "", timed_out=True), dict(quiet, max_reason=50000)), ["timeout_reasoning"])
        self.assertEqual(rsi2.tag_run(rec("a", "f", 1, 1.0), dict(quiet, misspelled=2)), ["path_misspell"])
        self.assertEqual(rsi2.tag_run(rec("a", "f", 1, 1.0), quiet), [])

    def test_mine_causes_ranks_fixable_causes_by_lost_score_over_all_runs(self):
        log = self.tmp / "q.agent.log"
        write_log(log)
        records = [(rec("a", "fa", 1, 0.86, "ValueError not raised"), log), (rec("a", "fa", 2, 0.86, "IndexError not raised"), log),
                   (rec("b", "fb", 1, 0.5, "Lists differ"), log), (rec("c", "fc", 1, 1.0), log), (rec("d", "fd", 1, 0.86, "Starter project"), log)]
        m = rsi2.mine_causes(records)
        by = {r["cause"]: r for r in m["causes"]}
        self.assertEqual(by["error_contract"]["runs"], 2)
        self.assertEqual(by["error_contract"]["tasks"], ["a"])
        self.assertEqual(by["logic"]["fixable"], False)
        self.assertEqual(m["causes"][0]["cause"], "error_contract", "the biggest fixable cause first")
        self.assertEqual(m["causes"][-1]["cause"], "logic", "the model-limit cause last")
        self.assertAlmostEqual(sum(r["lost"] for r in m["causes"]), m["lost_total"], places=3)


class Sets(unittest.TestCase):
    def test_growth_takes_tasks_with_room_and_safety_takes_ceiling_tasks_by_hash(self):
        tasks = [task("a", "fa"), task("b", "fb"), task("c", "fc"), task("d", "fd")]
        records = [(rec("a", "fa", 1, 1.0), None), (rec("a", "fa", 2, 0.5), None), (rec("b", "fb", 1, 1.0), None), (rec("b", "fb", 2, 1.0), None),
                   (rec("c", "fc", 1, 0.95), None), (rec("c", "fc", 2, 0.95), None), (rec("d", "fd", 1, 0.0), None), (rec("d", "fd", 2, 0.2), None)]
        growth = [t["id"] for t in rsi2.growth_set(tasks, records)]
        self.assertEqual(growth, ["d", "a"], "most room first; c (0.95) has less than MIN_ROOM")
        self.assertEqual([t["id"] for t in rsi2.safety_set(tasks, records)], ["b"])
        self.assertEqual([t["id"] for t in rsi2.growth_set(tasks, records, max_tasks=1)], ["d"])
        self.assertEqual(rsi2.growth_set(tasks[1:2], records), [], "only the allowed tasks (never test) are considered")


class Rule(unittest.TestCase):
    def setUp(self):
        self.th = accept.load_thresholds()

    def arms(self, delta, n=24, r=2, noise=0.0):
        base, cand = [], []
        for i in range(n):
            fam = f"f{i // 2}"
            for rep in range(1, r + 1):
                b = 0.7 + (0.05 if (i + rep) % 2 else -0.05) * (1 if noise else 0)
                base.append(rec(f"t{i}", fam, rep, b, seconds=100, tokens=1000))
                cand.append(rec(f"t{i}", fam, rep, min(1.0, b + delta), seconds=100, tokens=1000))
        return cand, base

    def test_a_clear_gain_passes_and_a_null_does_not_even_with_one_candidate(self):
        cand, base = self.arms(0.15, noise=1)
        self.assertTrue(rsi2.decide2(cand, base, k=1, thresholds=self.th)["accept"])
        cand, base = self.arms(0.0, noise=1)
        self.assertFalse(rsi2.decide2(cand, base, k=1, thresholds=self.th)["accept"])

    def test_the_family_t_bound_is_the_rule_and_the_bootstrap_is_only_reported(self):
        pairs = [(f"f{i}", d) for i, d in enumerate([0.1, 0.2, 0.0, 0.15, 0.05, 0.1, 0.2, 0.1])]
        mean, lo, n = rsi2.family_lower(pairs, 0.05)
        self.assertEqual(n, 8)
        self.assertAlmostEqual(mean, 0.1125, places=4)
        self.assertTrue(0 < lo < mean)
        self.assertLess(rsi2.family_lower(pairs, 0.01)[1], lo, "a smaller alpha gives a lower bound")
        self.assertAlmostEqual(rsi2.t_quantile(0.95, 1000), 1.645, places=2)
        self.assertAlmostEqual(rsi2.t_quantile(0.95, 7), 1.895, places=1)
        v = rsi2.decide2(*self.arms(0.15, noise=1), k=1, thresholds=self.th)
        self.assertIn("bootstrap_lower", v["score"])

    def test_more_candidates_tighten_the_bound(self):
        cand, base = self.arms(0.15, noise=1)
        v1, v6 = rsi2.decide2(cand, base, k=1, thresholds=self.th), rsi2.decide2(cand, base, k=6, thresholds=self.th)
        self.assertLess(v6["alpha_each"], v1["alpha_each"])
        self.assertLessEqual(v6["score"]["lower"], v1["score"]["lower"])
        self.assertEqual(v6["rule"], "RSI2D decide2 (pre-registered)")

    def test_a_cost_blowup_refuses_a_gain(self):
        cand, base = self.arms(0.15, noise=1)
        for c in cand:
            c["tokens"] = 2000
        self.assertFalse(rsi2.decide2(cand, base, k=1, thresholds=self.th, cost_decides=True)["accept"], "on the confirmation the cost decides")
        self.assertTrue(rsi2.decide2(cand, base, k=1, thresholds=self.th)["accept"], "on a candidate the cost is informational (RSI2E)")

    def test_the_safety_set_catches_a_drop_and_a_broken_task(self):
        base = [rec("a", "fa", 1, 1.0), rec("a", "fa", 2, 1.0), rec("b", "fb", 1, 1.0), rec("b", "fb", 2, 1.0)]
        same = [dict(r) for r in base]
        self.assertTrue(rsi2.safety_ok(same, base)["ok"])
        broke = [rec("a", "fa", 1, 0.8), rec("a", "fa", 2, 0.8), rec("b", "fb", 1, 1.0), rec("b", "fb", 2, 1.0)]
        s = rsi2.safety_ok(broke, base)
        self.assertFalse(s["ok"])
        self.assertEqual(s["broke"], ["a"])

    def test_drift_check_trusts_reused_runs_only_when_fresh_ones_agree(self):
        reused = [rec("a", "f", 1, 0.7), rec("a", "f", 2, 0.7), rec("b", "f", 1, 0.8), rec("b", "f", 2, 0.8)]
        self.assertTrue(rsi2.drift_ok([rec("a", "f", 3, 0.72), rec("b", "f", 3, 0.78)], reused)["ok"])
        self.assertFalse(rsi2.drift_ok([rec("a", "f", 3, 0.2), rec("b", "f", 3, 0.3)], reused)["ok"])

    def test_the_old_bootstrap_default_is_unchanged(self):
        pairs = [(f"f{i}", 0.1) for i in range(10)]
        self.assertEqual(accept.paired_bootstrap(pairs), accept.paired_bootstrap(pairs, alpha=0.05))


class Simulation(unittest.TestCase):
    def test_a_real_effect_is_found_more_often_than_a_null_and_the_run_count_is_right(self):
        levels = [0.5, 0.7, 0.8, 0.85, 0.7, 0.6, 0.75, 0.8]
        resid = [-0.2, -0.1, 0.0, 0.1, 0.2, -0.15, 0.15, 0.05, -0.05]
        out = rsi2.simulate(levels, resid, n_tasks=24, r=3, k=3, deltas=[0.15, 0.0, 0.0], cycles=40, boot_reps=120)
        self.assertEqual(out["runs"], 24 * 3 * 4)
        self.assertGreater(out["accept"][0], out["accept"][1])
        self.assertLessEqual(out["false_accept_cycle"], 0.3)


class Fake2Runner:
    """Scores by the marker: without it a task fails one error-contract test (0.86) and the tail says so; with it 1.0."""

    def __init__(self, help_score=1.0):
        self.calls = []
        self.help_score = help_score

    def run(self, task, root, model, arm, rep, out):
        text = (Path(root) / model / "harness.md").read_text(encoding="utf-8")
        good = "FAKE-MARK-7" in text
        self.calls.append((task["id"], arm, rep, good))
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_log(out / f"{task['id']}__{arm}__{rep}.agent.log", misspelled=1)
        score = self.help_score if good else 0.8571
        return {"arm": arm, "task": task["id"], "family": task["family"], "kind": task["kind"], "rep": rep, "model": model,
                "harness": "x", "resolved": score >= 1, "score": score, "seconds": 10.0, "tokens": 100, "prompt_tokens": 100, "requests": 2,
                "agent_exit": 0, "timed_out": False, "test_tail": "" if good else "AssertionError: ValueError not raised"}


class Night(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rsi2n-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.saved = (driver.RUNS, editors.CLAUDE_BIN, pilot.HARNESS_CFG)
        self.addCleanup(self._restore)
        driver.RUNS = self.tmp / "runs"
        cfg = self.tmp / "cfg" / "tiel"
        cfg.mkdir(parents=True)
        (cfg / "harness.md").write_text("Base rule.\n", encoding="utf-8")
        pilot.HARNESS_CFG = self.tmp / "cfg"
        wrapper = self.tmp / "claude"
        wrapper.write_text(f"#!/bin/sh\nexec python3 -B {pilot.STUB_EDITOR} \"$@\"\n")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
        editors.CLAUDE_BIN = str(wrapper)
        self.th = accept.load_thresholds()
        self.train = [task(f"js_t{i}-implement", f"js_t{i}") for i in range(6)]
        self.dev = [task(f"py_d{i}-implement", f"py_d{i}") for i in range(6)]

    def _restore(self):
        driver.RUNS, editors.CLAUDE_BIN, pilot.HARNESS_CFG = self.saved

    def make(self, runner, name="n1", **kw):
        opts = dict(thresholds=self.th, repeats=2, library=self.tmp / "library", fake=True)
        opts.update(kw)
        return rsi2.Cycle2(name, "tiel", {"train": self.train, "dev": self.dev, "test": []}, runner, **opts)

    def test_one_cause_one_candidate_measured_on_the_growth_set_and_adopted(self):
        runner = Fake2Runner()
        c = self.make(runner)
        status = c.run_cycle()
        self.assertEqual(status, "done: version adopted", status)
        steps = driver.load_state("n1")["steps"]
        self.assertEqual(steps["sets"]["value"]["growth"], sorted(t["id"] for t in self.train + self.dev), "every task has room in this fake")
        self.assertEqual(steps["causes"]["value"]["causes"][0]["cause"], "error_contract")
        self.assertIn("propose.error_contract", steps)
        self.assertNotIn("propose.logic", steps, "a model-limit cause gets no editor")
        v = steps["measure.error_contract"]["value"]["verdict"]
        self.assertTrue(v["accept"])
        self.assertEqual(v["k"], 1)
        self.assertEqual(steps["decision"]["value"]["accepted"], ["error_contract"])
        self.assertTrue(steps["confirm"]["value"]["ok"])
        self.assertIn("FAKE-MARK-7", (c.acc_dir / "harness.md").read_text(encoding="utf-8"))
        self.assertTrue((self.tmp / "library" / "harnesses" / "cp001" / "harness.md").exists())
        arms = {a for _, a, _, _ in runner.calls}
        self.assertTrue(any(a.startswith("b2-") for a in arms) and any(a.startswith("c2.error_contract-") for a in arms))
        self.assertFalse(any(a.startswith("final") for a in arms), "no test tasks were given, so no final test")

    def test_a_candidate_that_does_not_help_is_kept_out_by_the_rule_not_by_an_editor(self):
        runner = Fake2Runner(help_score=0.8571)
        c = self.make(runner, name="n2")
        status = c.run_cycle()
        self.assertEqual(status, "done: KEEP_BASE (no candidate passed the rule)")
        self.assertEqual((c.acc_dir / "harness.md").read_text(encoding="utf-8"), "Base rule.\n")
        self.assertNotIn("combine", driver.load_state("n2")["steps"])

    def test_a_crash_after_the_measurement_resumes_without_repeating_runs(self):
        runner = Fake2Runner()
        c = self.make(runner, name="n3", crash_after="measure.error_contract")
        with self.assertRaises(RuntimeError):
            c.run_cycle()
        n_before = len(runner.calls)
        c2 = self.make(runner, name="n3")
        self.assertEqual(c2.run_cycle(), "done: version adopted")
        new = runner.calls[n_before:]
        self.assertFalse(any(a.startswith("b2-") or a.startswith("c2.") for _, a, _, _ in new), "base and candidate runs come from the state")

    def test_a_pool_at_the_ceiling_ends_without_a_cycle(self):
        runner = Fake2Runner()
        orig = runner.run

        def run(task, root, model, arm, rep, out):
            r = orig(task, root, model, arm, rep, out)
            return dict(r, score=1.0, resolved=True, test_tail="")
        runner.run = run
        self.assertTrue(self.make(runner, name="n4").run_cycle().startswith("done: no task with room"))


class Delivery(unittest.TestCase):
    """RSI2D: the plugin must live inside the scratch (bwrap cannot see /home) and every run must prove the harness reached the model."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rsi2d-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_the_plugin_is_copied_next_to_the_experiment_config(self):
        cfg = self.tmp / "scratch" / "cfg"
        cfg.mkdir(parents=True)
        (cfg / "shuracode.json").write_text(json.dumps({"provider": {"eval": {"models": {"m": {"name": "x"}}}}, "model": "eval/m", "plugin": []}))
        executor.wire_harness(cfg, Path("/roots/a"), "tiel")
        out = json.loads((cfg / "shuracode.json").read_text())
        entry = out["plugin"][-1][0]
        self.assertTrue(entry.startswith(str(self.tmp / "scratch" / "plugin")), entry)
        self.assertTrue(Path(entry).exists())
        self.assertTrue((self.tmp / "scratch" / "plugin" / "lib" / "model-harness.ts").exists(), "the plugin's lib import ships with it")
        self.assertNotIn("/home/alisher/Projects", entry)

    def _requests(self, system_texts):
        p = self.tmp / "r.requests.jsonl"
        rows = [{"body": {"messages": [{"role": "system", "content": s} for s in system_texts] + [{"role": "user", "content": "go"}]}}]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return p

    def test_delivery_is_read_from_system_messages_not_from_tool_output(self):
        root = self.tmp / "root"
        (root / "tiel" / "skills" / "probe-skill").mkdir(parents=True)
        (root / "tiel" / "harness.md").write_text("- Before finishing, call every error case the task names.\n")
        (root / "tiel" / "skills" / "probe-skill" / "SKILL.md").write_text("---\nname: probe-skill\ndescription: x\n---\n")
        ok = executor.harness_delivered(self._requests(["prefix", "- Before finishing, call every error case the task names.", "<available_skills>probe-skill</available_skills>"]), root, "tiel")
        self.assertEqual((ok["checked"], ok["delivered"]), (True, True))
        miss = executor.harness_delivered(self._requests(["prefix only"]), root, "tiel")
        self.assertEqual((miss["checked"], miss["delivered"]), (True, False))
        self.assertEqual(miss["missing_skills"], ["probe-skill"])
        # tool output that quotes the file is not delivery
        p = self.tmp / "t.requests.jsonl"
        p.write_text(json.dumps({"body": {"messages": [{"role": "system", "content": "prefix"}, {"role": "tool", "content": "- Before finishing, call every error case the task names."}]}}) + "\n")
        self.assertFalse(executor.harness_delivered(p, root, "tiel")["delivered"])
        empty = self.tmp / "e"
        (empty / "tiel").mkdir(parents=True)
        (empty / "tiel" / "harness.md").write_text("")
        self.assertFalse(executor.harness_delivered(p, empty, "tiel")["checked"])
        self.assertFalse(executor.harness_delivered(self.tmp / "none.jsonl", root, "tiel").get("delivered"))

    def test_a_run_without_delivery_is_discarded_and_stops_the_cycle(self):
        stand = type("S", (), {"fake": True, "expect": lambda self, t: None, "proxy_port": 0, "proxy_log": str(self.tmp / "p.jsonl")})()
        runner = executor.Runner(stand, self.tmp / "scratch")
        root = self.tmp / "root"
        (root / "tiel").mkdir(parents=True)
        (root / "tiel" / "harness.md").write_text("- A rule the model must have seen in its system text.\n")
        out = self.tmp / "out"
        out.mkdir()
        (out / "x__arm__1.requests.jsonl").write_text(json.dumps({"body": {"messages": [{"role": "system", "content": "prefix"}]}}) + "\n")
        with mock.patch.object(executor, "run_task", return_value={"requests": 1}):
            with self.assertRaises(SystemExit):
                runner.run({"id": "x"}, root, "tiel", "arm", 1, out)
        self.assertEqual(len(runner.discarded), 1)
        self.assertIn("harness_missing", runner.discarded[0])

    @unittest.skipUnless(Path(executor.harness.ENGINE).exists() if hasattr(executor.harness, "ENGINE") else False, "no agent engine here")
    def test_on_the_real_engine_the_marker_reaches_the_fake_model_only_for_the_candidate(self):
        """The fake server's policy reads the marker from the SYSTEM text: with the fix the candidate harness is seen (right files,
        marker true in the fake log) and the empty base harness is not."""
        task = pilot.select_tasks(1, 1)["dev"][0]          # a real v3 task folder (start/, hidden/, solution/), the way the night picks them
        cand, base = self.tmp / "cand", self.tmp / "base"
        for r in (cand, base):
            (r / "tiel").mkdir(parents=True)
        (cand / "tiel" / "harness.md").write_text("- FAKE-MARK-7: finish only after the project's tests pass.\n")
        (base / "tiel" / "harness.md").write_text("")
        with executor.FakeStand(self.tmp / "fake") as stand:
            runner = executor.Runner(stand, self.tmp / "scratch")
            rc = runner.run(task, cand, "tiel", "cand", 1, self.tmp / "out")
            rb = runner.run(task, base, "tiel", "base", 1, self.tmp / "out")
            log = [json.loads(l) for l in open(stand.log)]
        self.assertTrue(rc["harness_delivery"]["delivered"], rc["harness_delivery"])
        self.assertFalse(rb["harness_delivery"]["checked"], "an empty base harness has nothing to check")
        self.assertTrue(any(e["marker"] for e in log), "the fake model saw the marker in the candidate run")
        self.assertGreater(rc["score"], rb["score"], (rc["score"], rb["score"]))


class OldPilotUntouched(unittest.TestCase):
    def test_the_driver_keeps_the_pilot_step_and_adds_cycle2(self):
        import argparse
        src = (HERE / "driver.py").read_text(encoding="utf-8")
        self.assertIn('"pilot"', src)
        self.assertIn('"cycle2"', src)
        self.assertIn('"simulate"', src)
        self.assertIn("cause.md", src)
        self.assertTrue((HERE / "prompts" / "cause.md").exists())



class Mechanisms(unittest.TestCase):
    """RSI2E: a mechanism plugin is a candidate measured by the same rule; the cost is informational on candidates and decides on the
    confirmation; a plugin that never fired is not accepted; the delivery check also wants the plugin's on-line."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rsi2e-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.th = accept.load_thresholds()

    def test_cost_is_informational_on_candidates_and_decides_on_confirmation(self):
        cand, base = Rule.arms(Rule(), 0.15, noise=1)
        for c in cand:
            c["tokens"] = 2000
        self.assertTrue(rsi2.decide2(cand, base, k=1, thresholds=self.th, cost_decides=False)["accept"])
        v = rsi2.decide2(cand, base, k=1, thresholds=self.th, cost_decides=True)
        self.assertFalse(v["accept"])
        self.assertFalse(v["cost_ok"])

    def test_a_mechanism_diff_adds_plugins_json_and_the_stand_reads_it(self):
        c = rsi2.Cycle2.__new__(rsi2.Cycle2)
        diff = c._mechanism_diff("done-gate")
        self.assertTrue(pilot.surface_ok(diff))
        root = self.tmp / "root" / "tiel"
        root.mkdir(parents=True)
        (root / "harness.md").write_text("")
        self.assertTrue(driver.apply_diff(root, diff))
        self.assertEqual(executor.plugin_candidates(self.tmp / "root", "tiel"), [{"name": "done-gate", "options": {"doneGate": True}}])
        (root / "plugins.json").write_text(json.dumps({"plugins": [{"name": "nope"}]}))
        with self.assertRaises(ValueError):
            executor.plugin_candidates(self.tmp / "root", "tiel")
        with self.assertRaises(pilot.Stop):
            c._mechanism_diff("nope")

    def test_wire_harness_copies_the_mechanism_plugin_and_delivery_wants_its_line(self):
        root = self.tmp / "root" / "tiel"
        root.mkdir(parents=True)
        (root / "harness.md").write_text("")
        (root / "plugins.json").write_text(json.dumps({"plugins": [{"name": "action-forcing"}]}))
        cfg = self.tmp / "scratch" / "cfg"
        cfg.mkdir(parents=True)
        (cfg / "shuracode.json").write_text(json.dumps({"provider": {"eval": {"models": {"m": {"name": "x"}}}}, "model": "eval/m", "plugin": []}))
        executor.wire_harness(cfg, self.tmp / "root", "tiel")
        out = json.loads((cfg / "shuracode.json").read_text())
        last = out["plugin"][-1]
        self.assertTrue(last[0].endswith("plugin/rsi2/action-forcing.ts") and Path(last[0]).exists())
        self.assertEqual(last[1], {"actionForcing": True})
        self.assertTrue((self.tmp / "scratch" / "plugin" / "rsi2" / "lib.ts").exists())
        req = self.tmp / "r.jsonl"
        req.write_text(json.dumps({"body": {"messages": [{"role": "system", "content": "prefix"}]}}) + "\n")
        miss = executor.harness_delivered(req, self.tmp / "root", "tiel")
        self.assertEqual((miss["checked"], miss["delivered"], miss["missing_plugins"]), (True, False, ["action-forcing"]))
        req.write_text(json.dumps({"body": {"messages": [{"role": "system", "content": "prefix\nshuracode-rsi2:action-forcing: on"}]}}) + "\n")
        self.assertTrue(executor.harness_delivered(req, self.tmp / "root", "tiel")["delivered"])

    def test_mechanism_events_are_collected_from_the_project(self):
        work = self.tmp / "work"
        (work / ".harness" / "events").mkdir(parents=True)
        (work / ".harness" / "events" / "done-gate.jsonl").write_text('{"event":"nudge"}\n{"event":"nudge"}\n')
        self.assertEqual({k: len(v) for k, v in executor.mechanism_events(work).items()}, {"done-gate": 2})
        self.assertEqual(executor.mechanism_events(self.tmp / "none"), {})

    def test_russian_role_tasks_come_from_the_dev_part_only(self):
        ru = rsi2.ru_safety_tasks(3)
        self.assertEqual(len(ru), 3)
        fam = json.loads((HERE.parent / "splits.json").read_text())["families"]
        self.assertTrue(all(t["prompt_lang"] == "ru" and fam[t["family"]] == "dev" for t in ru))
        self.assertEqual([t["id"] for t in ru], [t["id"] for t in rsi2.ru_safety_tasks(3)], "stable choice")
        self.assertEqual(rsi2.ru_safety_tasks(0), [])


class MechanismNight(Night):
    """A mechanism candidate in the fake night: refused when its plugin never fired, adopted when the runs show its events."""

    def test_a_plugin_that_never_fired_is_not_accepted_even_with_a_better_score(self):
        runner = Fake2Runner()
        orig = runner.run

        def run(task, root, model, arm, rep, out):      # the fake runner cannot run plugins: plugins.json makes the score better, but no events
            r = orig(task, root, model, arm, rep, out)
            if (Path(root) / model / "plugins.json").exists():
                r = dict(r, score=1.0, resolved=True, test_tail="")
            return r
        runner.run = run
        c = self.make(runner, name="m1", mechanisms=("done-gate",), max_candidates=0)
        status = c.run_cycle()
        self.assertEqual(status, "done: KEEP_BASE (no candidate passed the rule)")
        v = driver.load_state("m1")["steps"]["measure.plugin-done-gate"]["value"]["verdict"]
        self.assertEqual(v.get("mechanism_never_fired"), "done-gate")

    def test_a_plugin_whose_events_are_seen_can_be_adopted(self):
        runner = Fake2Runner()
        orig = runner.run

        def run(task, root, model, arm, rep, out):
            r = orig(task, root, model, arm, rep, out)
            if (Path(root) / model / "plugins.json").exists():
                r = dict(r, score=1.0, resolved=True, test_tail="", mechanism_events={"done-gate": 1})
            return r
        runner.run = run
        c = self.make(runner, name="m2", mechanisms=("done-gate",), max_candidates=0)
        self.assertEqual(c.run_cycle(), "done: version adopted")
        self.assertTrue((c.acc_dir / "plugins.json").exists())

    def test_the_screen_picks_the_sets_before_the_full_base(self):
        runner = Fake2Runner()
        c = self.make(runner, name="m3", screen_reps=1, safety_ru=rsi2.ru_safety_tasks(1))
        self.assertEqual(c.run_cycle(), "done: version adopted")
        steps = driver.load_state("m3")["steps"]
        self.assertTrue(any(k.startswith("s2.") for k in steps))
        sets = steps["sets"]["value"]
        self.assertEqual(len(sets["safety_ru"]), 1)
        self.assertIn(sets["safety_ru"][0], sets["safety"])
        confirm = steps["confirm"]["value"]
        self.assertTrue(confirm["growth"]["cost_decides"])
        self.assertGreater(confirm["safety"]["pairs"], 0, "the Russian role task was run in the safety set")



class MechanismsOnTheEngine(unittest.TestCase):
    """RSI2E: each mechanism plugin on the real engine with the fake server: its on-line is delivered (harness_delivered) and, where a short
    fake run can trigger it, its event file shows it fired. requirements-lock: the file is written and the agent cannot edit it;
    task-state: seeded; action-forcing: the cap applied; done-gate: the fake edits files and stops without a test -> a nudge;
    compact-preserve: no compaction in a short run, so delivery only."""

    @unittest.skipUnless(Path(executor.harness.ENGINE).exists(), "no agent engine here")
    def test_every_mechanism_is_delivered_and_the_triggerable_ones_fire(self):
        task = pilot.select_tasks(1, 1)["dev"][0]
        tmp = Path(tempfile.mkdtemp(prefix="rsi2e-engine-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        expect_events = {"requirements-lock": "written", "task-state": "seeded", "action-forcing": "cap", "done-gate": "nudge", "compact-preserve": None}
        with executor.FakeStand(tmp / "fake") as stand:
            runner = executor.Runner(stand, tmp / "scratch")
            for name, event in expect_events.items():
                root = tmp / name
                (root / "tiel").mkdir(parents=True)
                (root / "tiel" / "harness.md").write_text("")
                (root / "tiel" / "plugins.json").write_text(json.dumps({"plugins": [{"name": name}]}))
                rec = runner.run(task, root, "tiel", f"m-{name}", 1, tmp / "out")
                self.assertTrue(rec["harness_delivery"]["delivered"], (name, rec["harness_delivery"]))
                if event:
                    events = json.loads((tmp / "out" / f"{task['id']}__m-{name}__1.events.json").read_text())
                    self.assertIn(name, events, (name, rec.get("mechanism_events")))
                    self.assertIn(event, [e.get("event") for e in events[name]], (name, events[name]))



class LongHorizonFinal(Night):
    """LHSET1: the LH dev tasks run once at the end under original and accepted, with the script metrics, reported only."""

    def test_the_lh_final_block_reports_both_harnesses_with_metrics(self):
        if not rsi2.lh_tasks("dev"):
            self.skipTest("no LH tasks built")
        runner = Fake2Runner()
        c = self.make(runner, name="lh1", lh_final=1)
        self.assertEqual(c.run_cycle(), "done: version adopted")
        block = driver.load_state("lh1")["steps"]["lh.final"]["value"]
        self.assertEqual(len(block["tasks"]), 1)
        self.assertIn("summary", block)
        self.assertEqual(block["resolved"]["of"], 2)
        self.assertIn("hallucinated_paths", block["summary"])
        arms = {a for _, a, _, _ in runner.calls}
        self.assertTrue(any(a.startswith("lh-orig-") for a in arms) and any(a.startswith("lh-acc-") for a in arms))



class Calibration(Night):
    def test_lhcal_runs_the_lh_part_under_the_product_harness_and_summarizes(self):
        if not rsi2.lh_tasks("dev"):
            self.skipTest("no LH tasks built")
        runner = Fake2Runner()
        summary = rsi2.lh_calibration("cal1", "tiel", runner, part="dev", n=2, repeats=1)
        self.assertEqual(summary["of"], 2)
        self.assertIn("means", summary)
        steps = driver.load_state("cal1")["steps"]
        self.assertEqual(sum(1 for k in steps if k.startswith("lh.") and k != "lh.summary"), 2)
        again = rsi2.lh_calibration("cal1", "tiel", runner, part="dev", n=2, repeats=1)
        self.assertEqual(len(runner.calls), 2, "a rerun takes the runs from the state")
        self.assertEqual(again["of"], 2)


if __name__ == "__main__":
    unittest.main()