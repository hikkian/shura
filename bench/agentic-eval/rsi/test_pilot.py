"""RSIL0b tests (offline: no engine, no server, no model; the editors are the stub program, the runs are an injected fake).
Run: cd bench/agentic-eval && python3 -B -m unittest discover -s rsi -p 'test_pilot.py'

The cycle's steps, the resume after an interruption, the budgets, KEEP_BASE, the admission and the library are exercised with
a fake runner that scores the candidate by the marker line the stub editor adds. The engine path (wire_harness, the manifest of the
--fake stand, the policy of the fake worker) is checked by its own tests.
"""
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rsi import accept, budget, driver, editors, executor, pilot, trajectory  # noqa: E402


class FakeRunner:
    """Scores a run by the marker in the harness it was given: the same numbers for every task, so a candidate with the marker wins."""

    def __init__(self):
        self.calls = []

    def run(self, task, root, model, arm, rep, out):
        text = (Path(root) / model / "harness.md").read_text(encoding="utf-8")
        good = "FAKE-MARK-7" in text
        self.calls.append((task["id"], arm, rep, good))
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{task['id']}__{arm}__{rep}.agent.log").write_text('{"type":"text","part":{"text":"done"}}\n')
        return {"arm": arm, "task": task["id"], "family": task["family"], "kind": task["kind"], "rep": rep, "model": model,
                "harness": executor.fingerprint(Path(root) / model / "harness.md"), "resolved": good, "score": 1.0 if good else 0.0,
                "seconds": 10.0, "tokens": 100, "prompt_tokens": 100, "requests": 2, "agent_exit": 0, "timed_out": False}


def task(tid, family, kind="bugfix", difficulty="medium"):
    return {"id": tid, "family": family, "kind": kind, "difficulty": difficulty, "dir": Path("/nonexistent"), "prompt": "p",
            "test_cmd": "true", "timeout_s": 60}


class PilotCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rsil0b-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.saved = (driver.RUNS, editors.CLAUDE_BIN, pilot.HARNESS_CFG, os.environ.get("RSI_FAKE_COMPOSER"))
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
        os.environ.pop("RSI_FAKE_COMPOSER", None)
        self.library = self.tmp / "library"
        self.train = [task("js_a-bugfix", "js_a")]
        self.dev = [task("js_d1-bugfix", "js_d1"), task("js_d2-bugfix", "js_d2")]
        self.thresholds = accept.load_thresholds()

    def _restore(self):
        driver.RUNS, editors.CLAUDE_BIN, pilot.HARNESS_CFG = self.saved[0], self.saved[1], self.saved[2]
        if self.saved[3] is None:
            os.environ.pop("RSI_FAKE_COMPOSER", None)
        else:
            os.environ["RSI_FAKE_COMPOSER"] = self.saved[3]

    def make(self, runner, name="p1", **kw):
        opts = dict(thresholds=self.thresholds, repeats=2, library=self.library, fake=True)
        opts.update(kw)
        tasks = {"train": self.train, "dev": self.dev, "test": getattr(self, "test", [])}
        return pilot.Pilot(name, "tiel", tasks, runner, **opts)

    def steps(self, name="p1"):
        return driver.load_state(name)["steps"]


class Cycle(PilotCase):
    def test_a_marked_candidate_is_measured_admitted_adopted_and_entered_in_the_library(self):
        runner = FakeRunner()
        p = self.make(runner)
        status = p.run_cycle()
        self.assertEqual(status, "done")
        steps = self.steps()
        self.assertTrue(steps["t00-js_a-bugfix.ev.invariant"]["value"]["verdict"]["accept"])
        self.assertFalse(steps["t00-js_a-bugfix.ev.specific"]["value"]["verdict"]["accept"])
        self.assertIn("FAKE-MARK-7", p.h_file.read_text(encoding="utf-8"))
        self.assertTrue(steps["t00-js_a-bugfix.version"]["value"]["applied"])
        self.assertTrue(steps["cp.001"]["value"]["accepted"], "the checkpoint accepts the marked version against the base")
        entry = self.library / "harnesses" / "cp001" / "harness.md"
        self.assertIn("FAKE-MARK-7", entry.read_text(encoding="utf-8"))
        self.assertIn("checkpoint 1: accepted", (self.library / "EXPERIENCE.md").read_text(encoding="utf-8"))
        self.assertIn("harnesses/cp001/harness.md", (self.library / "INDEX.md").read_text(encoding="utf-8"))
        self.assertIn("FAKE-MARK-7", (p.acc_dir / "harness.md").read_text(encoding="utf-8"))

    def test_the_editors_work_in_sandboxed_folders_and_the_usage_is_logged(self):
        p = self.make(FakeRunner())
        p.run_cycle()
        rows = [json.loads(line) for line in (p.usage).read_text().splitlines()]
        self.assertEqual([r["role"] for r in rows if r["role"] in ("invariant", "specific")], ["invariant", "specific"])
        self.assertTrue(all(r["editor"] in ("opus", "fable", "sonnet", "sol") for r in rows))

    def test_keep_base_stops_before_any_admission_and_the_harness_stays(self):
        os.environ["RSI_FAKE_COMPOSER"] = "keep"
        runner = FakeRunner()
        p = self.make(runner)
        p.run_cycle()
        steps = self.steps()
        self.assertTrue(steps["t00-js_a-bugfix.compose"]["value"]["keep"])
        self.assertNotIn("t00-js_a-bugfix.version", steps)
        self.assertNotIn("cp.001", steps, "nothing was composed, so no checkpoint")
        self.assertEqual(p.h_file.read_text(encoding="utf-8"), "Base rule.\n")
        self.assertFalse((self.library / "harnesses").exists())
        self.assertIn("KEEP_BASE", (self.library / "EXPERIENCE.md").read_text(encoding="utf-8"))
        self.assertFalse(any(good for (_, arm, _, good) in runner.calls if arm.startswith("t00-js_a-bugfix-composed")))

    def test_an_editor_budget_stops_the_cycle_cleanly_and_a_rerun_with_more_budget_continues(self):
        runner = FakeRunner()
        # in this fake every base run fails, so the specific editor is Fable (RSIL0e E3) and its budget is the one that stops
        p = self.make(runner, limits={"calls": {"opus": 1, "fable": 0}})
        status = p.run_cycle()
        self.assertEqual(status, "stopped: budget: fable calls 0 of 0")
        steps = self.steps()
        self.assertIn("t00-js_a-bugfix.propose.invariant", steps)
        self.assertNotIn("t00-js_a-bugfix.propose.specific", steps)
        p2 = self.make(runner, limits={"calls": {"opus": 10, "fable": 10}})
        self.assertEqual(p2.run_cycle(), "done")
        self.assertIn("t00-js_a-bugfix.reflect", self.steps())

    def test_a_crash_between_steps_resumes_without_repeating_the_finished_steps(self):
        runner = FakeRunner()
        p = self.make(runner, crash_after="t00-js_a-bugfix.propose.invariant")
        with self.assertRaises(RuntimeError):
            p.run_cycle()
        runs_before = len(runner.calls)
        rows_before = len(pilot.usage_rows(self.tmp / "runs" / "p1" / "usage.jsonl"))
        p2 = self.make(runner)
        self.assertEqual(p2.run_cycle(), "done")
        self.assertEqual(sum(1 for c in runner.calls[:runs_before] if c[1] == "base"), 2, "the base runs (two repeats) came first")
        self.assertEqual(sum(1 for c in runner.calls if c[1] == "base"), 2, "the base runs are not repeated after the crash")
        rows = pilot.usage_rows(self.tmp / "runs" / "p1" / "usage.jsonl")
        self.assertEqual(sum(1 for r in rows if r["role"] == "invariant"), 1, "the invariant editor is not called twice")
        self.assertGreater(len(rows), rows_before)

    def test_the_morning_report_names_the_outcome_and_the_usage(self):
        self.make(FakeRunner()).run_cycle()
        text = pilot.morning("p1")
        self.assertIn("checkpoint 1: accepted", text)
        self.assertIn("| opus |", text)
        self.assertTrue((self.tmp / "runs" / "p1" / "MORNING.md").exists())


class Gates(unittest.TestCase):
    def test_the_thresholds_are_read_from_the_dataset_section_five(self):
        th = accept.load_thresholds()
        self.assertEqual((th["score_ci_low_min"], th["resolved_ci_low_min"], th["repeats_min"]), (0.0, -0.03, 2))
        self.assertIn("§5", th["source"])

    def test_a_missing_dataset_is_an_error_not_a_default(self):
        with self.assertRaises(FileNotFoundError):
            accept.load_thresholds(Path("/nonexistent/DATASET.md"))

    def test_the_real_pilot_refuses_fewer_than_two_repeats(self):
        import argparse
        args = argparse.Namespace(repeats=1, fake=True)
        with self.assertRaises(SystemExit):
            pilot.run_cli(args)

    def test_an_edit_outside_harness_md_is_refused_because_the_plugin_would_not_read_it(self):
        self.assertTrue(pilot.surface_ok("--- a/harness.md\n+++ b/harness.md\n@@ -0,0 +1 @@\n+x\n"))
        self.assertTrue(pilot.surface_ok("--- /dev/null\n+++ b/harness.md\n@@ -0,0 +1 @@\n+x\n"))
        self.assertFalse(pilot.surface_ok("--- a/AGENTS.md\n+++ b/AGENTS.md\n@@ -0,0 +1 @@\n+x\n"))
        self.assertFalse(pilot.surface_ok(""))

    def test_the_time_limit_is_1_35_only_when_the_score_is_significantly_better(self):
        def runs(score, seconds):
            base = [{"task": f"f{f}-{r}", "family": f"f{f}", "score": 0.0, "resolved": False, "seconds": 10.0, "tokens": 100, "rep": r}
                    for f in range(4) for r in (1, 2)]
            cand = [dict(b, score=score, seconds=seconds, resolved=score > 0) for b in base]
            return cand, base
        th = accept.load_thresholds()
        cand, base = runs(0.5, 13.0)                 # time x1.3, score significantly up in every family
        self.assertTrue(accept.decide(cand, base, thresholds=th)["accept"])
        cand, base = runs(0.0, 13.0)                 # time x1.3 without a gain: the 1.15 limit applies
        self.assertFalse(accept.decide(cand, base, thresholds=th)["accept"])
        self.assertEqual(accept.decide(cand, base, thresholds=th)["time_limit"], 1.15)

    def test_the_size_budget_also_limits_characters(self):
        self.assertTrue(budget.check("x" * 1000)["ok"])
        self.assertFalse(budget.check("x" * 7500)["ok"], "7 500 characters are over the 7 000 the editors are told")
        self.assertEqual(budget.added_lines("--- a\n+++ b\n@@\n+one\n two\n+three\n"), "one\nthree")


class Wiring(unittest.TestCase):
    def test_the_model_harness_is_wired_under_the_model_id_the_plugin_reads(self):
        tmp = Path(tempfile.mkdtemp(prefix="rsil0b-wire-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        conf = {"provider": {"eval": {"models": {"m": {"name": "model under test"}}}}, "model": "eval/m", "plugin": [["x.ts", {}]]}
        (tmp / "shuracode.json").write_text(json.dumps(conf))
        executor.wire_harness(tmp, Path("/roots/a"), "tiel")
        out = json.loads((tmp / "shuracode.json").read_text())
        self.assertEqual(out["model"], "eval/tiel-coder")
        self.assertIn("tiel-coder", out["provider"]["eval"]["models"])
        self.assertEqual(out["plugin"][-1][1], {"harnessDir": "/roots/a"})
        self.assertTrue(out["plugin"][-1][0].endswith("plugin/model-harness.ts"))

    def test_the_fake_stand_writes_the_right_and_the_wrong_files_from_the_solution(self):
        tmp = Path(tempfile.mkdtemp(prefix="rsil0b-exp-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "solution" / "src").mkdir(parents=True)
        (tmp / "solution" / "src" / "a.js").write_text("export const a = 1\n")
        stand = executor.FakeStand(tmp / "fake")
        stand.manifest = tmp / "fake" / "manifest.json"
        stand.manifest.parent.mkdir(parents=True)
        stand.expect({"dir": tmp})
        m = json.loads(stand.manifest.read_text())
        self.assertEqual(m["right"], [{"path": "src/a.js", "content": "export const a = 1\n"}])
        self.assertEqual(m["wrong"][0]["path"], "src/a.js")
        self.assertNotEqual(m["wrong"][0]["content"], m["right"][0]["content"])

    def test_the_fake_worker_writes_the_right_files_only_when_the_marker_is_in_the_system_text(self):
        import importlib
        fw = importlib.import_module("rsi.fake_worker")
        tmp = Path(tempfile.mkdtemp(prefix="rsil0b-fw-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "m.json").write_text(json.dumps({"right": [{"path": "a.js", "content": "R"}],
                                                "wrong": [{"path": "a.js", "content": "W"}]}))
        fw.MANIFEST = str(tmp / "m.json")
        fw.LOG = ""
        system = {"role": "system", "content": "Working directory: /w/proj\n" + fw.MARKER}
        marked = fw.policy({"messages": [system, {"role": "user", "content": "go"}]})
        self.assertEqual(marked["tool_calls"][0]["arguments"], {"filePath": "/w/proj/a.js", "content": "R"})
        plain = fw.policy({"messages": [{"role": "system", "content": "Working directory: /w/proj"}, {"role": "user", "content": "go"}]})
        self.assertEqual(plain["tool_calls"][0]["arguments"]["content"], "W")
        done = fw.policy({"messages": [system, {"role": "tool", "content": "ok"}]})
        self.assertEqual(done, {"text": "готово"})


class TwoLevels(PilotCase):
    def test_a_refused_checkpoint_puts_the_accepted_harness_back_and_writes_the_numbers(self):
        p = self.make(FakeRunner())
        p.run_cycle()                                              # the marked version is accepted at checkpoint 1
        self.assertIn("FAKE-MARK-7", (p.acc_dir / "harness.md").read_text(encoding="utf-8"))
        p.h_file.write_text("Worse rule.\n", encoding="utf-8")     # a working version without the marker
        result = p._checkpoint(2)
        self.assertFalse(result["accepted"])
        self.assertIn("FAKE-MARK-7", p.h_file.read_text(encoding="utf-8"), "the working harness is back to the accepted one")
        self.assertIn("checkpoint 2: refused", (self.library / "EXPERIENCE.md").read_text(encoding="utf-8"))

    def test_the_test_tasks_run_only_in_the_final_test_and_only_as_original_against_accepted(self):
        self.test = [task("js_t1-bugfix", "js_t1")]
        runner = FakeRunner()
        p = self.make(runner)
        self.assertEqual(p.run_cycle(), "done")
        test_calls = [c for c in runner.calls if c[0] == "js_t1-bugfix"]
        self.assertEqual(len(test_calls), 2 * 2, "original and accepted, two repeats each")
        for _, arm, _, _ in test_calls:
            self.assertTrue(arm.startswith(("orig-", "acc-")), arm)
        self.assertIn("reported only", self.steps()["final.test"]["value"]["note"])
        self.assertNotIn("js_t1-bugfix", [t["id"] for t in self.train + self.dev])


class DevSelection(unittest.TestCase):
    """The checkpoints' dev part: the pilot uses the six pre-registered ideas (48 tasks); the full dev part is for RSI1."""

    def test_the_pilot_dev_is_the_six_pre_registered_ideas_with_48_tasks(self):
        tasks = pilot.select_tasks(0, 0, 0, dev_scope="subset")
        expected = json.loads(pilot.DEV_SUBSET.read_text(encoding="utf-8"))["ideas"]
        self.assertEqual(len(expected), 6)
        self.assertEqual(sorted({t["family"].split("_", 1)[-1] for t in tasks["dev"]}), expected)
        self.assertEqual(len(tasks["dev"]), 48)

    def test_the_subset_file_is_the_hash_rule_it_was_registered_with(self):
        import hashlib
        fam = json.loads(pilot.SPLITS.read_text(encoding="utf-8"))["families"]
        # the rule was registered for the families that existed then; the ce2 families (CODEEVAL2, 2026-10-30) came later and have the tag "ce2"
        ce2 = {t["family"] for t in executor.all_tasks(pilot.TASKS_V3) if "ce2" in (t.get("tags") or [])}
        ideas = sorted({f.split("_", 1)[1] for f, s in fam.items() if s == "dev" and f.split("_", 1)[0] in ("py", "js") and f not in ce2})
        ranked = sorted(ideas, key=lambda i: hashlib.sha256(f"{pilot.SEED}:{i}".encode()).hexdigest())
        self.assertEqual(sorted(ranked[:6]), json.loads(pilot.DEV_SUBSET.read_text(encoding="utf-8"))["ideas"])

    def test_the_all_scope_has_the_whole_dev_part(self):
        # 96 tasks of the original v3 dev families (2026-10-22) plus the dev families of CODEEVAL2 (the count comes from splits.json)
        splits = json.loads(pilot.SPLITS.read_text(encoding="utf-8"))
        expected = sum(n for f, n in splits["tasks"].items() if splits["families"].get(f) == "dev" and splits["sets"].get(f) == "v3")
        self.assertEqual(len(pilot.select_tasks(0, 0, 0, dev_scope="all")["dev"]), expected)
        self.assertGreaterEqual(expected, 96)

    def test_no_test_tasks_unless_asked_for_so_the_night_runs_no_final_test_by_default(self):
        self.assertEqual(pilot.select_tasks(2, 2)["test"], [])
        self.assertEqual(len(pilot.select_tasks(2, 2, 2)["test"]), 2)

    def test_the_size_argument_takes_that_many_dev_tasks(self):
        self.assertEqual(len(pilot.select_tasks(0, 5, 0, dev_scope="subset")["dev"]), 5)


class ReviewItems(PilotCase):
    """RSIR1 review items with their own tests: kinds rotate in the selection (D4), the growth limit (D6), the library the editors see (D9),
    the night's call limits (D10), the runs side by side in the evidence (D8)."""

    def test_the_selection_rotates_the_kinds_of_tasks(self):
        train = pilot.select_tasks(8, 0, 0, dev_scope="subset")["train"]
        self.assertGreaterEqual(len({t["kind"] for t in train}), 3, "one kind must not take the whole selection")

    def test_the_growth_of_the_always_on_text_is_measured_against_the_original(self):
        self.assertTrue(pilot.growth("short\n", "short\n")["ok"])
        self.assertFalse(pilot.growth("x" * 20000, "")["ok"])
        self.assertEqual(pilot.growth("abc", "abcdef")["chars"], -3)

    def test_the_editors_see_only_the_accepted_checkpoints_and_the_notes(self):
        p = self.make(FakeRunner())
        (self.library / "versions" / "t00").mkdir(parents=True)
        (self.library / "versions" / "t00" / "harness.md").write_text("archive\n", encoding="utf-8")
        (self.library / "harnesses" / "cp001").mkdir(parents=True)
        (self.library / "harnesses" / "cp001" / "harness.md").write_text("accepted\n", encoding="utf-8")
        (self.library / "EXPERIENCE.md").write_text("notes\n", encoding="utf-8")
        (self.library / "INDEX.md").write_text("index\n", encoding="utf-8")
        view = p._library_view()
        names = sorted(str(x.relative_to(view)) for x in view.rglob("*") if x.is_file())
        self.assertEqual(names, ["EXPERIENCE.md", "INDEX.md", "harnesses/cp001/harness.md"])

    def test_the_night_defaults_for_the_calls_come_from_the_number_of_train_tasks(self):
        limits = pilot.parse_limits("", None, 12)
        self.assertEqual(limits["calls"], {"opus": 26, "sol": 26, "fable": 26, "sonnet": 14})
        self.assertEqual(pilot.parse_limits("opus=3", None, 12)["calls"], {"opus": 3})


class Routing(unittest.TestCase):
    """RSIL0e E3 (RSIR2 §6): which editor writes the specific proposal. The invariant is always Opus."""

    def test_the_specific_editor_follows_the_kind_and_the_base_repeats(self):
        ok = [{"resolved": True}, {"resolved": True}]
        failed = [{"resolved": False}, {"resolved": False}]
        split = [{"resolved": True}, {"resolved": False}]
        self.assertEqual(pilot.editor_for("invariant", task("t", "js_a", "implement"), failed), "opus")
        # RSI1: the main editor alternates by train task; Fable keeps the hard kinds
        self.assertEqual(pilot.editor_for("invariant", task("t", "js_a", "bugfix"), ok, 1, ("opus", "sol")), "sol")
        self.assertEqual(pilot.editor_for("specific", task("t", "js_a", "bugfix"), ok, 1, ("opus", "sol")), "sol")
        self.assertEqual(pilot.editor_for("specific", task("t", "js_a", "implement"), ok, 1, ("opus", "sol")), "fable")
        self.assertEqual(pilot.editor_for("invariant", task("t", "js_a", "bugfix"), ok, 2, ("opus", "sol")), "opus")
        self.assertEqual(pilot.editor_for("specific", task("t", "js_a", "bugfix"), ok), "opus")
        self.assertEqual(pilot.editor_for("specific", task("t", "js_a", "bugfix"), split), "fable")
        self.assertEqual(pilot.editor_for("specific", task("t", "js_a", "bugfix"), failed), "fable")
        for kind in ("implement", "refactor", "tests-only"):
            self.assertEqual(pilot.editor_for("specific", task("t", "js_a", kind), ok), "fable")


class StubSol(unittest.TestCase):
    def test_the_fake_answers_for_sol_the_way_codex_does(self):
        work = Path(tempfile.mkdtemp(prefix="rsi1-sol-", dir="/dev/shm" if os.path.isdir("/dev/shm") else None))
        self.addCleanup(shutil.rmtree, work, True)
        saved = (editors.CLAUDE_BIN, editors.CODEX_BIN)
        self.addCleanup(lambda: (setattr(editors, "CLAUDE_BIN", saved[0]), setattr(editors, "CODEX_BIN", saved[1])))
        pilot.setup_stub_editor(work)
        reply = editors.call("sol", "invariant", "# role: invariant\n", cwd=work, usage_log=work / "usage.jsonl")
        self.assertTrue(reply.ok, reply.error)
        self.assertIn("FAKE-MARK-7", reply.text)
        self.assertEqual((reply.model, reply.tokens_out), ("gpt-6.1-sol", 1200))


class PoolEntries(unittest.TestCase):
    """RSIL0e E1/E4: both repeats of a base task are pool entries; the header says which repeat it is and whether the repeats disagree."""

    def test_both_repeats_enter_and_a_disagreeing_pair_is_marked_and_ranked_first(self):
        tmp = Path(tempfile.mkdtemp(prefix="rsil0e-pool-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        log = tmp / "run.agent.log"
        log.write_text('{"type":"text","part":{"text":"done"}}\n')
        t = task("py_flags-refactor", "py_flags", "refactor")
        r1 = {"task": "py_flags-refactor", "family": "py_flags", "kind": "refactor", "rep": 1, "resolved": True,
              "tests_run_project": True, "agent_exit": 0, "timed_out": False, "seconds": 634}
        r2 = {"task": "py_flags-refactor", "family": "py_flags", "kind": "refactor", "rep": 2, "resolved": False,
              "tests_run_project": False, "agent_exit": None, "timed_out": True, "seconds": 900}
        e1 = trajectory.entry(r1, t, log, repeats=2, others=[r2])
        e2 = trajectory.entry(r2, t, log, repeats=2, others=[r1])
        self.assertNotEqual(e1["id"], e2["id"])
        self.assertIn("rep 2 of 2 | other repeat: resolved, 634 s | repeats disagree", e2["text"])
        self.assertIn("repeats disagree", e1["fail"])
        self.assertIn("repeats disagree", e2["fail"])
        same = trajectory.entry(dict(r1, rep=1, resolved=True), t, log, repeats=2, others=[dict(r1, rep=2)])
        self.assertNotIn("repeats disagree", same["text"])
        self.assertNotIn("repeats disagree", same["fail"])


class RoundText(PilotCase):
    """RSIL0e E6 and the KEEP_BASE line: a round without a lesson is one line; the reflector's own title is not doubled."""

    def test_a_keep_base_round_without_a_lesson_is_one_line_and_no_title_is_doubled(self):
        p = self.make(FakeRunner())
        with mock.patch.object(pilot.editors, "call", return_value=SimpleNamespace(text="NO_ENTRY", ok=True, error="")):
            p._reflect("t00-js_a-bugfix", task("t00-js_a-bugfix", "js_a"), "KEEP_BASE (does not apply)", {}, {"keep": True})
        with mock.patch.object(pilot.editors, "call",
                               return_value=SimpleNamespace(text="# EXPERIENCE.md\n## t01\n- one lesson\n", ok=True, error="")):
            p._reflect("t01-js_a-bugfix", task("t01-js_a-bugfix", "js_a"), "KEEP_BASE", {}, {"keep": True})
        text = (self.library / "EXPERIENCE.md").read_text(encoding="utf-8")
        self.assertIn("## t00-js_a-bugfix — KEEP_BASE (does not apply)\n- KEEP_BASE: no new lesson\n", text)
        self.assertIn("## t01-js_a-bugfix — KEEP_BASE\n## t01\n- one lesson\n", text)
        self.assertNotIn("# EXPERIENCE.md", text)

    def test_the_runs_are_paired_by_repeat_for_the_evidence(self):
        cand = [{"rep": 1, "resolved": True, "score": 1.0, "seconds": 9.0}, {"rep": 2, "resolved": False, "score": 0.5, "seconds": 8.0}]
        base = [{"rep": 1, "resolved": False, "score": 0.0, "seconds": 10.0}, {"rep": 2, "resolved": False, "score": 0.0, "seconds": 11.0}]
        pairs = pilot.paired_runs(cand, base)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["base"]["score"], 0.0)
        self.assertEqual(pairs[1]["candidate"]["score"], 0.5)


class Surface(unittest.TestCase):
    def test_skills_and_scripts_are_the_surface_and_subagents_are_not(self):
        skill = "--- a/skills/verify/SKILL.md\n+++ b/skills/verify/SKILL.md\n@@ -0,0 +1 @@\n+x\n"
        script = "--- /dev/null\n+++ b/skills/verify/scripts/check.py\n@@ -0,0 +1 @@\n+x\n"
        agent = "--- /dev/null\n+++ b/agents/review/AGENT.md\n@@ -0,0 +1 @@\n+x\n"
        self.assertTrue(pilot.surface_ok(skill))
        self.assertTrue(pilot.surface_ok(script))
        self.assertFalse(pilot.surface_ok(agent))

    def test_a_skill_body_over_its_limit_is_refused_and_the_always_on_text_keeps_its_budget(self):
        big = "--- /dev/null\n+++ b/skills/verify/SKILL.md\n@@ -0,0 +1 @@\n+" + "x" * 5000 + "\n"
        self.assertFalse(pilot.surface_size(big)["ok"])
        small = "--- a/harness.md\n+++ b/harness.md\n@@ -0,0 +1 @@\n+" + "y" * 100 + "\n"
        self.assertTrue(pilot.surface_size(small)["ok"])


if __name__ == "__main__":
    unittest.main()
