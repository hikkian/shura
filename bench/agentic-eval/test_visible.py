"""QTTS1q: the visible-check registry. Every registry check is honest (its literals come from the prompt or start/), passes on its
reference tree, fails on the start stub, is picked up by bestof's step (a) for implement/refactor, and is left out of the agent's own
checks. No GPU, no model.

    cd bench/agentic-eval && python3 -B -m unittest test_visible
"""
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bestof  # noqa: E402
import harness  # noqa: E402
import visible  # noqa: E402

FAMS = sorted(p.name for p in visible.REGISTRY.iterdir() if (p / "provenance.json").is_file())
TASKS60 = {x.strip() for x in re.split(r"[,\s]+", Path("/home/alisher/Desktop/AI/reports/qual0b/tasks60.txt").read_text()) if x.strip()}


def task(family, kind):
    return harness.load_task(visible.task_dirs(family)[kind])


class Registry(unittest.TestCase):
    def test_every_family_has_a_provenance_with_sources_and_an_existing_file(self):
        self.assertGreaterEqual(len(FAMS), 27)
        for fam in FAMS:
            prov = visible.provenance(fam)
            self.assertTrue((visible.REGISTRY / fam / prov["file"]).is_file(), fam)
            self.assertTrue(prov["sources"] and prov["author"] and prov["applies_to"], fam)
            self.assertNotIn("hidden", json.dumps(prov).lower().replace("hidden/ not opened", ""), fam)

    def test_no_line_shared_with_the_hidden_tests_goes_beyond_the_prompt(self):
        leaks = {fam: visible.guard(fam)["leaks"] for fam in FAMS}
        self.assertEqual({k: v for k, v in leaks.items() if v}, {})

    def test_every_registry_check_passes_on_its_reference_tree(self):
        failed = {}
        for fam in FAMS:
            res = visible.verify(fam)
            if not res["ok"]:
                failed[fam] = res["output"][-300:]
        self.assertEqual(failed, {})


class Signal(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="visible-t-", dir="/dev/shm" if Path("/dev/shm").is_dir() else None))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def tree(self, t, solved, tag=""):
        work = self.tmp / f"{t['id']}-{'s' if solved else 'u'}{tag}"
        shutil.copytree(t["dir"] / "start", work)
        if solved:
            shutil.copytree(t["dir"] / "solution", work, dirs_exist_ok=True)
            for rel in t.get("solution_remove") or []:
                (work / rel).unlink(missing_ok=True)
        return work

    def broken(self, t):
        work = self.tree(t, False, tag="-broken")
        for f in list(work.glob("*.py")) + list(work.glob("*.js")):
            f.write_text("raise RuntimeError('broken')\n" if f.suffix == ".py" else "throw new Error('broken')\n")
        return work

    def test_implement_stub_fails_and_the_solution_passes(self):
        for fam in ("py_retry_schedule", "js_route_match", "f22_roman", "py_event_journal"):
            t = task(fam, "implement")
            self.assertEqual(visible.spec(t)["source"], "registry", fam)
            self.assertFalse(visible.run_in_copy(t, self.tree(t, False))["ok"], fam)
            self.assertTrue(visible.run_in_copy(t, self.tree(t, True))["ok"], fam)

    def test_refactor_start_and_split_solution_pass_and_a_broken_refactor_fails(self):
        for fam in ("py_deps_plan", "js_headers", "py_rollout"):
            t = task(fam, "refactor")
            self.assertEqual(visible.spec(t)["source"], "registry", fam)
            self.assertTrue(visible.run_in_copy(t, self.tree(t, False))["ok"], fam)     # the working module as given
            self.assertTrue(visible.run_in_copy(t, self.tree(t, True))["ok"], fam)      # the split package
            self.assertFalse(visible.run_in_copy(t, self.broken(t))["ok"], fam)

    def test_bugfix_keeps_its_own_public_repro(self):
        t = harness.load_task(HERE / "tasks_v3" / "py_account" / "bugfix")
        s = visible.spec(t)
        self.assertEqual((s["source"], list(s["files"])), ("task", ["public_repro.py"]))
        t1 = harness.load_task(HERE / "tasks" / "f01_duration" / "bugfix")          # the first set names its repro in the prompt
        s1 = visible.spec(t1)
        self.assertEqual((s1["source"], list(s1["files"])), ("task", ["test_repro.py"]))
        self.assertFalse(visible.run_in_copy(t1, self.tree(t1, False))["ok"])
        self.assertTrue(visible.run_in_copy(t1, self.tree(t1, True))["ok"])

    def test_a_kind_the_check_does_not_apply_to_gets_no_signal(self):
        t = task("py_event_journal", "implement")
        self.assertEqual(visible.provenance("py_event_journal")["applies_to"], ["implement"])
        t_other = dict(t, kind="tests-only")
        self.assertIsNone(visible.spec(t_other))

    def test_the_agent_cannot_edit_the_registry_check_in_its_tree(self):
        t = task("py_retry_schedule", "implement")
        work = self.tree(t, False)
        (work / "visible_check.py").write_text("import unittest\nclass T(unittest.TestCase):\n    def test_ok(self):\n        pass\n")
        self.assertFalse(visible.run_in_copy(t, work)["ok"])          # the registry's file ran, not the planted one

    def test_bestof_uses_the_registry_and_keeps_it_out_of_the_agents_own_tests(self):
        t = task("py_retry_schedule", "implement")
        work = self.tree(t, True)
        log = self.tmp / "a.log"
        log.write_text("")
        self.assertTrue(bestof.visible_check(t, work)["ok"])
        self.assertEqual(bestof.visible_files(t), ["visible_check.py"])
        self.assertFalse(bestof.own_check(t, work, log)["has"])          # no own tests: the registry file is not one of them


class Coverage(unittest.TestCase):
    def test_tasks60_implement_and_refactor_are_covered(self):
        cov = visible.coverage(visible.all_tasks(), TASKS60)
        self.assertEqual((cov["implement"]["tasks"], cov["implement"]["with_signal"]), (21, 21), cov["implement"])
        self.assertEqual((cov["refactor"]["tasks"], cov["refactor"]["with_signal"]), (11, 11), cov["refactor"])
        self.assertEqual(cov["bugfix"]["with_signal"], cov["bugfix"]["tasks"])


if __name__ == "__main__":
    unittest.main()
