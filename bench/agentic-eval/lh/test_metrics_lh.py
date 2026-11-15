"""LHSET1: the metrics on synthetic logs (no model). Run: cd bench/agentic-eval && python3 -B -m unittest lh.test_metrics_lh"""
import json, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lh import metrics_lh as M  # noqa: E402


def tool(name, args, status="completed", output="ok", error=None):
    st = {"status": status, "input": args, "output": output}
    if error:
        st["error"] = error
    return {"type": "tool_use", "part": {"tool": name, "state": st}}


def log(path, events):
    Path(path).write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


class Metrics(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lhm-"))

    def test_hallucinated_paths_count_until_the_first_successful_read_of_that_name(self):
        p = log(self.tmp / "a.log", [
            tool("read", {"filePath": "/home/al/proj/work/app.py"}, "error", "", "File not found: /home/al/proj/work/app.py"),
            tool("read", {"filePath": "/w/work/app.py"}),
            tool("read", {"filePath": "/w/work/app.py"}, "error", "", "File not found"),      # after a success: not counted
            tool("read", {"filePath": "/w/work/ghost.py"}, "error", "", "ENOENT"),
            {"type": "text", "part": {"text": "done"}}])
        m = M.compute(p)
        self.assertEqual(m["hallucinated_paths"], 2)

    def test_hallucinated_symbols_are_imports_of_modules_absent_from_the_tree(self):
        work = self.tmp / "work"
        (work / "pkg").mkdir(parents=True)
        (work / "pkg" / "real.py").write_text("x = 1\n")
        p = log(self.tmp / "b.log", [tool("write", {"filePath": str(work / "pkg" / "new.py"), "content": "from .real import x\nfrom .phantom import y\nimport json\n"})])
        self.assertEqual(M.compute(p, work=work)["hallucinated_symbols"], ["phantom"])

    def test_repeats_loops_and_back_and_forth_edits(self):
        p = log(self.tmp / "c.log", [
            tool("bash", {"command": "ls"}), tool("bash", {"command": "ls"}), tool("bash", {"command": "ls"}),
            tool("write", {"filePath": "/w/a.py", "content": "v1"}), tool("write", {"filePath": "/w/a.py", "content": "v2"}),
            tool("write", {"filePath": "/w/a.py", "content": "v1"})])
        m = M.compute(p)
        self.assertEqual(m["repeated_calls"], 3, "two extra `ls` and the write of v1 again")
        self.assertEqual(m["longest_identical_run"], 3)
        self.assertEqual(m["edits_back_and_forth"], 1)

    def test_unverified_done_needs_a_claim_and_no_test_after_the_last_edit(self):
        base = [tool("write", {"filePath": "/w/a.py", "content": "x"}), tool("bash", {"command": "python3 -m pytest -q"})]
        p = log(self.tmp / "d.log", base + [{"type": "text", "part": {"text": "All tests pass, done."}}])
        self.assertEqual(M.compute(p)["unverified_done"], 0)
        p2 = log(self.tmp / "e.log", base + [tool("edit", {"filePath": "/w/a.py"}), {"type": "text", "part": {"text": "Готово, всё работает."}}])
        self.assertEqual(M.compute(p2)["unverified_done"], 1)
        p3 = log(self.tmp / "f.log", base + [tool("edit", {"filePath": "/w/a.py"}), {"type": "text", "part": {"text": "I could not finish."}}])
        self.assertEqual(M.compute(p3)["unverified_done"], 0, "no claim, no violation")

    def test_recovery_cost_after_a_compaction(self):
        p = log(self.tmp / "g.log", [
            tool("read", {"filePath": "/w/a.py"}), tool("read", {"filePath": "/w/b.py"}),
            {"type": "reasoning", "part": {"text": "Let me analyze the conversation and summarize"}},
            tool("read", {"filePath": "/w/a.py"}), tool("bash", {"command": "ls"}), tool("read", {"filePath": "/w/c.py"}),
            tool("edit", {"filePath": "/w/a.py"})])
        m = M.compute(p)
        self.assertEqual(m["compactions"], 1)
        self.assertEqual(m["recovery"], [{"calls_to_next_edit": 4, "rereads": 1}])
        self.assertEqual(m["recovery_calls_total"], 4)

    def test_start_rule_regressions_and_exact_contract_come_from_the_grade(self):
        tail = 'noise\n{"score": 0.5, "passed": false, "checks": 4, "ok": 2, "note": "", "parts": {"regression": {"unexpected_failures": ["tests.test_rb.T.test_x"]}, "rule": {"rc": 1}, "exact_contract": {"failed": ["test_exact_message"]}}}\n'
        g = M.from_grade(M.grade_parts(tail))
        self.assertEqual(g, {"regressions": ["tests.test_rb.T.test_x"], "start_rule_violation": 1, "exact_contract_failed": ["test_exact_message"]})
        self.assertEqual(M.from_grade(M.grade_parts("no json here")), {})
        rule = self.tmp / "rule.py"
        rule.write_text("import sys; sys.exit(0)\n")
        self.assertEqual(M.compute(log(self.tmp / "h.log", []), work=self.tmp, rule=str(rule))["start_rule_violation"], 0)
        p4 = log(self.tmp / "i.log", [])
        self.assertEqual(M.compute(p4, pre={"a": True, "b": True}, post={"a": True, "b": False})["regressions"], ["b"])


if __name__ == "__main__":
    unittest.main()
