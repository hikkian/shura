"""RSIL0 tests (offline: the editors are stub programs, no model is called).
Run: cd bench/agentic-eval && python3 -m unittest discover -s rsi -p 'test_rsi.py'

The leak test is the card's requirement: trying to read the hidden tests from an editor's folder must fail."""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rsi import accept, budget, driver, editors, sandbox  # noqa: E402


def script(path, body):
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rsi-sb-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_hidden_tests_cannot_be_copied_into_an_editor_folder(self):
        src = self.tmp / "task" / "hidden"
        src.mkdir(parents=True)
        (src / "check.py").write_text("secret\n")
        with self.assertRaises(sandbox.LeakError):
            sandbox.build(self.tmp / "editor", {"hidden_tests": src})

    def test_solutions_and_verdicts_are_refused_by_name(self):
        for name in ("solution/answer.json", "verdicts/v1.md", "dev/part.json", "test/part.json", "splits.json"):
            with self.assertRaises(sandbox.LeakError, msg=name):
                sandbox.build(self.tmp / "editor", {name: self.tmp / "x.txt"})

    def test_a_folder_that_holds_hidden_files_is_not_clean(self):
        dest = self.tmp / "editor"
        dest.mkdir()
        (dest / "hidden").mkdir()
        (dest / "hidden" / "check.py").write_text("x")
        with self.assertRaises(sandbox.LeakError):
            sandbox.assert_clean(dest)

    def test_the_allowed_material_is_copied_and_the_folder_is_clean(self):
        src = self.tmp / "trajectories"
        src.mkdir()
        (src / "t1.jsonl").write_text("{}\n")
        out = sandbox.build(self.tmp / "editor", {"trajectories": src})
        self.assertTrue((out / "trajectories" / "t1.jsonl").exists())
        sandbox.assert_clean(out)

    def test_a_source_on_dev_shm_is_not_taken_for_the_dev_split(self):
        src = self.tmp / "ok.txt"
        src.write_text("fine")
        out = sandbox.build(self.tmp / "editor", {"ok.txt": src})
        self.assertTrue((out / "ok.txt").exists())


class EditorCalls(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rsi-ed-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.saved = (editors.CLAUDE_BIN, editors.CODEX_BIN)
        self.addCleanup(lambda: setattr(editors, "CLAUDE_BIN", self.saved[0]) or setattr(editors, "CODEX_BIN", self.saved[1]))

    def test_a_claude_reply_is_parsed_with_its_usage_and_cost(self):
        claude = script(self.tmp / "claude", (
            "import json,sys\n"
            "print(json.dumps({'result':'OK','is_error':False,'total_cost_usd':0.12,"
            "'usage':{'input_tokens':3,'cache_creation_input_tokens':100,'cache_read_input_tokens':0,'output_tokens':2}}))\n"))
        editors.CLAUDE_BIN = str(claude)
        log = self.tmp / "usage.jsonl"
        r = editors.call("sonnet", "reflector", "hi", cwd=self.tmp, usage_log=log, allow_fallback=False)
        self.assertTrue(r.ok)
        self.assertEqual((r.text, r.tokens_in, r.tokens_out, r.cost_usd), ("OK", 103, 2, 0.12))
        row = json.loads(log.read_text().splitlines()[0])
        self.assertEqual((row["model"], row["role"]), ("claude-sonnet-5-5", "reflector"))

    def test_a_limit_error_moves_the_call_to_the_fallback_and_logs_both(self):
        claude = script(self.tmp / "claude", (
            "import json\nprint(json.dumps({'result':'Claude usage limit reached, try later','is_error':True}))\nraise SystemExit(1)\n"))
        codex = script(self.tmp / "codex", (
            "import sys\nargs=sys.argv\nout=args[args.index('-o')+1]\n"
            "open(out,'w').write('OK from sol')\nsys.stderr.write('tokens used\\n1,234\\n')\n"))
        editors.CLAUDE_BIN, editors.CODEX_BIN = str(claude), str(codex)
        log = self.tmp / "usage.jsonl"
        r = editors.call("opus", "invariant", "hi", cwd=self.tmp, usage_log=log)
        self.assertTrue(r.ok)
        self.assertEqual((r.text, r.switched_from, r.model), ("OK from sol", "opus", "gpt-6.1-sol"))
        rows = [json.loads(x) for x in log.read_text().splitlines()]
        self.assertEqual([x["model"] for x in rows], ["claude-opus-5-5", "gpt-6.1-sol"])
        self.assertEqual(rows[1]["tokens_out"], 1234)

    def counting_claude(self, replies):
        """A stub claude that answers the n-th call with replies[n] (a result text, or None = hang until killed)."""
        state = self.tmp / "count"
        state.write_text("0")
        return script(self.tmp / "claude", (
            "import json,sys,time\n"
            f"p={str(state)!r}\nn=int(open(p).read()); open(p,'w').write(str(n+1))\n"
            f"r={replies!r}[min(n,{len(replies) - 1})]\n"
            "sys.stdin.read()\n"
            "if r is None: time.sleep(30)\n"
            "print(json.dumps({'result':r,'is_error':False,'usage':{'input_tokens':1,'output_tokens':1}}))\n"))

    def test_the_editor_has_its_own_system_prompt_and_no_tools(self):
        cmd = editors._claude_cmd(editors.EDITORS["sonnet"])
        self.assertEqual(cmd[cmd.index("--tools") + 1], "")
        self.assertEqual(cmd[cmd.index("--system-prompt") + 1], editors.EDITOR_SYSTEM)
        self.assertIn("no tools", editors.EDITOR_SYSTEM)

    def test_tool_call_markup_instead_of_text_is_repeated_once(self):
        editors.CLAUDE_BIN = str(self.counting_claude(['<invoke name="Read"><parameter name="file_path">x</parameter></invoke>', "```diff\nok\n```"]))
        log = self.tmp / "usage.jsonl"
        r = editors.call("sonnet", "reflector", "hi", cwd=self.tmp, usage_log=log, allow_fallback=False)
        self.assertEqual((r.ok, r.text, r.extra.get("retry")), (True, "```diff\nok\n```", "tool markup"))
        self.assertEqual(len(log.read_text().splitlines()), 2)              # both attempts are in the usage log

    def test_a_hung_call_is_killed_at_the_timeout_and_repeated(self):
        editors.CLAUDE_BIN = str(self.counting_claude([None, "second try"]))
        r = editors.call("sonnet", "reflector", "hi", cwd=self.tmp, timeout=2, allow_fallback=False)
        self.assertEqual((r.ok, r.text, r.extra.get("retry")), (True, "second try", "timeout"))

    def test_a_reply_that_keeps_failing_is_returned_after_the_retries_not_hidden(self):
        editors.CLAUDE_BIN = str(self.counting_claude(["<invoke name='Read'></invoke>"]))
        r = editors.call("sonnet", "reflector", "hi", cwd=self.tmp, allow_fallback=False)
        self.assertEqual(r.extra.get("retry"), "tool markup")
        self.assertIn("<invoke", r.text)                                    # the caller sees it and finds no diff


class PromptOnStdin(unittest.TestCase):
    """RSIR1 D1: an argument is limited to 131 072 bytes, so the prompt goes on stdin. A 300 000-byte prompt must reach the editor."""

    def test_a_300000_byte_prompt_reaches_the_editor_on_stdin(self):
        tmp = Path(tempfile.mkdtemp(prefix="rsi-stdin-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        claude = script(tmp / "claude", (
            "import json,sys\n"
            "p=sys.stdin.read()\n"
            "print(json.dumps({'result': str(len(p.encode('utf-8'))), 'is_error': False, 'total_cost_usd': 0.0, "
            "'usage': {'input_tokens': 1, 'output_tokens': 1}}))\n"))
        saved = editors.CLAUDE_BIN
        editors.CLAUDE_BIN = str(claude)
        self.addCleanup(lambda: setattr(editors, "CLAUDE_BIN", saved))
        r = editors.call("sonnet", "reflector", "é" * 150000, cwd=tmp, allow_fallback=False)   # 300 000 bytes in UTF-8
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.text, "300000")

    def test_the_claude_call_is_isolated_from_the_users_settings(self):
        cmd = editors._claude_cmd(editors.EDITORS["opus"])
        self.assertIn("--setting-sources", cmd)
        self.assertEqual(cmd[cmd.index("--setting-sources") + 1], "project,local")
        self.assertIn("--exclude-dynamic-system-prompt-sections", cmd)
        self.assertNotIn("300000", " ".join(cmd))

    def test_the_codex_call_reads_the_prompt_from_stdin_and_ignores_the_user_config(self):
        cmd = editors._codex_cmd(editors.EDITORS["sol"], Path("/tmp/x.txt"))
        self.assertEqual(cmd[-1], "-")
        self.assertIn("--ignore-user-config", cmd)
        self.assertIn("--ignore-rules", cmd)


class BudgetAndAcceptance(unittest.TestCase):
    def test_a_small_addition_fits_and_a_large_one_does_not(self):
        self.assertTrue(budget.check("short rule\n")["ok"])
        self.assertFalse(budget.check("x" * 20000)["ok"])

    def test_acceptance_needs_an_explicit_go_past_the_stub_thresholds(self):
        base = [{"task": "a", "family": "f", "score": 0.5, "resolved": False, "seconds": 10, "tokens": 100}]
        cand = [dict(base[0], score=0.9, resolved=True)]
        with self.assertRaises(RuntimeError):
            accept.decide(cand, base)
        verdict = accept.decide(cand, base, allow_stub=True)
        self.assertTrue(verdict["stub"])

    def test_a_family_regression_blocks_acceptance(self):
        base = [{"task": f"{f}-{i}", "family": f, "score": 0.5, "resolved": True, "seconds": 10, "tokens": 100}
                for f in ("a", "b", "c") for i in range(2)]
        cand = [dict(r, score=0.9) if r["family"] != "c" else dict(r, score=0.1) for r in base]
        self.assertFalse(accept.decide(cand, base, allow_stub=True)["accept"])


class DiffRoundTrip(unittest.TestCase):
    def test_the_diff_applies_to_a_copy_and_the_original_is_untouched(self):
        tmp = Path(tempfile.mkdtemp(prefix="rsi-diff-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "AGENTS.md").write_text("one\ntwo\n")
        text = "```diff\n--- a/AGENTS.md\n+++ b/AGENTS.md\n@@ -1,2 +1,2 @@\n one\n-two\n+two, edited\n```"
        copy = tmp / "copy"
        shutil.copytree(tmp, copy, ignore=shutil.ignore_patterns("copy"))
        self.assertTrue(driver.apply_diff(copy, driver.extract_diff(text)))
        self.assertIn("two, edited", (copy / "AGENTS.md").read_text())
        self.assertEqual((tmp / "AGENTS.md").read_text(), "one\ntwo\n")


class HunkCounts(unittest.TestCase):
    """RSIL0e E2: a hunk whose counts were written by hand is recounted from its body (RSIR2 §1: `@@ -0,0 +1,34 @@` over 32 added lines)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rsi-hunk-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.copy = self.tmp / "copy"
        self.copy.mkdir()

    def test_a_new_file_with_a_wrong_count_is_refused_raw_and_created_after_the_fix(self):
        body = "".join(f"+line {n}\n" for n in range(1, 33))
        text = "--- /dev/null\n+++ b/NOTES.md\n@@ -0,0 +1,34 @@\n" + body
        raw = self.tmp / "raw.diff"
        raw.write_text(text)
        self.assertNotEqual(subprocess.run(["git", "apply", "--check", str(raw)], cwd=self.copy, capture_output=True).returncode, 0)
        self.assertTrue(driver.apply_diff(self.copy, text))
        self.assertEqual((self.copy / "NOTES.md").read_text(), "".join(f"line {n}\n" for n in range(1, 33)))

    def test_a_wrong_count_in_a_modifying_hunk_is_fixed(self):
        (self.copy / "AGENTS.md").write_text("one\ntwo\nthree\n")
        text = "--- a/AGENTS.md\n+++ b/AGENTS.md\n@@ -1,9 +1,9 @@\n one\n-two\n+two, edited\n three\n"
        self.assertTrue(driver.apply_diff(self.copy, text))
        self.assertEqual((self.copy / "AGENTS.md").read_text(), "one\ntwo, edited\nthree\n")

    def test_a_correct_diff_comes_back_unchanged(self):
        good = "--- a/AGENTS.md\n+++ b/AGENTS.md\n@@ -1,2 +1,2 @@\n one\n-two\n+two, edited\n"
        self.assertEqual(driver.normalize_hunks(good), good)
        self.assertEqual(driver.normalize_hunks(driver.normalize_hunks(good)), good)

    def test_a_blank_context_line_counts_and_the_next_file_closes_the_hunk(self):
        text = ("--- a/A.md\n+++ b/A.md\n@@ -1,2 +1,5 @@\n one\n\n-two\n+TWO\n"
                "--- a/B.md\n+++ b/B.md\n@@ -1,1 +1,1 @@\n-x\n+y\n")
        fixed = driver.normalize_hunks(text)
        self.assertEqual(fixed.count("@@ -1,3 +1,3 @@"), 1)
        self.assertIn("@@ -1,1 +1,1 @@", fixed)
        self.assertIn("--- a/B.md", fixed)


class CycleSteps(unittest.TestCase):
    """propose / compose / admit with a stub editor: the candidate is saved, the composer may keep the base, and a too-big edit is refused."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rsi-cycle-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.saved = (editors.CLAUDE_BIN, driver.RUNS, driver.USAGE)
        self.addCleanup(self._restore)
        driver.RUNS = self.tmp / "runs"
        driver.USAGE = self.tmp / "usage.jsonl"
        self.harness = self.tmp / "harness"
        self.harness.mkdir()
        (self.harness / "AGENTS.md").write_text("one\ntwo\n")
        self.run_dir = self.tmp / "run"
        (self.run_dir / "agent").mkdir(parents=True)
        (self.run_dir / "agent" / "t1.agent.log").write_text('{"type":"text","part":{"text":"done"}}\n')

    def _restore(self):
        editors.CLAUDE_BIN, driver.RUNS, driver.USAGE = self.saved

    def stub(self, text):
        body = "import json,sys\nprint(json.dumps({'result': " + repr(text) + ", 'is_error': False, 'total_cost_usd': 0.0, "
        body += "'usage': {'input_tokens': 1, 'output_tokens': 1}}))\n"
        editors.CLAUDE_BIN = str(script(self.tmp / "claude", body))

    def test_propose_saves_the_candidate_and_logs_the_call(self):
        self.stub("```diff\n--- a/AGENTS.md\n+++ b/AGENTS.md\n@@ -1,2 +1,2 @@\n one\n-two\n+two, checked\n```")
        diff = driver.propose("r1", "invariant", self.harness, self.run_dir)
        self.assertIn("two, checked", diff)
        self.assertTrue((driver.RUNS / "r1" / "candidates" / "invariant.diff").exists())
        self.assertEqual(len(driver.USAGE.read_text().splitlines()), 1)

    def test_compose_may_keep_the_base_with_an_empty_answer(self):
        self.stub("I keep the base: neither candidate helps.")
        (driver.RUNS / "r2" / "candidates").mkdir(parents=True)
        self.assertIsNone(driver.compose("r2", self.harness, {"dev": "no change"}))
        self.assertFalse((driver.RUNS / "r2" / "candidates" / "composed.diff").exists())

    def test_admit_refuses_an_edit_over_the_budget_and_accepts_a_measured_gain_only_with_a_go(self):
        big = "--- a/AGENTS.md\n+++ b/AGENTS.md\n@@ -1,2 +1,2 @@\n one\n-two\n+" + ("word " * 20000) + "\n"
        self.assertFalse(driver.admit("r3", self.harness, big, [], [])["accept"])
        small = "--- a/AGENTS.md\n+++ b/AGENTS.md\n@@ -1,2 +1,2 @@\n one\n-two\n+two, checked\n"
        base = [{"task": "a", "family": "f", "score": 0.5, "resolved": False, "seconds": 10, "tokens": 100}]
        cand = [dict(base[0], score=0.9, resolved=True)]
        with self.assertRaises(RuntimeError):
            driver.admit("r3", self.harness, small, cand, base)
        self.assertTrue(driver.admit("r3", self.harness, small, cand, base, allow_stub=True)["accept"])


if __name__ == "__main__":
    unittest.main()


class FindClaudeBinary(unittest.TestCase):
    """The preflight of RSI1 found the claude path pinned to an extension version that no longer existed."""

    def test_env_wins_then_path_then_the_newest_extension_build(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as home:
            for v in ("2.1.9", "2.1.10", "2.0.99"):
                b = Path(home) / ".vscode" / "extensions" / f"anthropic.claude-code-{v}-linux-x64" / "resources" / "native-binary" / "claude"
                b.parent.mkdir(parents=True)
                b.write_text("")
            nothing = lambda name: None
            self.assertEqual(editors.find_claude({"RSI_CLAUDE_BIN": "/x/claude"}, home, nothing), "/x/claude")
            self.assertEqual(editors.find_claude({}, home, lambda name: "/usr/bin/claude"), "/usr/bin/claude")
            newest = editors.find_claude({}, home, nothing)
            self.assertIn("claude-code-2.1.10-", newest)         # 2.1.10 is newer than 2.1.9 (numbers, not strings)
            self.assertEqual(editors.find_claude({}, tempfile.gettempdir() + "/no-such-home", nothing), "claude")
