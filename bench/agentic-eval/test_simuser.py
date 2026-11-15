"""SIMUSER1: the question policy (fixed / sim / reject / none), the answer cache, the leak guard, the harness loop that applies it,
and the ga_ tasks that need a question.   cd bench/agentic-eval && python3 -B -m unittest test_simuser
No model and no network: the claude CLI call is replaced."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import simuser  # noqa: E402

REQUEST = {"id": "que_1", "questions": [{"header": "Format", "question": "Which  format?", "options": [{"label": "JSON", "description": "a"},
                                                                                                         {"label": "CSV", "description": "b"}]}]}


def make_task(root, intent="Use CSV with the header row.", hidden_secret="needle_value_77"):
    d = Path(root) / "t1"
    (d / "start").mkdir(parents=True)
    (d / "hidden").mkdir()
    (d / "start" / "data.txt").write_text("rows\n")
    (d / "hidden" / "check.py").write_text(f"assert 'x' == '{hidden_secret}'  # also {hidden_secret}\n")
    if intent is not None:
        (d / "hidden" / "user_intent.md").write_text(intent + "\n")
    return {"id": "t1-general-01", "dir": d, "prompt": "Export the data."}


def fake_cli(answer_json):
    calls = []

    def call(prompt, model, **kw):
        calls.append(prompt)
        return answer_json, {"cost_usd": 0.01}
    call.calls = calls
    return call


STUB_ENGINE = r"""#!/usr/bin/env python3
# a stand-in for `shuracode serve`: like the real engine, POST /session/s1/message returns only when the turn is over, and the turn
# is over when the pending question has been answered (or rejected)
import json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
state = {"asked": False, "answered": None, "done": threading.Event()}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, obj, code=200):
        raw = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def body(self):
        n = int(self.headers.get("Content-Length") or 0); return json.loads(self.rfile.read(n) or "null")
    def do_GET(self):
        if self.path == "/session": return self.send([])
        if self.path == "/question":
            return self.send([{"id": "que_1", "sessionID": "s1", "questions": [{"question": "Which zone?", "header": "Zone", "options": []}]}] if state["asked"] and not state["done"].is_set() else [])
        if self.path == "/session/s1/message":
            tool = {"type": "tool", "tool": "question", "state": {"status": "completed" if state["done"].is_set() else "running"}}
            parts = [{"type": "step-start"}, tool] + ([{"type": "step-finish"}] if state["done"].is_set() else [])
            return self.send([{"info": {"role": "assistant"}, "parts": parts}])
        self.send(None, 404)
    def do_POST(self):
        b = self.body()
        if self.path == "/session": return self.send({"id": "s1"})
        if self.path == "/session/s1/message":
            state["asked"] = True; state["done"].wait(120); return self.send({})
        if self.path.startswith("/question/que_1/"):
            state["answered"] = [self.path.rsplit("/", 1)[1], b]; state["done"].set(); return self.send(True)
        self.send(None, 404)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
"""


class ServeLoopAnswersWhileTheTurnIsOpen(unittest.TestCase):
    """The engine holds POST /session/{id}/message open until the turn ends and a pending question keeps the turn open: the harness must
    answer the question from another thread. (QSTAND1 did not: the run waited for its deadline.)"""

    def run_stub(self, mode, die=False):
        import os
        tmp = Path(tempfile.mkdtemp(prefix="simuser-stub-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        engine = tmp / "engine.py"
        engine.write_text(STUB_ENGINE.replace('state["done"].wait(120); return self.send({})', 'import os; os._exit(139)') if die else STUB_ENGINE)
        engine.chmod(0o755)
        work = tmp / "scratch" / "work"
        work.mkdir(parents=True)
        task = make_task(tmp / "task")
        env = {"AGENTIC_EVAL_NO_SANDBOX": "1", "AGENTIC_EVAL_QUESTIONS": mode}
        with mock.patch.dict(os.environ, env), mock.patch.object(harness, "ENGINE", str(engine)), mock.patch.object(harness, "QUESTION_WAIT_S", 0), \
                mock.patch.object(simuser, "FIXED_WAIT_S", 0):
            t0 = __import__("time").monotonic()
            with harness.question_policy(task, tmp / "out") as policy:
                policy.wait_s = 0
                result = harness.run_agent_serve(work, "do it", tmp / "cfg", tmp / "data", 60, tmp / "agent.log", settle_s=0.5)
            took = __import__("time").monotonic() - t0
        return result, took, policy, (tmp / "agent.log.question.jsonl").read_text()

    def test_a_fixed_reply_ends_the_turn_long_before_the_deadline(self):
        (code, timed_out, secs), took, policy, qlog = self.run_stub("fixed")
        self.assertFalse(timed_out)
        self.assertLess(took, 30)
        self.assertEqual(policy.stats()["questions"], 1)
        self.assertIn('"reply_status": 200', qlog)

    def test_an_abort_ends_a_run_that_waits_for_a_dead_server_at_once(self):
        # RSI1: the llama-server crashed, the agent waits for an answer that never comes; the stand's abort check ends the run, not the deadline
        import time
        with harness.run_abort(lambda: True):
            t0 = time.monotonic()
            (code, timed_out, secs), took, policy, qlog = self.run_stub("none")
        self.assertLess(took, 30)
        self.assertFalse(timed_out)
        self.assertIsNone(code)

    def test_an_engine_that_dies_during_the_turn_is_reported_as_an_engine_failure(self):
        (code, timed_out, secs), took, policy, qlog = self.run_stub("none", die=True)
        self.assertEqual(code, 1)
        self.assertIsNotNone(harness.engine_failure())
        self.assertLess(took, 30)

    def test_a_healthy_run_reports_no_engine_failure(self):
        self.run_stub("fixed")
        self.assertIsNone(harness.engine_failure())

    def test_a_reject_ends_the_turn_too(self):
        (code, timed_out, secs), took, policy, qlog = self.run_stub("reject")
        self.assertFalse(timed_out)
        self.assertEqual(policy.events[-1]["source"], "reject")


class Policy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.task = make_task(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_default_is_the_fixed_qstand1_reply_after_five_seconds(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("AGENTIC_EVAL_QUESTIONS", None)
            p = simuser.QuestionPolicy(self.task)
        self.assertEqual((p.mode, p.wait_s), ("fixed", 5))
        self.assertEqual(p.answer(REQUEST), ("reply", [[simuser.QUESTION_TEXT]]))
        self.assertEqual(harness.QUESTION_TEXT, simuser.QUESTION_TEXT)       # QLEV1b's text is not changed

    def test_bad_mode_is_refused(self):
        with self.assertRaises(ValueError):
            simuser.QuestionPolicy(self.task, mode="maybe")

    def test_reject_and_none(self):
        self.assertEqual(simuser.QuestionPolicy(self.task, mode="reject").answer(REQUEST), ("reject", None))
        self.assertEqual(simuser.QuestionPolicy(self.task, mode="none").answer(REQUEST), ("none", None))

    def test_sim_answers_from_the_model_and_the_cache_serves_the_second_arm(self):
        cache = Path(self.tmp.name) / "cache"
        cli = fake_cli('{"answers": [["CSV"]]}')
        with mock.patch.object(simuser, "call_cli", cli):
            a = simuser.QuestionPolicy(self.task, mode="sim", cache_dir=cache)
            self.assertEqual(a.answer(REQUEST), ("reply", [["CSV"]]))
            again = {**REQUEST, "id": "que_9", "questions": [{**REQUEST["questions"][0], "question": "which format?"}]}   # whitespace, case, another id
            b = simuser.QuestionPolicy(self.task, mode="sim", cache_dir=cache)
            self.assertEqual(b.answer(again), ("reply", [["CSV"]]))
        self.assertEqual(len(cli.calls), 1)
        self.assertEqual(b.events[-1]["source"], "cache")
        self.assertEqual(b.stats()["questions"], 1)
        prompt = cli.calls[0]
        self.assertIn("Use CSV with the header row.", prompt)                     # the intent and the task text
        self.assertIn("Export the data.", prompt)
        self.assertNotIn("needle_value_77", prompt)                                 # never the hidden tests

    def test_the_cache_is_per_task(self):
        cache = Path(self.tmp.name) / "cache"
        other = {**self.task, "id": "t2-general-01"}
        cli = fake_cli('{"answers": [["JSON"]]}')
        with mock.patch.object(simuser, "call_cli", cli):
            simuser.QuestionPolicy(self.task, mode="sim", cache_dir=cache).answer(REQUEST)
            simuser.QuestionPolicy(other, mode="sim", cache_dir=cache).answer(REQUEST)
        self.assertEqual(len(cli.calls), 2)

    def test_a_leaked_hidden_literal_is_replaced_and_logged(self):
        cli = fake_cli('{"answers": [["Print needle_value_77 in the third column"]]}')
        with mock.patch.object(simuser, "call_cli", cli):
            p = simuser.QuestionPolicy(self.task, mode="sim", cache_dir=Path(self.tmp.name) / "c")
            verdict, answers = p.answer(REQUEST)
        self.assertEqual(answers, [[simuser.DONT_KNOW["en"]]])
        self.assertEqual(p.events[-1]["source"], "leak-guard")
        self.assertIn("needle_value_77", p.events[-1]["leaks"])
        self.assertEqual(p.stats()["question_leaks"], 1)

    def test_a_literal_the_task_text_or_the_intent_already_gives_is_no_leak(self):
        task = make_task(self.tmp.name + "/b", intent="Use needle_value_77 as the marker.")
        cli = fake_cli('{"answers": [["Use needle_value_77"]]}')
        with mock.patch.object(simuser, "call_cli", cli):
            p = simuser.QuestionPolicy(task, mode="sim")
            self.assertEqual(p.answer(REQUEST), ("reply", [["Use needle_value_77"]]))

    def test_a_broken_model_reply_or_a_failed_cli_means_dont_know_and_is_not_cached(self):
        cache = Path(self.tmp.name) / "cache"
        with mock.patch.object(simuser, "call_cli", fake_cli("no json here")):
            p = simuser.QuestionPolicy(self.task, mode="sim", cache_dir=cache)
            self.assertEqual(p.answer(REQUEST)[1], [[simuser.DONT_KNOW["en"]]])
        with mock.patch.object(simuser, "call_cli", side_effect=RuntimeError("limit")):
            q = simuser.QuestionPolicy(self.task, mode="sim", cache_dir=cache)
            self.assertEqual(q.answer(REQUEST)[1], [[simuser.DONT_KNOW["en"]]])
            self.assertTrue(q.events[-1]["source"].startswith("model-failed"))
        self.assertEqual(list(cache.glob("*.json")), [])                              # neither case is cached

    def test_the_wrong_number_of_answers_is_not_accepted(self):
        self.assertIsNone(simuser.parse_answers('{"answers": [["a"], ["b"]]}', 1))
        self.assertEqual(simuser.parse_answers('text {"answers": ["a", ["b", "c"]]} text', 2), [["a"], ["b", "c"]])

    def test_the_language_of_the_dont_know_follows_the_task(self):
        task = {**self.task, "prompt": "Выгрузи данные в файл."}
        with mock.patch.object(simuser, "call_cli", fake_cli("no json")):
            p = simuser.QuestionPolicy(task, mode="sim")
            self.assertEqual(p.answer(REQUEST)[1], [[simuser.DONT_KNOW["ru"]]])


class HarnessLoop(unittest.TestCase):
    """answer_questions against a fake engine API."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.task = make_task(self.tmp.name)
        self.posts = []
        self.pending = [dict(REQUEST)]

        def call(method, path, body=None, timeout=10):
            if method == "GET":
                return 200, list(self.pending)
            self.posts.append((path, body))
            return 200, True
        self.call = call
        self.log = open(Path(self.tmp.name) / "q.jsonl", "w", encoding="utf-8")

    def tearDown(self):
        self.log.close()
        self.tmp.cleanup()

    def lines(self):
        self.log.flush()
        return [json.loads(x) for x in (Path(self.tmp.name) / "q.jsonl").read_text().splitlines()]

    def test_no_policy_is_the_old_qstand1_behaviour(self):
        seen = {}
        harness.answer_questions(self.call, self.log, seen)
        self.assertEqual(self.posts, [])                                          # not before five seconds
        seen["que_1"][0] -= 6
        harness.answer_questions(self.call, self.log, seen)
        self.assertEqual(self.posts, [("/question/que_1/reply", {"answers": [[harness.QUESTION_TEXT]]})])
        harness.answer_questions(self.call, self.log, seen)
        self.assertEqual(len(self.posts), 1)                                      # answered once

    def test_sim_answers_at_once_and_logs_the_source(self):
        p = simuser.QuestionPolicy(self.task, mode="sim")
        with mock.patch.object(simuser, "call_cli", fake_cli('{"answers": [["CSV"]]}')):
            harness.answer_questions(self.call, self.log, {}, p)
        self.assertEqual(self.posts, [("/question/que_1/reply", {"answers": [["CSV"]]})])
        row = self.lines()[0]
        self.assertEqual((row["source"], row["answers"], row["reply_status"]), ("model", [["CSV"]], 200))

    def test_reject_posts_a_reject(self):
        harness.answer_questions(self.call, self.log, {}, simuser.QuestionPolicy(self.task, mode="reject"))
        self.assertEqual(self.posts, [("/question/que_1/reject", {})])

    def test_none_leaves_the_question_pending_and_asks_the_policy_once(self):
        p = simuser.QuestionPolicy(self.task, mode="none")
        seen = {}
        for _ in range(3):
            harness.answer_questions(self.call, self.log, seen, p)
        self.assertEqual(self.posts, [])
        self.assertEqual(p.stats()["questions"], 1)


class Summary(unittest.TestCase):
    def test_questions_summary_splits_asked_and_silent_runs(self):
        rows = [{"arm": "A", "resolved": True, "questions": 1, "simuser_cost_usd": 0.02}, {"arm": "A", "resolved": False, "questions": 0},
                {"arm": "A", "resolved": True, "questions": 0}, {"arm": "B", "resolved": False, "questions": 2, "question_leaks": 1}]
        s = simuser.questions_summary(rows)
        self.assertEqual((s["A"]["asked_runs"], s["A"]["resolved_rate_asked"], s["A"]["resolved_rate_silent"], s["A"]["cost_usd"]), (1, 1.0, 0.5, 0.02))
        self.assertEqual((s["B"]["questions"], s["B"]["leaks"], s["B"]["resolved_rate_silent"]), (2, 1, None))


class Usefulness(unittest.TestCase):
    def test_table_splits_tasks_that_need_a_question_from_the_rest_per_policy(self):
        fixed = [{"task": "ga_x-general-01", "arm": "A", "resolved": False, "score": 0.5, "questions": 1}, {"task": "gb_y-general-01", "arm": "A", "resolved": True, "score": 1.0}]
        sim = [{"task": "ga_x-general-01", "arm": "A", "resolved": True, "score": 1.0, "questions": 1, "simuser_cost_usd": 0.004}]
        t = simuser.usefulness({"fixed": fixed, "sim": sim}, {"ga_x-general-01"})
        self.assertEqual(t[("fixed", "A", "need")]["score"], 0.5)
        self.assertEqual((t[("sim", "A", "need")]["resolved"], t[("sim", "A", "need")]["cost_usd"]), (1, 0.004))
        self.assertEqual(t[("fixed", "A", "other")]["asked"], 0)


class AskTasks(unittest.TestCase):
    """The ga_ (clarify) tasks: the prompt must not carry the answer, the intent must, and the leak guard must have something to guard."""

    @classmethod
    def setUpClass(cls):
        cls.tasks = [harness.load_task(p.parent) for p in sorted((HERE / "tasks_general").glob("ga_*/*/task.json"))]

    def test_there_are_eight_that_need_a_question(self):
        self.assertEqual(len(self.tasks), 8)
        for t in self.tasks:
            self.assertTrue(t.get("needs_question"), t["id"])
            self.assertTrue((t["dir"] / "hidden" / "user_intent.md").is_file(), t["id"])

    def test_no_prompt_tells_the_agent_to_ask(self):
        for t in self.tasks:
            self.assertNotRegex(t["prompt"].lower(), r"\bask\b|спроси|уточни", t["id"])

    def test_the_intent_is_not_in_the_prompt_or_the_start_files(self):
        for t in self.tasks:
            intent = (t["dir"] / "hidden" / "user_intent.md").read_text(encoding="utf-8").strip().lower()
            visible = (t["prompt"] + "\n" + simuser.read_tree(t["dir"] / "start")).lower()
            self.assertNotIn(intent, visible, t["id"])

    def test_the_leak_guard_has_secrets_to_guard_but_never_the_intent_itself(self):
        for t in self.tasks:
            intent = (t["dir"] / "hidden" / "user_intent.md").read_text(encoding="utf-8").strip()
            guard = simuser.LeakGuard(t, intent)
            self.assertEqual(guard.leaks(intent), [], t["id"])                      # what the user may say is never a leak


if __name__ == "__main__":
    unittest.main()
