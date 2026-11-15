"""QTTS1p: best-of-N without a GPU or a model. A stand-in agent writes the attempt's tree by a script (solution, a broken tree, its
own tests); the selection must follow the rule, the hidden tests must not take part in it, N = 1 must give harness.run_one's
record byte for byte, and the seeds must reach the (fake) llama-server, one per attempt.

    cd bench/agentic-eval && python3 -B -m unittest test_bestof
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bestof  # noqa: E402
import harness  # noqa: E402
import proxy as proxy_mod  # noqa: E402
import run_experiment  # noqa: E402

FAKE = Path(os.environ.get("SHURA_QA", "/home/alisher/Projects/shura-qa")) / "fake_llama" / "fake_llama_server.py"
TASKS = {t["id"]: t for t in harness.all_tasks(HERE / "tasks_v3")}


def shm():
    return "/dev/shm" if Path("/dev/shm").is_dir() else None


def scripted(script, by_prompt):
    """A runner whose i-th call (per task) does script[i]: "solve" copies the solution, "break" writes a wrong module, "solve+own"
    also adds a passing test file of the agent's own, "own-fail" adds a failing one, "nothing" leaves the start tree."""
    calls = {}

    def runner(work, prompt, cfg, data, timeout, log_path):
        task = by_prompt[prompt]
        i = calls.get(task["id"], 0)
        calls[task["id"]] = i + 1
        what = script[i % len(script)]
        Path(log_path).write_text('{"type":"text"}\n')
        work = Path(work)
        if what.startswith("solve"):
            shutil.copytree(task["dir"] / "solution", work, dirs_exist_ok=True)
        if what == "break":
            for f in work.glob("*.py"):
                if not f.name.startswith(("public_", "test")):
                    f.write_text("raise RuntimeError('broken by the attempt')\n")
            for f in work.glob("*.js"):
                if not f.name.endswith(".test.js"):
                    f.write_text("throw new Error('broken by the attempt')\n")
        if what.endswith("+own"):
            if task["lang"] == "python":
                (work / "test_own.py").write_text("import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n"
                                                  "        self.assertTrue(True)\n    def test_b(self):\n        self.assertTrue(True)\n")
            else:
                (work / "own.test.js").write_text("const t = require('node:test'); const a = require('node:assert');\n"
                                                  "t('a', () => a.ok(true)); t('b', () => a.ok(true));\n")
        if what == "own-fail":
            if task["lang"] == "python":
                (work / "test_own.py").write_text("import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n"
                                                  "        self.fail('x')\n")
            else:
                (work / "own.test.js").write_text("const t = require('node:test'); const a = require('node:assert');\n"
                                                  "t('a', () => a.ok(false));\n")
        return 0, False, 0.25
    return runner


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="bestof-", dir=shm()))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.by_prompt = {t["prompt"]: t for t in TASKS.values()}
        self.proxy_log = self.tmp / "proxy.jsonl"

    def run_bo(self, task, script, n, seed=7, arm="bo"):
        return bestof.run_best_of(task, arm, 0, 1, self.proxy_log, self.tmp / "logs", n=n, seed=seed,
                                  scratch_root=str(self.tmp / "scratch"), runner=scripted(script, self.by_prompt))


class Rule(Base):
    def test_visible_tests_pick_the_fixed_attempt_over_the_broken_one(self):
        for tid in ("py_account-bugfix", "js_account-bugfix"):
            rec = self.run_bo(TASKS[tid], ["break", "solve"], 2)
            bo = rec["best_of"]
            self.assertEqual((bo["chosen"], bo["reason"]), (1, "visible"), (tid, bo))
            self.assertTrue(rec["resolved"])
            self.assertTrue(bo["attempts"][1]["visible"]["ok"])
            self.assertFalse(bo["attempts"][0]["visible"]["ok"])
            self.assertEqual(bo["first_resolved"], False)
            self.assertTrue(bo["oracle_resolved"])

    def test_an_edited_repro_does_not_flatter_the_attempt(self):
        task = TASKS["py_account-bugfix"]

        def cheat(work, prompt, cfg, data, timeout, log_path):   # the attempt deletes the public repro and leaves the bug
            Path(log_path).write_text("")
            (Path(work) / "public_repro.py").write_text("import unittest\nclass T(unittest.TestCase):\n    def test_ok(self):\n"
                                                        "        pass\n")
            return 0, False, 0.1
        rec = bestof.run_best_of(task, "bo", 0, 1, self.proxy_log, self.tmp / "logs", n=2, seed=1,
                                 scratch_root=str(self.tmp / "scratch"), runner=cheat)
        self.assertFalse(rec["best_of"]["attempts"][0]["visible"]["ok"])   # the original repro ran, not the rewritten one
        self.assertFalse(rec["resolved"])

    def test_without_visible_tests_the_agents_own_tests_decide(self):
        for tid in ("py_account-implement", "js_account-implement"):
            task = TASKS[tid]
            self.assertEqual(bestof.visible_files(task), [], tid)
            rec = self.run_bo(task, ["nothing", "solve+own", "solve"], 3)
            bo = rec["best_of"]
            self.assertEqual((bo["chosen"], bo["reason"]), (1, "own_pass"), (tid, bo))
            self.assertFalse(bo["attempts"][0]["visible"]["has"])
            self.assertTrue(bo["attempts"][1]["own"]["ok"])
            self.assertEqual(bo["attempts"][1]["own"]["total"], 2)
            self.assertFalse(bo["attempts"][2]["own"]["has"])

    def test_failing_own_tests_rank_below_passing_ones_and_no_signal_means_the_first(self):
        task = TASKS["py_account-refactor"]
        rec = self.run_bo(task, ["own-fail", "solve+own"], 2)
        self.assertEqual(rec["best_of"]["chosen"], 1)
        rec = self.run_bo(task, ["nothing", "nothing", "nothing", "nothing"], 4)
        self.assertEqual((rec["best_of"]["chosen"], rec["best_of"]["reason"]), (0, "first"))

    def test_select_is_pure_and_never_reads_the_hidden_result(self):
        attempts = [{"index": 0, "visible": {"has": False}, "own": {"has": False}, "hidden": {"resolved": False}},
                    {"index": 1, "visible": {"has": False}, "own": {"has": False}, "hidden": {"resolved": True}}]
        self.assertEqual(bestof.select(attempts), (0, "first"))       # the hidden pass of attempt 1 must not move the choice
        attempts[0]["hidden"], attempts[1]["hidden"] = attempts[1]["hidden"], attempts[0]["hidden"]
        self.assertEqual(bestof.select(attempts), (0, "first"))

    def test_the_price_is_summed_over_attempts_and_seeds_differ(self):
        rec = self.run_bo(TASKS["py_account-implement"], ["nothing", "solve"], 4, seed=3)
        bo = rec["best_of"]
        self.assertEqual(len(set(bo["seeds"])), 4)
        self.assertAlmostEqual(rec["seconds"], sum(a["seconds"] for a in bo["attempts"]), places=1)
        self.assertEqual(rec["tokens"], sum(a["tokens"] for a in bo["attempts"]))
        again = self.run_bo(TASKS["py_account-implement"], ["nothing", "solve"], 4, seed=3)
        self.assertEqual(again["best_of"]["seeds"], bo["seeds"])       # a rerun gets the same seeds
        other = self.run_bo(TASKS["py_account-implement"], ["nothing", "solve"], 4, seed=4)
        self.assertNotEqual(other["best_of"]["seeds"], bo["seeds"])

    def test_no_scratch_is_left_behind(self):
        self.run_bo(TASKS["py_account-implement"], ["solve"], 2)
        self.assertEqual(list((self.tmp / "scratch").iterdir()), [])


class NOne(Base):
    def test_n_1_is_run_one_byte_for_byte(self):
        task = TASKS["py_account-bugfix"]
        a = bestof.run_best_of(task, "x", 0, 1, self.proxy_log, self.tmp / "logs", n=1, seed=9,
                               scratch_root=str(self.tmp / "s1"), runner=scripted(["solve"], self.by_prompt))
        b = harness.run_one(task, "x", 0, 1, self.proxy_log, self.tmp / "logs", scratch_root=str(self.tmp / "s2"),
                            runner=scripted(["solve"], self.by_prompt))
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertNotIn("best_of", a)

    def test_n_2_keeps_run_ones_fields_and_adds_best_of(self):
        task = TASKS["py_account-bugfix"]
        one = harness.run_one(task, "x", 0, 1, self.proxy_log, self.tmp / "logs", scratch_root=str(self.tmp / "s2"),
                              runner=scripted(["solve"], self.by_prompt))
        two = self.run_bo(task, ["solve"], 2)
        self.assertEqual(set(one) | {"best_of"}, set(two))

    def test_the_experiment_runner_routes_by_the_arm_field(self):
        ids = ["py_account-bugfix", "js_account-bugfix"]
        res = run_experiment.run_arm_round({"name": "bo2", "model": "x", "best_of": 2}, ids, TASKS, reps=1, parallel=1,
                                           server_cfg={}, out_dir=str(self.tmp / "exp"), base_port=1, seed=5,
                                           runner=scripted(["break", "solve"], self.by_prompt),
                                           server_starter=lambda *a: object(), server_stopper=lambda s: None)
        self.assertEqual([r["best_of"]["chosen"] for r in res], [1, 1])
        plain = run_experiment.run_arm_round({"name": "bo1", "model": "x"}, ids, TASKS, reps=1, parallel=1, server_cfg={},
                                             out_dir=str(self.tmp / "exp1"), base_port=1, seed=5,
                                             runner=scripted(["solve"], self.by_prompt),
                                             server_starter=lambda *a: object(), server_stopper=lambda s: None)
        self.assertTrue(all("best_of" not in r for r in plain))


class Summary(unittest.TestCase):
    def test_summary_reports_solved_price_and_how_often_not_the_first(self):
        runs = []
        for t in ("a", "b", "c", "d"):
            runs.append({"arm": "n1", "task": t, "rep": 0, "resolved": t in "ab", "seconds": 100, "tokens": 1000})
            runs.append({"arm": "n4", "task": t, "rep": 0, "resolved": t in "abc", "seconds": 380, "tokens": 3900,
                         "best_of": {"n": 4, "chosen": 2 if t == "c" else 0, "reason": "visible" if t == "c" else "first",
                                     "first_resolved": t in "ab", "oracle_resolved": t in "abc"}})
        res = bestof.summary(runs)
        self.assertEqual(res["control"], "n1")
        n4 = res["arms"]["n4"]
        self.assertEqual((n4["n"], n4["tasks"], n4["solved"]), (4, 4, 0.75))
        self.assertEqual(n4["not_first"], 0.25)
        self.assertEqual(n4["by_reason"], {"first": 3, "visible": 1})
        self.assertAlmostEqual(n4["seconds_x"], 3.8)
        self.assertAlmostEqual(n4["tokens_x"], 3.9)
        self.assertEqual((n4["first_attempt_solved"], n4["oracle_solved"]), (0.5, 0.75))
        text = bestof.summary_text(res)
        self.assertIn("| n4 | 4 | 4 | 75.0%", text)


@unittest.skipUnless(FAKE.exists(), "the shura-qa fake llama-server is not installed")
class FakeServer(Base):
    """The real chain: attempt proxy -> arm proxy -> fake llama-server. The stand-in agent sends one chat request per attempt
    through the engine's base URL; the fake must receive one request per attempt, each with its own seed, and the attempts'
    tokens must be counted apart."""

    def test_each_attempt_reaches_the_server_with_its_own_seed(self):
        work = self.tmp / "fake"
        work.mkdir()
        cfgf = work / "fake.json"
        cfgf.write_text(json.dumps({"n_tokens": 7, "token_delay_ms": 1, "load_seconds": 0}))
        import socket
        s = socket.socket(); s.bind(("127.0.0.1", 0)); fport = s.getsockname()[1]; s.close()
        proc = subprocess.Popen([sys.executable, "-B", str(FAKE), "--host", "127.0.0.1", "--port", str(fport), "-m", "fake.gguf"],
                                env={**os.environ, "FAKE_LLAMA_CONFIG": str(cfgf)}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        self.addCleanup(lambda: (proc.terminate(), proc.wait(10)))
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{fport}/health", timeout=1).read()
                break
            except OSError:
                time.sleep(0.2)
        arm = proxy_mod.serve(0, f"http://127.0.0.1:{fport}", str(self.proxy_log), {"chat_template_kwargs": {"terse": False}})
        threading.Thread(target=arm.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (arm.shutdown(), arm.server_close()))

        def agent(work, prompt, cfg, data, timeout, log_path):
            conf = json.loads((Path(cfg) / "shuracode.json").read_text())
            url = conf["provider"]["eval"]["options"]["baseURL"] + "/chat/completions"
            body = json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}], "stream": False}).encode()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                self.assertEqual(r.status, 200)
                r.read()
            Path(log_path).write_text("")
            return 0, False, 0.1
        rec = bestof.run_best_of(TASKS["py_account-implement"], "bo3", 0, arm.server_port, self.proxy_log, self.tmp / "logs", n=3,
                                 seed=11, scratch_root=str(self.tmp / "scratch"), runner=agent)
        seen = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{fport}/__fake/log", timeout=5).read())
        posts = [e["body"] for e in seen if e.get("method") == "POST" and e.get("path", "").endswith("/chat/completions")]
        self.assertEqual(len(posts), 3)
        self.assertEqual([p.get("seed") for p in posts], rec["best_of"]["seeds"])
        self.assertTrue(all(p.get("chat_template_kwargs") == {"terse": False} for p in posts))   # the arm's own fields stay
        self.assertEqual([a["requests"] for a in rec["best_of"]["attempts"]], [1, 1, 1])
        self.assertEqual(rec["requests"], 3)
        self.assertEqual(rec["tokens"], sum(a["tokens"] for a in rec["best_of"]["attempts"]))
        self.assertGreater(rec["tokens"], 0)


if __name__ == "__main__":
    unittest.main()
