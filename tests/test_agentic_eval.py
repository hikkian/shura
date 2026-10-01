"""The measuring proxy and the statistics of the model comparison (bench/agentic-eval). Offline, no GPU, no model."""
import json
import shutil
import socketserver
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench" / "agentic-eval"))
import analyze  # noqa: E402
import harness  # noqa: E402
import overnight  # noqa: E402
import safety  # noqa: E402
import proxy  # noqa: E402
import run_experiment  # noqa: E402
import validate_tasks  # noqa: E402


class Quick(HTTPServer):
    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


class FakeLlama(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        req = json.loads(raw)
        FakeLlama.seen.append(req)
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            events = [b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n',
                      b'data: {"choices":[{"delta":{"content":"llo"}}],"usage":{"prompt_tokens":11,"completion_tokens":2},'
                      b'"timings":{"predicted_per_second":40.5}}\n\n', b"data: [DONE]\n\n"]
            for e in events:
                self.wfile.write(f"{len(e):x}\r\n".encode() + e + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        else:
            out = json.dumps({"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)


class Proxy(unittest.TestCase):
    def setUp(self):
        FakeLlama.seen.clear()
        self.up = Quick(("127.0.0.1", 0), FakeLlama)
        threading.Thread(target=self.up.serve_forever, daemon=True).start()
        self.tmp = tempfile.TemporaryDirectory()
        self.log = str(Path(self.tmp.name) / "run.jsonl")
        self.px = proxy.serve(0, f"http://127.0.0.1:{self.up.server_port}", self.log,
                              {"chat_template_kwargs": {"terse": False}})
        threading.Thread(target=self.px.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.px.server_port}"

    def tearDown(self):
        self.px.shutdown()
        self.px.server_close()
        self.up.shutdown()
        self.up.server_close()
        self.tmp.cleanup()

    def post(self, body):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read()

    def test_spelling_extra_fields_and_usage_request_reach_the_model(self):
        self.post({"model": "m", "messages": [], "reasoningEffort": "medium", "stream": True})
        seen = FakeLlama.seen[0]
        self.assertEqual(seen["reasoning_effort"], "medium")
        self.assertNotIn("reasoningEffort", seen)
        self.assertIs(seen["chat_template_kwargs"]["terse"], False)
        self.assertTrue(seen["stream_options"]["include_usage"])

    def wait_logged(self, n=1):
        import time
        for _ in range(100):                                  # the proxy logs after it has sent the last chunk
            if proxy.totals(self.log)["requests"] >= n:
                break
            time.sleep(0.05)
        return proxy.totals(self.log)

    def test_a_stream_arrives_complete_and_its_usage_is_logged(self):
        data = self.post({"model": "m", "messages": [], "stream": True})
        self.assertIn(b"llo", data)
        self.assertIn(b"[DONE]", data)
        t = self.wait_logged()
        self.assertEqual((t["requests"], t["prompt_tokens"], t["completion_tokens"], t["errors"]), (1, 11, 2, 0))

    def test_a_plain_response_and_other_paths_pass_through(self):
        out = json.loads(self.post({"model": "m", "messages": []}))
        self.assertEqual(out["choices"][0]["message"]["content"], "hi")
        with urllib.request.urlopen(self.base + "/health", timeout=20) as r:
            self.assertEqual(json.load(r)["status"], "ok")
        self.assertEqual(self.wait_logged()["requests"], 1)                    # only completions are counted

    def test_a_dead_upstream_is_a_502_not_a_hang(self):
        self.up.shutdown()
        self.up.server_close()
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post({"model": "m", "messages": []})
        self.assertEqual(cm.exception.code, 502)

    def test_usage_scanner_handles_json_sse_and_garbage(self):
        self.assertEqual(proxy.scan_usage(b'{"usage":{"completion_tokens":4}}')["usage"]["completion_tokens"], 4)
        self.assertEqual(proxy.scan_usage(b"data: nonsense\n\ndata: [DONE]\n"), {})
        self.assertEqual(proxy.scan_usage(b""), {})


def runs_for(shifts, tasks=40, reps=2, seed=1):
    import random
    rng = random.Random(seed)
    out = []
    for t in range(tasks):
        p = rng.uniform(0.15, 0.85)
        for arm, d in shifts.items():
            for r in range(reps):
                out.append({"arm": arm, "task": f"t{t}", "rep": r, "resolved": rng.random() < min(0.99, max(0.01, p + d)),
                            "tokens": 10_000 + (0 if arm == "cheap" else 5_000), "seconds": 1.0})
    return out


class Statistics(unittest.TestCase):
    def test_an_obvious_winner_is_declared(self):
        res = analyze.compare(runs_for({"A": 0.30, "B": 0.0, "C": 0.0}, tasks=60), flips=4000, boot=800)
        self.assertEqual(analyze.verdict(res)["kind"], "winner")
        self.assertEqual(analyze.verdict(res)["arm"], "A")

    def test_equal_arms_are_a_tie_and_the_cheaper_one_wins_the_tie_break(self):
        runs = runs_for({"cheap": 0.0, "dear": 0.0}, tasks=40, seed=3)
        # make the two arms IDENTICAL in outcomes, so that the quality verdict is a tie by construction
        by = {(r["task"], r["rep"]): r["resolved"] for r in runs if r["arm"] == "cheap"}
        for r in runs:
            r["resolved"] = by[(r["task"], r["rep"])]
        text, res, ver = analyze.report(runs, flips=2000, boot=500)
        self.assertEqual(ver["kind"], "tie")
        self.assertIn("Tie-break", text)
        self.assertIn("cheap", text.split("Tie-break")[1].splitlines()[0])

    def test_holm_is_monotone_and_never_below_the_raw_p(self):
        adj = analyze.holm({"x": 0.01, "y": 0.04, "z": 0.03})
        self.assertAlmostEqual(adj["x"], 0.03)
        self.assertGreaterEqual(adj["z"], adj["x"])
        self.assertTrue(all(adj[k] >= p for k, p in {"x": 0.01, "y": 0.04, "z": 0.03}.items()))

    def test_repetitions_of_one_task_are_averaged_before_testing(self):
        runs = [{"arm": a, "task": "t", "rep": r, "resolved": a == "A", "tokens": 1, "seconds": 1}
                for a in ("A", "B") for r in range(5)]
        self.assertEqual(analyze.per_task_scores(runs), {"A": {"t": 1.0}, "B": {"t": 0.0}})

    def test_an_incomplete_design_is_an_error_not_a_silent_gap(self):
        with self.assertRaises(ValueError):
            analyze.per_task_scores([{"arm": "A", "task": "t1", "resolved": True},
                                     {"arm": "B", "task": "t2", "resolved": True}])

    def test_the_sign_flip_test_does_not_see_a_difference_that_is_not_there(self):
        self.assertEqual(analyze.sign_flip_p([0.0, 0.0, 0.0]), 1.0)
        self.assertGreater(analyze.sign_flip_p([0.5, -0.5, 0.5, -0.5, 0.5, -0.5], flips=4000), 0.5)
        self.assertLess(analyze.sign_flip_p([0.5] * 12, flips=4000), 0.01)


class Tasks(unittest.TestCase):
    """The shipped task set must be valid, otherwise the experiment measures the tasks and not the models."""

    def test_every_shipped_task_is_valid(self):
        tasks = harness.all_tasks()
        self.assertGreaterEqual(len(tasks), 2)
        bad = {t["id"]: validate_tasks.check(t) for t in tasks}
        self.assertEqual({k: v for k, v in bad.items() if v}, {})

    def test_ids_are_unique_and_every_family_has_both_variants(self):
        tasks = harness.all_tasks()
        ids = [t["id"] for t in tasks]
        self.assertEqual(len(ids), len(set(ids)))
        by_family = {}
        for t in tasks:
            by_family.setdefault(t["family"], set()).add(t["kind"])
        self.assertTrue(all(k == {"implement", "bugfix"} for k in by_family.values()))

    def test_prompts_never_leak_hidden_material(self):
        for t in harness.all_tasks():
            for p in (t["dir"] / "hidden").rglob("*.py"):
                self.assertNotIn(p.name, t["prompt"])


@unittest.skipUnless(shutil.which("bwrap"), "bubblewrap is not installed")
class Sandbox(unittest.TestCase):
    """An unattended agent must not be able to touch anything outside its scratch folder."""

    def run_in(self, script):
        import subprocess
        with tempfile.TemporaryDirectory(dir="/dev/shm") as scratch:
            (Path(scratch) / "work").mkdir()
            (Path(scratch) / "home").mkdir()
            argv = harness.sandboxed(["sh", "-c", script], scratch, Path(scratch) / "work")
            r = subprocess.run(argv, capture_output=True, text=True, timeout=60)
            return r, sorted(p.name for p in (Path(scratch) / "work").iterdir())

    def test_scratch_is_writable_and_the_system_is_not(self):
        r, files = self.run_in("echo hi > ok.txt; touch /usr/x; touch /etc/x; touch /opt/x; echo end")
        self.assertEqual(files, ["ok.txt"])
        self.assertIn("Read-only", r.stderr)
        self.assertIn("end", r.stdout)

    def test_the_users_home_and_environment_are_invisible(self):
        marker = Path.home() / ".agentic_eval_sandbox_probe"
        marker.write_text("secret")
        try:
            r, _ = self.run_in(f"cat {marker} 2>&1; echo \"token=$GITHUB_TOKEN$ANTHROPIC_API_KEY\"; ls -a {Path.home()} 2>&1 | head -3")
        finally:
            marker.unlink()
        self.assertNotIn("secret", r.stdout)
        self.assertIn("token=", r.stdout)
        self.assertNotIn(".ssh", r.stdout)

    def test_a_process_cannot_outlive_the_sandbox_and_sees_no_host_processes(self):
        r, _ = self.run_in("ps -e 2>/dev/null | wc -l")
        self.assertLess(int(r.stdout.strip() or 0), 20)


class Plumbing(unittest.TestCase):
    """The runner end to end with a stand-in agent: no model, no GPU. The stand-in copies the reference solution with a
    probability that depends on the arm, so the statistics must see the difference that was put in."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tasks = {t["id"]: t for t in harness.all_tasks()}
        self.by_prompt = {t["prompt"]: t for t in self.tasks.values()}

    def tearDown(self):
        self.tmp.cleanup()

    def agent(self, p_solve, seed):
        import random
        import shutil
        rng = random.Random(seed)

        def runner(work, prompt, cfg, data, timeout, log_path):
            Path(log_path).write_text('{"type":"text"}\n')
            if rng.random() < p_solve:
                shutil.copytree(self.by_prompt[prompt]["dir"] / "solution", work, dirs_exist_ok=True)
            return 0, False, 0.5
        return runner

    def test_rounds_are_a_seeded_partition(self):
        a = run_experiment.make_rounds(list(self.tasks.values()), 5, seed=1)
        b = run_experiment.make_rounds(list(self.tasks.values()), 5, seed=1)
        c = run_experiment.make_rounds(list(self.tasks.values()), 5, seed=2)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(sorted(sum(a, [])), sorted(self.tasks))

    def test_one_arm_round_writes_a_result_per_run(self):
        ids = sorted(self.tasks)[:4]
        arm = {"name": "good", "model": "x.gguf"}
        res = run_experiment.run_arm_round(arm, ids, self.tasks, reps=2, parallel=2, server_cfg={}, out_dir=self.tmp.name,
                                           base_port=1, seed=1, runner=self.agent(1.0, 1),
                                           server_starter=lambda *a: object(), server_stopper=lambda s: None)
        self.assertEqual(len(res), 8)
        self.assertTrue(all(r["resolved"] for r in res))
        self.assertEqual(len((Path(self.tmp.name) / "results.jsonl").read_text().splitlines()), 8)
        self.assertEqual({r["arm"] for r in res}, {"good"})

    def test_a_do_nothing_agent_resolves_nothing(self):
        ids = sorted(self.tasks)[:4]
        res = run_experiment.run_arm_round({"name": "idle", "model": "x"}, ids, self.tasks, 1, 2, {}, self.tmp.name, 1, 1,
                                           runner=self.agent(0.0, 1), server_starter=lambda *a: object(),
                                           server_stopper=lambda s: None)
        self.assertFalse(any(r["resolved"] for r in res))

    def test_the_statistics_see_the_difference_that_was_put_in(self):
        ids = sorted(self.tasks)
        runs = []
        for name, p in (("strong", 0.95), ("weak", 0.15)):
            runs += run_experiment.run_arm_round({"name": name, "model": "x"}, ids, self.tasks, 1, 4, {}, self.tmp.name, 1, 3,
                                                 runner=self.agent(p, 11), server_starter=lambda *a: object(),
                                                 server_stopper=lambda s: None)
        text, res, ver = analyze.report(runs, flips=4000, boot=800)
        self.assertEqual(ver["kind"], "winner")
        self.assertEqual(ver["arm"], "strong")

    def test_the_hidden_tests_are_not_visible_to_the_agent(self):
        seen = []

        def runner(work, prompt, cfg, data, timeout, log_path):
            seen.append(sorted(p.name for p in Path(work).rglob("*") if p.is_file()))
            Path(log_path).write_text("")
            return 0, False, 0.1
        t = self.tasks[sorted(self.tasks)[0]]
        harness.run_one(t, "a", 0, 1, Path(self.tmp.name) / "p.jsonl", self.tmp.name, runner=runner)
        self.assertFalse(any(n.startswith("test_") and "hidden" in n for n in seen[0]))
        self.assertNotIn("__init__.py", [n for n in seen[0] if "hidden" in n])
        self.assertFalse(any(p.name == "hidden_tests" for p in Path(self.tmp.name).rglob("*")))

    def test_interim_does_not_stop_early_on_little_data_or_ties(self):
        stop, _ = run_experiment.interim([{"arm": "a", "task": "t", "rep": 0, "resolved": True, "tokens": 1, "seconds": 1},
                                          {"arm": "b", "task": "t", "rep": 0, "resolved": False, "tokens": 1, "seconds": 1}],
                                         ["a", "b"])
        self.assertFalse(stop)


class Rehearsal(unittest.TestCase):
    """The whole night, with a stand-in for the agent and for the server: every step runs, nothing touches the GPU or the system."""

    def test_the_night_end_to_end(self):
        import random
        import shutil
        from unittest import mock
        rng = random.Random(5)
        tasks = {t["prompt"]: t for t in harness.all_tasks()}

        def agent(work, prompt, cfg, data, timeout, log_path):
            Path(log_path).write_text("")
            if rng.random() < 0.85:
                shutil.copytree(tasks[prompt]["dir"] / "solution", work, dirs_exist_ok=True)
            return 0, False, 0.2

        class Proc:
            def poll(self):
                return None
        with tempfile.TemporaryDirectory(dir="/dev/shm") as d:
            d = Path(d)
            eval_dir = d / "eval"
            eval_dir.mkdir()
            tiel = d / "tiel.gguf"
            tiel.write_bytes(b"x")
            for n in ("Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "occamy-1.0-IQ4_XS.gguf"):
                (eval_dir / n).write_bytes(b"x")
            calls = []
            patches = [mock.patch.object(overnight, "TIEL", tiel), mock.patch.object(overnight, "EVAL_DIR", eval_dir),
                       mock.patch.object(overnight, "RUNNER", agent), mock.patch.object(overnight, "wait_for_free_gpu", return_value=True),
                       mock.patch.object(overnight, "plumbing_ok", return_value=True),
                       mock.patch.object(overnight, "shura", side_effect=lambda c: calls.append(c) or True),
                       mock.patch.object(overnight.rx, "start_server", return_value=Proc()),
                       mock.patch.object(overnight.rx, "stop_server", return_value=None),
                       mock.patch.object(harness, "ENGINE", sys.executable)]
            for p in patches:
                p.start()
            try:
                code = overnight.main(["--out", str(d / "out"), "--max-hours", "0.2", "--reps", "1", "--keep-models", "--min-disk-gb", "1"])
            finally:
                for p in patches:
                    p.stop()
            out = d / "out"
            self.assertEqual(code, 0, (out / "overnight.log").read_text()[-2000:])
            self.assertEqual(json.loads((out / "status.json").read_text())["state"], "done")
            self.assertTrue((out / "MORNING.md").exists())
            self.assertIn("off", calls)
            self.assertEqual(calls[-1], "on")                         # the gateway is always switched back on
            runs = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
            self.assertEqual({r["arm"] for r in runs}, {"tiel", "tiel-no-terse", "base", "occamy"})
            self.assertEqual(len(runs), 4 * len(tasks))
            self.assertEqual(sum(1 for r in runs if r["arm"] == "base"), len(tasks))

    def test_a_failure_still_puts_the_machine_back(self):
        from unittest import mock
        calls = []
        with tempfile.TemporaryDirectory(dir="/dev/shm") as d:
            with mock.patch.object(overnight, "wait_for_free_gpu", return_value=False), \
                    mock.patch.object(overnight, "shura", side_effect=lambda c: calls.append(c) or True), \
                    mock.patch.object(harness, "ENGINE", sys.executable):
                code = overnight.main(["--out", d + "/out", "--max-hours", "0.01"])
            self.assertEqual(code, 1)
            self.assertIn("не закончился", (Path(d) / "out" / "MORNING.md").read_text())
            self.assertEqual(json.loads((Path(d) / "out" / "status.json").read_text())["state"], "failed")


class Night(unittest.TestCase):
    def test_watchdog_trips_on_heat_low_vram_low_ram_and_full_disk_but_not_on_one_bad_reading(self):
        def run(readings, limits=None):
            it = iter(readings)
            last = [readings[-1]]

            def reader():
                try:
                    last[0] = next(it)
                except StopIteration:
                    pass
                return last[0]
            w = safety.Watchdog("/dev/shm", limits={"hot_seconds": 0.05, **(limits or {})}, period=0.01, gpu=False, reader=reader)
            w.start()
            import time
            time.sleep(0.4)
            w.stop()
            return w
        ok = {"gpu_c": 60, "cpu_c": 50, "vram_free": 900, "vram_used": 5000, "ram": 8.0, "disk": 400.0}
        self.assertFalse(run([ok] * 5).tripped.is_set())
        self.assertFalse(run([ok, {**ok, "vram_free": 10}, ok, ok]).tripped.is_set())          # a single low reading is tolerated
        self.assertIn("VRAM", run([{**ok, "vram_free": 10}] * 5).reason)
        self.assertIn("RAM", run([{**ok, "ram": 0.5}] * 5).reason)
        self.assertIn("disk", run([{**ok, "disk": 3.0}] * 5).reason)
        self.assertIn("hot", run([{**ok, "gpu_c": 90}] * 100).reason)
        self.assertFalse(run([{**ok, "gpu_c": 90}, ok, ok, ok]).tripped.is_set())               # a short spike is tolerated

    def test_cleanup_never_touches_a_file_that_was_not_downloaded_for_the_test(self):
        with tempfile.TemporaryDirectory() as d:
            users = Path(d) / "tiel.gguf"
            users.write_bytes(b"x")
            eval_dir = Path(d) / "eval"
            eval_dir.mkdir()
            base, occ = eval_dir / "base.gguf", eval_dir / "occ.gguf"
            base.write_bytes(b"x")
            occ.write_bytes(b"x")
            arms = [{"name": "tiel", "model": str(users)}, {"name": "tiel-no-terse", "model": str(users)},
                    {"name": "base", "model": str(base)}, {"name": "occamy", "model": str(occ)}]
            old = overnight.EVAL_DIR
            overnight.EVAL_DIR = eval_dir
            try:
                removed, kept = overnight.cleanup_models(d, "base", arms)
            finally:
                overnight.EVAL_DIR = old
            self.assertTrue(users.exists())                       # the user's own model is never deleted
            self.assertTrue(base.exists())                        # the winner stays
            self.assertFalse(occ.exists())                        # the loser goes
            self.assertEqual(removed, [str(occ)])
            self.assertEqual(kept, [str(base)])

    def test_a_tiel_win_removes_both_downloads(self):
        with tempfile.TemporaryDirectory() as d:
            users = Path(d) / "tiel.gguf"
            users.write_bytes(b"x")
            eval_dir = Path(d) / "eval"
            eval_dir.mkdir()
            files = {n: eval_dir / f"{n}.gguf" for n in ("base", "occ")}
            for f in files.values():
                f.write_bytes(b"x")
            arms = [{"name": "tiel", "model": str(users)}, {"name": "base", "model": str(files["base"])},
                    {"name": "occamy", "model": str(files["occ"])}]
            old = overnight.EVAL_DIR
            overnight.EVAL_DIR = eval_dir
            try:
                overnight.cleanup_models(d, "tiel", arms)
            finally:
                overnight.EVAL_DIR = old
            self.assertTrue(users.exists())
            self.assertFalse(any(f.exists() for f in files.values()))

    def test_decide_uses_only_cells_every_arm_finished(self):
        runs = []
        for t in range(30):
            for arm, p in (("a", 1), ("b", 0)):
                runs.append({"arm": arm, "task": f"t{t}", "rep": 0, "resolved": bool(p) and t % 3 != 0, "tokens": 10, "seconds": 1.0})
        runs.append({"arm": "a", "task": "t99", "rep": 0, "resolved": True, "tokens": 10, "seconds": 1.0})      # arm b never ran it
        keep, text, ver = overnight.decide_and_clean("/nonexistent", runs, [])
        self.assertEqual(keep, "a")
        self.assertEqual(ver["kind"], "winner")

    def test_one_arm_is_not_a_comparison(self):
        keep, text, ver = overnight.decide_and_clean("/x", [{"arm": "a", "task": "t", "rep": 0, "resolved": True, "tokens": 1, "seconds": 1}], [])
        self.assertIsNone(keep)
        self.assertIn("fewer than two", text)

    def test_server_layouts_are_tried_from_the_best_down(self):
        self.assertEqual(overnight.LAYOUTS[0], (4, 24))
        self.assertTrue(all(p >= 1 for p, _ in overnight.LAYOUTS))
        cfg = overnight.server_cfg(16)
        self.assertIn("--moe-cache-slots", cfg["args"])
        self.assertNotIn("--spec-type", cfg["args"])                    # MTP is off for every arm: the same conditions
        self.assertNotIn("--moe-cache-slots", overnight.server_cfg(0)["args"])
        self.assertEqual(cfg["prefix"], ["nice", "-n", "10"])


if __name__ == "__main__":
    unittest.main()
