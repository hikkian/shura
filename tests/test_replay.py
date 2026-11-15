import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench/agentic-eval"))
from replay import hidden_from_rev, replay_log, replay_safe_shell, replay_safe_shell_detail, tree_hash


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.work = self.root / "work"
        self.work.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def log(self, events):
        path = self.root / "run.agent.log"
        path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        return path

    @staticmethod
    def event(tool, state):
        return {"type": "tool_use", "part": {"tool": tool, "state": state}}

    def test_hidden_from_rev_exports_committed_files_not_worktree(self):
        import subprocess
        repo = self.root / "repo"
        hidden = repo / "t" / "hidden"
        hidden.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (hidden / "a.test.js").write_text("v1", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                        "commit", "-q", "-m", "v1"], check=True)
        (hidden / "a.test.js").write_text("v2", encoding="utf-8")   # work tree changed after the commit
        out = hidden_from_rev(hidden, "HEAD", self.root / "scratch")
        self.assertEqual((out / "a.test.js").read_text(encoding="utf-8"), "v1")
        self.assertNotEqual(tree_hash(out), tree_hash(hidden))

    def test_tree_hash_is_order_stable_and_tracks_files(self):
        (self.work / "b.txt").write_text("b")
        (self.work / "a.txt").write_text("a")
        first = tree_hash(self.work)
        (self.work / "__pycache__").mkdir()
        (self.work / "__pycache__/x.pyc").write_bytes(b"volatile")
        (self.work / "node_modules/.cache").mkdir(parents=True)
        (self.work / "node_modules/.cache/x").write_bytes(b"volatile")
        self.assertEqual(first, tree_hash(self.work))
        (self.work / "a.txt").write_text("changed")
        self.assertNotEqual(first, tree_hash(self.work))

    def test_replay_applies_logged_write_and_edit(self):
        (self.work / "src").mkdir()
        (self.work / "src/app.py").write_text("VALUE = 1\n")
        log = self.log([
            self.event("write", {"status": "completed", "input": {"filePath": "/old/session/work/new/file.txt", "content": "hello\n"}, "output": "Wrote file successfully."}),
            self.event("edit", {"status": "completed", "input": {"filePath": "src/app.py", "oldString": "VALUE = 1", "newString": "VALUE = 2"}, "output": "Edit applied successfully."}),
        ])
        stats = replay_log(log, self.work)
        self.assertEqual((stats["applied"], stats["rerun"], stats["skipped"]), (2, 0, 0))
        self.assertEqual((self.work / "new/file.txt").read_text(), "hello\n")
        self.assertEqual((self.work / "src/app.py").read_text(), "VALUE = 2\n")

    def test_edit_uses_recorded_diff_when_exact_old_text_has_drifted(self):
        path = self.work / "sample.js"
        path.write_text("test('case', () => {\n  assert.equal(actual, 1);\n});\n")
        patch = """Index: sample.js
--- sample.js
+++ sample.js
@@ -1,3 +1,3 @@
 test('case', () => {
-  assert.equal(actual, 0);
+  assert.equal(actual, 2);
 });
"""
        log = self.log([self.event("edit", {"status": "completed", "input": {
            "filePath": "sample.js", "oldString": "not present", "newString": "unused"
        }, "output": "Edit applied successfully.", "metadata": {"diff": patch}})])
        stats = replay_log(log, self.work)
        self.assertEqual(stats["applied"], 1)
        self.assertEqual(path.read_text(), "test('case', () => {\n  assert.equal(actual, 2);\n});\n")

    def test_replay_rejects_paths_outside_work(self):
        log = self.log([self.event("write", {"status": "completed", "input": {"filePath": "../outside", "content": "x"}, "output": "ok"})])
        stats = replay_log(log, self.work)
        self.assertEqual(stats["applied"], 0)
        self.assertEqual(len(stats["errors"]), 1)

    def test_shell_replays_confined_mkdir_and_remove_only(self):
        done, skipped, _ = replay_safe_shell("mkdir -p pkg/sub && rm -f pkg/sub/old.py && python3 -m unittest", self.work)
        self.assertEqual((done, skipped), (2, 1))
        self.assertTrue((self.work / "pkg/sub").is_dir())

    def test_shell_refuses_redirection_and_external_paths(self):
        done, skipped, _ = replay_safe_shell("cat > result.txt <<'EOF'\nunsafe\nEOF", self.work)
        self.assertEqual((done, skipped), (0, 1))
        done, skipped, _ = replay_safe_shell("rm -rf /etc", self.work)
        self.assertEqual((done, skipped), (0, 1))
        self.assertFalse((self.work / "result.txt").exists())

    def sh(self, command):
        r = replay_safe_shell_detail(command, self.work)
        return r["counts"]

    def test_shell_mv_cp_inside(self):
        (self.work / "a.txt").write_text("A")
        (self.work / "d").mkdir()
        (self.work / "d/x.txt").write_text("X")
        c = self.sh("mv -f a.txt b.txt && cp b.txt c.txt && cp -r d e && mv c.txt d")
        self.assertEqual((c["mv"], c["cp"], c["skipped"]), (2, 2, 0))
        self.assertFalse((self.work / "a.txt").exists())
        self.assertEqual((self.work / "b.txt").read_text(), "A")
        self.assertEqual((self.work / "e/x.txt").read_text(), "X")
        self.assertEqual((self.work / "d/c.txt").read_text(), "A")

    def test_shell_mv_outside_refused(self):
        (self.work / "a.txt").write_text("A")
        outside = self.root / "out.txt"
        c = self.sh(f"mv a.txt {outside}")
        self.assertEqual((c["mv"], c["skipped"]), (0, 1))
        c = self.sh("mv a.txt ../out.txt")
        self.assertEqual(c["skipped"], 1)
        self.assertTrue((self.work / "a.txt").exists())
        self.assertFalse(outside.exists())

    def test_shell_path_traversal_refused(self):
        c = self.sh("echo x > ../../etc/evil && mkdir -p ../../etc/z && cp ../../etc/passwd p")
        self.assertEqual((c["write"], c["mkdir"], c["cp"], c["skipped"]), (0, 0, 0, 3))

    def test_shell_symlink_outside_refused(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("S")
        (self.work / "link").symlink_to(outside)
        c = self.sh("cp link/secret.txt got.txt && echo hi > link/new.txt && mv link/secret.txt here.txt")
        self.assertEqual(c["skipped"], 3)
        self.assertFalse((outside / "new.txt").exists())
        self.assertTrue((outside / "secret.txt").exists())
        self.assertFalse((self.work / "got.txt").exists())

    def test_shell_sed_inplace(self):
        f = self.work / "f.txt"
        f.write_text("foo foo\nbar foo\nkeep\n")
        c = self.sh("sed -i 's/foo/baz/' f.txt")
        self.assertEqual((c["sed"], c["skipped"]), (1, 0))
        self.assertEqual(f.read_text(), "baz foo\nbar baz\nkeep\n")
        self.sh("sed -i -E 's/(a)z/\\1Z/g' f.txt")
        self.assertEqual(f.read_text(), "baZ foo\nbar baZ\nkeep\n")
        self.sh("sed -i '2s/bar/BAR/' f.txt")
        self.assertEqual(f.read_text(), "baZ foo\nBAR baZ\nkeep\n")
        self.sh("sed -i '/keep/d' f.txt")
        self.assertEqual(f.read_text(), "baZ foo\nBAR baZ\n")

    def test_shell_sed_unsafe_forms_skipped(self):
        f = self.work / "f.txt"
        f.write_text("foo\n")
        c = self.sh("sed -i 's/foo/a|b/' f.txt | cat")
        self.assertEqual((c["sed"], c["skipped"]), (0, 1))
        for cmd in ("sed -i.bak 's/foo/x/' f.txt", "sed 's/foo/x/' f.txt", "sed -i 'w /tmp/x' f.txt",
                    "sed -i 's/foo/x/' ../f.txt"):
            self.assertEqual(self.sh(cmd)["sed"], 0, cmd)
        self.assertEqual(f.read_text(), "foo\n")

    def test_shell_echo_printf_write_and_append(self):
        c = self.sh("echo 'hello world' > f.txt && echo more >> f.txt && printf '%s\\n' 'x y' >> f.txt && printf 'a\\tb\\n' > g.txt")
        self.assertEqual((c["write"], c["skipped"]), (4, 0))
        self.assertEqual((self.work / "f.txt").read_text(), "hello world\nmore\nx y\n")
        self.assertEqual((self.work / "g.txt").read_text(), "a\tb\n")
        c = self.sh("echo -n 'raw' > n.txt")
        self.assertEqual((self.work / "n.txt").read_text(), "raw")

    def test_shell_echo_text_is_not_executed(self):
        c = self.sh("echo '$(touch pwned); rm -rf x' > f.txt")
        self.assertEqual(c["write"], 1)
        self.assertEqual((self.work / "f.txt").read_text(), "$(touch pwned); rm -rf x\n")
        self.assertFalse((self.work / "pwned").exists())
        c = self.sh('''echo "$HOME" > h.txt && echo 'a\\nb' > i.txt && echo x 2> j.txt && echo x > k.txt > l.txt''')
        self.assertEqual((c["write"], c["skipped"]), (0, 4))

    def test_shell_heredoc_quoted_replayed_unquoted_skipped(self):
        c = self.sh("mkdir -p src && cat <<'EOF' > src/a.txt\nline $1 `x`\nsecond\nEOF\nrm -f nothing")
        self.assertEqual((c["write"], c["mkdir"], c["rm"], c["skipped"]), (1, 1, 1, 0))
        self.assertEqual((self.work / "src/a.txt").read_text(), "line $1 `x`\nsecond\n")
        c = self.sh("cat <<EOF > b.txt\nvalue $HOME\nEOF")
        self.assertEqual(c["write"], 0)
        self.assertGreaterEqual(c["skipped"], 1)
        self.assertFalse((self.work / "b.txt").exists())
        c = self.sh("cat <<'EOF' >> a.txt\nnever ends")
        self.assertEqual((c["write"], c["skipped"]), (0, 1))

    def test_shell_substitution_unknown_and_pipes_skipped(self):
        c = self.sh("mv $(echo a) b && echo `id` > f.txt && frobnicate --now && ls | wc -l && sleep 1 & && echo ok > ok.txt")
        self.assertEqual(c["write"], 0)          # a skipped clause fails the && chain: the last write does not run
        self.assertEqual(c["mv"], 0)
        self.assertGreaterEqual(c["skipped"], 4)
        c = self.sh("mv $(echo a) b; echo ok > ok.txt")
        self.assertEqual(c["write"], 1)          # after ; the next clause runs anyway
        c = self.sh("python3 run.py")
        self.assertEqual((c["skipped"], c["write"]), (1, 0))

    def test_replay_log_reports_shell_counts(self):
        log = self.log([self.event("bash", {"status": "completed", "input": {"command": "echo hi > x.txt && npm test"}})])
        stats = replay_log(log, self.work)
        self.assertEqual((stats["rerun"], stats["skipped"]), (1, 1))
        self.assertEqual((stats["shell"]["write"], stats["shell"]["skipped"]), (1, 1))

    def test_review_cp_of_tree_with_symlink_is_skipped(self):
        # a symlink in the replayed tree must not let copytree write outside the work folder
        (self.work / "a" / "b").mkdir(parents=True)
        (self.work / "a" / "b" / "l").symlink_to("../../..")
        r = replay_safe_shell_detail("mkdir -p q && cp -r a q/src && mkdir -p s2/src/l && echo x > s2/src/l/pwn && cp -r s2/src q", self.work)
        self.assertFalse((self.work.parent / "pwn").exists())
        self.assertFalse((self.work.parent.parent / "pwn").exists())
        self.assertGreaterEqual(r["skipped"], 1)

    def test_review_unknown_heredoc_body_is_not_run(self):
        # an unquoted heredoc is not replayed; its body must not run as commands either
        r = replay_safe_shell_detail("cat <<EOS > b\nmkdir -p pwned\nEOS\ncat <<'EOF' > c\nok\nEOF", self.work)
        self.assertFalse((self.work / "pwned").exists())
        self.assertGreaterEqual(r["skipped"], 1)

    def test_review_exponential_sed_regex_is_refused(self):
        (self.work / "f").write_text("a" * 40 + "c\n", encoding="utf-8")
        r = replay_safe_shell_detail("sed -E -i 's/(a+)+b/x/' f", self.work)
        self.assertEqual((self.work / "f").read_text(encoding="utf-8"), "a" * 40 + "c\n")
        self.assertGreaterEqual(r["skipped"], 1)

    def test_review_overflowing_regex_skips_only_that_clause(self):
        (self.work / "f").write_text("x\n", encoding="utf-8")
        r = replay_safe_shell_detail("sed -E -i 's/a{99999999999}/b/' f; mkdir -p ok", self.work)
        self.assertTrue((self.work / "ok").is_dir())
        self.assertGreaterEqual(r["skipped"], 1)

    def test_and_chain_stops_after_a_failed_clause(self):
        # cd to a missing folder fails in bash, so the mkdir after && must not run
        r = replay_safe_shell_detail("cd nodir && mkdir -p x", self.work)
        self.assertFalse((self.work / "x").exists())
        self.assertGreaterEqual(r["skipped"], 1)
        # rm of a missing file without -f fails, so the && chain stops
        replay_safe_shell_detail("rm nosuch && mkdir -p y", self.work)
        self.assertFalse((self.work / "y").exists())

    def test_or_runs_only_after_a_failure(self):
        replay_safe_shell_detail("rm nosuch || mkdir -p z", self.work)
        self.assertTrue((self.work / "z").is_dir())
        replay_safe_shell_detail("mkdir -p w || mkdir -p v", self.work)
        self.assertTrue((self.work / "w").is_dir())
        self.assertFalse((self.work / "v").exists())

    def test_semicolon_runs_after_a_failure(self):
        replay_safe_shell_detail("rm nosuch; mkdir -p u", self.work)
        self.assertTrue((self.work / "u").is_dir())

    def test_skipped_clause_counts_as_failed_for_and_chain(self):
        # the unsafe pipe clause is skipped, so the && clause after it does not run
        replay_safe_shell_detail("echo a | cat && mkdir -p t", self.work)
        self.assertFalse((self.work / "t").exists())

    def test_echo_option_e_is_not_replayed(self):
        r = replay_safe_shell_detail("echo -e 'a\\tb' > f", self.work)
        self.assertFalse((self.work / "f").exists())
        self.assertGreaterEqual(r["skipped"], 1)
        replay_safe_shell_detail("echo -n hi > g", self.work)
        self.assertEqual((self.work / "g").read_text(encoding="utf-8"), "hi")

    def test_rm_options_are_checked(self):
        (self.work / "d").mkdir()
        (self.work / "d" / "f").write_text("x", encoding="utf-8")
        r = replay_safe_shell_detail("rm --help d", self.work)
        self.assertTrue((self.work / "d").exists())
        self.assertGreaterEqual(r["skipped"], 1)
        replay_safe_shell_detail("rm --recursive d", self.work)
        self.assertFalse((self.work / "d").exists())
        (self.work / "e").mkdir()
        replay_safe_shell_detail("rm -rf e", self.work)
        self.assertFalse((self.work / "e").exists())
        replay_safe_shell_detail("rm -f nosuch && mkdir -p ok", self.work)
        self.assertTrue((self.work / "ok").is_dir())

    def test_incomplete_tool_call_is_skipped(self):
        log = self.log([self.event("write", {"status": "running", "input": {"filePath": "x", "content": "x"}})])
        stats = replay_log(log, self.work)
        self.assertEqual((stats["applied"], stats["skipped"]), (0, 1))
        self.assertFalse((self.work / "x").exists())


if __name__ == "__main__":
    unittest.main()


# ------------------------------------------------------------------------------------------------- FORK1b: continue from step k

import http.client  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

import fork  # noqa: E402

OLD_WORK = "/dev/shm/agentic-eval/py_x-implement-base-0-ab12cd/work"


def _step_events(i, tools, text="", reasoning="", finish="tool-calls", tokens=None):
    ev = [{"type": "step_start", "part": {"type": "step-start", "messageID": f"m{i}"}}]
    if reasoning:
        ev.append({"type": "reasoning", "part": {"type": "reasoning", "text": reasoning}})
    if text:
        ev.append({"type": "text", "part": {"type": "text", "text": text}})
    for tid, name, inp in tools:
        ev.append({"type": "tool_use", "part": {"type": "tool", "callID": tid, "tool": name,
                                                "state": {"status": "completed", "input": inp, "output": "ok"}}})
    ev.append({"type": "step_finish", "part": {"type": "step-finish", "reason": finish,
                                               "tokens": tokens or {"input": 100 * (i + 1), "output": 10, "reasoning": 0,
                                                                    "cache": {"read": 0, "write": 0}}}})
    return ev


def _record(root):
    events = [{"type": "user", "text": "do it"}]
    events += _step_events(0, [("c0", "bash", {"command": f"ls {OLD_WORK}"})], reasoning="look first")
    events += _step_events(1, [("c1", "write", {"filePath": f"{OLD_WORK}/a.py", "content": "x = 1\n"})], text="writing")
    events += [{"type": "user", "text": ""}]
    events += _step_events(2, [], text="## Objective\nsummary", finish="stop",
                           tokens={"input": 30000, "output": 50, "reasoning": 0, "cache": {"read": 0, "write": 0}})
    events += [{"type": "user", "text": "Continue if you have next steps"}]
    events += _step_events(3, [("c3", "bash", {"command": "python3 a.py"})])
    events += _step_events(4, [], text="done", finish="stop")
    p = Path(root) / "rec.agent.log"
    p.write_text("".join(json.dumps(e) + "\n" for e in events))
    return p


def _post(port, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, data


def _sse_message(data):
    """Rebuild the assistant message from an SSE answer, the way an OpenAI-compatible client does."""
    msg = {"content": "", "reasoning": "", "tool_calls": {}}
    finish, usage = None, None
    for line in data.decode().splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        obj = json.loads(line[6:])
        usage = obj.get("usage") or usage
        ch = obj["choices"][0]
        finish = ch.get("finish_reason") or finish
        d = ch.get("delta") or {}
        msg["content"] += d.get("content") or ""
        msg["reasoning"] += d.get("reasoning_content") or ""
        for tc in d.get("tool_calls") or []:
            slot = msg["tool_calls"].setdefault(tc["index"], {"id": tc.get("id"), "name": "", "arguments": ""})
            slot["name"] += tc["function"].get("name") or ""
            slot["arguments"] += tc["function"].get("arguments") or ""
    return msg, finish, usage


class ForkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.log = _record(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_steps_and_middle(self):
        steps, events = fork.parse_steps(self.log)
        self.assertEqual(len(steps), 5)
        self.assertEqual([s["finish"] for s in steps], ["tool-calls", "tool-calls", "stop", "tool-calls", "stop"])
        self.assertEqual(steps[0]["reasoning"], "look first")
        self.assertEqual(steps[1]["tools"][0]["input"]["content"], "x = 1\n")
        self.assertEqual(fork.mid_step(steps), 2)
        self.assertEqual(sum(1 for e in events if "user" in e), 3)
        self.assertEqual(fork.recorded_work_path(self.log), OLD_WORK)

    def test_step_message_moves_the_whole_run_dir(self):
        step = {"text": "", "reasoning": "", "tools": [{"id": "c1", "name": "bash",
                "input": {"command": "ls /s/t-base-0-ab/cfg /s/t-base-0-ab/work/x.py"}}]}
        msg = fork.step_message(step, "/s/t-base-0-ab/work", "/n/fork-1/t-base-0-cd/work")
        cmd = json.loads(msg["tool_calls"][0]["function"]["arguments"])["command"]
        self.assertEqual(cmd, "ls /n/fork-1/t-base-0-cd/cfg /n/fork-1/t-base-0-cd/work/x.py")

    def test_replay_log_stops_after_k_steps(self):
        log = self.root / "files.agent.log"
        ev = _step_events(0, [("a", "write", {"filePath": "one.txt", "content": "1"})])
        ev += _step_events(1, [("b", "write", {"filePath": "two.txt", "content": "2"})])
        for e in ev:
            if e["type"] == "tool_use":
                e["part"]["state"]["output"] = "Wrote file successfully."
        log.write_text("".join(json.dumps(e) + "\n" for e in ev))
        w1 = self.root / "w1"
        w1.mkdir()
        replay_log(log, w1, max_steps=1)
        self.assertEqual(sorted(p.name for p in w1.iterdir()), ["one.txt"])
        w2 = self.root / "w2"
        w2.mkdir()
        replay_log(log, w2)
        self.assertEqual(sorted(p.name for p in w2.iterdir()), ["one.txt", "two.txt"])

    def test_replay_server_answers_the_record_and_checks_the_history(self):
        steps, _ = fork.parse_steps(self.log)
        new_work = "/dev/shm/fork-x/py_x-implement-base-0-zz/work"
        st = fork.ReplayState(steps, 4, None, OLD_WORK, lambda: new_work)
        srv = fork.serve_replay(st)
        port = srv.server_port
        tools = [{"type": "function", "function": {"name": "bash"}}]
        try:
            # a title request: canned, not part of the record
            code, data = _post(port, {"stream": True, "messages": [{"role": "system", "content": "You are a title generator."},
                                                                      {"role": "user", "content": "x"}]})
            self.assertEqual(st.titles, 1)
            self.assertEqual(st.served, 0)
            # step 0: reasoning + a tool call, with the recorded work path rewritten to the new one
            code, data = _post(port, {"stream": True, "tools": tools, "messages": [{"role": "user", "content": "do it"}]})
            msg, finish, usage = _sse_message(data)
            self.assertEqual((code, finish), (200, "tool_calls"))
            self.assertEqual(msg["reasoning"], "look first")
            self.assertEqual(msg["tool_calls"][0]["id"], "c0")
            self.assertEqual(json.loads(msg["tool_calls"][0]["arguments"]), {"command": f"ls {new_work}"})
            self.assertEqual(usage["prompt_tokens"], 100)          # the recorded counts (the engine compacts from them)
            # step 1: the request must carry step 0's call as its last assistant message
            hist = [{"role": "user", "content": "do it"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "type": "function"}]},
                    {"role": "tool", "tool_call_id": "c0", "content": "ok"}]
            code, data = _post(port, {"stream": False, "tools": tools, "messages": hist})
            obj = json.loads(data)
            self.assertEqual(obj["choices"][0]["message"]["tool_calls"][0]["id"], "c1")
            self.assertIn(new_work, obj["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
            # step 2 is the engine's compaction (no tools): answered with the recorded summary
            code, data = _post(port, {"stream": True, "messages": hist + [{"role": "user", "content": "summarize"}]})
            msg, finish, _ = _sse_message(data)
            self.assertEqual((finish, msg["content"].startswith("## Objective")), ("stop", True))
            self.assertEqual(st.compactions, 1)
            # step 3: after the compaction the last assistant message is still step 1's call
            hist2 = hist + [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function"}]},
                            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                            {"role": "user", "content": "Continue if you have next steps"}]
            code, data = _post(port, {"stream": True, "tools": tools, "messages": hist2})
            msg, finish, _ = _sse_message(data)
            self.assertEqual(msg["tool_calls"][0]["id"], "c3")
            self.assertIsNone(st.diverged)
            self.assertEqual(st.served, 4)
            # past k without a model: a plain stop, and the hand-over is noted
            code, data = _post(port, {"stream": True, "tools": tools, "messages": hist2})
            msg, finish, _ = _sse_message(data)
            self.assertEqual((finish, st.extra_stops), ("stop", 1))
            self.assertIsNotNone(st.handover_at)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_divergence_is_detected(self):
        steps, _ = fork.parse_steps(self.log)
        st = fork.ReplayState(steps, 4, None, OLD_WORK, lambda: None)
        srv = fork.serve_replay(st)
        tools = [{"type": "function", "function": {"name": "bash"}}]
        try:
            _post(srv.server_port, {"stream": True, "tools": tools, "messages": [{"role": "user", "content": "do it"}]})
            wrong = [{"role": "assistant", "content": "", "tool_calls": [{"id": "OTHER", "type": "function"}]}]
            _post(srv.server_port, {"stream": True, "tools": tools, "messages": wrong})
            self.assertEqual(st.diverged["at_step"], 1)
            self.assertEqual(st.served, 1)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_past_k_requests_go_to_the_model(self):
        got = []

        class Up(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                data = b'data: {"choices":[{"index":0,"delta":{"content":"model"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        up = ThreadingHTTPServer(("127.0.0.1", 0), Up)
        threading.Thread(target=up.serve_forever, daemon=True).start()
        steps, _ = fork.parse_steps(self.log)
        handed = []
        st = fork.ReplayState(steps, 1, f"http://127.0.0.1:{up.server_port}", OLD_WORK, lambda: None, handed.append)
        srv = fork.serve_replay(st)
        tools = [{"type": "function", "function": {"name": "bash"}}]
        try:
            _post(srv.server_port, {"stream": True, "tools": tools, "messages": [{"role": "user", "content": "do it"}]})
            hist = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "type": "function"}]}]
            code, data = _post(srv.server_port, {"stream": True, "tools": tools, "messages": hist})
            msg, finish, _ = _sse_message(data)
            self.assertEqual(msg["content"], "model")
            self.assertEqual(len(got), 1)
            self.assertEqual(got[0]["messages"], hist)              # forwarded as the engine sent it
            self.assertEqual(len(handed), 1)
            self.assertEqual(st.forwarded, 1)
        finally:
            srv.shutdown()
            srv.server_close()
            up.shutdown()
            up.server_close()

    def test_tree_diff_lists_changed_and_missing_files(self):
        a, b = self.root / "a", self.root / "b"
        for d in (a, b):
            (d / "hidden_tests").mkdir(parents=True)
            (d / "same.txt").write_text("1")
        (a / "only_a.txt").write_text("x")
        (a / "mod.txt").write_text("1")
        (b / "mod.txt").write_text("2")
        (a / "hidden_tests" / "t.py").write_text("ignored")
        self.assertEqual(fork.tree_diff(a, b), ["mod.txt", "only_a.txt"])
