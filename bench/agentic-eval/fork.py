#!/usr/bin/env python3
"""FORK1b: continue an agentic-eval task from step k of a recorded run ("доигрывание с шага k").

A recorded run (agent log of harness.run_one) is a list of model steps: each step is one model answer (reasoning, text, tool
calls). To continue from step k the engine is started on the task as usual, through the same harness, but its model endpoint is
a replay server:
  * the first k agent requests are answered from the record (the same reasoning, text and tool calls, with the recorded work
    path rewritten to the new one); the engine itself executes the tool calls, so the project tree and the dialogue history are
    rebuilt by the engine exactly as it builds them (including shell mv/cp that replay.py does not repeat);
  * from request k on, requests go to the real model (through proxy.py, which counts tokens like a normal run).
Each replayed request is checked against the record: the previous step's tool calls must be the last assistant message of the
request. A mismatch marks the fork as diverged (it is then answered by the model, and the result is not a valid fork).
At the hand-over the project tree is hashed and compared with the file-level reconstruction of replay.py (writes and edits up to
step k on the same start), so a difference shows what the record alone cannot restore.

    fork.py check --log RUN.agent.log --task TASK_ID [--tasks-root DIR]     # full replay through the engine, no model: k = all steps
    fork.py run --log RUN.agent.log --task TASK_ID --k mid --upstream http://127.0.0.1:PORT --out DIR [--rep N]
"""
from __future__ import annotations

import argparse
import copy
import glob
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

WORK_PATH_RE = re.compile(r"(/[A-Za-z0-9_./-]*?/[A-Za-z0-9_.-]+-[A-Za-z0-9_]+-\d+-[A-Za-z0-9_]+)/work\b")


# ------------------------------------------------------------------------------------------------- the record

def parse_steps(log_path):
    """Model steps of a recorded run, in order. Each: {"reasoning", "text", "tools": [{"id", "name", "input", "status"}],
    "finish", "tokens"}. User events (prompt and plugin follow-ups) are kept in order too, as {"user": text}."""
    steps, cur, events = [], None, []
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        t, p = e.get("type"), e.get("part") or {}
        if t == "user":
            events.append({"user": e.get("text", "")})
        elif t == "step_start":
            cur = {"reasoning": "", "text": "", "tools": [], "finish": None, "tokens": None}
        elif cur is not None and t == "reasoning":
            cur["reasoning"] += p.get("text") or ""
        elif cur is not None and t == "text":
            cur["text"] += p.get("text") or ""
        elif cur is not None and t == "tool_use":
            st = p.get("state") or {}
            cur["tools"].append({"id": p.get("callID"), "name": p.get("tool"), "input": st.get("input") or {},
                                 "status": st.get("status")})
        elif cur is not None and t == "step_finish":
            cur["finish"] = p.get("reason")
            cur["tokens"] = p.get("tokens")
            steps.append(cur)
            events.append({"step": len(steps) - 1})
            cur = None
    return steps, events


def recorded_work_path(log_path):
    """The absolute work folder of the recorded run (its tool calls name it), or None."""
    counts = {}
    for m in WORK_PATH_RE.finditer(Path(log_path).read_text(encoding="utf-8", errors="replace")):
        counts[m.group(1) + "/work"] = counts.get(m.group(1) + "/work", 0) + 1
    return max(counts, key=counts.get) if counts else None


def mid_step(steps):
    """k = the middle of the trajectory, counted in model steps (at least 1, at most n - 1)."""
    n = len(steps)
    return max(1, min(n - 1, n // 2)) if n > 1 else n


# ------------------------------------------------------------------------------------------------- synthetic answers

def _rewrite(obj, old, new):
    if not old or not new:
        return obj
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, list):
        return [_rewrite(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite(v, old, new) for k, v in obj.items()}
    return obj


def step_message(step, old_work=None, new_work=None):
    """The recorded step as an OpenAI chat message (role assistant). Paths are moved from the recorded run dir to the new one: the whole
    run dir (the parent of `work`), so that commands looking at the sandbox around `work` (cfg/, home/) find the new one too."""
    if old_work and new_work and old_work.endswith("/work") and new_work.endswith("/work"):
        old_work, new_work = old_work[:-len("/work")], new_work[:-len("/work")]
    msg = {"role": "assistant", "content": _rewrite(step["text"], old_work, new_work)}
    if step["reasoning"]:
        msg["reasoning_content"] = _rewrite(step["reasoning"], old_work, new_work)
    calls = [{"id": t["id"], "type": "function",
              "function": {"name": t["name"], "arguments": json.dumps(_rewrite(t["input"], old_work, new_work), ensure_ascii=False)}}
             for t in step["tools"]]
    if calls:
        msg["tool_calls"] = calls
    return msg


def step_usage(step):
    """The recorded token counts of the step as the server's usage: the engine decides to compact the history from them, so a
    replay must report what the real server reported (otherwise the engine never compacts where the record did)."""
    tok = step.get("tokens") or {}
    cache = (tok.get("cache") or {}).get("read") or 0
    prompt = int(tok.get("input") or 0) + int(cache)
    out = int(tok.get("output") or 0)
    usage = {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out,
             "prompt_tokens_details": {"cached_tokens": int(cache)}}
    if tok.get("reasoning"):
        usage["completion_tokens_details"] = {"reasoning_tokens": int(tok["reasoning"])}
    return usage


def is_side_request(body):
    """A session-title request (never part of the agent's history). Compaction requests are NOT side requests: the record has
    their answer (the summary step)."""
    msgs = body.get("messages") or []
    first = msgs[0] if msgs and isinstance(msgs[0], dict) else {}
    return first.get("role") == "system" and "title generator" in str(first.get("content") or "")


def synth_json(step, model, old_work=None, new_work=None):
    msg = step_message(step, old_work, new_work)
    finish = "tool_calls" if msg.get("tool_calls") else "stop"
    usage = step_usage(step)
    return {"id": "chatcmpl-replay", "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}


def synth_sse(step, model, old_work=None, new_work=None):
    """The same answer as llama-server streams it: role, reasoning, content, tool calls, finish, usage."""
    msg = step_message(step, old_work, new_work)
    base = {"id": "chatcmpl-replay", "object": "chat.completion.chunk", "created": int(time.time()), "model": model}

    def chunk(delta, finish=None, extra=None):
        obj = dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": finish}])
        if extra:
            obj.update(extra)
        return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()

    out = [chunk({"role": "assistant", "content": None})]
    if msg.get("reasoning_content"):
        out.append(chunk({"reasoning_content": msg["reasoning_content"]}))
    if msg.get("content"):
        out.append(chunk({"content": msg["content"]}))
    for i, call in enumerate(msg.get("tool_calls") or []):
        out.append(chunk({"tool_calls": [{"index": i, "id": call["id"], "type": "function",
                                          "function": {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}}]}))
    finish = "tool_calls" if msg.get("tool_calls") else "stop"
    out.append(chunk({}, finish, {"usage": step_usage(step)}))
    out.append(b"data: [DONE]\n\n")
    return out


# ------------------------------------------------------------------------------------------------- the replay server

class ReplayState:
    def __init__(self, steps, k, upstream=None, old_work=None, find_work=None, on_handover=None):
        self.steps, self.k, self.upstream = steps, int(k), upstream
        self.old_work, self.find_work, self.on_handover = old_work, find_work, on_handover
        self.new_work = None
        self.served = 0
        self.forwarded = 0
        self.titles = 0
        self.extra_stops = 0
        self.compactions = 0
        self.last_agent_ids = []
        self.after_compaction = False
        self.diverged = None
        self.handover_at = None
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.log = []

    def expected_ids(self):
        """Tool-call ids the request's last assistant message must carry: those of the last step served to an agent request
        (a compaction summary is not an assistant message of the history, so it does not move this)."""
        return list(self.last_agent_ids)


def _last_assistant_ids(messages):
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return [c.get("id") for c in m.get("tool_calls") or []]
    return []


class ReplayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: ReplayState = None

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for c in chunks:
            self.wfile.write(f"{len(c):x}\r\n".encode() + c + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _forward(self, raw):
        st = self.state
        u = urlparse(st.upstream)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
        try:
            conn.request(self.command, self.path, body=raw, headers={k: v for k, v in self.headers.items()
                                                                      if k.lower() not in ("host", "content-length", "connection")}
                         | ({"Content-Length": str(len(raw))} if raw is not None else {}))
            resp = conn.getresponse()
            self.send_response(resp.status)
            chunked = False
            for k, v in resp.getheaders():
                if k.lower() in ("transfer-encoding", "connection", "content-length"):
                    continue
                self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            chunked = True
            while True:
                data = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                if not data:
                    break
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
        finally:
            conn.close()

    def do_GET(self):
        st = self.state
        if st.upstream:
            return self._forward(None)
        return self._send(404, {"error": "replay server: no upstream"})

    def do_POST(self):
        st = self.state
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if not self.path.endswith("/chat/completions"):
            return self._forward(raw) if st.upstream else self._send(404, {"error": "replay server: unknown path"})
        try:
            body = json.loads(raw)
        except ValueError:
            return self._send(400, {"error": "bad json"})
        model = body.get("model", "m")
        stream = bool(body.get("stream"))
        if os.environ.get("FORK_DEBUG"):
            tail = [{"role": m.get("role"), "ids": [c.get("id") for c in m.get("tool_calls") or []],
                     "tool_call_id": m.get("tool_call_id"), "content": str(m.get("content"))[:80]}
                    for m in (body.get("messages") or [])[-4:] if isinstance(m, dict)]
            with open(os.environ["FORK_DEBUG"], "a") as f:
                f.write(json.dumps({"served": st.served, "tools": bool(body.get("tools")), "n": len(body.get("messages") or []),
                                    "tail": tail}, ensure_ascii=False) + "\n")
        with st.lock:
            if is_side_request(body):
                # session titles: they never change the project or the history; answered canned until the hand-over
                if st.upstream and st.served >= st.k:
                    st.log.append({"t": time.monotonic() - st.started, "kind": "side-forward"})
                    action = "forward"
                else:
                    st.titles += 1
                    action = "title"
            elif st.served < st.k and st.diverged is None:
                # an agent request, or the engine's compaction request (no tools): the next recorded step answers it
                want, got = st.expected_ids(), _last_assistant_ids(body.get("messages"))
                if body.get("tools") and st.after_compaction and got == []:
                    # the engine replaced the history by "What did we do so far?" + the summary (an assistant message without
                    # calls); its other compaction mode keeps the last call. Both are the record's own state.
                    want = got
                if not body.get("tools"):
                    # the engine's compaction request: it rebuilds the history for the summary, so tool-call ids are not there;
                    # the record answers it with its summary step (checked: that step has no tool calls)
                    st.compactions += 1
                    if st.steps[st.served]["tools"]:
                        want, got = ["<summary step>"], [t["id"] for t in st.steps[st.served]["tools"]]
                    else:
                        want = got
                if want != got:
                    st.diverged = {"at_step": st.served, "want": want, "got": got}
                    action = "forward" if st.upstream else "stop"
                else:
                    if st.new_work is None and st.find_work:
                        st.new_work = st.find_work()
                    step = st.steps[st.served]
                    if body.get("tools"):
                        st.last_agent_ids = [t["id"] for t in step["tools"]]
                    st.after_compaction = not body.get("tools")
                    st.served += 1
                    st.log.append({"t": time.monotonic() - st.started, "kind": "replay", "step": st.served - 1})
                    action = ("replay", step)
                    if st.served == st.k:
                        st.handover_pending = True
            elif st.upstream:
                if st.handover_at is None:
                    st.handover_at = time.monotonic() - st.started
                    if st.on_handover:
                        st.on_handover(st)
                st.forwarded += 1
                st.log.append({"t": time.monotonic() - st.started, "kind": "forward"})
                action = "forward"
            else:
                if st.handover_at is None:
                    st.handover_at = time.monotonic() - st.started
                    if st.on_handover:
                        st.on_handover(st)
                st.extra_stops += 1
                action = "stop"
        if action == "title":
            step = {"reasoning": "", "text": "Task", "tools": [], "finish": "stop", "tokens": None}
            return self._stream(synth_sse(step, model)) if stream else self._send(200, synth_json(step, model))
        if action == "stop":
            step = {"reasoning": "", "text": "Done.", "tools": [], "finish": "stop", "tokens": None}
            return self._stream(synth_sse(step, model)) if stream else self._send(200, synth_json(step, model))
        if action == "forward":
            return self._forward(raw)
        _, step = action
        args = (step, model, st.old_work, st.new_work)
        return self._stream(synth_sse(*args)) if stream else self._send(200, synth_json(*args))


def serve_replay(state, port=0):
    handler = type("H", (ReplayHandler,), {"state": state})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ------------------------------------------------------------------------------------------------- one fork

def _reference_tree(task, log_path, k, dest):
    """The file-level reconstruction of the project at step k: start + the harness's project rules + replay.py's writes and
    edits (and safe mkdir/rm) of steps 0..k-1."""
    import harness
    import replay
    work = Path(dest)
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(task["dir"] / "start", work) if (task["dir"] / "start").is_dir() else work.mkdir(parents=True)
    harness.write_project_rules(work, agent_tools=harness.agent_tools_on(task))
    stats = replay.replay_log(Path(log_path), work, max_steps=k)
    return work, stats


def run_fork(task, log_path, k, out_dir, upstream=None, extra=None, rep=0, arm="base", scratch_root=None, keep=False):
    """Continue `task` from step k of the recorded run `log_path`. upstream = llama-server URL (None: no model; then every
    request past the record is answered with a plain stop). Returns harness.run_one's record plus a "fork" section."""
    import harness
    import proxy as proxy_mod
    import replay
    steps, _events = parse_steps(log_path)
    if k == "mid":
        k = mid_step(steps)
    k = int(min(int(k), len(steps)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(scratch_root or tempfile.mkdtemp(prefix=f"fork-{task['id']}-", dir="/dev/shm"))
    scratch_root.mkdir(parents=True, exist_ok=True)
    proxy_srv, model_url = None, None
    proxy_log = out_dir / f"proxy-{task['id']}-{rep}.jsonl"
    if upstream:
        proxy_srv = proxy_mod.serve(0, upstream, str(proxy_log), extra or {})
        threading.Thread(target=proxy_srv.serve_forever, daemon=True).start()
        model_url = f"http://127.0.0.1:{proxy_srv.server_port}"

    def find_work():
        hits = sorted(glob.glob(str(scratch_root / "*" / "work")))
        return hits[-1] if hits else None

    hand = {}

    def on_handover(st):
        work = Path(st.new_work or find_work() or "")
        if work.is_dir():
            hand["engine_tree"] = replay.tree_hash(work)
            snap = scratch_root / "engine-at-k"
            shutil.copytree(work, snap, symlinks=True, ignore=shutil.ignore_patterns("node_modules", "__pycache__"))
            hand["engine_snapshot"] = str(snap)

    state = ReplayState(steps, k, model_url, recorded_work_path(log_path), find_work, on_handover)
    srv = serve_replay(state)
    old_keep = os.environ.get("AGENTIC_EVAL_KEEP_SCRATCH")
    os.environ["AGENTIC_EVAL_KEEP_SCRATCH"] = "1"          # the scratch is removed below, after the tree checks
    try:
        rec = harness.run_one(task, arm, rep, srv.server_port, str(proxy_log), out_dir, scratch_root=str(scratch_root))
        if "engine_snapshot" not in hand and state.served == k:
            # nothing came after the replayed steps (k = all of them): the tree at the end is the tree at step k
            on_handover(state)
    finally:
        if old_keep is None:
            os.environ.pop("AGENTIC_EVAL_KEEP_SCRATCH", None)
        else:
            os.environ["AGENTIC_EVAL_KEEP_SCRATCH"] = old_keep
        srv.shutdown()
        srv.server_close()
        if proxy_srv:
            proxy_srv.shutdown()
            proxy_srv.server_close()
    fork = {"k": k, "steps_recorded": len(steps), "served": state.served, "forwarded": state.forwarded, "titles": state.titles,
            "extra_stops": state.extra_stops, "compactions": state.compactions, "diverged": state.diverged, "handover_s": state.handover_at,
            "recorded_work": state.old_work, "valid": state.diverged is None and state.served == k}
    if "engine_snapshot" in hand:
        ref, stats = _reference_tree(task, log_path, k, scratch_root / "reference-at-k")
        a, b = Path(hand["engine_snapshot"]), ref
        fork["tree_engine"] = hand["engine_tree"]
        fork["tree_reference"] = replay.tree_hash(ref)
        fork["tree_diff"] = tree_diff(a, b)
        fork["tree_match"] = not fork["tree_diff"]
        fork["reference_stats"] = {k2: v for k2, v in stats.items() if k2 != "errors"}
    rec["fork"] = fork
    rec["env"] = environment_fingerprint()           # FORK1c step 8: a record made in another environment is marked, not mixed in
    if not keep:
        shutil.rmtree(scratch_root, ignore_errors=True)
    return rec


def environment_fingerprint() -> dict:
    """Versions that can change tool output: Python, Node, the engine, and a hash of the plugin folder (names and sizes)."""
    import hashlib
    import harness
    def ver(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or "unknown"
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"
    plugin_root = Path(harness.PLUGIN_SRC)
    digest = hashlib.sha256()
    for p in sorted(plugin_root.rglob("*")):
        if p.is_file() and "node_modules" not in p.parts:
            digest.update(f"{p.relative_to(plugin_root)} {p.stat().st_size}\n".encode())
    return {"python": sys.version.split()[0], "node": ver(["node", "--version"]),
            "engine": ver([str(Path(harness.ENGINE).resolve()), "--version"]), "plugins": digest.hexdigest()[:16]}


def same_environment(rec: dict, env: dict | None = None) -> bool:
    """True if a record was made in this environment (Python, Node, engine, plugin hash all equal)."""
    env = env or environment_fingerprint()
    return rec.get("env") == env


def tree_diff(a, b, ignore=("node_modules", "__pycache__", ".cache", "hidden_tests")):
    """Relative paths whose presence or bytes differ between two trees (ignoring caches)."""
    def files(root):
        out = {}
        for p in Path(root).rglob("*"):
            rel = p.relative_to(root)
            if any(x in rel.parts for x in ignore) or not p.is_file():
                continue
            out[rel.as_posix()] = p.read_bytes()
        return out
    fa, fb = files(a), files(b)
    return sorted(x for x in set(fa) | set(fb) if fa.get(x) != fb.get(x))


# ------------------------------------------------------------------------------------------------- CLI

def _task(task_id, tasks_root):
    import harness
    tasks = {t["id"]: t for t in harness.all_tasks(Path(tasks_root))}
    if task_id not in tasks:
        raise SystemExit(f"task not found: {task_id}")
    return tasks[task_id]


def probe(results, logs_dir, tasks_root, out, seed=20261113, per_outcome=5, k=None, json_out=None):
    """The FORK1 step-0 sample (replay.select_runs), each run replayed through the engine without a model: the hidden-test
    result must equal the record. With k (or "mid") the tree at step k is compared instead of the end."""
    import harness
    import replay
    rows = replay.read_jsonl(Path(results))
    selected = replay.select_runs(rows, Path(logs_dir), seed=seed, per_outcome=per_outcome)
    tasks = {t["id"]: t for t in harness.all_tasks(Path(tasks_root))}
    outcomes = []
    for row in selected:
        log = Path(logs_dir) / f"{row['task']}__{row['arm']}__{row['rep']}.agent.log"
        steps, _ = parse_steps(log)
        kk = mid_step(steps) if k == "mid" else (int(k) if k else len(steps))
        rec = run_fork(tasks[row["task"]], log, kk, out, None, rep=row["rep"])
        f = rec["fork"]
        o = {"task": row["task"], "rep": row["rep"], "recorded": f"{row['tests_passed']}/{row['tests_total']}",
             "replayed": f"{rec.get('tests_passed')}/{rec.get('tests_total')}", "k": kk, "steps": len(steps),
             "score_match": rec.get("tests_passed") == row["tests_passed"] and rec.get("tests_total") == row["tests_total"],
             "valid": f["valid"], "diverged": f["diverged"], "compactions": f.get("compactions"), "tree_match": f.get("tree_match"),
             "tree_diff": f.get("tree_diff"), "seconds": rec.get("seconds"), "recorded_seconds": row.get("seconds")}
        outcomes.append(o)
        print(json.dumps(o, ensure_ascii=False), flush=True)
    summary = {"total": len(outcomes), "score_match": sum(o["score_match"] for o in outcomes),
               "valid": sum(bool(o["valid"]) for o in outcomes), "tree_match": sum(bool(o["tree_match"]) for o in outcomes)}
    if json_out:
        Path(json_out).write_text(json.dumps({"summary": summary, "outcomes": outcomes}, indent=1, ensure_ascii=False) + "\n")
    return summary, outcomes


# ------------------------------------------------------------------------------------------------- the GPU batch

VRAM_SYSFS = "/sys/class/drm/card1/device/"


def _free_vram_gib():
    return (int(open(VRAM_SYSFS + "mem_info_vram_total").read()) - int(open(VRAM_SYSFS + "mem_info_vram_used").read())) / 2**30


def _pid_on_port(port):
    """The listening process of a port (exact PID from `ss`, never a name pattern)."""
    import subprocess
    out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True).stdout
    for part in out.replace(",", " ").split():
        if part.startswith("pid="):
            return int(part[4:])
    return None


def _stop_pid(pid):
    import signal
    if not pid:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    for _ in range(15):
        time.sleep(1)
        try:
            os.kill(pid, 0)
        except OSError:
            return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def batch(selection, server, out, reps=(0, 1), logs_dir=None, tasks_root=None, min_free_gib=1.5):
    """GPU acceptance: start the stand server, continue every chosen task from the middle of each recorded repeat, one at a
    time; results.jsonl after every fork (resumable). A VRAM watchdog stops the server below min_free_gib."""
    import subprocess
    import urllib.request
    import harness
    logs_dir = Path(logs_dir or Path.home() / "Desktop/AI/reports/qlev1/raw/main/agent-logs")
    tasks = {t["id"]: t for t in harness.all_tasks(Path(tasks_root or HERE / "tasks_v3"))}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = {(r["task"], r["rep"]) for r in (json.loads(x) for x in res_path.read_text().splitlines() if x.strip())} \
        if res_path.exists() else set()
    port = int(server.get("port", 18195))
    argv = [*server.get("prefix", []), server["exe"], "-m", server["model"], "--host", "127.0.0.1", "--port", str(port), *server["args"]]
    env = dict(os.environ, **server.get("env", {}))
    log = open(out / "server.log", "ab")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    stop = {"why": None, "min_free": 99.0}
    pid = None
    for _ in range(600):
        pid = _pid_on_port(port)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200 and pid:
                    break
        except OSError:
            pass
        if proc.poll() is not None and not pid:
            raise RuntimeError("the server exited while loading; see server.log")
        time.sleep(1)
    else:
        _stop_pid(_pid_on_port(port))
        raise RuntimeError("the server did not become healthy")

    def watchdog():
        while stop["why"] is None:
            f = _free_vram_gib()
            stop["min_free"] = min(stop["min_free"], f)
            if f < min_free_gib:
                time.sleep(1)
                if _free_vram_gib() < min_free_gib:
                    stop["why"] = f"free VRAM {f:.2f} GiB < {min_free_gib}"
                    _stop_pid(pid)
                    return
            time.sleep(0.25)
    threading.Thread(target=watchdog, daemon=True).start()
    summary = []
    try:
        for tid in selection:
            for rep in reps:
                if (tid, rep) in done or stop["why"]:
                    continue
                logp = logs_dir / f"{tid}__base__{rep}.agent.log"
                t0 = time.monotonic()
                rec = run_fork(tasks[tid], logp, "mid", out / "runs", f"http://127.0.0.1:{port}", rep=rep)
                rec["wall_s"] = round(time.monotonic() - t0, 1)
                rec["min_free_vram_gib"] = round(stop["min_free"], 2)
                with open(res_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                summary.append(rec)
                fk = rec["fork"]
                print(f"{tid} rep {rep}: score {rec.get('score')} ({rec.get('tests_passed')}/{rec.get('tests_total')}), "
                      f"{rec.get('seconds')} s, k {fk['k']}/{fk['steps_recorded']}, valid {fk['valid']}, tree {fk.get('tree_match')}",
                      flush=True)
    finally:
        stop["why"] = stop["why"] or "done"
        _stop_pid(_pid_on_port(port) or pid)
        try:
            proc.wait(timeout=30)
        except Exception:
            pass
    return {"stopped": stop["why"], "min_free_vram_gib": round(stop["min_free"], 2), "forks": len(summary)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("--results", default=str(Path.home() / "Desktop/AI/reports/qlev1/raw/main/results.jsonl"))
    pr.add_argument("--logs", default=str(Path.home() / "Desktop/AI/reports/qlev1/raw/main/agent-logs"))
    pr.add_argument("--tasks-root", default=str(HERE / "tasks_v3"))
    pr.add_argument("--out", default="/dev/shm/FORK1B/probe")
    pr.add_argument("--k", default=None)
    pr.add_argument("--json-out", default=None)
    bt = sub.add_parser("batch")
    bt.add_argument("--selection", required=True, help="JSON with a 'chosen' list of task ids")
    bt.add_argument("--server", required=True, help="JSON: exe, model, args, prefix, env, port")
    bt.add_argument("--out", required=True)
    bt.add_argument("--reps", default="0,1")
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--log", required=True)
        p.add_argument("--task", required=True)
        p.add_argument("--tasks-root", default=str(HERE / "tasks_v3"))
        p.add_argument("--out", default="/dev/shm/FORK1B/out")
        p.add_argument("--rep", type=int, default=0)
        p.add_argument("--keep", action="store_true")
        if name == "run":
            p.add_argument("--k", default="mid")
            p.add_argument("--upstream", required=True)
            p.add_argument("--extra", default="{}")
    a = ap.parse_args(argv)
    if a.cmd == "batch":
        sel = json.loads(Path(a.selection).read_text())["chosen"]
        res = batch(sel, json.loads(Path(a.server).read_text()), a.out, tuple(int(x) for x in a.reps.split(",")))
        print(json.dumps(res))
        return 0
    if a.cmd == "probe":
        summary, _ = probe(a.results, a.logs, a.tasks_root, a.out, k=a.k, json_out=a.json_out)
        print(json.dumps(summary))
        return 0
    task = _task(a.task, a.tasks_root)
    if a.cmd == "check":
        steps, _ = parse_steps(a.log)
        rec = run_fork(task, a.log, len(steps), a.out, None, rep=a.rep, keep=a.keep)
    else:
        rec = run_fork(task, a.log, a.k, a.out, a.upstream, json.loads(a.extra), rep=a.rep, keep=a.keep)
    print(json.dumps(rec, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
