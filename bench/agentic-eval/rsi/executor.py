"""RSIL0b: one run of the executor for the pilot, with a candidate harness wired in, and the two stands.

- FakeStand (--fake): the shura-qa fake llama-server with the policy of fake_worker.py. No GPU, no model: it writes the right files
  (solution/) when the harness the engine sent carries the candidate marker, and wrong files otherwise, with the same number of tool calls.
- GpuStand (real): llama-server under safe-run, a VRAM guard from the first start, a stop by its own PID. The command line comes from a
  file (--server-argv): no command line for the Tiel profile is verified in this repository yet, so it is not built here. Not run in
  RSIL0b (no GPU).

run_task() returns the same fields as harness.run_one, plus the model and the harness fingerprint. The harness is wired in by
wire_harness(): the model-harness plugin (plugin/model-harness.ts in shuracode-integration) reads <harnessDir>/<model>/harness.md by the
model id of the request, so the model key of the config must contain the model name ("tiel-coder" for tiel).
"""
import collections
import hashlib
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402
import proxy  # noqa: E402

FAKE_WORKER = HERE / "fake_worker.py"
SAFE_RUN = Path("/home/alisher/Desktop/AI/reports/tools/safe-run.sh")
MODEL_HARNESS_PLUGIN = Path(os.environ.get("RSI_MODEL_HARNESS_PLUGIN",
                                           "/home/alisher/Projects/shuracode-integration/plugin/model-harness.ts"))
MODEL_IDS = {"tiel": "tiel-coder", "sage": "sage", "gemma": "gemma"}   # RSI1: Tiel and Sage; Occamy removed (2026-10-28)
# RSI2E: the mechanism plugins (shuracode-integration/plugin/rsi2), copied into the scratch like model-harness; a candidate names them
# in <harness>/<model>/plugins.json as {"plugins": [{"name": "done-gate", "options": {"doneGate": true}}]} (off by default in the product)
RSI2_PLUGIN_SRC = Path(os.environ.get("RSI2_PLUGIN_SRC", "/home/alisher/Projects/shuracode-integration/plugin/rsi2"))
RSI2_PLUGINS = {"requirements-lock": "requirementsLock", "task-state": "taskState", "compact-preserve": "compactPreserve",
                "done-gate": "doneGate", "action-forcing": "actionForcing"}
ENGINE_CONTEXT = int(os.environ.get("RSI_ENGINE_CONTEXT", "200000"))   # RSI2E: the product's limit.context for tiel-coder (shuracode.json), not 32768
VRAM_USED = Path("/sys/class/drm/card1/device/mem_info_vram_used")
VRAM_TOTAL = Path("/sys/class/drm/card1/device/mem_info_vram_total")
KEEP_FREE_GIB = 1.5          # the desktop's share of the card (reports/tools prompts, VRAM rule)
GUARD_SECONDS = 1.0          # the guard stops the server after this long below the share
GUARD_POLL_S = 0.1           # VRAM sample interval of the guard (0.2 s before VRAMDIP1: a dip shorter than that could pass between samples)
RELEASE_TIMEOUT_S = 60.0     # after the stop: wait at most this long for VRAM to come back (RSIL0c, orchestrator's decision)
RELEASE_MARGIN_GIB = 0.5     # ... above the desktop's share plus this margin, and above the free VRAM before the start, less 1 GiB
REAL_PORT = 8091
SAFE_MAX_GB = "15"    # RSIL1 (2026-10-22 02:54): the llama-server was killed by the 13 GB cgroup limit (anon 5.8 + shmem 7.8 GB)
SAFE_MIN_FREE_GB = "12"   # RSIL1 stand decision (2026-10-22): start threshold 12 GiB (was 16); the 3.5 GiB kill guard of safe-run is unchanged
RUN_ATTEMPTS = 3     # a run on a dead or failing server is not counted: it is repeated after a restart
SERVER_MEM_LIMIT_GIB = 6.0   # RSIL1 fix (orchestrator, 2026-10-22): heap + /dev/zero of the server above this: a planned restart before the next run
PID_FILE = "server.pid"      # the stand's own server: PID, port and argv, written at each start (in the stand's work folder)
PORT_FREE_TIMEOUT_S = 30.0   # a start waits this long for the port to be free; a listener that is not the stand's is left alone

def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    return 0.0


def wait_mem(limit_s=300):
    """safe-run refuses to start under 16 GiB MemAvailable (exit 75): wait up to 5 minutes, then start anyway and let it report."""
    t0 = time.time()
    while mem_available_gib() < float(SAFE_MIN_FREE_GB) and time.time() - t0 < limit_s:
        time.sleep(10)
    return round(time.time() - t0, 1)


load_task = harness.load_task
all_tasks = harness.all_tasks


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_port(port: int, timeout: float = 30.0):
    deadline = time.time() + timeout
    while True:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return
        except OSError:
            if time.time() > deadline:
                raise RuntimeError(f"nothing listens on {port}")
            time.sleep(0.3)


def fingerprint(path) -> str:
    """A file's hash, or a folder's (the harness of a model): the sorted relative names and the bytes of every file in it."""
    p = Path(path)
    h = hashlib.sha1()
    if p.is_dir():
        for f in sorted(x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
            h.update(f.relative_to(p).as_posix().encode("utf-8") + b"\0" + f.read_bytes() + b"\0")
    else:
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def stop_own(proc: subprocess.Popen, grace: float = 5.0):
    """SIGTERM to the process group this process started, then SIGKILL after `grace` seconds. Never by name."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=grace)
    except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


SMAPS_HEADER = re.compile(r"^[0-9a-f]+-[0-9a-f]+\s")


def parse_smaps_kib(text: str):
    """Resident kB of the [heap] mapping and of the /dev/zero mappings, from a process's smaps text."""
    heap = devzero = 0
    name = ""
    for line in text.splitlines():
        if SMAPS_HEADER.match(line):
            parts = line.split()
            name = parts[5] if len(parts) > 5 else ""
        elif line.startswith("Rss:"):
            kib = int(line.split()[1])
            if name == "[heap]":
                heap += kib
            elif name == "/dev/zero":
                devzero += kib
    return heap, devzero


def server_memory_gib(pid: int, proc: str = "/proc"):
    """(heap, /dev/zero) in GiB: the two numbers of the RSIL1 memory rule (the orchestrator's decision, 2026-10-22)."""
    heap, devzero = parse_smaps_kib(Path(proc, str(pid), "smaps").read_text(encoding="utf-8"))
    return heap / 2 ** 20, devzero / 2 ** 20


def over_memory_limit(heap_gib: float, devzero_gib: float, limit_gib: float = SERVER_MEM_LIMIT_GIB) -> bool:
    return heap_gib + devzero_gib > limit_gib


def pid_running(pid: int, proc: str = "/proc") -> bool:
    """A zombie counts as stopped: the process has ended, only its parent has not collected it yet."""
    try:
        stat = Path(proc, str(pid), "stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def cmdline(pid: int, proc: str = "/proc"):
    try:
        raw = Path(proc, str(pid), "cmdline").read_bytes()
    except OSError:
        return []
    return [x.decode("utf-8", "replace") for x in raw.split(b"\0") if x]


def find_server_pids(argv, proc: str = "/proc"):
    """PIDs whose command line is exactly argv: the stand's llama-server, as safe-run starts it (checked on the live system)."""
    return [int(p.name) for p in Path(proc).iterdir() if p.name.isdigit() and cmdline(int(p.name), proc) == list(argv)]


def listen_inodes(port: int, net_texts) -> set:
    """Inodes of the sockets in LISTEN state (0A) on `port`, from the texts of /proc/net/tcp and /proc/net/tcp6."""
    suffix = f":{port:04X}"
    out = set()
    for text in net_texts:
        for line in text.splitlines()[1:]:
            fields = line.split()
            if len(fields) > 9 and fields[3] == "0A" and fields[1].endswith(suffix):
                out.add(int(fields[9]))
    return out


def listener_pids(port: int, proc: str = "/proc", net_files=("/proc/net/tcp", "/proc/net/tcp6")) -> set:
    """PIDs that hold a LISTEN socket on `port`. The stand's owner check: a server it did not start is never used."""
    texts = []
    for name in net_files:
        try:
            texts.append(Path(name).read_text(encoding="utf-8"))
        except OSError:
            pass
    inodes = listen_inodes(port, texts)
    pids = set()
    if not inodes:
        return pids
    for p in Path(proc).iterdir():
        if not p.name.isdigit():
            continue
        try:
            for fd in (p / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith("socket:[") and int(target[8:-1]) in inodes:
                    pids.add(int(p.name))
                    break
        except OSError:
            continue
    return pids


def health_ok(port: int, timeout: float = 3.0) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def stop_pid(pid: int, grace: float = 5.0):
    """SIGTERM to one PID this stand started, then SIGKILL after `grace` seconds. The caller has checked that it is the stand's server."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace
    while pid_running(pid) and time.time() < deadline:
        time.sleep(0.2)
    if pid_running(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def install_model_harness_plugin(cfg_dir):
    """RSI2D: copy the model-harness plugin and its lib next to the other plugins of the experiment (<scratch>/plugin/), because the
    engine runs under bwrap which cannot see /home: in RSI1 the plugin was named by its absolute /home path, the engine never loaded
    it, and NO candidate text reached the model (0 of 34 candidate runs; `<available_skills>` stayed empty). Returns the copied path."""
    dest = Path(cfg_dir).parent / "plugin"
    (dest / "lib").mkdir(parents=True, exist_ok=True)
    src = MODEL_HARNESS_PLUGIN
    shutil.copy2(src, dest / src.name)
    lib = src.parent / "lib" / src.name
    if lib.exists():
        shutil.copy2(lib, dest / "lib" / src.name)
    return dest / src.name


def wire_harness(cfg_dir, harness_root, model, sampling=None):
    """Add the per-model harness plugin to the experiment's config and name the model the way the plugin reads it. `sampling` (the
    gateway stand: the model's own, from the candidate layout) replaces the engine's request options, which win over the server's.
    The plugin is copied into the scratch first (install_model_harness_plugin): a /home path is invisible inside the sandbox."""
    cfg = Path(cfg_dir) / "shuracode.json"
    conf = json.loads(cfg.read_text())
    entry = next(iter(conf["provider"]["eval"]["models"].values()))
    if sampling:
        entry["options"] = dict(sampling)
        conf["agent"]["build"] = {k: v for k, v in sampling.items() if k in ("temperature", "top_p")}
    model_id = MODEL_IDS[model]
    conf["provider"]["eval"]["models"] = {model_id: entry}
    conf["model"] = conf["small_model"] = f"eval/{model_id}"
    conf["plugin"].append([str(install_model_harness_plugin(cfg_dir)), {"harnessDir": str(harness_root)}])
    for entry in plugin_candidates(harness_root, model):          # RSI2E: the candidate's mechanism plugins, copied into the scratch too
        conf["plugin"].append([str(install_rsi2_plugin(cfg_dir, entry["name"])), dict(entry.get("options") or {})])
    cfg.write_text(json.dumps(conf, indent=2))
    return cfg


def plugin_candidates(harness_root, model):
    """The mechanism plugins a harness folder asks for: <root>/<model>/plugins.json -> [{"name", "options"}], names from RSI2_PLUGINS."""
    f = Path(harness_root) / model / "plugins.json"
    if not f.exists():
        return []
    data = json.loads(f.read_text(encoding="utf-8"))
    out = []
    for e in data.get("plugins", []):
        if e.get("name") not in RSI2_PLUGINS:
            raise ValueError(f"plugins.json names an unknown mechanism {e.get('name')!r}; known: {sorted(RSI2_PLUGINS)}")
        out.append({"name": e["name"], "options": e.get("options") or {RSI2_PLUGINS[e["name"]]: True}})
    return out


def install_rsi2_plugin(cfg_dir, name):
    """Copy one mechanism plugin and the shared lib into <scratch>/plugin/rsi2/ (bwrap cannot see /home)."""
    dest = Path(cfg_dir).parent / "plugin" / "rsi2"
    dest.mkdir(parents=True, exist_ok=True)
    for fn in (f"{name}.ts", "lib.ts"):
        shutil.copy2(RSI2_PLUGIN_SRC / fn, dest / fn)
    return dest / f"{name}.ts"


def mechanism_events(work_dir):
    """The events the mechanism plugins wrote in the run's project (<work>/.harness/events/<name>.jsonl): {name: [events]}."""
    out = {}
    folder = Path(work_dir) / ".harness" / "events"
    if folder.is_dir():
        for f in sorted(folder.glob("*.jsonl")):
            rows = []
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
            out[f.stem] = rows
    return out


def harness_probe(harness_root, model):
    """What a delivered harness must show in the system text: the first long line of harness.md (if any) and the skill names."""
    folder = Path(harness_root) / model
    text = (folder / "harness.md").read_text(encoding="utf-8", errors="replace") if (folder / "harness.md").exists() else ""
    line = next((l.strip() for l in text.splitlines() if len(l.strip()) >= 20), "")
    skills = sorted(p.parent.name for p in folder.glob("skills/*/SKILL.md")) if (folder / "skills").exists() else []
    return {"line": line, "skills": skills}


def harness_delivered(requests_path, harness_root, model):
    """RSI2D guard: did the harness text reach the model? Reads the run's request bodies (AGENTIC_EVAL_LOG_REQUESTS=1) and looks for
    the probe line and every skill name in SYSTEM/developer messages (a `read` of .harness/ by the model is tool output, not delivery).
    {"checked": False} when there is nothing to deliver (empty harness, no skills) or no request log to read."""
    probe = harness_probe(harness_root, model)
    plugins = [e["name"] for e in plugin_candidates(harness_root, model)]
    if not probe["line"] and not probe["skills"] and not plugins:
        return {"checked": False, "reason": "empty harness"}
    path = Path(requests_path) if requests_path else None
    if path is None or not path.exists():
        return {"checked": False, "reason": "no request log (AGENTIC_EVAL_LOG_REQUESTS=1 needed)", "delivered": False}
    system = []
    with open(path, errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            body = d.get("body") if isinstance(d, dict) and isinstance(d.get("body"), dict) else d
            for m in (body.get("messages") or []) if isinstance(body, dict) else []:
                if m.get("role") in ("system", "developer"):
                    c = m.get("content")
                    system.append(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))
    joined = "\n".join(system)
    line_ok = (probe["line"] in joined) if probe["line"] else True
    missing = [n for n in probe["skills"] if n not in joined]
    missing_plugins = [n for n in plugins if f"shuracode-rsi2:{n}: on" not in joined]   # each mechanism announces itself (plugin/rsi2/lib.ts)
    return {"checked": True, "delivered": bool(line_ok and not missing and not missing_plugins), "line_ok": line_ok, "missing_skills": missing,
            "missing_plugins": missing_plugins, "system_messages": len(system)}


def run_task(task, harness_root, model, arm, rep, port, proxy_log, out_dir, scratch_root, sampling=None, abort=None, context=None):
    """One run: the same steps as harness.run_one, with the candidate harness wired in. `context`: the engine's limit.context (RSI2E:
    the product's 200000 by default, so the engine compacts where the product would, and the gateway's long layout takes over at 28k)."""
    scratch = Path(tempfile.mkdtemp(prefix=f"{task['id']}-{arm}-{rep}-", dir=_ensure(scratch_root)))
    try:
        work = scratch / "work"
        # a task without start files has no start/ folder in git (empty folders are not tracked): an empty project then
        shutil.copytree(task["dir"] / "start", work) if (task["dir"] / "start").is_dir() else work.mkdir(parents=True)
        tools_on = harness.agent_tools_on(task)       # AGTOOLS1: render/frames tools for design, document and media tasks (auto)
        harness.write_project_rules(work, agent_tools=tools_on)
        # The model's folder (harness.md, skills, scripts) is copied into the work folder: the sandbox does not see /home, and the
        # skill pointer names paths the model can read with its own tools (RSIL0c).
        staged = work / ".harness"
        if (Path(harness_root) / model).exists():
            shutil.copytree(Path(harness_root) / model, staged / model)
        # LHSET1: a task may pin the engine's window (LH-B: a compaction forced by a small window) and send follow-up user messages
        # (LH-C: the requirement changes after the first turn); the hidden tests of a big project may need more than 120 s
        task_context = task.get("engine_context") or context or ENGINE_CONTEXT
        cfg = harness.write_config(scratch / "cfg", port, context=task_context, workdir=work, task=task)
        wire_harness(cfg, staged, model, sampling)
        Path(proxy_log).unlink(missing_ok=True)
        bodies = Path(f"{proxy_log}.bodies.jsonl")      # AGENTIC_EVAL_LOG_REQUESTS=1: the request bodies of this run
        bodies.unlink(missing_ok=True)
        log_path = Path(out_dir) / f"{task['id']}__{arm}__{rep}.agent.log"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        extra = {"ro_binds": harness.agent_tools_binds()} if tools_on else {}
        # SIMUSER1: who answers the agent's questions (AGENTIC_EVAL_QUESTIONS: fixed | sim | reject | none), the same for every run of the night
        with harness.question_policy(task, out_dir) as policy, harness.run_abort(abort):
            code, timed_out, secs = harness.run_agent_serve(work, task["prompt"], cfg, scratch / "data", task.get("timeout_s", 900), log_path,
                                                            followups=tuple(task.get("followups") or ()), **extra)
        if bodies.is_file():
            shutil.move(str(bodies), str(Path(out_dir) / f"{task['id']}__{arm}__{rep}.requests.jsonl"))
        engine_failed = harness.engine_failure()
        passed, tail = harness.run_tests(work, task["dir"] / "hidden", task["test_cmd"], timeout=int(task.get("test_timeout_s") or 120))
        soft = harness.LAST_TEST_RUN.get("soft") or {}
        tot = proxy.totals(proxy_log)
        rec = {"arm": arm, "task": task["id"], "family": task["family"], "kind": task["kind"], "rep": rep, "model": model,
               "harness": fingerprint(Path(harness_root) / model), "resolved": bool(passed),
               "score": 1.0 if passed else float(soft.get("score", 0.0)), "tests_total": soft.get("tests_total"),
               "tests_passed": soft.get("tests_passed"), "timed_out": timed_out, "agent_exit": code, "seconds": round(secs, 1),
               "tokens": tot["completion_tokens"], "prompt_tokens": tot["prompt_tokens"], "requests": tot["requests"],
               "request_errors": tot["errors"], "client": client_verdict(proxy_log),
               **harness.count_events(log_path), **harness.project_checks(log_path), **policy.stats()}
        if engine_failed:
            rec["engine_failed"] = engine_failed
        if not passed:
            rec["test_tail"] = tail[-400:]
        events = mechanism_events(work)                     # RSI2E: proof that a mechanism fired, kept with the record and in out_dir
        if events:
            rec["mechanism_events"] = {k: len(v) for k, v in events.items()}
            (Path(out_dir) / f"{task['id']}__{arm}__{rep}.events.json").write_text(json.dumps(events, ensure_ascii=False, indent=0))
        return rec
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _ensure(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


class FakeStand:
    """The shura-qa fake server with fake_worker.py as its policy, and the measuring proxy in front of it (as in the real experiment)."""
    fake = True

    def __init__(self, work):
        self.work = Path(work)

    def __enter__(self):
        self.work.mkdir(parents=True, exist_ok=True)
        self.manifest = self.work / "manifest.json"
        self.log = self.work / "fake_log.jsonl"
        cfg = self.work / "fake.json"
        cfg.write_text(json.dumps({"text": "ok", "n_tokens": 5, "token_delay_ms": 5, "load_seconds": 0.1}))
        self.port = free_port()
        env = {**os.environ, "FAKE_LLAMA_CONFIG": str(cfg), "RSI_FAKE_MANIFEST": str(self.manifest), "RSI_FAKE_LOG": str(self.log)}
        self.proc = subprocess.Popen([sys.executable, "-B", str(FAKE_WORKER), "--host", "127.0.0.1", "--port", str(self.port),
                                      "-m", "fake.gguf"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     start_new_session=True)
        wait_port(self.port)
        self.proxy_log = self.work / "proxy.jsonl"
        self.proxy_port = free_port()
        self.srv = proxy.serve(self.proxy_port, f"http://127.0.0.1:{self.port}", str(self.proxy_log), extra={})
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.srv.shutdown()
        self.srv.server_close()
        stop_own(self.proc)

    def expect(self, task, extra=None):
        """The files the fake writes for this task: the solution's files (right) and the same paths with wrong content (the base).
        `extra`: steps appended to both scripts, e.g. [{"tool": "render", "args": {...}}] (AGTOOLS1) or [{"run": "sh x.sh"}]."""
        sol = Path(task["dir"]) / "solution"
        files = [p for p in sorted(sol.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
        right = [{"path": p.relative_to(sol).as_posix(), "content": p.read_text(encoding="utf-8", errors="replace")} for p in files]
        wrong = [{"path": f["path"], "content": "# fake: a wrong answer\n"} for f in right]
        if task.get("solution_script"):   # GEVAL1b media tasks: the solution is a script that produces the answer file — run it after writing it
            right.append({"path": task["solution_script"], "run": f"sh {task['solution_script']}"})
            wrong.append({"path": task["solution_script"], "run": f"sh {task['solution_script']}"})
        for step in extra or []:
            right.append(dict(step))
            wrong.append(dict(step))
        self.manifest.write_text(json.dumps({"right": right, "wrong": wrong}, ensure_ascii=False))

    def requests(self):
        """What the fake server received so far (the last 200 requests with their bodies), from its /__fake/log."""
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/__fake/log", timeout=10) as r:
            return json.loads(r.read().decode())


def used_mib() -> int:
    """VRAM in use on the card, in MiB (the stand reads it before a start, when its own server is down: this is the desktop's share)."""
    return int(VRAM_USED.read_text()) // 2 ** 20


def last_prompt_tokens(log_path):
    """prompt_tokens of the last request in the proxy's request log (the context the run reached), or None."""
    try:
        lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            usage = (json.loads(line).get("usage") or {})
        except ValueError:
            continue
        if "prompt_tokens" in usage:
            return usage["prompt_tokens"]
    return None


def guard_step(low_since, low_min, free_gib, now, keep=None, window=None):
    """One sample of the VRAM guard. Returns (low_since, low_min, trigger): trigger is (low_seconds, min_free_gib) when the stop is due.
    Free VRAM at or above the keep level resets the window."""
    keep = KEEP_FREE_GIB if keep is None else keep
    window = GUARD_SECONDS if window is None else window
    if free_gib >= keep:
        return None, None, None
    low_since = now if low_since is None else low_since
    low_min = free_gib if low_min is None else min(low_min, free_gib)
    if now - low_since > window:
        return None, None, (round(now - low_since, 2), round(low_min, 3))
    return low_since, low_min, None


class DipRecorder:
    """VRAMDIP1: every time the free VRAM falls below the keep level is a dip, however short. The guard only stops the server after
    GUARD_SECONDS; a dip below that left no trace (QLEV1b/c: five runs with a minimum of 1.32-1.44 GiB, no kill, no duration). Each
    dip is recorded with its length, its minimum, the samples before and during it (`trace`: seconds from its start, free GiB) and,
    taken when it starts, who held the card's memory (`snapshot()`, see vram_by_process)."""

    def __init__(self, keep=None, trace_before=20, trace_max=200, snapshot=None):
        self.keep = KEEP_FREE_GIB if keep is None else keep
        self.before = collections.deque(maxlen=trace_before)
        self.trace_max = trace_max
        self.snapshot = snapshot
        self.open = None

    def step(self, free_gib, now, wall=None):
        """One sample; returns the dips that ended with it (a list of 0 or 1 records)."""
        done = []
        if free_gib < self.keep:
            if self.open is None:
                self.open = {"wall": wall, "start": now, "min_free_GiB": free_gib, "samples": list(self.before), "by_process": None}
                if self.snapshot:
                    try:
                        self.open["by_process"] = self.snapshot()
                    except Exception as error:   # the record of a dip must never stop the guard
                        self.open["by_process"] = {"error": repr(error)[:120]}
            self.open["min_free_GiB"] = min(self.open["min_free_GiB"], free_gib)
            if len(self.open["samples"]) < self.trace_max:
                self.open["samples"].append((now, free_gib))
        else:
            if self.open is not None:
                self.open["samples"].append((now, free_gib))
                done.append(self._finish(now))
        self.before.append((now, free_gib))
        return done

    def flush(self, now):
        """The dip still open when the server stops (or the guard ends)."""
        return [self._finish(now, ended=False)] if self.open is not None else []

    def _finish(self, now, ended=True):
        d, self.open = self.open, None
        start = d["start"]
        return {"wall": d["wall"], "duration_s": round(now - start, 2), "ended": ended, "min_free_GiB": round(d["min_free_GiB"], 3),
                "threshold_GiB": self.keep, "by_process": d["by_process"],
                "trace": [(round(t - start, 2), round(f, 3)) for t, f in d["samples"]]}


def vram_by_process(proc_root="/proc", limit=6):
    """Who holds VRAM on the card right now: the amdgpu DRM clients in /proc/<pid>/fdinfo (drm-memory-vram, KiB), summed per process,
    largest first. Only processes of this user can be read; the sum is not the card's total (kernel buffers and other users are not in it)."""
    out = []
    root = Path(proc_root)
    for pid_dir in root.iterdir():
        if not pid_dir.name.isdigit():
            continue
        clients, comm = {}, None
        try:
            for fd in (pid_dir / "fdinfo").iterdir():
                try:
                    text = fd.read_text(errors="replace")
                except OSError:
                    continue
                if "drm-driver:\tamdgpu" not in text:
                    continue
                client = re.search(r"drm-client-id:\s*(\d+)", text)
                vram = re.search(r"drm-memory-vram:\s*(\d+)\s*KiB", text)
                if vram:
                    clients[client.group(1) if client else fd.name] = max(clients.get(client.group(1) if client else fd.name, 0), int(vram.group(1)))
            if clients:
                comm = (pid_dir / "comm").read_text().strip()
        except OSError:
            continue
        if clients:
            out.append({"pid": int(pid_dir.name), "comm": comm, "vram_MiB": round(sum(clients.values()) / 1024)})
    return sorted(out, key=lambda r: -r["vram_MiB"])[:limit]


def guard_stop_record(*, t, min_free_gib, low_s, desktop_mib, context_tokens, run, threshold=None, window=None):
    """The reason of one VRAM-guard stop, as written to the stand's guard_stops.jsonl and its server.log (RSIL0f)."""
    return {"t": t, "reason": "VRAM guard", "threshold_GiB": KEEP_FREE_GIB if threshold is None else threshold,
            "window_s": GUARD_SECONDS if window is None else window, "min_free_GiB": round(min_free_gib, 3),
            "low_s": round(low_s, 2), "desktop_used_MiB_before_start": desktop_mib, "context_tokens": context_tokens, "run": run}


class GpuStand:
    """The real stand: llama-server under safe-run, one VRAM guard for every start, and every stop by a PID. The server's PID is kept in
    the stand's pid file together with the argv it was started with (RSIL1 fixes, 2026-10-22): alive(), the memory rule and the stop use
    only that PID. A process that answers on the port without being this stand's server is never used and never stopped."""
    fake = False

    def __init__(self, argv_file, work, port=REAL_PORT):
        self.argv = json.loads(Path(argv_file).read_text())
        self.work = Path(work)
        self.port = port
        self.proc = None
        self.restarts = []
        self.stop_reason = None
        self.lock = threading.RLock()
        self.desktop_mib_before = None   # the card's used VRAM when the last start began (llama down: the desktop's share)
        self.current_run = ""           # the run the Runner is in, for the guard's record
        self.guard_stops = []
        self.guard_dips = []             # VRAMDIP1: every time free VRAM fell below the keep level, however short (guard_dips.jsonl)
        self._done = threading.Event()
        self._old_handlers = {}

    def __enter__(self):
        self.work.mkdir(parents=True, exist_ok=True)
        if not (VRAM_USED.exists() and VRAM_TOTAL.exists()):
            raise RuntimeError("no VRAM counters under /sys/class/drm/card1: refuse to start the real stand")
        self.free_before = self._free_gib()
        self._old_handlers = {sig: signal.signal(sig, self._on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            (self.work / "server.log").write_text("", encoding="utf-8")
            with self.lock:
                self._start("start")
            self.proxy_log = self.work / "proxy.jsonl"
            self.proxy_port = free_port()
            self.srv = proxy.serve(self.proxy_port, f"http://127.0.0.1:{self.port}", str(self.proxy_log), extra={})
            threading.Thread(target=self.srv.serve_forever, daemon=True).start()
            threading.Thread(target=self._guard, daemon=True).start()
        except BaseException:
            self.stop_server()     # __exit__ is not called when __enter__ fails: the server must not stay alive
            self._restore_handlers()
            raise
        return self

    def __exit__(self, *exc):
        self._done.set()
        self.srv.shutdown()
        self.srv.server_close()
        self.stop_server()
        self._restore_handlers()
        # VRAM comes back a few seconds after the process ends: wait for it (with a timeout) and keep both numbers
        self.free_after_stop = self._free_gib()
        self.release_target = max(KEEP_FREE_GIB + RELEASE_MARGIN_GIB, self.free_before - 1.0)
        t0 = time.time()
        while self._free_gib() < self.release_target and time.time() - t0 < RELEASE_TIMEOUT_S:
            time.sleep(0.5)
        self.free_released = self._free_gib()
        self.release_wait_s = round(time.time() - t0, 1)
        self.released = self.free_released >= self.release_target

    def alive(self) -> bool:
        """The stand's own server runs, holds the port and answers /health. Another process on the port is not used."""
        pid = self._own_pid()
        return pid is not None and pid in listener_pids(self.port) and health_ok(self.port)

    def restart(self, reason=""):
        """Stop this stand's server by its PID and start it again with the same argv, port and proxy. The record keeps the reason and
        the memory numbers measured before the stop."""
        with self.lock:
            mem = self.server_memory()
            note = reason if mem is None else f"{reason}; heap {mem[0]:.2f} GiB, /dev/zero {mem[1]:.2f} GiB before the stop"
            if self.stop_reason:
                note = f"{note}; {self.stop_reason}"
                self.stop_reason = None
            self.restarts.append({"t": time.time(), "reason": note[:200]})
            self.stop_server()
            self._start(f"restart: {note}")

    def stop_server(self):
        """Stop this stand's own llama-server by its PID from the pid file, then its wrapper (safe-run and its scope). Never by name."""
        with self.lock:
            pid = self._own_pid()
            if pid is not None:
                stop_pid(pid)
            if self.proc is not None:
                stop_own(self.proc)
            (self.work / PID_FILE).unlink(missing_ok=True)

    def server_memory(self):
        """(heap, /dev/zero) in GiB of this stand's server, or None when it does not run."""
        pid = self._own_pid()
        if pid is None:
            return None
        try:
            return server_memory_gib(pid)
        except OSError:
            return None

    def memory_over_limit(self) -> bool:
        mem = self.server_memory()
        return mem is not None and over_memory_limit(*mem)

    def _own_pid(self):
        """The PID in the pid file, while that process still runs with the argv this stand started it with."""
        try:
            pid = int(json.loads((self.work / PID_FILE).read_text(encoding="utf-8"))["pid"])
        except (OSError, ValueError, KeyError):
            return None
        return pid if pid_running(pid) and cmdline(pid) == list(self.argv) else None

    def _start(self, label):
        """Start the server under safe-run and wait for /health; then record the PID of the llama-server this stand started."""
        self._wait_port_free()
        self.desktop_mib_before = used_mib()
        with open(self.work / "server.log", "a") as log:
            log.write(f"\n--- {label}; waited {wait_mem()} s for MemAvailable >= {SAFE_MIN_FREE_GB} GiB\n")
            self.proc = subprocess.Popen(["bash", str(SAFE_RUN), SAFE_MAX_GB, SAFE_MIN_FREE_GB, "--", *self.argv],
                                         start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        self._wait_health()
        self._adopt()

    def _adopt(self):
        """The one llama-server with this argv that holds the port is the stand's own: its PID goes to the pid file."""
        pids = [p for p in find_server_pids(self.argv) if p in listener_pids(self.port)]
        if len(pids) != 1:
            raise RuntimeError(f"expected one llama-server with this argv on port {self.port}, found {pids}")
        (self.work / PID_FILE).write_text(json.dumps({"pid": pids[0], "port": self.port, "argv": self.argv}), encoding="utf-8")

    def _wait_port_free(self, timeout=PORT_FREE_TIMEOUT_S):
        """Nothing may listen on the port before a start. A listener that is not this stand's is reported and left alone."""
        deadline = time.time() + timeout
        while True:
            busy = listener_pids(self.port)
            if not busy:
                return
            if time.time() > deadline:
                raise SystemExit(f"port {self.port} is held by pid {sorted(busy)}, not by this stand: nothing is stopped or started")
            time.sleep(0.5)

    def _on_signal(self, signum, frame):
        """SIGINT or SIGTERM: the stand's own server is stopped by its PID first; the `with` block then ends (__exit__ waits for VRAM)."""
        self.stop_server()
        raise SystemExit(128 + signum)

    def _restore_handlers(self):
        for sig, handler in self._old_handlers.items():
            signal.signal(sig, handler)

    def _free_gib(self) -> float:
        return (int(VRAM_TOTAL.read_text()) - int(VRAM_USED.read_text())) / 2 ** 30

    def _guard(self):
        """One VRAM guard for the whole stand, every start included: VRAM below the desktop's share for longer than GUARD_SECONDS stops
        the server by its PID; the next run's alive() check starts it again."""
        low_since = low_min = None
        dips = DipRecorder(snapshot=vram_by_process)
        while not self._done.is_set():
            running = self.proc is not None and self.proc.poll() is None
            if not running:
                low_since = low_min = None
                for d in dips.flush(time.time()):
                    self._record_dip(d)
            else:
                now, free = time.time(), self._free_gib()
                for d in dips.step(free, now, time.strftime("%Y-%m-%dT%H:%M:%S") + f".{int(now * 1000) % 1000:03d}"):
                    self._record_dip(d)
                low_since, low_min, trigger = guard_step(low_since, low_min, free, now)
                if trigger:
                    low_s, min_free = trigger
                    with self.lock:
                        self._record_guard_stop(min_free, low_s)
                        self.stop_server()
                    low_since = low_min = None
            time.sleep(GUARD_POLL_S)

    def _record_dip(self, dip):
        """VRAMDIP1: a dip below the keep level (it did not reach the stop, or it did): length, minimum, trace, who held the memory."""
        dip = dict(dip, run=self.current_run or "(no run)",
                   context_tokens=last_prompt_tokens(getattr(self, "proxy_log", self.work / "proxy.jsonl")))
        self.guard_dips.append(dip)
        with open(self.work / "guard_dips.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(dip, ensure_ascii=False) + "\n")

    def _record_guard_stop(self, min_free, low_s):
        """RSIL0f: the stop's reason goes to guard_stops.jsonl and server.log of the stand, and to the next restart's note."""
        rec = guard_stop_record(t=time.strftime("%Y-%m-%dT%H:%M:%S"), min_free_gib=min_free, low_s=low_s,
                                desktop_mib=self.desktop_mib_before,
                                context_tokens=last_prompt_tokens(getattr(self, "proxy_log", self.work / "proxy.jsonl")),
                                run=self.current_run or "(no run)")
        self.guard_stops.append(rec)
        self.stop_reason = (f"VRAM guard: free {rec['min_free_GiB']} GiB below {rec['threshold_GiB']} GiB for {rec['low_s']} s, "
                            f"desktop {rec['desktop_used_MiB_before_start']} MiB, context {rec['context_tokens']} tokens, run {rec['run']}")
        with open(self.work / "guard_stops.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(self.work / "server.log", "a", encoding="utf-8") as log:
            log.write(f"\n--- VRAM guard stop: {self.stop_reason}\n")

    def _wait_health(self, timeout=900):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early (see {self.work / 'server.log'})")
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                conn.request("GET", "/health")
                if conn.getresponse().status == 200:
                    return
            except OSError:
                pass
            time.sleep(1)
        raise RuntimeError("llama-server did not become healthy")


def client_verdict(log_path):
    """RSI1 criterion «200 и корректно»: every request of the run got HTTP 200 and an answer that finished (a finish_reason, or a
    stream's [DONE]). A request the agent itself hung up on (its timeout, its abort) is the model's run, not the stand's fault. Returns
    {"ok", "requests", "bad": [{"status", "why"}]} (at most 5 listed)."""
    bad, n = [], 0
    try:
        lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        n += 1
        if r.get("status") != 200:
            bad.append({"status": r.get("status"), "why": "status"})
        elif not (r.get("done") or r.get("finish_reason")) and not r.get("client_closed"):
            bad.append({"status": 200, "why": "unfinished"})
    return {"ok": n >= 1 and not bad, "requests": n, "bad": bad[:5], "bad_total": len(bad)}


class Runner:
    """One run on a stand. `root` is a folder holding <model>/harness.md (the candidate or the base)."""

    def __init__(self, stand, scratch_root, context=None):
        self.stand = stand
        self.scratch_root = Path(scratch_root)
        self.context = context            # None: run_task's default (ENGINE_CONTEXT, the product's); set: the stand's or the command line's
        self.discarded = []          # runs not counted (stand or client failures), for the morning report
        os.environ["AGENTIC_EVAL_LOG_REQUESTS"] = "1"   # RSI2D: every run keeps its request bodies, the delivery check reads them

    def _check_delivery(self, task, root, model, arm, rep, out, rec):
        """RSI2D: a run whose harness text did not reach the model is not a measurement of that harness: it is discarded and the
        cycle stops with an error (a silent miss is how RSI1 measured 17 candidates that the model never saw)."""
        requests = Path(out) / f"{task['id']}__{arm}__{rep}.requests.jsonl"
        check = harness_delivered(requests, root, model)
        rec["harness_delivery"] = check
        if check.get("checked") and not check.get("delivered"):
            self.discarded.append({"t": time.time(), "run": f"{task['id']} {arm} {rep}", "harness_missing": check})
            raise SystemExit(f"the harness text did not reach the model in {task['id']} {arm} {rep}: {check}; the run is not counted "
                             f"and the cycle is stopped (check the plugin wiring before continuing)")
        return rec

    def run(self, task, root, model, arm, rep, out):
        if self.stand.fake:
            self.stand.expect(task)
            rec = run_task(task, Path(root), model, arm, rep, self.stand.proxy_port, self.stand.proxy_log, Path(out), self.scratch_root,
                           **({"context": self.context} if self.context else {}))
            return self._check_delivery(task, root, model, arm, rep, out, rec)
        sampling = getattr(self.stand, "sampling", None)
        strict = getattr(self.stand, "client_strict", False)
        for attempt in range(1, RUN_ATTEMPTS + 1):
            if not self.stand.alive():                       # before the run: a dead server is restarted, the run is not counted
                self.stand.restart(reason=f"before {task['id']} {arm} {rep} (attempt {attempt}): server not alive")
            elif self.stand.memory_over_limit():             # RSIL1 fix: a planned restart at the run boundary, before the run starts
                self.stand.restart(reason=f"planned before {task['id']} {arm} {rep}: heap + /dev/zero over {SERVER_MEM_LIMIT_GIB:g} GiB")
            self.stand.current_run = f"{task['id']} {arm} {rep} (attempt {attempt})"
            # a crash of the llama-server inside the run (RSI1 night 1: "corrupted size vs. prev_size"): the gateway does not restart it for
            # the agent that is already waiting, the agent hangs until the deadline and the run looked like the model's timeout. The stand's
            # crash counter ends the agent at once and the run is not counted.
            crashes = getattr(self.stand, "crashes", None)
            before = crashes() if crashes else 0
            abort = (lambda: crashes() > before) if crashes else None
            try:
                rec = run_task(task, Path(root), model, arm, rep, self.stand.proxy_port, self.stand.proxy_log, Path(out), self.scratch_root,
                               **({"context": self.context} if self.context else {}), **({"sampling": sampling} if sampling else {}),
                               **({"abort": abort} if abort else {}))
            except (OSError, RuntimeError, ValueError) as err:   # a connection or an output error during the run: not a record
                rec = {"requests": 0, "error": repr(err)[:200]}
            client = rec.get("client") or {}
            engine = rec.get("engine_failed")                  # the agent engine (Bun) crashed or exited: not the model's failure either
            crashed = (bool(crashes) and crashes() > before) or bool(engine)
            if not crashed and self.stand.alive() and rec.get("requests", 0) >= 1 and (not strict or client.get("ok")):
                return self._check_delivery(task, root, model, arm, rep, out, rec)
            # the server died during the run, the model got no request, or (gateway stand) the client got an error or an unfinished
            # answer: the record is discarded and kept for the report, the run is repeated on a fresh stand
            self.discarded.append({"t": time.time(), "run": f"{task['id']} {arm} {rep}", "attempt": attempt,
                                   "requests": rec.get("requests"), "client": client, "error": rec.get("error"),
                                   **({"server_crashed": True} if crashed and not engine else {}), **({"engine_failed": engine} if engine else {})})
            self.stand.restart(reason=f"after {task['id']} {arm} {rep}: requests {rec.get('requests')}, alive {self.stand.alive()}, "
                                      f"client bad {client.get('bad_total')}")
        raise SystemExit(f"the server kept failing on {task['id']} {arm} {rep}: stopped, the run is not counted")
