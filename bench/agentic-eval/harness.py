"""Runs ONE (arm, task, repetition): a clean work folder, the real agent (ShuraCode, one-shot), the hidden tests, a result record.

The agent is the product's own: `shuracode run --format json --auto`. Only what the experiment must control is replaced, and
identically for every arm: the configuration folder (no memory plugin, no personal rules), the model endpoint (the measuring
proxy) and the work folder (tmpfs). The hidden tests are copied in AFTER the agent has finished and decide the outcome.

Task layout (bench/agentic-eval/tasks/<family>/<variant>/):
    task.json   {id, family, kind, lang, prompt, test_cmd, timeout_s}
    start/      the files the agent starts with
    hidden/     the tests that decide (never shown to the agent)
    solution/   a reference solution overlaid on start/ (used only to validate the task)
"""
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = HERE / "tasks"
STOCK_RULES = """# Verification discipline

- Never claim a task is complete, a bug is fixed, or a test passes without actually running it and seeing the real output.
- If genuinely uncertain about a fact, say so explicitly rather than presenting a guess as confirmed.
- Before reporting a fix as done: re-run the failing test or reproduction and look at the real output.

# File paths in tool calls

- Refer to files by paths RELATIVE to the current project directory. Never type out an absolute /home/... path yourself.
"""


def load_task(task_dir):
    task_dir = Path(task_dir)
    t = json.loads((task_dir / "task.json").read_text())
    t["dir"] = task_dir
    return t


def all_tasks(root=TASKS):
    """Every task under `root`. The default folder is generated from families/*.py on first use (it is not stored in git)."""
    if Path(root) == TASKS and not any(Path(root).glob("*/*/task.json")):
        import build_tasks
        build_tasks.build(TASKS)
    return [load_task(p.parent) for p in sorted(Path(root).glob("*/*/task.json"))]


def write_config(cfg_dir, port, context=32768, output=8192, effort="medium"):
    """A ShuraCode configuration for the experiment: one provider (the measuring proxy), no plugins, stock rules only."""
    cfg_dir = Path(cfg_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "AGENTS.md").write_text(STOCK_RULES)
    conf = {
        "$schema": "https://opencode.ai/config.json", "autoupdate": False, "share": "disabled",
        "provider": {"eval": {"npm": "@ai-sdk/openai-compatible", "name": "eval", "options": {"baseURL": f"http://127.0.0.1:{port}/v1"},
                              "models": {"m": {"name": "model under test", "limit": {"context": context, "output": output},
                                               "settings": {"reasoningEffort": effort}}}}},
        "model": "eval/m", "small_model": "eval/m", "instructions": [str(cfg_dir / "AGENTS.md")],
        "agent": {"build": {"temperature": 0.6, "top_p": 0.95}},
        "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny", "external_directory": "deny"},
    }
    (cfg_dir / "shuracode.json").write_text(json.dumps(conf, indent=2))
    (cfg_dir / "tui.json").write_text("{}")
    return cfg_dir


def run_tests(workdir, hidden, test_cmd, timeout=120):
    """Copy the hidden tests over the work folder and run the task's test command. Returns (passed, output_tail)."""
    dest = Path(workdir) / "hidden_tests"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(hidden, dest)
    for cache in Path(workdir).rglob("__pycache__"):          # a stale .pyc (same size, same second) must never decide the outcome
        shutil.rmtree(cache, ignore_errors=True)
    try:
        r = subprocess.run(test_cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    except subprocess.TimeoutExpired:
        return False, "hidden tests timed out"
    return r.returncode == 0, (r.stdout + r.stderr)[-1500:]


ENGINE = os.environ.get("SHURACODE_ENGINE", str(Path.home() / ".local/share/shuracode/libexec/shuracode"))
ENGINE_ENV = {"OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_SHARE": "1", "OPENCODE_DISABLE_MODELS_FETCH": "1",
              "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1", "OPENCODE_DISABLE_TERMINAL_TITLE": "1",
              "OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "1", "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1", "NO_COLOR": "1"}


def sandboxed(argv, scratch, workdir, engine=None, env=None):
    """Wrap a command in bubblewrap so that an unattended agent can change NOTHING outside its scratch folder: the system is
    read-only, the user's home is not there at all (no keys, no dotfiles), the environment is cleared. Network stays on
    (the agent talks to the measuring proxy on 127.0.0.1). Returns the argv to run."""
    engine_dir = str(Path(engine or ENGINE).resolve().parent)
    scratch = str(scratch)
    cmd = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-ipc", "--clearenv",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
           "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin", "--ro-bind", "/etc", "/etc",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--ro-bind", engine_dir, engine_dir,
           "--bind", scratch, scratch, "--chdir", str(workdir),
           "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin", "--setenv", "HOME", f"{scratch}/home",
           "--setenv", "XDG_DATA_HOME", f"{scratch}/home/.local/share", "--setenv", "XDG_CACHE_HOME", f"{scratch}/home/.cache",
           "--setenv", "XDG_CONFIG_HOME", f"{scratch}/home/.config", "--setenv", "LANG", "C.UTF-8", "--setenv", "TERM", "dumb"]
    for k, v in (env or {}).items():
        cmd += ["--setenv", k, v]
    return cmd + list(argv)


def agent_argv(workdir, prompt):
    return [ENGINE, "run", "--format", "json", "--auto", "--dir", str(workdir), prompt]


def run_agent(workdir, prompt, cfg_dir, data_dir, timeout_s, log_path):
    """One-shot agent run in its own process group, killed at the deadline. Returns (exit_code or None, timed_out, seconds).
    The agent runs inside the sandbox (set AGENTIC_EVAL_NO_SANDBOX=1 only for a supervised dry run)."""
    scratch = Path(workdir).parent
    (scratch / "home").mkdir(parents=True, exist_ok=True)
    env = {**ENGINE_ENV, "OPENCODE_CONFIG": str(Path(cfg_dir) / "shuracode.json"), "OPENCODE_CONFIG_DIR": str(cfg_dir)}
    argv = agent_argv(workdir, prompt)
    if os.environ.get("AGENTIC_EVAL_NO_SANDBOX") == "1":
        run_env, argv = {**os.environ, **env, "HOME": f"{scratch}/home"}, argv
    else:
        argv, run_env = sandboxed(argv, scratch, workdir, env=env), dict(os.environ)
    t0 = time.monotonic()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(argv, cwd=workdir, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=run_env,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out, code = True, None
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    return code, timed_out, time.monotonic() - t0


def count_events(log_path):
    """Rough accounting of what the agent did, from its JSON event stream (the decisive numbers come from the proxy)."""
    n = errors = 0
    for line in Path(log_path).read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        n += 1
        text = json.dumps(ev)
        if '"error"' in text and ev.get("type") in ("tool_use", "tool", "error"):
            errors += 1
    return {"events": n, "tool_errors": errors}


def run_one(task, arm, rep, port, proxy_log, out_dir, scratch_root="/dev/shm/agentic-eval", context=32768, runner=run_agent):
    """Everything for one cell of the experiment. `runner` is replaceable so that the plumbing can be tested without a model."""
    scratch = Path(tempfile.mkdtemp(prefix=f"{task['id']}-{arm}-{rep}-", dir=_ensure(scratch_root)))
    try:
        work = scratch / "work"
        shutil.copytree(task["dir"] / "start", work)
        cfg = write_config(scratch / "cfg", port, context=context)
        Path(proxy_log).unlink(missing_ok=True)
        log_path = Path(out_dir) / f"{task['id']}__{arm}__{rep}.agent.log"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        code, timed_out, secs = runner(work, task["prompt"], cfg, scratch / "data", task.get("timeout_s", 900), log_path)
        passed, tail = run_tests(work, task["dir"] / "hidden", task["test_cmd"])
        from proxy import totals
        tot = totals(proxy_log)
        rec = {"arm": arm, "task": task["id"], "family": task["family"], "kind": task["kind"], "rep": rep, "resolved": bool(passed),
               "timed_out": timed_out, "agent_exit": code, "seconds": round(secs, 1),
               "tokens": tot["completion_tokens"], "prompt_tokens": tot["prompt_tokens"], "requests": tot["requests"],
               "request_errors": tot["errors"], **count_events(log_path)}
        if not passed:
            rec["test_tail"] = tail[-400:]
        return rec
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _ensure(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
