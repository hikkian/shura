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
import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
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

# Temporary files

- Put scratch files (probes, throwaway scripts, quick checks) inside this project, in ./.tmp/ (create it if needed).
  /tmp is outside the project and is not allowed.
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


# output: the product limit (shuracode/config/shuracode.json.in, limit.output 16384); was 8192, which cut long turns short
# QHARN1: the product's plugin code (read-only source) and its dependency; copied per run so that a run cannot change them.
PLUGIN_SRC = Path(os.environ.get("AGENTIC_EVAL_PLUGIN_SRC", "/home/alisher/Projects/shuracode-integration/plugin"))
RELIABILITY_PLUGIN_SRC = Path(os.environ.get("AGENTIC_EVAL_RELIABILITY_PLUGIN_SRC", "/home/alisher/Projects/shuracode-integration/plugin/reliability"))
NODE_MODULES_SRC = Path(os.environ.get("AGENTIC_EVAL_NODE_MODULES", str(Path.home() / ".config/shuracode/node_modules")))
PLUGIN_MODULES = ("server.ts", "prefix-stability.ts", "session-header.ts")
# imported by server.ts, not plugin entries: kept under plugin/lib so directory-based discovery does not load them alone
HELPER_MODULES = ("prefix-stability.ts", "session-header.ts")
RELIABILITY = {"completion-gate": "completionGate", "soft-edit": "softEdit", "loop-detector": "loopDetector", "task-card": "taskCard"}
# AGTOOLS1: the agent's eyes (render: Chromium screenshot + layout problems; frames: video frames + ffprobe) as a product plugin.
AGENT_TOOLS_SRC = Path(os.environ.get("AGENTIC_EVAL_AGENT_TOOLS_SRC", "/home/alisher/Projects/shuracode-integration/plugin/agent-tools"))
AGENT_TOOLS_CATEGORIES = {"design", "document", "media"}
QA_HOME = Path(os.environ.get("SHURA_QA_HOME", str(Path.home() / ".local/share/shura-qa")))


def agent_tools_on(task=None):
    """AGENTIC_EVAL_AGENT_TOOLS=1 (always), 0 (never) or auto (the default: on for the render/media task categories of general-v1)."""
    raw = os.environ.get("AGENTIC_EVAL_AGENT_TOOLS", "auto").strip().lower()
    if raw in ("1", "on", "all", "true"):
        return True
    if raw in ("0", "off", "none", "false"):
        return False
    return bool(task) and task.get("category") in AGENT_TOOLS_CATEGORIES


def agent_tools_binds():
    """Read-only folders the sandbox needs for the render tool: the shura-qa toolkit (Playwright + Chromium) and the uv Python its
    venv points at (the venv's python is a symlink into ~/.local/share/uv)."""
    binds = []
    if QA_HOME.is_dir():
        binds.append(str(QA_HOME))
        py = QA_HOME / "venv" / "bin" / "python"
        if py.exists():
            real = Path(os.path.realpath(py))
            uv = next((p for p in real.parents if p.name == "python" and p.parent.name == "uv"), None)
            if uv:
                binds.append(str(uv))
    return binds


def reliability_flags():
    """Which reliability plugins run in an arm: AGENTIC_EVAL_RELIABILITY=all, none (the default, as the shipped template) or a comma list."""
    raw = os.environ.get("AGENTIC_EVAL_RELIABILITY", "none").strip()
    if raw == "all":
        return set(RELIABILITY)
    return {x.strip() for x in raw.split(",") if x.strip() in RELIABILITY}


def install_plugins(cfg_dir, reliability=None, agent_tools=False):
    """Copy the product's plugins next to the experiment's config and return the `plugin` list for the config."""
    dest = Path(cfg_dir).parent / "plugin"
    (dest / "reliability").mkdir(parents=True, exist_ok=True)
    for name in PLUGIN_MODULES:
        target = dest / "lib" / name if name in HELPER_MODULES else dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PLUGIN_SRC / name, target)
    # The helpers (HELPER_MODULES) are imported, not plugin entries. Keep them below plugin/lib to avoid
    # directory-based plugin discovery trying to load it as a standalone module.
    server_file = dest / "server.ts"
    source = server_file.read_text(encoding="utf-8")
    for name in HELPER_MODULES:
        source = source.replace(f'"./{name}"', f'"./lib/{name}"')
    server_file.write_text(source, encoding="utf-8")
    for src in sorted(RELIABILITY_PLUGIN_SRC.glob("*.ts")):
        shutil.copy2(src, dest / "reliability" / src.name)
    # Plugin runtime dependencies must accompany copied plugins into scratch (bwrap cannot see /home).
    for package in ("@opencode-ai", "zod"):
        source = NODE_MODULES_SRC / package
        if source.is_dir():
            shutil.copytree(source, Path(cfg_dir).parent / "node_modules" / package, dirs_exist_ok=True)
    scratch_root = Path(cfg_dir).parent
    scratch_modules = scratch_root / "node_modules"
    for package_root in (Path(cfg_dir), scratch_root / "home" / ".config" / "opencode"):
        package_root.mkdir(parents=True, exist_ok=True)
        resolver_modules = package_root / "node_modules"
        if scratch_modules.is_dir() and not resolver_modules.exists():
            resolver_modules.symlink_to(scratch_modules, target_is_directory=True)
    on = reliability_flags() if reliability is None else set(reliability)
    entries = [str(dest / "server.ts")]
    for name, key in RELIABILITY.items():
        entries.append([str(dest / "reliability" / f"{name}.ts"), {key: name in on}])
    if agent_tools and AGENT_TOOLS_SRC.is_dir():
        shutil.copytree(AGENT_TOOLS_SRC, dest / "agent-tools", dirs_exist_ok=True)
        entries.append([str(dest / "agent-tools" / "agent-tools.ts"),
                        {"agentTools": True, "vision": "auto", "python": str(QA_HOME / "venv" / "bin" / "python"), "browsersPath": str(QA_HOME / "browsers")}])
    return entries


def searxng_mcp_source():
    """The cached mcp-searxng package: $GEVAL_MCP_SEARXNG (a node_modules folder holding mcp-searxng) or the newest npx cache entry."""
    env = os.environ.get("GEVAL_MCP_SEARXNG")
    if env:
        return Path(env)
    hits = sorted(Path.home().glob(".npm/_npx/*/node_modules/mcp-searxng/package.json"), key=lambda p: p.stat().st_mtime)
    return hits[-1].parent.parent if hits else None


def install_searxng_mcp(cfg_dir):
    """Copy the tool filter, its allowlist and the mcp-searxng package (with its node_modules) into the scratch folder; return that folder."""
    integ = Path(os.environ.get("SHURACODE_INTEGRATION", "/home/alisher/Projects/shuracode-integration"))
    dest = Path(cfg_dir).parent / "mcp"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("mcp-tool-filter.js", "searxng-tools-allowlist.json"):
        shutil.copy2(integ / "mcp" / name, dest / name)
    src = searxng_mcp_source()
    if src is None or not (src / "mcp-searxng").is_dir():
        raise RuntimeError("mcp-searxng is not cached (npx) and GEVAL_MCP_SEARXNG is not set: web tasks cannot run")
    if not (dest / "node_modules").is_dir():
        shutil.copytree(src, dest / "node_modules", symlinks=True)
    return dest


AGENT_TOOLS_RULE = ("\n\n## Seeing your work\n\nYou have two tools that show you results: `render` (an HTML file or URL -> screenshot at a viewport width "
                    "plus layout problems; use 390 and 1280) and `frames` (a video -> frames and an ffprobe summary). After writing or changing a page or a "
                    "video, look at it with them before you finish.\n")


def write_project_rules(work, agent_tools=False):
    """The rules the product keeps in the project's AGENTS.md. They used to sit in the config folder, which the model reached by an
    absolute path and was denied (QHARN1); now they are the project's own file, as in a real project."""
    path = Path(work) / "AGENTS.md"
    if not path.exists():
        path.write_text(STOCK_RULES + (AGENT_TOOLS_RULE if agent_tools else ""), encoding="utf-8")


def write_config(cfg_dir, port, context=32768, output=16384, effort="medium", plugins=True, reliability=None,
                 arm=None, workdir=None, task=None):
    """A ShuraCode config for the experiment. Optional `arm` applies one QLEV1 shoulder in this scratch config."""
    cfg_dir = Path(cfg_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    qlev_arm = None
    if arm is not None:
        from qlev.arms import normalize_arm, reliability_for_arm
        qlev_arm = normalize_arm(arm)
        reliability = reliability_for_arm(qlev_arm)
    tools_on = agent_tools_on(task)
    plugin_entries = install_plugins(cfg_dir, reliability, agent_tools=tools_on) if plugins else []
    conf = {
        "$schema": "https://opencode.ai/config.json", "autoupdate": False, "share": "disabled",
        "provider": {"eval": {"npm": "@ai-sdk/openai-compatible", "name": "eval", "options": {"baseURL": f"http://127.0.0.1:{port}/v1"},
                              "models": {"m": {"name": "model under test", "tool_call": True, "reasoning": True,
                                               # AGTOOLS1: the model under test has vision (mmproj in the candidate layouts), as the product config says;
                                               # without "image" here the agent-tools plugin sends text only
                                               **({"attachment": True, "modalities": {"input": ["text", "image"], "output": ["text"]}} if tools_on else {}),
                                               "limit": {"context": context, "output": output},
                                               "options": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
                                               "settings": {"reasoningEffort": effort},
                                               "variants": {"low": {"settings": {"reasoningEffort": "low"}}}}}}},
        "model": "eval/m", "small_model": "eval/m", "instructions": [],
        "agent": {"build": {"temperature": 0.6, "top_p": 0.95}},
        "plugin": plugin_entries,
        # RSI2E: external_directory "allow": the sandbox (bwrap) already isolates the run, and with "deny" a mistyped absolute path
        # (/home/al/...) was a permission refusal the model retried (124 refusals in 45 of 72 RSI1 base runs); now it is a plain not-found
        "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny", "external_directory": os.environ.get("AGENTIC_EVAL_EXTERNAL_DIRECTORY", "allow")},
    }
    if task is not None and task.get("needs") == "searxng" and os.environ.get("GEVAL_SEARXNG") == "1":
        # GEVAL1 web tasks: the product's SearXNG MCP (search + page read), the same allowlist as the shipped config. GEVAL1b: the sandbox
        # cannot see /home (no npx cache, no integration checkout), so the filter, the allowlist and the cached mcp-searxng package are
        # copied next to the config and started with node directly — nothing is downloaded at run time.
        mcp_dir = install_searxng_mcp(cfg_dir)
        conf["mcp"] = {"searxng": {"type": "local", "enabled": True,
                                   "command": ["node", str(mcp_dir / "mcp-tool-filter.js"), str(mcp_dir / "searxng-tools-allowlist.json"),
                                               "--", "node", str(mcp_dir / "node_modules" / "mcp-searxng" / "dist" / "cli.js")],
                                   "environment": {"SEARXNG_URL": os.environ.get("SEARXNG_URL", "http://127.0.0.1:8888")}}}
    if qlev_arm is not None:
        from qlev.arms import apply_arm
        apply_arm(conf, qlev_arm, cfg_dir, workdir=workdir, task=task)
    (cfg_dir / "shuracode.json").write_text(json.dumps(conf, indent=2))
    (cfg_dir / "tui.json").write_text("{}")
    return cfg_dir


def run_tests(workdir, hidden, test_cmd, timeout=120, log_path=None):
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
    except subprocess.TimeoutExpired as exc:
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            Path(log_path).write_text(stdout + stderr, encoding="utf-8")
        return False, "hidden tests timed out"
    output = r.stdout + r.stderr
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(output, encoding="utf-8")
    LAST_TEST_RUN["soft"] = soft_score(test_cmd, r.returncode, output)
    return r.returncode == 0, output[-1500:]


LAST_TEST_RUN = {"soft": None}   # the soft metric of the last run_tests call (run_one reads it; keeps the old signature)

_UNITTEST_RAN = re.compile(r"^Ran (\d+) tests?", re.M)
_UNITTEST_FAIL = re.compile(r"^FAILED \((.*)\)", re.M)
_NODE_TESTS = re.compile(r"^(?:#|\u2139)\s*tests (\d+)", re.M)     # tap ("# tests N") and spec ("ℹ tests N") reporters
_NODE_PASS = re.compile(r"^(?:#|\u2139)\s*pass (\d+)", re.M)
_GRADE_STAGES = (("production files must remain unchanged", 0.0), ("add discoverable tests", 0.25),
                 ("new tests must pass", 0.5), ("mutation anchor mismatch", 0.5), ("tests do not detect", 0.75))
_GRADE_EXIT_CJS = {2: 0.0, 3: 0.25, 4: 0.5, 5: 0.5, 6: 0.75}


def soft_score(test_cmd, returncode, output):
    """RSIT1: the soft metric of one hidden-test run: {"score": 0..1, "tests_total", "tests_passed", "how"} (reports/rsit1/FORMAT.md §4).
    unittest: passed = ran - failures - errors (an import or syntax error counts as 0 of the known total, unknown total = 0);
    node:test: "# pass P" of "# tests N"; grade scripts (tests-only): the stage reached; role checks: the last JSON line."""
    cmd = " ".join(str(c) for c in test_cmd)
    text = output or ""
    if "check.py" in cmd:
        for line in reversed(text.strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    score = float(obj.get("score", 1.0 if returncode == 0 else 0.0))
                    return {"score": max(0.0, min(1.0, score)), "tests_total": obj.get("checks"), "tests_passed": obj.get("ok"), "how": "score"}
                except (ValueError, TypeError):
                    break
        return {"score": 1.0 if returncode == 0 else 0.0, "tests_total": None, "tests_passed": None, "how": "score"}
    if "grade.py" in cmd or "grade.cjs" in cmd:
        if returncode == 0:
            return {"score": 1.0, "tests_total": 4, "tests_passed": 4, "how": "stages"}
        if "grade.cjs" in cmd and returncode in _GRADE_EXIT_CJS:
            score = _GRADE_EXIT_CJS[returncode]
        else:
            score = 0.0
            for marker, value in _GRADE_STAGES:
                if marker in text:
                    score = value
        return {"score": score, "tests_total": 4, "tests_passed": int(round(score * 4)), "how": "stages"}
    if "unittest" in cmd:
        ran = _UNITTEST_RAN.search(text)
        total = int(ran.group(1)) if ran else None
        if returncode == 0 and total:
            return {"score": 1.0, "tests_total": total, "tests_passed": total, "how": "counts"}
        if total is None:
            return {"score": 0.0, "tests_total": None, "tests_passed": 0, "how": "counts"}
        failed = 0
        m = _UNITTEST_FAIL.search(text)
        if m:
            for part in m.group(1).split(","):
                k, _, v = part.strip().partition("=")
                if k.strip() in ("failures", "errors") and v.strip().isdigit():
                    failed += int(v)
        passed = max(0, total - failed)
        return {"score": passed / total if total else 0.0, "tests_total": total, "tests_passed": passed, "how": "counts"}
    if "node" in cmd and "--test" in cmd:
        n, p = _NODE_TESTS.search(text), _NODE_PASS.search(text)
        if n and p and int(n.group(1)) > 0:
            total, passed = int(n.group(1)), int(p.group(1))
            return {"score": passed / total, "tests_total": total, "tests_passed": passed, "how": "counts"}
        return {"score": 1.0 if returncode == 0 else 0.0, "tests_total": None, "tests_passed": None, "how": "counts"}
    return {"score": 1.0 if returncode == 0 else 0.0, "tests_total": None, "tests_passed": None, "how": "exit"}


ENGINE = os.environ.get("SHURACODE_ENGINE", str(Path.home() / ".local/share/shuracode/libexec/shuracode"))
# FORK1d: npm_config_offline=true. Without it the engine's start looks up registry.npmjs.org (on a sandbox resolver that
# does not answer: 10 s per lookup, about 70 s per start); with it the start takes about 3 s. Set AGENTIC_EVAL_NPM_OFFLINE=0 to undo.
_NPM_OFFLINE = {"npm_config_offline": "true"} if os.environ.get("AGENTIC_EVAL_NPM_OFFLINE", "1") == "1" else {}
ENGINE_ENV = {**_NPM_OFFLINE, "OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_SHARE": "1", "OPENCODE_DISABLE_MODELS_FETCH": "1",
              "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1", "OPENCODE_DISABLE_TERMINAL_TITLE": "1",
              "OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "1", "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1", "NO_COLOR": "1"}


def sandboxed(argv, scratch, workdir, engine=None, env=None, ro_binds=(), unshare_net=False):
    """Wrap a command in bubblewrap so that an unattended agent can change NOTHING outside its scratch folder: the system is
    read-only, the user's home is not there at all (no keys, no dotfiles), the environment is cleared. Network stays on
    (the agent talks to the measuring proxy on 127.0.0.1) unless unshare_net=True: then the sandbox has only its own loopback
    and the outside is reached through unix sockets in scratch (see bridge_plan). Returns the argv to run."""
    engine_dir = str(Path(engine or ENGINE).resolve().parent)
    scratch = str(scratch)
    cmd = ["bwrap"] + (["--unshare-net"] if unshare_net else []) + ["--die-with-parent", "--unshare-pid", "--unshare-ipc", "--clearenv",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
           "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin", "--ro-bind", "/etc", "/etc",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--ro-bind", engine_dir, engine_dir,
           "--bind", scratch, scratch, "--chdir", str(workdir),
           "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin", "--setenv", "HOME", f"{scratch}/home",
           "--setenv", "XDG_DATA_HOME", f"{scratch}/home/.local/share", "--setenv", "XDG_CACHE_HOME", f"{scratch}/home/.cache",
           "--setenv", "XDG_CONFIG_HOME", f"{scratch}/home/.config", "--setenv", "LANG", "C.UTF-8", "--setenv", "TERM", "dumb"]
    for b in ro_binds:            # AGTOOLS1: extra read-only folders (the shura-qa Playwright for the render tool)
        cmd += ["--ro-bind", str(b), str(b)]
    for k, v in (env or {}).items():
        cmd += ["--setenv", k, v]
    return cmd + list(argv)


def model_port_from_config(cfg_dir):
    """Port of the model server from the first provider baseURL (http://127.0.0.1:PORT/v1) in <cfg_dir>/shuracode.json."""
    conf = json.loads((Path(cfg_dir) / "shuracode.json").read_text(encoding="utf-8"))
    for prov in (conf.get("provider") or {}).values():
        m = re.match(r"https?://[^/:]+:(\d+)", str((prov.get("options") or {}).get("baseURL", "")))
        if m:
            return int(m.group(1))
    raise RuntimeError("no provider baseURL with a port in shuracode.json")


def bridge_plan(scratch, model_port, engine_port, host_port, engine_argv):
    """SANDNET1: command lines of the unix-socket bridge for a sandbox without a network. Pure function, starts nothing.
    host_model: outside, <scratch>/model.sock -> the model server on 127.0.0.1:model_port.
    host_engine: outside, 127.0.0.1:host_port -> <scratch>/engine.sock (made by the engine side inside).
    inner: the sandbox command: loopback model_port -> model.sock, engine.sock -> loopback engine_port, then the engine."""
    q = shlex.quote
    msock, esock = f"{scratch}/model.sock", f"{scratch}/engine.sock"
    inner = ["sh", "-c",
             f"socat TCP-LISTEN:{model_port},bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:{q(msock)} & "
             f"socat UNIX-LISTEN:{q(esock)},fork,reuseaddr TCP:127.0.0.1:{engine_port} & "
             f"sleep 0.3; exec {' '.join(q(a) for a in engine_argv)}"]
    return {"model_sock": msock, "engine_sock": esock,
            "host_model": ["socat", f"UNIX-LISTEN:{msock},fork,reuseaddr", f"TCP:127.0.0.1:{model_port}"],
            "host_engine": ["socat", f"TCP-LISTEN:{host_port},bind=127.0.0.1,fork,reuseaddr", f"UNIX-CONNECT:{esock}"],
            "inner": inner}


def _stop_group(proc):
    """Stop a process group started with start_new_session=True: SIGTERM, SIGKILL after 5 s."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except (ProcessLookupError, OSError):
            pass


def _expendable():
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")                 # out of memory: the kernel kills the agent before anything the user is working in
    except OSError:
        pass


def agent_argv(workdir, prompt):
    return [str(Path(ENGINE).resolve()), "run", "--format", "json", "--auto", "--dir", str(workdir), prompt]


def run_agent(workdir, prompt, cfg_dir, data_dir, timeout_s, log_path, ro_binds=()):
    """One-shot agent run in its own process group, killed at the deadline. Returns (exit_code or None, timed_out, seconds).
    The agent runs inside the sandbox (set AGENTIC_EVAL_NO_SANDBOX=1 only for a supervised dry run)."""
    scratch = Path(workdir).parent
    (scratch / "home").mkdir(parents=True, exist_ok=True)
    env = {**ENGINE_ENV, "OPENCODE_CONFIG": str(Path(cfg_dir) / "shuracode.json"), "OPENCODE_CONFIG_DIR": str(cfg_dir)}
    argv = agent_argv(workdir, prompt)
    if os.environ.get("AGENTIC_EVAL_NO_SANDBOX") == "1":
        run_env, argv = {**os.environ, **env, "HOME": f"{scratch}/home"}, argv
    else:
        argv, run_env = sandboxed(argv, scratch, workdir, env=env, ro_binds=ro_binds), dict(os.environ)
    t0 = time.monotonic()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(argv, cwd=workdir, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=run_env,
                                start_new_session=True, preexec_fn=_expendable)
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


# The product's `question` tool waits for a person. In the stand nobody is there, so the run hung until its timeout (2026-10-30:
# 11 of 28 runs of arm C). Opus's rule (QSTAND1): after QUESTION_WAIT_S seconds every pending question gets the fixed text below, as if
# the user were busy; the call and the answer go to <agent log>.question.jsonl.
# SIMUSER1: simuser.QuestionPolicy decides who answers (AGENTIC_EVAL_QUESTIONS: fixed = the QSTAND1 behaviour, the default; sim = a model
# plays the user; reject; none = nobody). run_one puts the run's policy here; a run is one thread.
from simuser import FIXED_WAIT_S as QUESTION_WAIT_S, QUESTION_TEXT, QuestionPolicy  # noqa: E402,F401
_RUN = threading.local()


def engine_failure():
    """Why the last run_agent_serve of this thread lost its engine (the Bun process crashed or exited, the run could not go on), or None.
    RSI1: a Bun segfault at 16:17 left an empty agent log, agent_exit 1 and a partly scored run: not the model's failure."""
    return getattr(_RUN, "engine_failed", None)


def _aborted():
    """The run's abort check (run_abort): true when the stand under the run has failed and the run must end now."""
    check = getattr(_RUN, "abort", None)
    try:
        return bool(check and check())
    except Exception:
        return False


@contextlib.contextmanager
def run_abort(check):
    """`with run_abort(callable): run_agent_serve(...)`: the agent run ends at once when callable() is true (a crashed llama-server leaves
    the agent waiting for an answer that never comes until the task's deadline)."""
    _RUN.abort = check
    try:
        yield
    finally:
        _RUN.abort = None


@contextlib.contextmanager
def question_policy(task, out_dir):
    """The question policy of one agent run (the thread's): `with question_policy(task, out_dir) as policy: run the agent` -> policy.stats()."""
    policy = _RUN.policy = QuestionPolicy(task, out_dir=out_dir)
    try:
        yield policy
    finally:
        _RUN.policy = None


def answer_questions(call, question_log, seen, policy=None):
    """One pass over the engine's pending questions (GET /question). `seen` maps a request id to [first time seen, answered].
    `policy` (simuser.QuestionPolicy, None = fixed) says what each question gets and when."""
    try:
        status, pending = call("GET", "/question", timeout=5)
    except OSError:
        return
    if status != 200 or not isinstance(pending, list):
        return
    now = time.monotonic()
    wait_s = QUESTION_WAIT_S if policy is None else policy.wait_s
    for item in pending:
        rid = item.get("id") if isinstance(item, dict) else None
        if not rid:
            continue
        first, answered = seen.setdefault(rid, [now, False])
        if answered or now - first < wait_s:
            continue
        questions = item.get("questions") or []
        extra = {}
        if policy is None:
            verdict, answers = "reply", [[QUESTION_TEXT] for _ in range(max(1, len(questions)))]
        else:
            verdict, answers = policy.answer(item)
            event = policy.events[-1]
            extra = {"source": event["source"], "answers": answers, "leaks": event["leaks"], "cost_usd": event["cost_usd"]}
        status_r, data_r = None, "left pending"
        try:
            if verdict == "reply":
                status_r, data_r = call("POST", f"/question/{rid}/reply", {"answers": answers}, timeout=10)
            elif verdict == "reject":
                status_r, data_r = call("POST", f"/question/{rid}/reject", {}, timeout=10)
        except OSError as error:
            status_r, data_r = None, str(error)
        seen[rid][1] = status_r == 200 or verdict == "none"
        question_log.write(json.dumps({"time": time.strftime("%H:%M:%S"), "request": rid, "asked_s_ago": round(now - first, 1),
                                       "questions": [q.get("question", "")[:300] if isinstance(q, dict) else str(q)[:300]
                                                     for q in questions], "reply_status": status_r,
                                       "reply": str(data_r)[:200], **extra}, ensure_ascii=False) + "\n")
        question_log.flush()


def run_agent_serve(workdir, prompt, cfg_dir, data_dir, timeout_s, log_path, settle_s=10.0, variant=None, ro_binds=(), followups=()):
    """FABLE3: like run_agent, but the engine runs as its headless server (`shuracode serve`) and the prompt goes through its
    HTTP API, the way the TUI drives it. `shuracode run` exits as soon as the answer ends, so plugins that react to
    `session.idle` (the completion gate, the task card) never get a turn in it; here the run ends only when the session has
    stayed idle for `settle_s` after the last message (or at the deadline). The agent log is rebuilt from the session's
    messages in the same JSON-event shape as `run --format json` (step_start, tool_use, text, step_finish), plus `user`
    events for the follow-up messages the plugins inject. Returns (exit_code or None, timed_out, seconds)."""
    import http.client
    import socket
    scratch = Path(workdir).parent
    (scratch / "home").mkdir(parents=True, exist_ok=True)
    env = {**ENGINE_ENV, "OPENCODE_CONFIG": str(Path(cfg_dir) / "shuracode.json"), "OPENCODE_CONFIG_DIR": str(cfg_dir)}
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    argv = [str(Path(ENGINE).resolve()), "serve", "--port", str(port), "--hostname", "127.0.0.1"]
    api_port, bridge, bridge_procs = port, None, []     # SANDNET1: AGENTIC_EVAL_NET=bridge (default) | host (the old way)
    if os.environ.get("AGENTIC_EVAL_NO_SANDBOX") == "1":
        run_env, argv = {**os.environ, **env, "HOME": f"{scratch}/home"}, argv
    elif os.environ.get("AGENTIC_EVAL_NET", "bridge") == "host":
        argv, run_env = sandboxed(argv, scratch, workdir, env=env, ro_binds=ro_binds), dict(os.environ)
    else:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        api_port = sock.getsockname()[1]
        sock.close()
        bridge = bridge_plan(scratch, model_port_from_config(cfg_dir), port, api_port, argv)
        argv = sandboxed(bridge["inner"], scratch, workdir, env=env, ro_binds=ro_binds, unshare_net=True)
        run_env = dict(os.environ)
    t0 = time.monotonic()
    deadline = t0 + timeout_s

    def call(method, path, body=None, timeout=30):
        conn = http.client.HTTPConnection("127.0.0.1", api_port, timeout=timeout)
        try:
            data = json.dumps(body).encode() if body is not None else None
            conn.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
            resp = conn.getresponse()
            raw = resp.read()
            try:
                return resp.status, json.loads(raw.decode() or "null")
            except ValueError:
                return resp.status, raw.decode(errors="replace")
        finally:
            conn.close()

    def messages(session_id):
        status, data = call("GET", f"/session/{session_id}/message", timeout=30)
        return data if status == 200 and isinstance(data, list) else []

    def finished(msgs):
        if not msgs:
            return False
        last = msgs[-1]
        if (last.get("info") or {}).get("role") != "assistant":
            return False
        parts = last.get("parts") or []
        return any(p.get("type") == "step-finish" for p in parts) and not any(
            p.get("type") == "tool" and (p.get("state") or {}).get("status") in ("pending", "running") for p in parts)

    serve_log = open(str(log_path) + ".serve.txt", "wb")
    question_log = open(str(log_path) + ".question.jsonl", "a", encoding="utf-8")
    question_seen = {}
    code, timed_out, session_id, msgs = None, False, None, []
    _RUN.engine_failed = None
    proc = None
    try:
        if bridge:
            for old in (bridge["model_sock"], bridge["engine_sock"]):
                with contextlib.suppress(OSError):
                    os.unlink(old)
            for key in ("host_model", "host_engine"):
                bridge_procs.append(subprocess.Popen(bridge[key], stdin=subprocess.DEVNULL, stdout=serve_log, stderr=subprocess.STDOUT,
                                                     start_new_session=True))
        proc = subprocess.Popen(argv, cwd=workdir, stdin=subprocess.DEVNULL, stdout=serve_log, stderr=subprocess.STDOUT, env=run_env,
                                start_new_session=True, preexec_fn=_expendable)
        while time.monotonic() < min(deadline, t0 + 90):
            if proc.poll() is not None:
                break
            try:
                if call("GET", "/session", timeout=3)[0] == 200:
                    break
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("the engine server did not answer in time")
        if proc.poll() is not None:
            raise RuntimeError(f"the engine server exited at start with code {proc.returncode}")
        serve_log.write(f"\nFORK1C_READY {time.monotonic() - t0:.2f}\n".encode())   # FORK1c: stage marker (engine answers /session)
        status, data = call("POST", "/session", {}, timeout=30)
        if status != 200 or not isinstance(data, dict) or not data.get("id"):
            raise RuntimeError(f"could not create a session: {status} {str(data)[:200]}")
        session_id = data["id"]
        try:
            message_body = {"parts": [{"type": "text", "text": prompt}]}
            if variant:
                message_body["variant"] = variant
            # the engine answers this POST only when the whole turn is over, and a pending `question` keeps the turn open: so the POST runs
            # in a thread and this one answers the questions meanwhile (without it nobody could, and the run waited for its deadline)
            box = {}

            def post_message():
                try:
                    box["status"] = call("POST", f"/session/{session_id}/message", message_body, timeout=max(5, deadline - time.monotonic()))[0]
                except (OSError, http.client.HTTPException) as error:
                    box["error"] = error
            poster = threading.Thread(target=post_message, daemon=True)
            poster.start()
            while poster.is_alive() and time.monotonic() < deadline:
                answer_questions(call, question_log, question_seen, getattr(_RUN, "policy", None))
                if _aborted():
                    break
                poster.join(1.0)
            if _aborted():
                code, timed_out = None, False
            elif "status" in box:
                code = 0 if box["status"] == 200 else 1
            else:
                timed_out = time.monotonic() >= deadline
                code = None if timed_out else 1
        except (OSError, http.client.HTTPException):
            code = 1
        last_change, last_len = time.monotonic(), -1
        pending = list(followups)      # LHSET1 (LH-C): the user's next messages, each sent when the previous turn has finished and settled
        while time.monotonic() < deadline and not _aborted():
            answer_questions(call, question_log, question_seen, getattr(_RUN, "policy", None))
            msgs = messages(session_id)
            n = sum(len(m.get("parts") or []) for m in msgs)
            if n != last_len:
                last_len, last_change = n, time.monotonic()
            if finished(msgs) and time.monotonic() - last_change >= settle_s:
                if not pending:
                    break
                text = pending.pop(0)
                body = {"parts": [{"type": "text", "text": text}]}
                if variant:
                    body["variant"] = variant
                follow = {}

                def post_follow(body=body):
                    try:
                        follow["status"] = call("POST", f"/session/{session_id}/message", body, timeout=max(5, deadline - time.monotonic()))[0]
                    except (OSError, http.client.HTTPException) as error:
                        follow["error"] = error
                threading.Thread(target=post_follow, daemon=True).start()
                last_change, last_len = time.monotonic(), -1
                time.sleep(2.0)
                continue
            time.sleep(1.0)
        else:
            timed_out = not _aborted()
        msgs = messages(session_id) or msgs
    except (RuntimeError, OSError) as error:
        serve_log.write(f"\nrun_agent_serve: {error}\n".encode())
        code = 1 if code is None else code
        _RUN.engine_failed = f"the engine run failed: {error}"
    finally:
        if proc is not None and proc.poll() is not None and not _RUN.engine_failed:
            _RUN.engine_failed = f"the engine exited by itself with code {proc.returncode}"
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                except (ProcessLookupError, OSError):
                    pass
        for bp in bridge_procs:
            _stop_group(bp)
        if bridge:
            for old in (bridge["model_sock"], bridge["engine_sock"]):
                with contextlib.suppress(OSError):
                    os.unlink(old)
        serve_log.close()
        question_log.close()
    secs = time.monotonic() - t0
    kinds = {"step-start": "step_start", "tool": "tool_use", "text": "text", "step-finish": "step_finish", "reasoning": "reasoning",
             "patch": "patch", "file": "file"}
    with open(log_path, "w") as log:
        for m in msgs:
            info = m.get("info") or {}
            usage = info.get("tokens")
            if isinstance(usage, dict) and info.get("role") == "assistant":
                log.write(json.dumps({"type": "token_usage", "timestamp": info.get("time", {}).get("created"),
                                      "sessionID": session_id, "tokens": usage}, ensure_ascii=False) + "\n")
            if info.get("role") == "user":
                text = "\n".join(str(p.get("text", "")) for p in (m.get("parts") or []) if p.get("type") == "text")
                log.write(json.dumps({"type": "user", "timestamp": info.get("time", {}).get("created"), "sessionID": session_id,
                                      "text": text}, ensure_ascii=False) + "\n")
                continue
            for p in m.get("parts") or []:
                kind = kinds.get(p.get("type"))
                if kind is None:
                    continue
                log.write(json.dumps({"type": kind, "timestamp": p.get("time", {}).get("start") if isinstance(p.get("time"), dict) else None,
                                      "sessionID": session_id, "part": p}, ensure_ascii=False) + "\n")
    return code, timed_out, secs


def count_events(log_path):
    """Event accounting plus user turns and reasoning tokens from serve message metadata."""
    n = errors = user_messages = reasoning_tokens = 0
    for line in Path(log_path).read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        n += 1
        if ev.get("type") == "user":
            user_messages += 1
        usage = ev.get("tokens") or (ev.get("part") or {}).get("tokens") or {}
        try:
            reasoning_tokens += int(usage.get("reasoning", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            pass
        text = json.dumps(ev)
        if '"error"' in text and ev.get("type") in ("tool_use", "tool", "error"):
            errors += 1
    return {"events": n, "tool_errors": errors, "user_messages": user_messages, "reasoning_tokens": reasoning_tokens}


def engine_runner():
    """QHARN1: AGENTIC_EVAL_ENGINE_MODE=serve (default, as the product runs: the TUI's headless server) or run (the old one-shot)."""
    return run_agent if os.environ.get("AGENTIC_EVAL_ENGINE_MODE", "serve") == "run" else run_agent_serve


# QHARN1: what the agent did with the project's own checks versus its own throwaway scripts (the same verification pattern as the
# completion gate in the product's plugin, see fable3-completion-gate-verification.patch).
VERIFY_RE = re.compile(r"\b(pytest|unittest|tox|nox|python3?\s+-m\s+(pytest|unittest|tox)|python3?\s+\S*test\S*\.py|npm\s+(test|run\s+(test|build|check))|pnpm\s+(test|run\s+test)|yarn\s+test|npx\s+(jest|vitest|mocha|ava|playwright\s+test)|node\s+--test|node\s+\S*test\S*\.[cm]?js|bun\s+test|deno\s+test|jest|vitest|mocha|go\s+test|cargo\s+(test|check|build)|gradle(w)?\s+(test|build|check)|mvn\s+(test|verify)|dotnet\s+test|cmake\s+--build|ctest|make\s+(test|check)|rspec|php\s+artisan\s+test|phpunit)\b", re.I)
ADHOC_RE = re.compile(r"\b(?:python3?|node|bash|sh)\s+(?!-)[\w./-]+\.(?:py|js|mjs|cjs|ts|sh)\b|\bnode\s+-e\b|\bpython3?\s+-c\b")


def project_checks(log_path):
    """{tests_run_project, adhoc_checks}: did the run call the project's own test command, and how many throwaway checks did it run."""
    cmds = []
    for line in Path(log_path).read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        p = ev.get("part") or {}
        if p.get("tool") == "bash":
            command = ((p.get("state") or {}).get("input") or {}).get("command")
            if command:
                cmds.append(command)
    return {"tests_run_project": any(VERIFY_RE.search(c) for c in cmds), "adhoc_checks": sum(len(ADHOC_RE.findall(c)) for c in cmds)}


def request_prompt_peak(log_path):
    """Largest per-request prompt token count in this run (not the sum over turns)."""
    peak = 0
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        return 0
    for line in lines:
        try:
            usage = json.loads(line).get("usage") or {}
            peak = max(peak, int(usage.get("prompt_tokens", 0) or 0))
        except (ValueError, TypeError, AttributeError):
            continue
    return peak


def run_one(task, arm, rep, port, proxy_log, out_dir, scratch_root="/dev/shm/agentic-eval", context=32768, runner=None):
    """Everything for one cell of the experiment. `runner` is replaceable so that the plumbing can be tested without a model."""
    from qlev.arms import is_arm
    qlev_arm = arm if is_arm(arm) else None
    runner = runner or engine_runner()
    if qlev_arm == "T" and runner is run_agent_serve:
        runner = lambda work, prompt, cfg, data, timeout, log: run_agent_serve(
            work, prompt, cfg, data, timeout, log, variant="low")
    scratch = Path(tempfile.mkdtemp(prefix=f"{task['id']}-{arm}-{rep}-", dir=_ensure(scratch_root)))
    try:
        work = scratch / "work"
        # a task without start files has no start/ folder in git (empty folders are not tracked): an empty project then
        shutil.copytree(task["dir"] / "start", work) if (task["dir"] / "start").is_dir() else work.mkdir(parents=True)
        tools_on = agent_tools_on(task)
        write_project_rules(work, agent_tools=tools_on)
        cfg = write_config(scratch / "cfg", port, context=context, arm=qlev_arm, workdir=work, task=task)
        if tools_on and runner in (run_agent, run_agent_serve):
            base, binds = runner, agent_tools_binds()
            runner = lambda work_, prompt_, cfg_, data_, timeout_, log_: base(work_, prompt_, cfg_, data_, timeout_, log_, ro_binds=binds)
        Path(proxy_log).unlink(missing_ok=True)
        bodies = Path(f"{proxy_log}.bodies.jsonl")      # AGTOOLS1b: request bodies when AGENTIC_EVAL_LOG_REQUESTS=1
        bodies.unlink(missing_ok=True)
        log_path = Path(out_dir) / f"{task['id']}__{arm}__{rep}.agent.log"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with question_policy(task, out_dir) as policy:                       # SIMUSER1: who answers the agent's questions
            code, timed_out, secs = runner(work, task["prompt"], cfg, scratch / "data", task.get("timeout_s", 900), log_path)
        if bodies.is_file():
            shutil.move(str(bodies), str(Path(out_dir) / f"{task['id']}__{arm}__{rep}.requests.jsonl"))
        passed, tail = run_tests(work, task["dir"] / "hidden", task["test_cmd"])
        soft = LAST_TEST_RUN.get("soft") or {}
        from proxy import totals
        tot = totals(proxy_log)
        rec = {"arm": arm, "task": task["id"], "family": task["family"], "kind": task["kind"], "rep": rep, "resolved": bool(passed),
               "score": 1.0 if passed else float(soft.get("score", 0.0)), "tests_total": soft.get("tests_total"), "tests_passed": soft.get("tests_passed"),
               "timed_out": timed_out, "agent_exit": code, "seconds": round(secs, 1),
               "tokens": tot["completion_tokens"], "prompt_tokens": tot["prompt_tokens"], "requests": tot["requests"],
               "request_errors": tot["errors"], "max_prompt_tokens": request_prompt_peak(proxy_log),
                **count_events(log_path), **project_checks(log_path), **policy.stats(),
               # Keep the server's fields authoritative; old proxy logs simply contribute zero/empty values.
               "reasoning_tokens": int(tot.get("reasoning_tokens", 0) or 0),
               "switches": tot.get("switches", {}),
               "slice": {"lang": task.get("lang", "unknown"), "project": task["family"],
                         "kind": task["kind"]}}
        if not passed:
            rec["test_tail"] = tail[-400:]
        return rec
    finally:
        if os.environ.get("AGENTIC_EVAL_KEEP_SCRATCH") != "1":      # QLEV2S arm check: keep the engine config and log (off by default)
            shutil.rmtree(scratch, ignore_errors=True)


def _ensure(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
