"""From a plan to a running llama-server: command line, background start, status, stop.

Plans hold *intents* (context, KV type, expert layers in RAM, threads); this module turns them into the flags of an upstream
llama.cpp `llama-server` and manages the process with a pid file, so `shura start` / `shura stop` work the same on every OS.
The fork tier (turbo3 KV, expert cache, MTP) is launched by the Linux `setup.sh`, not here.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_PORT = 8080
_PROCS = {}                      # servers started by this process: polled so that a stopped one is reaped, not left a zombie


class LaunchError(Exception):
    pass


def home_dir(env=None, platform=None):
    """Where Shura keeps engines, models, logs and state. `SHURA_HOME` wins; otherwise the usual place of each OS."""
    env, platform = os.environ if env is None else env, platform or sys.platform
    if env.get("SHURA_HOME"):
        return Path(env["SHURA_HOME"])
    if platform.startswith("win"):
        return Path(env.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "shura"
    if platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "shura"
    return Path(env.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "shura"


def server_args(plan, server, model, *, host="127.0.0.1", port=DEFAULT_PORT, alias=None):
    """argv for llama-server from a plan. Raises LaunchError for a plan this module cannot launch."""
    if not plan.get("ok"):
        raise LaunchError("there is no plan to launch")
    if plan.get("tier") == "fork":
        raise LaunchError("the fork tier is launched by setup.sh (it needs the Shura CUDA fork)")
    s = plan["settings"]
    ngl = s.get("ngl", 0)
    argv = [str(server), "-m", str(model), "--host", host, "--port", str(port), "-c", str(s["context"]),
            "-np", str(s.get("parallel", 1)), "-t", str(s["threads"]), "-ngl", "99" if ngl == "all" else str(ngl),
            "-ctk", s["kv_type"], "-ctv", s["kv_type"], "-fa", "on" if s.get("flash_attn", True) else "off", "--jinja"]
    if s.get("n_cpu_moe"):
        argv += ["--n-cpu-moe", str(s["n_cpu_moe"])]
    if s.get("numa"):
        argv += ["--numa", s["numa"]]
    if s.get("mtp"):                          # the model's own MTP head drafts tokens: speculative decoding without a second model
        argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1", "--spec-draft-p-min", "0.2"]
    if alias:
        argv += ["--alias", alias]
    return argv


def server_env(server):
    """The server's own folder holds its shared libraries."""
    env, lib = dict(os.environ), str(Path(server).parent)
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        env[var] = lib + os.pathsep + env.get(var, "")
    return env


def health(port, host="127.0.0.1", timeout=2):
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def alive(pid):
    if not pid:
        return False
    if pid in _PROCS:
        return _PROCS[pid].poll() is None
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid(home):
    try:
        return int((Path(home) / "server.pid").read_text().strip())
    except (OSError, ValueError):
        return None


def start(home, argv, port, *, env=None, wait_s=900, log_name="server.log"):
    """Start the server in the background. Returns the pid once /health answers; raises LaunchError with the log tail."""
    home = Path(home)
    pid = read_pid(home)
    if alive(pid) and health(port):
        raise LaunchError(f"already running (pid {pid}); `shura stop` first")
    (home / "logs").mkdir(parents=True, exist_ok=True)
    log_path = home / "logs" / log_name
    kw = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                env=env or server_env(argv[0]), **kw)
    _PROCS[proc.pid] = proc
    (home / "server.pid").write_text(str(proc.pid))
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise LaunchError("the server exited while starting:\n" + tail(log_path))
        if health(port):
            return proc.pid
        time.sleep(0.5)
    stop(home)
    raise LaunchError(f"the server did not become healthy within {wait_s} s:\n" + tail(log_path))


def tail(path, n=1500):
    try:
        return Path(path).read_bytes()[-n:].decode(errors="replace")
    except OSError:
        return ""


def stop(home, timeout=15):
    """Stop the server started by `start`. Returns True if something was stopped."""
    pid = read_pid(home)
    if not alive(pid):
        (Path(home) / "server.pid").unlink(missing_ok=True)
        return False
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T"], capture_output=True)
        else:
            os.killpg(pid, signal.SIGTERM)
    except OSError:
        pass
    deadline = time.monotonic() + timeout
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    if alive(pid):
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            else:
                os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass
    (Path(home) / "server.pid").unlink(missing_ok=True)
    return True


def load_state(home):
    try:
        return json.loads((Path(home) / "state.json").read_text())
    except (OSError, ValueError):
        return None


def save_state(home, state):
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    tmp = home / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(home / "state.json")
