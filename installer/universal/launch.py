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


class AlreadyRunning(LaunchError):
    """A Shura server started earlier is still up. Not a failure of the build or of the settings."""


class PortBusy(LaunchError):
    """The port is held by another program (typically the Shura gateway on 8080): not a verdict on the build or the settings."""


HOME_MARKER = ".shura-home"      # written by `shura install`: this folder belongs to the universal installer


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
        if plan.get("tier") == "hipfork" and s.get("moe_cache_slots"):        # our fork's expert cache: hot experts in VRAM
            profile = Path(__file__).resolve().parent.parent.parent / s["moe_cache_profile"]
            argv += ["--moe-cache-profile", str(profile), "--moe-cache-slots", str(s["moe_cache_slots"])]
        if plan.get("tier") == "tqp":              # the TurboQuant+ build caches hot experts in VRAM: give it exactly our budget
            argv += ["--moe-cache", str(s["moe_cache_mib"]) if s.get("moe_cache_mib") else "off"]
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


def _is_llama_server(pid):
    """Is `pid` a llama-server? A pid from an old server.pid can belong to anything after a reboot or a crash; we must never
    signal a process that is not ours. True when it cannot be told (no /proc, no ps): the old behaviour."""
    try:
        if sys.platform.startswith("linux") and Path("/proc/self/cmdline").exists():
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
            return b"llama" in cmd.lower()
        if sys.platform == "darwin":
            out = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=5).stdout
            return "llama" in out.lower()
    except FileNotFoundError:
        return False
    except PermissionError:
        return False
    except (OSError, subprocess.SubprocessError):
        return True
    return True


def alive(pid):
    """Is `pid` a running Shura server? A live process that is not a llama-server (a recycled pid) does not count."""
    if not pid:
        return False
    if pid in _PROCS:
        return _PROCS[pid].poll() is None
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out and "llama" in out.lower()
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False                  # a process of another user cannot be our server
    return _is_llama_server(pid)


def port_free(port, host="127.0.0.1"):
    """Can we listen on host:port? False when something (for instance the Shura gateway on 8080) already holds it."""
    import socket
    with socket.socket() as s:
        if os.name != "nt":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)     # a port in TIME_WAIT after our last test launch is free
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def mark_home(home):
    """Remember that this folder is a Shura home of the universal installer (`shura uninstall` checks for it)."""
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    (home / HOME_MARKER).write_text("created by `shura install`; `shura uninstall` only deletes inside folders that have this file\n")


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
        raise AlreadyRunning(f"already running (pid {pid}); `shura stop` first")
    if not port_free(port):
        raise PortBusy(f"port {port} is already in use by another program (the Shura gateway uses 8080): "
                       f"pass another one with --port")
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


def _signal(pid, sig):
    """Signal the server's process group (it leads its own, see `start`); a pid that does not lead one gets only itself."""
    try:
        group = os.getpgid(pid) == pid
    except ProcessLookupError:
        return
    (os.killpg if group else os.kill)(pid, sig)


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
            _signal(pid, signal.SIGTERM)
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
                _signal(pid, signal.SIGKILL)
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
