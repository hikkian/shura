"""Pieces both gateway entry points (the full one and the preserved default one) share, so a fix is made once.

Everything here takes the state or the configuration it needs as an argument; nothing reads another module's globals.
Standard library only.
"""
import hashlib
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

LIFECYCLE_DEFAULTS = {
    "crashWindowSeconds": 900,        # crashes older than this stop counting towards crashRestartMaxAttempts
    "errorCooldownSeconds": 300,      # after the restart limit, the gateway leaves ERROR by itself (0 = only `ai-on` does)
    "healthProbeSeconds": 30,         # how often a READY server's /health is probed (0 = never)
    "healthProbeFailures": 3,         # consecutive failed probes before the server is declared hung and restarted
    "shutdownLockSeconds": 15,        # how long a stop signal waits for a loading thread before it kills the server itself
    "recoveryQueueSeconds": 600,      # crashRecovery: how long a second chat request waits for the first one
    "deadBackendWaitSeconds": 45,     # a server that stopped listening is waited for this long to finish dying before it is declared hung
}


# --- crashes ----------------------------------------------------------------------------------------------------------

def note_crash(st, config, now=None):
    """Count one server failure inside the sliding window; True when the restart limit is reached (the caller sets ERROR).
    Reaching READY does not reset the count: a model that loads but dies on its first answer must not restart for ever."""
    now = time.monotonic() if now is None else now
    st.crash_times.append(now)
    st.crash_count += 1
    while st.crash_times and now - st.crash_times[0] > config["crashWindowSeconds"]:
        st.crash_times.popleft()
        st.crash_count = max(0, st.crash_count - 1)
    if st.crash_count >= config["crashRestartMaxAttempts"]:
        st.error_since = now
        return True
    return False


def clear_error_state(st):
    st.crash_count, st.last_error, st.error_since = 0, "", None
    st.crash_times.clear()


def error_cooldown_over(st, config, now):
    """True when the gateway has been in ERROR (because of the crash limit) for errorCooldownSeconds."""
    return (st.status == "ERROR" and st.error_since is not None and config["errorCooldownSeconds"] > 0
            and now - st.error_since >= config["errorCooldownSeconds"])


# --- wall clock steps ---------------------------------------------------------------------------------------------------

def compensate_wall_clock(st, wall, mono, slack=5.0):
    """Idle time is measured on the wall clock (the state file and the status page speak in it), but a suspend or a forward
    clock step must not make a busy model look idle for hours: if the wall clock moved further than the monotonic one, the
    surplus is added to `last_request`. (A step backwards only keeps the model a little longer and is left alone.)"""
    ref = getattr(st, "clock_ref", None)
    st.clock_ref = (wall, mono)
    if ref is None:
        return 0.0
    drift = (wall - ref[0]) - (mono - ref[1])
    if drift > slack:
        st.last_request += drift
        return drift
    return 0.0


# --- locks ------------------------------------------------------------------------------------------------------------

@contextmanager
def try_lock(lock, seconds):
    """The lock, but never for longer than `seconds`: a model load holds it for minutes and a control request must still answer."""
    got = lock.acquire(timeout=seconds)
    try:
        yield got
    finally:
        if got:
            lock.release()


def snapshot_without_waiting(st, build):
    """Build a status snapshot while a model load holds st.lock (up to minutes): otherwise the status page and `shura status` hang
    for the whole load and never show LOADING. Without the lock the values are read as they are; a container that changes under
    the copy is read again."""
    with try_lock(st.lock, 0.25) as got:
        if got:
            return build()
    for _ in range(5):
        try:
            return build()
        except RuntimeError:
            time.sleep(0.01)
    return build()


# --- the llama-server command line ------------------------------------------------------------------------------------

_FLAG_SUPPORT = {}


def server_supports(exe, flag):
    """Does the llama-server binary know this option? Read once from its --help, so a build without it is never started with it."""
    try:
        info = Path(exe).stat()
    except (OSError, TypeError):
        return False
    key = (str(exe), info.st_mtime_ns)
    if key not in _FLAG_SUPPORT:
        try:
            out = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL)
            _FLAG_SUPPORT[key] = out.stdout + out.stderr
        except (OSError, subprocess.SubprocessError):
            _FLAG_SUPPORT[key] = ""
    return flag in _FLAG_SUPPORT[key]


def write_api_key_file(state_dir, key):
    """The key goes to a private file instead of the command line, where every local user could read it with `ps`."""
    state_dir = Path(state_dir)
    path = state_dir / "llama-api-key"
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(str(key) + "\n")
    os.chmod(path, 0o600)
    return str(path)


def cors_args(exe, llama_port):
    """llama-server answers any web page by default (Access-Control-Allow-Origin: the caller, with credentials). The gateway strips
    those headers, but the server's own port is reachable from a browser too: allow only its own origin there. The option takes
    exactly one value (the header carries it as it is), not a list."""
    if not server_supports(exe, "--cors-origins"):
        return []
    args = ["--cors-origins", f"http://127.0.0.1:{llama_port}"]
    if server_supports(exe, "--no-cors-credentials"):
        args.append("--no-cors-credentials")
    return args


def key_args(exe, state_dir, key):
    if not key:
        return []
    if server_supports(exe, "--api-key-file"):
        try:
            return ["--api-key-file", write_api_key_file(state_dir, key)]
        except OSError:
            pass                            # a read-only state folder: the older way still works
    return ["--api-key", str(key)]


def redacted_command(args):
    shown, hide = [], False
    for arg in args:
        shown.append("***" if hide else arg)
        hide = arg == "--api-key"
    return " ".join(shown)


def llama_headers(key, has_body=False):
    headers = {"Content-Type": "application/json"} if has_body else {}
    if key:
        headers["Authorization"] = "Bearer " + str(key)       # llama-server asks for it on /slots and friends
    return headers


# --- stopping a server that is not ours to wait for -------------------------------------------------------------------

def backend_listening(host, port, timeout=0.3):
    """Does anything accept connections on the server's port? (A server that is busy still does: the kernel completes the handshake.)"""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def kill_server(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


# --- the lease that lets a new gateway find the server an old one left behind -----------------------------------------

def process_identity(pid):
    root = Path("/proc") / str(pid)
    # comm may contain spaces or parentheses: start-time is field 22, after the final ')'.
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start": fields[19], "uid": root.stat().st_uid,
            "cmd": hashlib.sha256((root / "cmdline").read_bytes()).hexdigest()}


def server_exec_ready(pid, exe_path):
    """Has the process become the server? `nice` may still be exec'ing it when Popen returns. A launcher script counts too: the
    kernel then runs its interpreter and the script is the first argument."""
    wanted = Path(exe_path).resolve()
    if Path(os.readlink(f"/proc/{pid}/exe")).resolve() == wanted:
        return True
    argv = [a for a in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:2] if a]
    return any(Path(os.fsdecode(a)).resolve() == wanted for a in argv)


def stop_owned_orphan(root, llama_call, on_event=lambda *a, **k: None, identity=None):
    """Only a matching owned PID can be stopped after a gateway restart; never kill by pattern."""
    import json
    identity = identity or process_identity
    lease_path = Path(root) / "server.json"
    if not lease_path.exists():
        return
    if lease_path.is_symlink() or lease_path.stat().st_size > 16384:
        raise RuntimeError("Unsafe server lease; manual inspection required")
    saved = json.loads(lease_path.read_text())
    pid = saved.get("pid")
    if not isinstance(pid, int) or pid <= 1 or saved.get("uid") != os.getuid() or saved.get("slot_dir") != str(root):
        raise RuntimeError("Invalid owned server lease; manual inspection required")
    try:
        current = identity(pid)
    except FileNotFoundError:
        lease_path.unlink()
        return
    if any(current[key] != saved.get(key) for key in ("pid", "start", "uid", "cmd")):
        lease_path.unlink()
        on_event("orphan_pid_reused", "Old PID belongs to another process; left untouched", fallback=True)
        return
    code, slots = llama_call("GET", "/slots", timeout=3)
    if code != 200 or not isinstance(slots, list) or not slots or any(slot.get("is_processing", True) for slot in slots):
        raise RuntimeError(f"Owned server PID {pid} is busy or unverifiable; restart waits for the answer")
    if identity(pid) != current:
        raise RuntimeError("Server identity changed before recovery; left untouched")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if identity(pid) != current:
                break
            # A zombie has already released GPU memory.
            if (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                break
        except FileNotFoundError:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError(f"Owned server PID {pid} did not exit; refusing a second server")
    lease_path.unlink(missing_ok=True)
    on_event("orphan_recovered", f"Stopped idle owned server PID {pid} after gateway restart")


def write_lease(root, proc, exe_path, verify=False):
    """Record the identity of the server this gateway started. Called once the server has answered /health on our port, so any
    launcher script or `nice` in front of it has finished exec'ing and the identity is final; `verify` additionally insists that
    the process image is `exe_path` (only right when exe_path is the server binary itself)."""
    import json
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if verify:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if server_exec_ready(proc.pid, exe_path):
                break
            time.sleep(0.02)
        else:
            raise OSError("Server executable not ready for ownership lease")
    lease = {**process_identity(proc.pid), "slot_dir": str(root)}
    target = root / "server.json"
    pending = target.with_suffix(".json.tmp")
    pending.write_text(json.dumps(lease))
    os.replace(pending, target)
