#!/usr/bin/env python3
"""AI Gateway / Resource Guardian for a single local llama-server.

Sits in front of llama-server as an OpenAI-compatible reverse proxy and owns its lifecycle:
load on first request, switch text/vision mode by inspecting the request, unload when idle or
under memory pressure, restart after crashes, and persist the active KV slot to disk only when
the model is unloaded (so session switching stays in RAM and SSD writes stay rare).

Standard library only. Configuration: guardian.json + model-launch.json in $AI_GATEWAY_CONFIG
(default: ../config next to this file).
"""
import atexit
import hashlib
import http.client
import importlib.util
import json
import math
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_nvml import NVML
from game_mode import DEFAULTS as GAME_DEFAULTS, GamePolicy, GameSignals

CFG_DIR = Path(os.environ.get("AI_GATEWAY_CONFIG", Path(__file__).resolve().parent.parent / "config"))
STATE_DIR = Path(os.path.expanduser(os.environ.get("AI_GATEWAY_STATE", "~/.local/state/ai-gateway")))


PATH_KEYS = {"exePath", "modelPath", "mmprojPath", "moeCacheProfile", "slotSaveDir", "vramPressureSlotDir"}


def expand(key, value):
    """Expand ~ and $VARS in path settings; relative paths are resolved against the config directory."""
    if key not in PATH_KEYS or not isinstance(value, str) or not value:
        return value
    p = Path(os.path.expanduser(os.path.expandvars(value)))
    return str(p if p.is_absolute() else (CFG_DIR / p).resolve())


def load_config(name):
    cfg = json.loads((CFG_DIR / name).read_text())
    return {k: expand(k, v) for k, v in cfg.items()}


G = load_config("guardian.json")
M = load_config("model-launch.json")
M["models"] = {mid: {k: expand(k, v) for k, v in mc.items()} for mid, mc in M["models"].items()}
G = {**GAME_DEFAULTS, "vramGuard": False, "vramIdleSeconds": 90,
     "vramEmergencyMiB": 60, "vramEmergencySeconds": 10,
     "vramFreeCriticalMiB": 120, "vramPressureSeconds": 5,
     "vramReloadWaitSeconds": 30, "vramPressureSlotReserveMiB": 512,
     "vramPressureSlotDir": "/dev/shm", **G}
SLOT_DIR = Path(G["slotSaveDir"])
RAM_SLOT_DIR = None
OVERRIDE_FLAG = STATE_DIR / "override.flag"

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade", "content-length", "host"}


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}", flush=True)


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


_NVML_DEVICE = None
_NVML_LOCK = threading.Lock()

def nvml_device():
    global _NVML_DEVICE
    with _NVML_LOCK:
        if _NVML_DEVICE is None:
            _NVML_DEVICE = NVML()
        return _NVML_DEVICE

def close_nvml():
    global _NVML_DEVICE
    with _NVML_LOCK:
        if _NVML_DEVICE is not None:
            _NVML_DEVICE.close()
            _NVML_DEVICE = None

atexit.register(close_nvml)

def vram_free_gb():
    try:
        free_mib = nvml_device().card_free_mib()
        return free_mib / 1024 if isinstance(free_mib, int) and free_mib >= 0 else -1.0
    except (OSError, RuntimeError, ValueError, OverflowError):
        return -1.0


def process_vram_mib(pid):
    """Return the latest monitor-thread sample for this server, never query NVML inline."""
    proc = st.proc
    sample_age = time.time() - st.proc_vram_sample_at if st.proc_vram_sample_at is not None else math.inf
    if (proc is None or proc.pid != pid or st.proc_vram_pid != pid
            or sample_age > G.get("vramTelemetryStaleSeconds", 5)):
        return -1
    value = st.proc_vram_mib
    return value if isinstance(value, int) and value >= 0 else -1


class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.status = "UNLOADED"  # UNLOADED LOADING READY MEMORY_PRESSURE ERROR
        self.model_id = ""
        self.vision = False
        self.proc = None
        self.expected_exit = False
        self.override = "OFF" if OVERRIDE_FLAG.exists() and OVERRIDE_FLAG.read_text().strip() == "OFF" else "AUTO"
        self.memory_pressure = False
        self.pressure_sustain = 0
        self.multimedia_lock = False
        self.crash_count = 0
        self.last_error = ""
        self.last_request = time.time()
        self.busy = 0
        self.busy_lock = threading.Lock()
        self.vram_pressure = False
        self.vram_pressure_since = None
        self.last_yield_reason = ""
        self.last_yield_at = None
        self.ram_checkpoint = None
        self.last_ram_poll = 0.0
        self.vram_resume_min_gb = 0.0
        self.preload_vram_gb = 0.0
        self.vram_emergency_since = None
        self.vram_emergency = False
        self.pressure_notified = False
        self.guard_fallback = ""
        self.last_event = None
        self.event_counters = {}
        self.monitor_healthy = True
        self.monitor_last_tick = None
        self.monitor_error = ""
        self.vram_free_mib = None
        self.vram_sample_at = None
        self.proc_vram_mib = None
        self.proc_vram_pid = None
        self.proc_vram_sample_at = None
        self.ram_backend = False
        self.skip_disk_restore = False
        self.checkpoint_checked = False
        self.next_save_attempt = 0.0
        self.monitor_thread = None
        self.backend_blocked = ""
        self.ram_io_failed = False


st = State()
GAME = GamePolicy(G)
GAME_SIGNALS = GameSignals(G)
GAME_REQUESTS = {}


def event(reason, detail, fallback=False):
    """Transition-only logs and in-memory counters; no new persistent event store."""
    previous = st.last_event
    st.last_event = {"reason": reason, "detail": detail, "at": time.time()}
    st.event_counters[reason] = st.event_counters.get(reason, 0) + 1
    if fallback:
        st.guard_fallback = detail
    try:
        if not previous or (previous["reason"], previous["detail"]) != (reason, detail):
            log(f"Guardian {reason}: {detail}")
    except OSError:
        pass  # A closed journal pipe must not kill the guardian thread.


def reset_pressure():
    st.vram_pressure_since = st.vram_emergency_since = None
    st.vram_pressure = st.vram_emergency = st.pressure_notified = False


def legacy_main():
    """Run the preserved original entrypoint with no experimental lifecycle changes."""
    path = Path(__file__).with_name("_gateway_legacy.py")
    spec = importlib.util.spec_from_file_location("shura_legacy_gateway", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def model_config(model_id, vision):
    mc = dict(M["models"][model_id])
    if vision:
        mc.update(mc.get("visionOverrides", {}))
    return mc


def slot_filename(model_id, vision):
    return f"{model_id}-{'vision' if vision else 'text'}.bin"


def llama_call(method, path, body=None, timeout=10):
    conn = http.client.HTTPConnection(G["llamaHost"], G["llamaPort"], timeout=timeout)
    try:
        payload = json.dumps(body) if body is not None else None
        conn.request(method, path, body=payload, headers={"Content-Type": "application/json"} if payload else {})
        r = conn.getresponse()
        data = r.read()
        try:
            return r.status, json.loads(data) if data else None
        except ValueError:
            return r.status, None
    finally:
        conn.close()


def build_args(mc, vision):
    a = [M["exePath"], "--host", G["llamaHost"], "--port", str(G["llamaPort"]),
         "-m", mc["modelPath"], "-c", str(mc["ctxSize"]), "-np", str(mc["parallelSlots"]),
         "-t", str(mc["threads"]), "-tb", str(mc["threadsBatch"]),
         "-b", str(mc["batchSize"]), "-ub", str(mc["ubatchSize"]),
         "-ngl", str(mc["nGpuLayers"]), "-ncmoe", str(mc["nCpuMoe"]),
         "-ctk", mc["cacheTypeK"], "-ctv", mc["cacheTypeV"], "-fa", mc["flashAttn"],
         "--load-mode", mc["loadMode"],
         "--temp", str(mc["temperature"]), "--top-k", str(mc["topK"]), "--top-p", str(mc["topP"]),
         "--jinja", "--reasoning", mc["reasoning"],
         "--cache-ram", str(M["cacheRamMB"]), "--slot-save-path", str(server_slot_dir())]
    if mc.get("moeCacheProfile") and mc.get("moeCacheSlots", 0) > 0:
        a += ["--moe-cache-profile", mc["moeCacheProfile"], "--moe-cache-slots", str(mc["moeCacheSlots"])]
    if mc.get("specType") and mc["specType"] != "none":
        a += ["--spec-type", mc["specType"], "--spec-draft-n-max", str(mc["specDraftNMax"]),
              "--spec-draft-p-min", str(mc["specDraftPMin"])]
    checkpoints = int(G.get("slotSaveCheckpoints", 0))
    if GAME.enabled:
        checkpoints = max(1, checkpoints)
    if checkpoints > 0:
        a += ["--slot-save-checkpoints", str(checkpoints)]
    a += ["--mmproj", mc["mmprojPath"]] if vision else ["--no-mmproj-auto"]
    if M.get("apiKey"):
        a += ["--api-key", M["apiKey"]]
    if mc.get("niceLevel"):
        # Lower CPU priority: desktop apps win any contention; no speed is lost while the desktop is idle.
        a = ["nice", "-n", str(mc["niceLevel"])] + a
    return a


def server_slot_dir():
    st.ram_backend = False
    if G["vramGuard"] or GAME.enabled:
        try:
            path = prepare_ram_slot_dir()
            st.ram_backend = True
            return path
        except (OSError, RuntimeError) as e:
            event("ram_path_failed", f"RAM sessions unavailable; using ordinary disk slots: {e}", fallback=True)
    return SLOT_DIR


def preload_check(mc):
    if st.override == "OFF":
        return "AI is manually disabled (ai-off)"
    if st.multimedia_lock:
        return "GPU is busy with an audio/video job"
    ram = mem_available_gb()
    if ram < G["ramFreeMinGBToLoad"]:
        return f"Not enough free RAM to load the model ({ram:.1f} GB available, need {G['ramFreeMinGBToLoad']} GB)"
    sample_age = time.time() - st.vram_sample_at if st.vram_sample_at is not None else math.inf
    if sample_age > G.get("vramTelemetryStaleSeconds", 5) or st.monitor_error:
        if GAME.enabled:
            return "Fresh VRAM admission data unavailable; retry shortly"
        st.monitor_error = "Fresh NVML data unavailable; skipping VRAM admission check"
        event("telemetry_failed", st.monitor_error, fallback=True)
        reset_pressure()
        return ""
    vram = (st.vram_free_mib or 0) / 1024
    required = max(mc["vramFreeMinGBToLoad"], st.vram_resume_min_gb)
    if vram < required:
        return f"Not enough free VRAM ({vram:.1f} GB free, need {required:.2f} GB)"
    st.preload_vram_gb = vram
    return ""


def prepare_ram_slot_dir():
    """A private verified tmpfs path. Never silently write emergency sessions to SSD."""
    global RAM_SLOT_DIR
    if RAM_SLOT_DIR is not None:
        return RAM_SLOT_DIR
    parent = Path(G["vramPressureSlotDir"]).resolve(strict=True)
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        mount = Path(left.split()[4].replace("\\040", " "))
        if parent == mount or mount in parent.parents:
            mounts.append((len(str(mount)), right.split()[0]))
    if not mounts or max(mounts)[1] != "tmpfs":
        raise RuntimeError("vramPressureSlotDir must be on tmpfs (RAM)")
    identity = hashlib.sha256(str(STATE_DIR.resolve()).encode()).hexdigest()[:16]
    root = parent / f"shura-slots-{os.getuid()}-{identity}"
    if root.is_symlink():
        raise RuntimeError("RAM slot directory must not be a symlink")
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError("RAM slot directory must be private and owned by this user")
    RAM_SLOT_DIR = root
    try:
        recover_orphan(root)
    except (OSError, RuntimeError, ValueError) as e:
        RAM_SLOT_DIR = None
        st.backend_blocked = str(e)
        raise RuntimeError(str(e)) from e
    st.backend_blocked = ""
    metadata = root / "checkpoint.json"
    if metadata.exists():
        try:
            if metadata.is_symlink() or metadata.stat().st_size > 16384:
                raise ValueError("invalid checkpoint metadata")
            saved = json.loads(metadata.read_text())
            if saved["signature"] != checkpoint_signature(saved["model"], saved["vision"]):
                raise ValueError("checkpoint configuration changed")
            validate_checkpoint(root / "active.bin", saved)
            st.ram_checkpoint = {**saved, "path": root / "active.bin"}
            st.vram_resume_min_gb = saved.get("reload_min_gb", 0.0)
            st.last_yield_reason = "vram_pressure"
            st.last_yield_at = saved.get("saved_at")
            event("checkpoint_found", "Compatible RAM session found after gateway restart")
        except (OSError, ValueError, KeyError, TypeError) as e:
            event("checkpoint_invalid", f"RAM session discarded; next prompt will be re-read: {e}", fallback=True)
            st.skip_disk_restore = True
            clear_ram_checkpoint()
    elif (root / "active.bin").exists():
        event("checkpoint_orphan", "Incomplete RAM session discarded; next prompt will be re-read", fallback=True)
        clear_ram_checkpoint()
    # A killed gateway may leave an interrupted save/metadata update. Only private known names.
    (root / "active.pending.bin").unlink(missing_ok=True)
    (root / "checkpoint.json.tmp").unlink(missing_ok=True)
    return RAM_SLOT_DIR


def checkpoint_signature(model_id, vision):
    value = {"model": model_config(model_id, vision), "exe": M["exePath"]}
    for key, path in (("exe_stat", M["exePath"]), ("model_stat", value["model"]["modelPath"])):
        try:
            info = Path(path).stat()
            value[key] = [info.st_size, info.st_mtime_ns]
        except OSError:
            value[key] = None
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_checkpoint(path, saved):
    if (saved["model"] not in M["models"] or type(saved["vision"]) is not bool
            or type(saved["tokens"]) is not int or type(saved["bytes"]) is not int
            or not isinstance(saved.get("reload_min_gb", 0), (int, float))
            or not math.isfinite(saved.get("reload_min_gb", 0))
            or saved.get("reload_min_gb", 0) < 0
            or saved["tokens"] > model_config(saved["model"], saved["vision"])["ctxSize"]):
        raise ValueError("checkpoint metadata is invalid")
    if (path.is_symlink() or not path.is_file() or not isinstance(saved["tokens"], int)
            or saved["tokens"] < 1 or path.stat().st_size != saved["bytes"]
            or file_hash(path) != saved["sha256"]):
        raise ValueError("checkpoint hash, size or token count is invalid")


def clear_ram_checkpoint():
    st.ram_checkpoint = None
    if RAM_SLOT_DIR is None:
        return
    errors = []
    for name in ("active.bin", "active.pending.bin", "checkpoint.json", "checkpoint.json.tmp"):
        try:
            (RAM_SLOT_DIR / name).unlink(missing_ok=True)
        except OSError as e:
            errors.append(str(e))
    st.ram_io_failed = bool(errors)
    if errors:
        event("ram_cleanup_failed", "RAM checkpoint cleanup failed; further saves disabled: " + "; ".join(errors), fallback=True)
    return not errors


def ram_slot_space_ok(required_bytes=0):
    reserve = G["vramPressureSlotReserveMiB"] * 1024 * 1024
    needed = max(0, int(required_bytes)) + reserve
    root = prepare_ram_slot_dir()
    return (shutil.disk_usage(root).free >= needed
            and mem_available_gb() >= G["ramFreeMinGB"] + needed / 1024 ** 3)


def save_slot_locked(ram_only=False):
    """Pressure saves are atomic in RAM; ordinary saves retain the original SSD policy."""
    root = None
    target = None
    try:
        code, slots = llama_call("GET", "/slots")
        if code != 200 or not isinstance(slots, list) or not slots:
            raise RuntimeError("unable to inspect active slot")
        n = slots[0].get("n_prompt_tokens", 0)
        if n == 0 or (not ram_only and n < G["slotSaveMinTokens"]):
            if ram_only:
                clear_ram_checkpoint()
                st.skip_disk_restore = True
            return True
        if not st.ram_backend:
            if ram_only:
                raise RuntimeError("server has no RAM slot backend")
            name = slot_filename(st.model_id, st.vision)
            target = SLOT_DIR / f"{name}.pending"
            code, res = llama_call("POST", "/slots/0?action=save",
                                   {"filename": target.name}, timeout=120)
            if code != 200 or not res or res.get("n_saved") != n or not target.is_file():
                raise RuntimeError(f"slot save failed: HTTP {code}")
            os.replace(target, SLOT_DIR / name)
            log(f"Saved slot ({n} prompt tokens) atomically -> {name}")
            return True
        root = prepare_ram_slot_dir()
        if st.ram_io_failed:
            raise RuntimeError("RAM checkpoint cleanup failed; refusing another checkpoint")
        target = root / "active.pending.bin"
        if "n_slot_save_estimated_bytes" not in slots[0]:
            raise RuntimeError("server does not report the A2 slot-size estimate")
        estimate = int(slots[0]["n_slot_save_estimated_bytes"])
        if not ram_slot_space_ok(estimate):
            raise RuntimeError("not enough RAM/tmpfs space for the estimated slot and reserve")
        code, res = llama_call("POST", "/slots/0?action=save", {"filename": target.name},
                               timeout=10 if ram_only else 120)
        if (code != 200 or not res or res.get("n_saved") != n
                or target.is_symlink() or not target.is_file() or target.stat().st_size <= 0):
            raise RuntimeError(f"slot save failed or token count differs: HTTP {code}")
        if ram_only:
            saved = {"model": st.model_id, "vision": st.vision, "tokens": n,
                     "bytes": target.stat().st_size, "sha256": file_hash(target),
                     "reload_min_gb": st.vram_resume_min_gb,
                     "signature": checkpoint_signature(st.model_id, st.vision), "saved_at": time.time()}
            temporary = root / "checkpoint.json.tmp"
            temporary.write_text(json.dumps(saved))
            os.replace(target, root / "active.bin")
            os.replace(temporary, root / "checkpoint.json")
            st.ram_checkpoint = {**saved, "path": root / "active.bin"}
            event("checkpoint_saved", f"Saved {n} tokens to RAM")
        else:
            name = slot_filename(st.model_id, st.vision)
            fd, tmp = tempfile.mkstemp(prefix=".slot-", dir=SLOT_DIR)
            os.close(fd)
            try:
                shutil.copyfile(target, tmp)
                os.replace(tmp, SLOT_DIR / name)
            finally:
                Path(tmp).unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            log(f"Saved slot ({n} tokens) to disk")
        return True
    except Exception as e:  # noqa: BLE001 - this is the checkpoint integrity/fallback boundary.
        st.last_error = f"Session checkpoint failed: {e}"
        event("checkpoint_save_failed", st.last_error, fallback=True)
        if ram_only:
            clear_ram_checkpoint()
        elif target is not None:
            target.unlink(missing_ok=True)
        return False


def restore_slot_locked():
    checkpoint = st.ram_checkpoint
    from_ram = bool(checkpoint and checkpoint["model"] == st.model_id and checkpoint["vision"] == st.vision)
    if st.skip_disk_restore and not from_ram:
        st.skip_disk_restore = False
        return ""
    disk = SLOT_DIR / slot_filename(st.model_id, st.vision)
    if not from_ram and not disk.exists():
        return ""
    root = RAM_SLOT_DIR if st.ram_backend else SLOT_DIR
    target = checkpoint["path"] if from_ram else root / "restore.bin" if st.ram_backend else disk
    try:
        if from_ram:
            if checkpoint["signature"] != checkpoint_signature(st.model_id, st.vision):
                raise ValueError("RAM session configuration changed")
            validate_checkpoint(target, checkpoint)
        elif st.ram_backend:
            if not ram_slot_space_ok(disk.stat().st_size):
                raise RuntimeError("not enough RAM for disk checkpoint staging")
            shutil.copyfile(disk, target)
        code, res = llama_call("POST", "/slots/0?action=restore", {"filename": target.name}, timeout=120)
        if (code != 200 or not res or res.get("n_restored", 0) <= 0
                or (from_ram and res["n_restored"] != checkpoint["tokens"])):
            raise RuntimeError(f"slot restore failed or token count differs: HTTP {code}")
        if from_ram:
            cleaned = clear_ram_checkpoint()
            event("checkpoint_restored", f"Restored {res['n_restored']} RAM tokens; " +
                  ("checkpoint removed" if cleaned else "checkpoint cleanup failed"))
        else:
            log(f"Restored slot from disk ({res['n_restored']} tokens)")
        return ""
    except Exception as e:  # noqa: BLE001 - partially restored backends must be discarded safely.
        event("checkpoint_restore_failed", f"RAM/disk session unusable; prompt will be re-read: {e}", fallback=True)
        # A backend may have partially restored corrupt state. Force a fresh process before proxying.
        clear_ram_checkpoint()
        st.skip_disk_restore = True
        return "restart_empty"
    finally:
        if not from_ram and st.ram_backend:
            target.unlink(missing_ok=True)


def notify_vram_yield(message="Видеопамять освобождена. Модель загрузится при следующем запросе."):
    try:
        subprocess.run(["notify-send", "Shura", message],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass


def stop_llama_locked(reason, save=True):
    if st.proc and st.proc.poll() is None:
        if save and st.status == "READY":
            if not save_slot_locked() and GAME.enabled:
                event("game_save_failed", "Save failed; server stays loaded", fallback=True)
                return False
        log(f"Stopping llama-server (pid {st.proc.pid}): {reason}")
        st.expected_exit = True
        try:
            st.proc.terminate()
            try:
                st.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                st.proc.kill()
                st.proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as e:
            if st.proc.poll() is None:
                st.last_error = f"Unable to stop owned server PID {st.proc.pid}: {e}"
                event("stop_failed", st.last_error, fallback=True)
                st.expected_exit = False
                return False
    if RAM_SLOT_DIR is not None:
        (RAM_SLOT_DIR / "server.json").unlink(missing_ok=True)
    st.proc = None
    st.status = "UNLOADED"
    st.model_id = ""
    st.vision = False
    return True


def process_identity(pid):
    root = Path("/proc") / str(pid)
    # comm may contain spaces or parentheses: start-time is field 22, after the final ')'.
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start": fields[19], "uid": root.stat().st_uid,
            "cmd": hashlib.sha256((root / "cmdline").read_bytes()).hexdigest()}


def write_server_lease():
    if not st.ram_backend:
        return
    # nice may still be exec'ing the server when Popen returns. Record only the final executable.
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if Path(os.readlink(f"/proc/{st.proc.pid}/exe")).resolve() == Path(M["exePath"]).resolve():
            break
        time.sleep(0.02)
    else:
        raise OSError("Server executable not ready for ownership lease")
    lease = {**process_identity(st.proc.pid), "slot_dir": str(RAM_SLOT_DIR)}
    target = RAM_SLOT_DIR / "server.json"
    pending = target.with_suffix(".json.tmp")
    pending.write_text(json.dumps(lease))
    os.replace(pending, target)


def recover_orphan(root):
    """Only a matching owned PID can be stopped after a gateway restart; never kill by pattern."""
    lease_path = root / "server.json"
    if not lease_path.exists():
        return
    if lease_path.is_symlink() or lease_path.stat().st_size > 16384:
        raise RuntimeError("Unsafe server lease; manual inspection required")
    saved = json.loads(lease_path.read_text())
    pid = saved.get("pid")
    if not isinstance(pid, int) or pid <= 1 or saved.get("uid") != os.getuid() or saved.get("slot_dir") != str(root):
        raise RuntimeError("Invalid owned server lease; manual inspection required")
    try:
        current = process_identity(pid)
    except FileNotFoundError:
        lease_path.unlink()
        return
    if any(current[key] != saved.get(key) for key in ("pid", "start", "uid", "cmd")):
        lease_path.unlink()
        event("orphan_pid_reused", "Old PID belongs to another process; left untouched", fallback=True)
        return
    code, slots = llama_call("GET", "/slots", timeout=3)
    if code != 200 or not isinstance(slots, list) or not slots or any(slot.get("is_processing", True) for slot in slots):
        raise RuntimeError(f"Owned server PID {pid} is busy or unverifiable; restart waits for the answer")
    if process_identity(pid) != current:
        raise RuntimeError("Server identity changed before recovery; left untouched")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if process_identity(pid) != current:
                break
            # A zombie has already released GPU memory.
            if (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                break
        except FileNotFoundError:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError(f"Owned server PID {pid} did not exit; refusing a second server")
    lease_path.unlink(missing_ok=True)
    event("orphan_recovered", f"Stopped idle owned server PID {pid} after gateway restart")


def backend_port_in_use():
    try:
        with socket.create_connection((G["llamaHost"], G["llamaPort"]), timeout=0.3):
            return True
    except OSError:
        return False


def start_llama(model_id, vision, fresh=False):
    with st.lock:
        if (st.proc and st.proc.poll() is None and st.status == "READY"
                and st.model_id == model_id and st.vision == vision):
            return ""
        if (st.proc and st.proc.poll() is None and st.busy > 1
                and (st.model_id != model_id or st.vision != vision)):
            return "Another response is in progress; retry before switching model or vision mode"
        if st.status == "ERROR":
            return st.last_error
        if st.proc and st.proc.poll() is None and not stop_llama_locked(
                f"switching to model={model_id} vision={vision}", save=True):
            return st.last_error
        mc = model_config(model_id, vision)
        if vision and not mc.get("mmprojPath"):
            vision = False
        reason = preload_check(mc)
        if reason:
            log(f"Load blocked: {reason}")
            st.last_error = reason
            return reason
        st.status, st.model_id, st.vision = "LOADING", model_id, vision
        st.vram_pressure_since = None
        try:
            args = build_args(mc, vision)
        except (OSError, RuntimeError) as e:
            st.status = "UNLOADED"
            st.last_error = str(e)
            return st.last_error
        if st.backend_blocked or backend_port_in_use():
            st.status = "UNLOADED"
            st.last_error = st.backend_blocked or "Backend port is occupied; refusing a second server"
            event("backend_blocked", st.last_error, fallback=True)
            return st.last_error
        log("Starting llama-server: " + " ".join(args))
        st.expected_exit = False
        try:
            st.proc = subprocess.Popen(args, stdin=subprocess.DEVNULL)
            write_server_lease()
        except OSError as e:
            stop_llama_locked("spawn or lease failed", save=False)
            st.last_error = f"Could not spawn llama-server: {e}"
            return st.last_error
        deadline = time.time() + G["healthCheckTimeoutSeconds"]
        while time.time() < deadline:
            time.sleep(1)
            if st.proc.poll() is not None:
                st.crash_count += 1
                st.last_error = f"llama-server exited during load (code {st.proc.returncode})"
                st.proc = None
                st.status = "ERROR" if st.crash_count >= G["crashRestartMaxAttempts"] else "UNLOADED"
                log(st.last_error)
                return st.last_error
            try:
                code, _ = llama_call("GET", "/health", timeout=3)
                if code == 200:
                    st.status, st.crash_count, st.last_error = "READY", 0, ""
                    log(f"llama-server READY (pid {st.proc.pid}, vision={vision})")
                    error = "" if fresh else restore_slot_locked()
                    if error == "restart_empty":
                        if not stop_llama_locked("checkpoint restore failed; starting empty", save=False):
                            return st.last_error
                        st.skip_disk_restore = False
                        st.vram_resume_min_gb = 0.0
                        return start_llama(model_id, vision, fresh=True)
                    if not error:
                        st.vram_resume_min_gb = 0.0
                    return error
            except OSError:
                pass
        st.last_error = f"Health check timed out after {G['healthCheckTimeoutSeconds']}s"
        stop_llama_locked("load timeout", save=False)
        return st.last_error


def monitor_tick(now, free_mib, ram, wall_now=None, process_mib=None):
    """Monotonic pressure timers; idle means time since the last completed/admitted request."""
    with st.lock:
        st.monitor_last_tick = time.time()
        if math.isfinite(free_mib) and 0 <= free_mib <= 2097152:
            st.vram_free_mib = int(free_mib)
            st.vram_sample_at = st.monitor_last_tick
        if isinstance(process_mib, int) and process_mib >= 0:
            st.proc_vram_mib = process_mib
            st.proc_vram_pid = st.proc.pid if st.proc else None
            st.proc_vram_sample_at = st.monitor_last_tick
        if st.proc and st.proc.poll() is not None and not st.expected_exit:
            st.crash_count += 1
            st.last_error = f"llama-server crashed (exit {st.proc.returncode})"
            event("server_crashed", st.last_error, fallback=True)
            st.proc = None
            st.status = "ERROR" if st.crash_count >= G["crashRestartMaxAttempts"] else "UNLOADED"

        valid = math.isfinite(free_mib) and 0 <= free_mib <= 2097152
        if GAME.enabled and valid:
            st.monitor_error = ""
        idle = (time.time() if wall_now is None else wall_now) - st.last_request
        if G["vramGuard"]:
            if not valid:
                reset_pressure()
                if st.monitor_error != "VRAM telemetry unavailable":
                    event("telemetry_failed", "VRAM telemetry unavailable; using original lifecycle", fallback=True)
                st.monitor_error = "VRAM telemetry unavailable"
            else:
                if st.monitor_error:
                    event("telemetry_recovered", "VRAM readings available again")
                    if st.guard_fallback == "Fresh NVML data unavailable; skipping VRAM admission check":
                        st.guard_fallback = ""
                st.monitor_error = ""
                if st.status == "READY":
                    if free_mib < G["vramFreeCriticalMiB"]:
                        if st.vram_pressure_since is None:
                            st.vram_pressure_since = now
                        st.vram_pressure = now - st.vram_pressure_since >= G["vramPressureSeconds"]
                    else:
                        st.vram_pressure_since = None
                        st.vram_pressure = st.pressure_notified = False
                    if free_mib < G["vramEmergencyMiB"]:
                        if st.vram_emergency_since is None:
                            st.vram_emergency_since = now
                        st.vram_emergency = now - st.vram_emergency_since >= G["vramEmergencySeconds"]
                    else:
                        st.vram_emergency_since = None
                        st.vram_emergency = False
                    if st.vram_pressure and not st.pressure_notified:
                        st.pressure_notified = True
                        event("vram_pressure", "Desktop needs VRAM; active agent session stays loaded")
                        notify_vram_yield("Видеопамяти мало. Активная сессия продолжает работать.")
                    should_yield = st.busy == 0 and (st.vram_emergency or
                                   (st.vram_pressure and idle >= G["vramIdleSeconds"]))
                    if should_yield and (now >= st.next_save_attempt or st.vram_emergency):
                        used = process_vram_mib(st.proc.pid) if st.proc else -1
                        required = ((used + 250) / 1024
                                    if isinstance(used, int) and used >= 0 else st.preload_vram_gb)
                        st.vram_resume_min_gb = max(model_config(st.model_id, st.vision)["vramFreeMinGBToLoad"], required)
                        saved = save_slot_locked(ram_only=True)
                        if saved or st.vram_emergency:
                            reason = "vram_emergency" if st.vram_emergency else "vram_pressure"
                            if not saved:
                                clear_ram_checkpoint()
                                st.skip_disk_restore = True
                                event("emergency_context_lost", "Emergency unload without RAM session; next prompt will be re-read", fallback=True)
                            if stop_llama_locked(reason, save=False):
                                st.last_yield_reason, st.last_yield_at = reason, time.time()
                                event("vram_yield", f"Model unloaded: {reason}")
                                notify_vram_yield("Видеопамять освобождена. " +
                                                 ("Сессия сохранена в RAM." if saved else "Контекст будет прочитан заново."))
                        else:
                            # Original lifecycle stays active; retry at most once per 30 seconds.
                            st.next_save_attempt = now + 30
                            notify_vram_yield("Сессию не удалось сохранить в RAM. Модель остаётся загруженной.")
                else:
                    st.vram_pressure_since = st.vram_emergency_since = None
                    if free_mib >= G["vramFreeCriticalMiB"]:
                        st.vram_pressure = st.vram_emergency = st.pressure_notified = False
        else:
            reset_pressure()

        if st.status == "READY" and st.busy == 0 and idle > G["idleUnloadSeconds"] and not (GAME.enabled and GAME.active):
            stop_llama_locked(f"idle for {int(idle)}s", save=True)

        if now - st.last_ram_poll >= G["pollIntervalSeconds"]:
            st.last_ram_poll = now
            if st.status == "READY":
                st.pressure_sustain = st.pressure_sustain + 1 if ram < G["ramFreeMinGB"] else 0
                if st.pressure_sustain >= G["memoryPressureSustainPolls"] and st.busy == 0 and not (GAME.enabled and GAME.active):
                    stop_llama_locked("memory pressure", save=True)
                    st.memory_pressure = True
                    st.status = "MEMORY_PRESSURE"
            elif st.memory_pressure and ram >= G["ramFreeMinGBToLoad"]:
                st.memory_pressure = False
                st.status = "UNLOADED"
        st.monitor_healthy = True


def monitor(stop=None):
    stop = stop or threading.Event()
    while not stop.wait(1):
        try:
            free_mib = vram_free_gb() * 1024
            proc = st.proc
            proc_mib = None
            if proc is not None and proc.poll() is None:
                try:
                    proc_mib = nvml_device().proc_mib(proc.pid)
                except (OSError, RuntimeError, ValueError, OverflowError):
                    proc_mib = None
            monitor_tick(time.monotonic(), free_mib, mem_available_gb(), process_mib=proc_mib)
        except Exception as e:  # noqa: BLE001 - a failed sensor tick must be recorded, not kill the monitor.
            with st.lock:
                st.monitor_healthy = False
                st.monitor_error = str(e)
                reset_pressure()
                event("monitor_failed", f"Guard disabled for this tick; original lifecycle retried next tick: {e}", fallback=True)


def acquire_model(model_id, vision):
    """Reserve the request before checking/loading, closing the ready-to-proxy unload race."""
    deadline = time.monotonic() + G["vramReloadWaitSeconds"]
    while True:
        with st.lock:
            if GAME.enabled and (GAME.active or not GAME.ready):
                return "Model paused because of a game: " + (GAME.reason or "detector initializing")
            if (G["vramGuard"] or GAME.enabled) and RAM_SLOT_DIR is None:
                try:
                    prepare_ram_slot_dir()
                except (OSError, RuntimeError) as e:
                    event("ram_path_failed", f"RAM sessions unavailable; ordinary lifecycle retained: {e}", fallback=True)
            ready = (st.proc and st.proc.poll() is None and st.status == "READY"
                     and st.model_id == model_id and st.vision == vision)
            if ready:
                st.busy += 1
                st.last_request = time.time()
                return ""
            error = preload_check(model_config(model_id, vision))
            if not error:
                st.busy += 1
                try:
                    error = start_llama(model_id, vision)
                except BaseException:
                    st.busy -= 1
                    raise
                if error:
                    st.busy -= 1
                return error
            wait_for_vram = "VRAM" in error and st.override != "OFF" and not st.multimedia_lock
        if not wait_for_vram or time.monotonic() >= deadline:
            return error
        time.sleep(min(1, max(0, deadline - time.monotonic())))


def game_tick(now, reasons, errors=()):
    notifications = []
    with st.lock:
        changed = GAME.update(now, reasons, errors)
        if changed:
            event("game_active" if GAME.active else "game_quiet", GAME.reason or "Heavy GPU load ended")
            if not GAME.active:
                if st.status == "PARKED_BY_GAME":
                    st.status = "UNLOADED"
                GAME.parked = False
                notifications.append("Игра окончена. Модель загрузится по следующему запросу.")
        if not GAME.active:
            return notifications
        if st.busy:
            GAME.drain_since = now if GAME.drain_since is None else GAME.drain_since
            if now - GAME.drain_since >= G["gameDrainSeconds"]:
                for lease in list(GAME_REQUESTS.values()):
                    lease["cancel"].set()
                    sock = lease.get("sock") or lease["conn"].sock
                    if sock:
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                event("game_cancel_requested", "Drain timeout: client receives explicit error before checkpoint")
            return notifications
        if GAME.parked or now < GAME.next_save:
            return notifications
        if st.proc and st.proc.poll() is None:
            try:
                code, slots = llama_call("GET", "/slots")
                if code != 200 or any(slot.get("is_processing", False) for slot in slots):
                    return notifications
            except (OSError, TypeError):
                return notifications
            if not save_slot_locked(ram_only=True):
                GAME.next_save = now + 30
                event("game_save_failed", "A2 save failed; model stays loaded", fallback=True)
                notifications.append("Не удалось сохранить A2. Модель оставлена загруженной.")
                return notifications
            if not stop_llama_locked("game mode", save=False):
                return notifications
        st.status = "PARKED_BY_GAME"
        st.last_error = ""
        if st.guard_fallback.startswith(("A2 save failed", "Session checkpoint failed")):
            st.guard_fallback = ""
        GAME.next_save = 0
        GAME.parked = True
        event("game_parked", GAME.reason)
        notifications.append("Модель на паузе из-за игры. Контекст сохранён в RAM.")
    return notifications


def game_monitor(stop=None):
    stop = stop or threading.Event()
    while not stop.is_set():
        if not GAME.enabled:
            stop.wait(G["gamePollSeconds"])
            continue
        try:
            pid = st.proc.pid if st.proc else None
            reasons, errors = GAME_SIGNALS.collect(pid, nvml_device() if G["gameDetectNvml"] else None)
            for message in game_tick(time.monotonic(), reasons, errors):
                notify_vram_yield(message)
        except Exception as error:
            with st.lock:
                GAME.errors = [str(error)]
                event("game_detector_failed", str(error), fallback=True)
        stop.wait(G["gamePollSeconds"])


def status_snapshot():
    proc = st.proc
    return {
        "game_mode": GAME.snapshot(),
        "status": st.status, "model": st.model_id, "vision": st.vision, "override": st.override,
        "memory_pressure": st.memory_pressure, "multimedia_lock": st.multimedia_lock,
        "vram_pressure": st.vram_pressure, "last_yield_reason": st.last_yield_reason,
        "last_yield_at": st.last_yield_at, "vram_reload_min_gb": round(st.vram_resume_min_gb, 3),
        "ram_session_saved": st.ram_checkpoint is not None,
        "vram_guard_enabled": G["vramGuard"], "vram_emergency": st.vram_emergency,
        "guard_fallback": st.guard_fallback, "last_event": st.last_event,
        "event_counters": dict(st.event_counters), "monitor_healthy": st.monitor_healthy,
        "monitor_last_tick": st.monitor_last_tick, "monitor_error": st.monitor_error,
        "vram_free_mib": st.vram_free_mib,
        "vram_sample_age_seconds": (round(time.time() - st.vram_sample_at, 2) if st.vram_sample_at else None),
        "monitor_thread_alive": st.monitor_thread.is_alive() if st.monitor_thread else None,
        "backend_blocked": st.backend_blocked, "ram_io_failed": st.ram_io_failed,
        "busy_requests": st.busy, "crash_count": st.crash_count, "last_error": st.last_error,
        "ram_available_gb": round(mem_available_gb(), 2),
        "vram_free_gb": (round(st.vram_free_mib / 1024, 2) if st.vram_free_mib is not None else None),
        "proc_vram_mib": st.proc_vram_mib,
        "idle_seconds": int(time.time() - st.last_request),
        "idle_unload_after_seconds": G["idleUnloadSeconds"],
        "llama_pid": proc.pid if proc and proc.poll() is None else None,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def send_json(self, code, obj, retry_after=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if retry_after is not None:
            self.send_header("Retry-After", str(retry_after))
        self.end_headers()
        self.wfile.write(body)

    def guardian_endpoint(self, path):
        if path == "/guardian/status":
            self.send_json(200, status_snapshot())
        elif path == "/guardian/prepare-rollback":
            with st.lock:
                if st.busy:
                    return self.send_json(409, {"error": "Response in progress; rollback waits for completion"})
                previous = st.override
                st.override = "OFF"  # Hold admission before stopping; no status/stop race.
                if not stop_llama_locked("operator rollback", save=True):
                    st.override = previous
                    return self.send_json(503, {"error": st.last_error})
                try:
                    OVERRIDE_FLAG.write_text("OFF")
                except OSError as e:
                    st.override = previous
                    return self.send_json(503, {"error": f"Cannot persist rollback gate: {e}"})
                event("rollback_prepared", "Active answer complete; requests gated for operator rollback")
                self.send_json(200, {"ok": True, "previous_override": previous})
        elif path in ("/guardian/pause", "/guardian/resume"):
            if not GAME.enabled:
                return self.send_json(409, {"error": "Game mode is disabled in configuration"})
            with st.lock:
                was_active = GAME.active
                GAME.manual_switch(path.endswith("pause"), time.monotonic())
                if was_active and not GAME.active:
                    if st.status == "PARKED_BY_GAME":
                        st.status = "UNLOADED"
                    GAME.parked = False
                    notify_vram_yield("Игра окончена. Модель загрузится по следующему запросу.")
            self.send_json(200, {"ok": True, "game_mode": GAME.snapshot()})
        elif path == "/guardian/ai-on":
            OVERRIDE_FLAG.write_text("ON")
            with st.lock:
                st.override = "AUTO"
                if st.status == "ERROR":
                    st.status, st.crash_count, st.last_error = "UNLOADED", 0, ""
            self.send_json(200, {"ok": True, "override": "AUTO"})
        elif path == "/guardian/ai-off":
            OVERRIDE_FLAG.write_text("OFF")
            with st.lock:
                st.override = "OFF"
                stop_llama_locked("manual ai-off", save=True)
            self.send_json(200, {"ok": True, "override": "OFF"})
        elif path == "/guardian/unload":
            with st.lock:
                stop_llama_locked("manual unload", save=True)
            self.send_json(200, {"ok": True})
        elif path == "/guardian/multimedia-lock":
            with st.lock:
                stop_llama_locked("multimedia pipeline acquiring GPU", save=True)
                st.multimedia_lock = True
            self.send_json(200, {"ok": True})
        elif path == "/guardian/multimedia-unlock":
            st.multimedia_lock = False
            self.send_json(200, {"ok": True})
        else:
            self.send_json(404, {"error": f"unknown guardian endpoint {path}"})

    def handle_any(self):
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""

        if path.startswith("/guardian/"):
            return self.guardian_endpoint(path)

        st.last_request = time.time()
        if GAME.enabled and (GAME.active or not GAME.ready):
            return self.send_json(503, {"error": "Модель на паузе из-за игры", "reason": GAME.reason or "detector initializing"}, G["gameRetrySeconds"])
        if st.override == "OFF":
            return self.send_json(503, {"error": "AI is manually disabled (ai-off). POST /guardian/ai-on to re-enable."})
        if st.multimedia_lock:
            return self.send_json(503, {"error": "GPU is busy with an audio/video job; retry when it finishes."})
        if st.memory_pressure:
            return self.send_json(503, {"error": "AI paused due to system memory pressure; resumes automatically."})

        vision, model_id = False, M["defaultModel"]
        if body:
            text = body.decode("utf-8", errors="replace")
            if '"image_url"' in text or re.search(r'"type"\s*:\s*"image', text):
                vision = True
            m = re.search(r'"model"\s*:\s*"([^"]+)"', text)
            if m and m.group(1) in M["models"]:
                model_id = m.group(1)
            # The agent's AI-SDK client (ShuraCode / OpenCode) sends camelCase "reasoningEffort"; llama-server only reads "reasoning_effort".
            if '"reasoningEffort"' in text:
                body = re.sub(r'"reasoningEffort"\s*:', '"reasoning_effort":', text).encode()

        err = acquire_model(model_id, vision)
        if err:
            return self.send_json(503, {"error": f"Could not start model '{model_id}'", "reason": err}, G["gameRetrySeconds"] if GAME.enabled and GAME.active else None)

        try:
            if GAME.enabled:
                self.proxy_game(body)
            else:
                self.proxy(body)
        finally:
            with st.lock:
                st.busy -= 1
                st.last_request = time.time()

    def proxy_game(self, body):
        conn = http.client.HTTPConnection(G["llamaHost"], G["llamaPort"], timeout=3600)
        key = id(self)
        lease = {"conn": conn, "cancel": threading.Event()}
        sent = False
        chunked = False
        with st.lock:
            GAME_REQUESTS[key] = lease
        try:
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            if body or self.command in ("POST", "PUT", "PATCH"):
                headers["Content-Length"] = str(len(body))
            conn.request(self.command, self.path, body=body or None, headers=headers)
            lease["sock"] = conn.sock
            reply = conn.getresponse()
            if lease["cancel"].is_set():
                raise ConnectionAbortedError("game drain timeout")
            self.send_response(reply.status)
            for name, value in reply.getheaders():
                if name.lower() not in HOP_BY_HOP:
                    self.send_header(name, value)
            length = reply.getheader("Content-Length")
            chunked = length is None
            self.send_header("Transfer-Encoding", "chunked") if chunked else self.send_header("Content-Length", length)
            self.end_headers()
            sent = True
            while self.command != "HEAD":
                data = reply.read1(65536)
                if not data:
                    break
                self.wfile.write((f"{len(data):X}\r\n".encode() + data + b"\r\n") if chunked else data)
                self.wfile.flush()
            if chunked:
                if lease["cancel"].is_set():
                    raise ConnectionAbortedError("game drain timeout")
                self.wfile.write(b"0\r\n\r\n")
        except (OSError, http.client.HTTPException) as error:
            if lease["cancel"].is_set():
                message = "Генерация остановлена для игрового режима; повторите запрос после игры."
                if not sent:
                    self.send_json(503, {"error": message, "type": "game_pause_timeout"}, G["gameRetrySeconds"])
                elif chunked:
                    data = ("data: " + json.dumps({"error": {"message": message, "type": "game_pause_timeout"}}) + "\n\ndata: [DONE]\n\n").encode()
                    try:
                        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n0\r\n\r\n")
                        self.wfile.flush()
                    except OSError:
                        pass
            elif not sent:
                self.send_json(502, {"error": f"llama-server unreachable: {error}"})
        finally:
            conn.close()
            with st.lock:
                GAME_REQUESTS.pop(key, None)

    def proxy(self, body):
        conn = http.client.HTTPConnection(G["llamaHost"], G["llamaPort"], timeout=3600)
        try:
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            if body or self.command in ("POST", "PUT", "PATCH"):
                headers["Content-Length"] = str(len(body))
            conn.request(self.command, self.path, body=body or None, headers=headers)
            r = conn.getresponse()
        except OSError as e:
            conn.close()
            return self.send_json(502, {"error": f"llama-server unreachable: {e}"})
        try:
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            clen = r.getheader("Content-Length")
            if clen is not None:
                self.send_header("Content-Length", clen)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(r.read())
                return
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away; closing upstream makes llama-server abort the generation
        finally:
            conn.close()

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = handle_any


def shutdown(signum, _frame):
    log(f"Signal {signum}: saving active slot and stopping llama-server")
    with st.lock:
        stop_llama_locked("gateway shutdown", save=True)
    sys.exit(0)


def main():
    if G["vramGuard"] is not True and int(G.get("slotSaveCheckpoints", 0)) == 0 and not GAME.enabled:
        return legacy_main()
    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    st.monitor_thread = threading.Thread(target=monitor, daemon=True, name="shura-vram-guard")
    st.monitor_thread.start()
    if GAME.enabled:
        threading.Thread(target=game_monitor, daemon=True, name="shura-game-mode").start()
    server = ThreadingHTTPServer((G["gatewayHost"], G["gatewayPort"]), Handler)
    server.daemon_threads = True
    log(f"Gateway listening on http://{G['gatewayHost']}:{G['gatewayPort']} -> llama-server :{G['llamaPort']} "
        f"(config {CFG_DIR}, override={st.override}, idle unload {G['idleUnloadSeconds']}s)")
    server.serve_forever()


if __name__ == "__main__":
    main()
