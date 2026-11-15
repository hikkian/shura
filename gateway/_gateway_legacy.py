#!/usr/bin/env python3
"""AI Gateway / Resource Guardian for a single local llama-server.

Sits in front of llama-server as an OpenAI-compatible reverse proxy and owns its lifecycle:
load on first request, switch text/vision mode by inspecting the request, unload when idle or
under memory pressure, restart after crashes, and persist the active KV slot to disk only when
the model is unloaded (so session switching stays in RAM and SSD writes stay rare).

Standard library only. Configuration: guardian.json + model-launch.json in $AI_GATEWAY_CONFIG
(default: ../config next to this file).
"""
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import gateway_common as common
from gpu_nvml import NVML
from proxy_core import DEFAULTS as PROXY_DEFAULTS, error_body, forward_request, inspect_body
from web_guard import BadFraming, BoundedServer, DEFAULTS as WEB_DEFAULTS, Policy as WebPolicy, content_length, guarded_handler
from monitor import Monitor, gateway_snapshot, monitor_authorized, monitor_route, read_expert_cache_metrics
from monitor_sources import sample_gpu

CFG_DIR = Path(os.environ.get("AI_GATEWAY_CONFIG", Path(__file__).resolve().parent.parent / "config"))
STATE_DIR = Path(os.path.expanduser(os.environ.get("AI_GATEWAY_STATE", "~/.local/state/ai-gateway")))


PATH_KEYS = {"exePath", "modelPath", "mmprojPath", "moeCacheProfile", "slotSaveDir"}


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
G = {**WEB_DEFAULTS, **PROXY_DEFAULTS, **common.LIFECYCLE_DEFAULTS, **G}
SLOT_DIR = Path(G["slotSaveDir"])
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


def vram_free_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return float(out.strip().splitlines()[0]) / 1024
    except Exception:
        return -1.0


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
        self.crash_times = deque()
        self.error_since = None
        self.loading_since = None
        self.health_failures = 0
        self.last_health_probe = 0.0
        self.shutting_down = threading.Event()
        self.slot_dirty = True


st = State()
st.events = deque(maxlen=30)
st.event_counters = {}
MONITOR = Monitor()
_MONITOR_NVML = None


def event(reason, detail):
    with st.lock:
        st.event_counters[reason] = st.event_counters.get(reason, 0) + 1
        st.events.append({"reason": str(reason)[:100], "detail": str(detail)[:240], "at": time.time()})
    log(f"Gateway {reason}: {detail}")


def _monitor_gpu(status):
    global _MONITOR_NVML
    def nvidia_sample():
        global _MONITOR_NVML
        if _MONITOR_NVML is None:
            _MONITOR_NVML = NVML()
        return _MONITOR_NVML.monitor_info(status.get("llama_pid"))
    return sample_gpu("/sys", nvidia_sample)


def _monitor_slots(status):
    if not status.get("llama_pid") or status.get("status") != "READY":
        return None
    try:
        code, value = llama_call("GET", "/slots", timeout=0.5)
        return value if code == 200 and isinstance(value, list) else None
    except Exception:
        return None


def _monitor_state_payload():
    def status_provider():
        return {"status": st.status, "model": st.model_id, "vision": st.vision,
                "override": st.override, "memory_pressure": st.memory_pressure,
                "multimedia_lock": st.multimedia_lock, "busy_requests": st.busy,
                "crash_count": st.crash_count, "last_error": st.last_error,
                "llama_pid": st.proc.pid if st.proc and st.proc.poll() is None else None,
                "default_model": M["defaultModel"], "events": list(st.events), "slot_dir": str(SLOT_DIR)}
    return gateway_snapshot(MONITOR, status_provider, model_config,
        _monitor_gpu, _monitor_slots,
        lambda status: read_expert_cache_metrics(G["llamaHost"], G["llamaPort"], M.get("apiKey"))
        if status.get("llama_pid") else None)


def _monitor_authorized(handler):
    return monitor_authorized(handler.headers, {**G, "apiKey": G.get("apiKey") or M.get("apiKey")})


def _monitor_begin(handler, path, body):
    try:
        payload = json.loads(body) if body else {}
        model = payload.get("model", M["defaultModel"]) if isinstance(payload, dict) else M["defaultModel"]
        handler._monitor_request_id = MONITOR.begin_request(model, path, handler.command)
    except Exception:
        handler._monitor_request_id = None


def _monitor_finish(handler, status=None, error=False):
    ident = getattr(handler, "_monitor_request_id", None)
    if ident is not None:
        try:
            MONITOR.finish_request(ident, status=status, error=error)
        except Exception:
            pass
        handler._monitor_request_id = None


def _monitor_call(method, *args):
    try:
        return getattr(MONITOR, method)(*args)
    except Exception:
        return None


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
        conn.request(method, path, body=payload, headers=common.llama_headers(M.get("apiKey"), bool(payload)))
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
         "--cache-ram", str(M["cacheRamMB"]), "--slot-save-path", str(SLOT_DIR)]
    if mc.get("moeCacheProfile") and mc.get("moeCacheSlots", 0) > 0:
        a += ["--moe-cache-profile", mc["moeCacheProfile"], "--moe-cache-slots", str(mc["moeCacheSlots"])]
    if mc.get("specType") and mc["specType"] != "none":
        a += ["--spec-type", mc["specType"], "--spec-draft-n-max", str(mc["specDraftNMax"]),
              "--spec-draft-p-min", str(mc["specDraftPMin"])]
    a += ["--mmproj", mc["mmprojPath"]] if vision else ["--no-mmproj-auto"]
    a += common.cors_args(M.get("exePath"), G["llamaPort"])
    a += common.key_args(M.get("exePath"), STATE_DIR, M.get("apiKey"))
    if mc.get("niceLevel"):
        # Lower CPU priority: desktop apps win any contention; no speed is lost while the desktop is idle.
        a = ["nice", "-n", str(mc["niceLevel"])] + a
    return a


def preload_check(mc):
    if st.override == "OFF":
        return "AI is manually disabled (ai-off)"
    if st.multimedia_lock:
        return "GPU is busy with an audio/video job"
    ram = mem_available_gb()
    if ram < G["ramFreeMinGBToLoad"]:
        return f"Not enough free RAM to load the model ({ram:.1f} GB available, need {G['ramFreeMinGBToLoad']} GB)"
    vram = vram_free_gb()
    if 0 <= vram < mc["vramFreeMinGBToLoad"]:
        return f"Not enough free VRAM ({vram:.1f} GB free, need {mc['vramFreeMinGBToLoad']} GB)"
    return ""


def save_slot_locked():
    """Persist the active slot only when it is large enough that re-prefilling would be slow."""
    pending_path = None
    try:
        _, slots = llama_call("GET", "/slots")
        n = (slots or [{}])[0].get("n_prompt_tokens", 0)
        if n < G["slotSaveMinTokens"]:
            log(f"Slot has {n} tokens (< {G['slotSaveMinTokens']}), not saving to disk")
            return
        name = slot_filename(st.model_id, st.vision)
        final_path = SLOT_DIR / name
        if not st.slot_dirty and final_path.is_file() and not final_path.is_symlink():
            log("Slot unchanged since it was restored; not written to disk again")
            return
        pending_path = SLOT_DIR / f"{name}.pending"
        code, res = llama_call("POST", "/slots/0?action=save", {"filename": pending_path.name}, timeout=120)
        if code != 200 or not pending_path.is_file():
            raise RuntimeError(f"slot save returned HTTP {code} or did not create its pending file")
        os.replace(pending_path, final_path)
        log(f"Saved slot ({n} prompt tokens) atomically -> {name}: HTTP {code} {res and res.get('n_saved')} tokens")
    except Exception as e:  # noqa: BLE001 - slot persistence must not take down the gateway
        if pending_path is not None:
            try:
                pending_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                log(f"Could not remove incomplete slot {pending_path.name}: {cleanup_error}")
        log(f"Slot save failed (non-fatal): {e}")


def restore_slot_locked():
    st.slot_dirty = True
    name = slot_filename(st.model_id, st.vision)
    if not (SLOT_DIR / name).exists():
        return
    try:
        code, res = llama_call("POST", "/slots/0?action=restore", {"filename": name}, timeout=120)
        st.slot_dirty = not (code == 200 and (res or {}).get("n_restored", 0) > 0)
        log(f"Restored slot {name}: HTTP {code} {res and res.get('n_restored')} tokens")
    except Exception as e:
        log(f"Slot restore failed (non-fatal, falls back to normal prefill): {e}")


def lease_root():
    return STATE_DIR


def stop_llama_locked(reason, save=True):
    was_running = bool(st.proc and st.proc.poll() is None)
    crashed = False
    if st.proc and st.proc.poll() is not None and not st.expected_exit:
        # Dead before we got here and nobody had noticed: that is a crash, not a clean stop.
        st.last_error = f"llama-server crashed (exit {st.proc.returncode})"
        event("server_crashed", st.last_error)
        crashed = common.note_crash(st, G)
    elif st.proc and st.proc.poll() is None:
        if save and st.status == "READY":
            save_slot_locked()
        log(f"Stopping llama-server (pid {st.proc.pid}): {reason}")
        st.expected_exit = True
        st.proc.terminate()
        try:
            st.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            st.proc.kill()
            st.proc.wait()
    try:
        (lease_root() / "server.json").unlink(missing_ok=True)
    except OSError:
        pass
    st.proc = None
    # ERROR is left only by `ai-on` or the cooldown, never by an unrelated stop.
    st.status = "ERROR" if (crashed or st.status == "ERROR") else "UNLOADED"
    st.model_id = ""
    st.vision = False
    if was_running:
        event("model_unloaded", reason)


def backend_port_in_use():
    try:
        with socket.create_connection((G["llamaHost"], G["llamaPort"]), timeout=0.3):
            return True
    except OSError:
        return False


def recover_orphans_at_start():
    """A server left behind by a gateway that died keeps holding its RAM and VRAM. Stop it now, before the first request: that request
    would otherwise be refused by the memory check (the orphan's own memory counts against it). Only a process whose identity matches
    the lease is touched."""
    try:
        if (lease_root() / "server.json").exists():
            common.stop_owned_orphan(lease_root(), llama_call, lambda reason, detail, **_: event(reason, detail))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, http.client.HTTPException) as error:
        log(f"Orphaned server left alone: {error}")


def claim_port():
    """True when the backend port is free, or held by an idle server this gateway started before it died (that one is stopped)."""
    if not backend_port_in_use():
        return True
    try:
        common.stop_owned_orphan(lease_root(), llama_call, lambda reason, detail, **_: event(reason, detail))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, http.client.HTTPException) as e:
        st.last_error = f"Backend port is occupied; refusing a second server ({e})"
        return False
    for _ in range(30):               # the stopped server's socket takes a moment to disappear
        if not backend_port_in_use():
            return True
        time.sleep(0.1)
    return False


def start_llama(model_id, vision):
    with st.lock:
        if st.shutting_down.is_set():
            return "Gateway is shutting down"
        if st.proc and st.proc.poll() is not None and not st.expected_exit:
            st.last_error = f"llama-server crashed (exit {st.proc.returncode})"
            event("server_crashed", st.last_error)
            st.proc = None
            st.model_id, st.vision = "", False
            st.status = "ERROR" if common.note_crash(st, G) else "UNLOADED"
        if (st.proc and st.proc.poll() is None and st.status == "READY"
                and st.model_id == model_id and st.vision == vision):
            return ""
        if (st.proc and st.proc.poll() is None and st.busy > 1
                and (st.model_id != model_id or st.vision != vision)):
            return "Another response is in progress; retry before switching model or vision mode"
        if st.status == "ERROR":
            return st.last_error
        mc = model_config(model_id, vision)
        if vision and not mc.get("mmprojPath"):
            vision = False
        # Admission is decided before the running model is touched: a refused switch must leave it loaded.
        reason = preload_check(mc)
        if reason:
            log(f"Load blocked: {reason}")
            st.last_error = reason
            return reason
        if st.proc and st.proc.poll() is None:
            stop_llama_locked(f"switching to model={model_id} vision={vision}", save=True)
        st.status, st.model_id, st.vision = "LOADING", model_id, vision
        st.loading_since = time.time()
        event("model_loading", f"model={model_id} vision={vision}")
        try:
            return load_server_locked(model_id, vision, mc)
        except Exception as e:  # noqa: BLE001 - nothing may leave LOADING behind with a child running and no answer.
            log(f"Load failed unexpectedly: {type(e).__name__}: {e}")
            st.last_error = f"Could not start model: {type(e).__name__}: {e}"
            try:
                stop_llama_locked("load failed", save=False)
            except Exception:  # noqa: BLE001
                pass
            if st.status == "LOADING":
                st.status = "UNLOADED"
            return st.last_error
        finally:
            st.loading_since = None


def load_server_locked(model_id, vision, mc):
    try:
        args = build_args(mc, vision)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as e:
        st.status = "UNLOADED"
        st.last_error = f"Invalid model configuration: {type(e).__name__}: {e}"
        return st.last_error
    if not claim_port():
        st.status = "UNLOADED"
        st.last_error = st.last_error or "Backend port is occupied; refusing a second server"
        event("backend_blocked", st.last_error)
        return st.last_error
    log("Starting llama-server: " + common.redacted_command(args))
    st.expected_exit = False
    st.proc = subprocess.Popen(args, stdin=subprocess.DEVNULL)
    deadline = time.monotonic() + G["healthCheckTimeoutSeconds"]
    while time.monotonic() < deadline:
        if st.shutting_down.wait(0.5):
            stop_llama_locked("gateway shutdown during load", save=False)
            st.last_error = "Gateway is shutting down"
            return st.last_error
        if st.override == "OFF" or st.multimedia_lock:
            stop_llama_locked("switched off during load", save=False)
            st.last_error = "AI is manually disabled (ai-off)" if st.override == "OFF" else "GPU is busy with an audio/video job"
            return st.last_error
        if st.proc.poll() is not None:
            st.last_error = f"llama-server exited during load (code {st.proc.returncode})"
            st.proc = None
            st.status = "ERROR" if common.note_crash(st, G) else "UNLOADED"
            log(st.last_error)
            return st.last_error
        try:
            code, _ = llama_call("GET", "/health", timeout=3)
        except (OSError, ValueError, http.client.HTTPException):
            continue
        if code != 200:
            continue
        st.status, st.last_error = "READY", ""
        st.health_failures, st.last_health_probe = 0, time.monotonic()
        try:
            # Written when the server has answered on our port: any launcher script or `nice` has finished exec'ing by now.
            common.write_lease(lease_root(), st.proc, M["exePath"])
        except OSError as lease_error:
            # The lease only helps a later gateway to find its orphan; without one the port check still protects.
            log(f"Server lease not written ({lease_error}); an orphaned server would have to be stopped by hand")
        log(f"llama-server READY (pid {st.proc.pid}, vision={vision})")
        event("server_started", f"model={model_id} vision={vision}")
        restore_slot_locked()
        return ""
    st.last_error = f"Health check timed out after {G['healthCheckTimeoutSeconds']}s"
    if common.note_crash(st, G):
        st.status = "ERROR"
    stop_llama_locked("load timeout", save=False)
    return st.last_error


def declare_hung(proc, reason):
    """The server is alive but does not answer: stop it, count it as a failure, and let the next request start a fresh one."""
    acquired = st.lock.acquire(timeout=G["shutdownLockSeconds"])
    try:
        if not acquired:
            return False                      # a load is still running; the next probe or stall tries again
        if st.proc is not proc or proc.poll() is not None:
            return False
        st.last_error = f"llama-server hung ({reason})"
        event("server_hung", st.last_error)
        st.expected_exit = True
        try:
            proc.kill()                       # a hung server cannot save a session or answer SIGTERM (a stopped process never even sees it)
            proc.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            if proc.poll() is None:
                return False
        st.proc = None
        st.model_id, st.vision = "", False
        st.status = "ERROR" if common.note_crash(st, G) else "UNLOADED"
        return True
    finally:
        if acquired:
            st.lock.release()


def settle_dead_backend():
    """A killed server closes its socket first and goes away only when the driver has released its GPU memory (seconds for a 10 GB
    model); until then `proc.poll()` says "running" although nothing listens. Wait for it to finish, so the crash is counted and the
    model restarted by the normal path; if it never finishes, treat it as hung."""
    proc = st.proc
    if proc is None or st.status != "READY" or proc.poll() is not None:
        return
    if common.backend_listening(G["llamaHost"], G["llamaPort"]):
        return
    end = time.monotonic() + G["deadBackendWaitSeconds"]
    while time.monotonic() < end:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    declare_hung(proc, "stopped listening but did not exit")


def health_probe(now):
    """A READY server that stops answering /health several times in a row is declared hung. This runs outside st.lock."""
    every = G["healthProbeSeconds"]
    proc = st.proc
    if not every or st.status != "READY" or proc is None or proc.poll() is not None or now - st.last_health_probe < every:
        return
    st.last_health_probe = now
    try:
        ok = llama_call("GET", "/health", timeout=5)[0] == 200
    except (OSError, ValueError, http.client.HTTPException):
        ok = False
    if ok:
        st.health_failures = 0
        return
    st.health_failures += 1
    event("health_probe_failed", f"llama-server /health failed ({st.health_failures} in a row)")
    if st.health_failures >= G["healthProbeFailures"]:
        st.health_failures = 0
        declare_hung(proc, f"/health failed {G['healthProbeFailures']} times in a row")


def monitor():
    while True:
        time.sleep(G["pollIntervalSeconds"])
        try:
            health_probe(time.monotonic())
            with common.try_lock(st.lock, G["pollIntervalSeconds"]) as got:
                if not got:
                    continue            # a load is running; the decisions below wait for the next poll
                if common.error_cooldown_over(st, G, time.monotonic()):
                    common.clear_error_state(st)
                    st.status = "UNLOADED"
                    event("error_cooldown_over", "Restart limit cooldown is over; the next request starts the model again")
                if st.proc and st.proc.poll() is not None and not st.expected_exit:
                    st.last_error = f"llama-server crashed (exit {st.proc.returncode})"
                    log(st.last_error)
                    event("server_crashed", st.last_error)
                    st.proc = None
                    st.model_id, st.vision = "", False
                    st.status = "ERROR" if common.note_crash(st, G) else "UNLOADED"

                common.compensate_wall_clock(st, time.time(), time.monotonic())
                idle = time.time() - st.last_request
                if st.status == "READY" and st.busy == 0 and idle > G["idleUnloadSeconds"]:
                    stop_llama_locked(f"idle for {int(idle)}s", save=True)

                ram = mem_available_gb()
                if st.status == "READY":
                    st.pressure_sustain = st.pressure_sustain + 1 if ram < G["ramFreeMinGB"] else 0
                    if st.pressure_sustain >= G["memoryPressureSustainPolls"] and st.busy == 0:
                        log(f"MEMORY PRESSURE ({ram:.1f} GB available) - unloading model")
                        stop_llama_locked("memory pressure", save=True)
                        st.memory_pressure = True
                        st.status = "MEMORY_PRESSURE"
                elif st.memory_pressure and ram >= G["ramFreeMinGBToLoad"]:
                    st.memory_pressure = False
                    if st.status == "MEMORY_PRESSURE":
                        st.status = "UNLOADED"       # ERROR and the others are left to their own ways out
                    log(f"Memory pressure cleared ({ram:.1f} GB available)")
        except Exception as e:
            log(f"Monitor error: {e}")


def status_snapshot():
    return common.snapshot_without_waiting(st, _status_snapshot_locked)


def _status_snapshot_locked():
    return {
        "status": st.status, "model": st.model_id, "vision": st.vision, "override": st.override,
        "loading_seconds": round(time.time() - st.loading_since, 1) if st.loading_since else None,
        "memory_pressure": st.memory_pressure, "multimedia_lock": st.multimedia_lock,
        "busy_requests": st.busy, "crash_count": st.crash_count, "last_error": st.last_error,
        "event_counters": dict(st.event_counters),
        "ram_available_gb": round(mem_available_gb(), 2),
        "vram_free_gb": None if st.loading_since else round(vram_free_gb(), 2),     # nvidia-smi can take seconds while the driver loads a model
        "idle_seconds": int(time.time() - st.last_request),
        "idle_unload_after_seconds": G["idleUnloadSeconds"],
        "llama_pid": st.proc.pid if st.proc and st.proc.poll() is None else None,
        "default_model": M["defaultModel"], "events": list(st.events), "slot_dir": str(SLOT_DIR),
    }


PASSIVE_METHODS = ("GET", "HEAD")


def gateway_key():
    return G.get("gatewayApiKey") or G.get("monitorApiKey") or G.get("apiKey") or M.get("apiKey") or None


def web_policy():
    return WebPolicy(G, G.get("gatewayHost", "127.0.0.1"), gateway_key())


def model_is_ready():
    """Is a model running that a read-only request can be passed to? Never starts one (a health probe, a model list, a favicon
    must not load 20 GB onto the GPU). Such requests are not counted as busy: they must not block an unload or a model switch,
    and one that loses its server meanwhile just gets a clean 502."""
    return bool(st.proc and st.proc.poll() is None and st.status == "READY" and st.override != "OFF")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def send_json(self, code, obj, retry_after=None):
        request_id = getattr(self, "_monitor_request_id", None)
        if request_id is not None:
            _monitor_call("set_status", request_id, code)
            if code >= 400:
                _monitor_finish(self, status=code, error=True)
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if retry_after is not None:
            self.send_header("Retry-After", str(retry_after))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_api_error(self, code, message, kind="api_error", error_code=None, retry_after=None, **extra):
        """An error in the shape OpenAI-style clients read: they show `error.message` to the user."""
        return self.send_json(code, error_body(message, kind, error_code, **extra), retry_after)

    def guardian_endpoint(self, path):
        busy_reply = {"error": "The gateway is loading a model; retry in a moment"}
        if path == "/guardian/status":
            self.send_json(200, status_snapshot())
        elif path == "/guardian/ai-on":
            try:
                OVERRIDE_FLAG.write_text("ON")
            except OSError as e:
                return self.send_json(503, {"error": f"Cannot persist the switch: {e}"})
            st.override = "AUTO"
            with common.try_lock(st.lock, 5) as got:
                if got and st.status == "ERROR":
                    st.status = "UNLOADED"
                    common.clear_error_state(st)
            self.send_json(200, {"ok": True, "override": "AUTO"})
        elif path == "/guardian/ai-off":
            try:
                OVERRIDE_FLAG.write_text("OFF")
            except OSError as e:
                return self.send_json(503, {"error": f"Cannot persist the switch, so nothing was changed: {e}"})
            st.override = "OFF"          # a model load in progress notices this and stops
            with common.try_lock(st.lock, 30) as got:
                if not got:
                    return self.send_json(503, {"error": "AI is off, but the model could not be stopped yet (a load is still running)"})
                stop_llama_locked("manual ai-off", save=True)
                if st.proc and st.proc.poll() is None:
                    return self.send_json(503, {"error": "AI is off, but the server could not be stopped"})
            self.send_json(200, {"ok": True, "override": "OFF"})
        elif path == "/guardian/unload":
            force = "force=1" in getattr(self, "_query", "").split("&")
            end = time.monotonic() + 2           # an answer that has just been delivered may still be closing its books
            while st.busy and not force and time.monotonic() < end:
                time.sleep(0.05)
            with common.try_lock(st.lock, 5) as got:
                if not got:
                    return self.send_json(503, busy_reply)
                if st.busy and not force:
                    return self.send_json(409, {"error": "Response in progress; unload would cut it (add ?force=1 to do it anyway)",
                                                "busy_requests": st.busy})
                stop_llama_locked("manual unload", save=True)
            self.send_json(200, {"ok": True})
        elif path == "/guardian/multimedia-lock":
            st.multimedia_lock = True    # no new load starts, and one in progress stops
            with common.try_lock(st.lock, 30) as got:
                if not got:
                    return self.send_json(503, {"error": "GPU is reserved, but a model load is still running; retry"})
                stop_llama_locked("multimedia pipeline acquiring GPU", save=True)
            self.send_json(200, {"ok": True})
        elif path == "/guardian/multimedia-unlock":
            st.multimedia_lock = False
            self.send_json(200, {"ok": True})
        else:
            self.send_json(404, {"error": f"unknown guardian endpoint {path}"})

    def read_body(self):
        try:
            length = content_length(self.headers, int(G["maxRequestBytes"]))
        except BadFraming as error:
            self.close_connection = True
            self.send_json(error.status, {"error": error.message})
            return None
        return self.rfile.read(length) if length else b""

    def handle_any(self):
        path = self.path.split("?", 1)[0]
        self._query = self.path.partition("?")[2]
        if path.startswith("/monitor") and path not in ("/monitor", "/monitor/", "/monitor/api/state", "/monitor/api/requests", "/monitor/api/stream"):
            return self.send_json(404, {"error": "Unknown monitor endpoint"})
        if path in ("/monitor", "/monitor/", "/monitor/api/state", "/monitor/api/requests", "/monitor/api/stream"):
            return monitor_route(self, path, _monitor_state_payload, MONITOR.requests,
                                 authorized=_monitor_authorized(self), monitor_instance=MONITOR)
        body = self.read_body()
        if body is None:
            return None

        if path.startswith("/guardian/"):
            return self.guardian_endpoint(path)

        if self.command in PASSIVE_METHODS:
            return self.handle_passive(path)
        return self.handle_generation(path, body)

    def handle_passive(self, path):
        """GET and HEAD never start a model and never count as activity: a browser tab, a health check or a client listing models
        must not load 20 GB onto the GPU or keep it from unloading when idle."""
        if path in ("/health", "/v1/health"):
            return self.send_json(200, {"status": "ok", "model_loaded": st.status == "READY", "gateway_status": st.status})
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            return self.end_headers()
        if path in ("/v1/models", "/models"):
            # Always the configured ids, loaded or not: the list must not change (llama-server's own list names the model file).
            return self.send_json(200, {"object": "list", "data": [{"id": mid, "object": "model", "owned_by": "shura"} for mid in M["models"]]})
        if model_is_ready():
            return self.proxy(b"")
        return self.send_api_error(503, "The model is not loaded. Send a generation request to load it.", "service_unavailable",
                                   "SHURA_MODEL_UNLOADED")

    def handle_generation(self, path, body):
        if path.startswith("/v1/") or path in ("/completion", "/infill", "/completions"):
            _monitor_begin(self, path, body)

        st.last_request = time.time()
        if st.override == "OFF":
            return self.send_api_error(503, "AI is manually disabled (ai-off). POST /guardian/ai-on to re-enable.", "service_unavailable", "SHURA_AI_OFF")
        if st.multimedia_lock:
            return self.send_api_error(503, "GPU is busy with an audio/video job; retry when it finishes.", "service_unavailable", "SHURA_GPU_BUSY")
        if st.memory_pressure:
            return self.send_api_error(503, "AI paused due to system memory pressure; resumes automatically.", "service_unavailable", "SHURA_MEMORY_PRESSURE")

        vision, model_id, body = inspect_body(body, M["models"], M["defaultModel"])
        if vision and not model_config(model_id, True).get("mmprojPath"):
            vision = False         # no projector configured: the request is served as text, and the server must not restart for it
        if body:
            _monitor_call("set_request_model", getattr(self, "_monitor_request_id", None), model_id)

        settle_dead_backend()
        # The request is counted before the model is started or switched (as the full gateway does): a switch for another model must not
        # stop a server that is answering someone else, and an unload cannot slip in between the start and the first byte.
        with st.lock:
            st.busy += 1
            try:
                err = start_llama(model_id, vision)
            except BaseException:
                st.busy -= 1
                raise
            if err:
                st.busy -= 1
        if err:
            return self.send_api_error(503, f"Could not start model '{model_id}': {err}", "service_unavailable", "SHURA_MODEL_UNAVAILABLE", reason=err)

        st.slot_dirty = True
        try:
            self.proxy(body)
        finally:
            with st.lock:
                st.busy -= 1
            st.last_request = time.time()
            _monitor_finish(self, status=500, error=True)
        return None

    def observe(self, kind, value):
        request_id = getattr(self, "_monitor_request_id", None)
        if kind == "status":
            _monitor_call("set_status", request_id, value)
        elif kind == "chunk":
            _monitor_call("observe_sse", request_id, value)
        elif kind == "json":
            _monitor_call("observe_json", request_id, value)

    def stalled(self):
        proc = st.proc
        if proc is not None:
            threading.Thread(target=declare_hung, args=(proc, f"no data for {G['upstreamIdleSeconds']} s in the middle of an answer"),
                             daemon=True, name="shura-hung-server").start()

    def proxy(self, body):
        result = forward_request(self, body, G["llamaHost"], G["llamaPort"], G, observe=self.observe, on_stall=self.stalled)
        _monitor_finish(self, status=result["status"], error=result["error"])

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = handle_any


def sweep_slot_leftovers():
    """A save interrupted by a kill leaves a half-written file behind; nothing else would ever remove it."""
    try:
        for pattern in ("*.pending", ".slot-*", "*.json.tmp"):
            for leftover in SLOT_DIR.glob(pattern):
                if leftover.is_file() and not leftover.is_symlink():
                    leftover.unlink(missing_ok=True)
    except OSError:
        pass


def shutdown(signum, _frame):
    log(f"Signal {signum}: saving active slot and stopping llama-server")
    st.shutting_down.set()            # no new load starts, and one in progress gives up within a second
    got = st.lock.acquire(timeout=G["shutdownLockSeconds"])
    try:
        if got:
            stop_llama_locked("gateway shutdown", save=True)
        else:
            common.kill_server(st.proc)      # a loading thread still holds the lock: the child must not outlive the gateway
    finally:
        if got:
            st.lock.release()
    sys.exit(0)


def main():
    os.umask(0o077)                   # slot files (a session's KV state) and state files are for this user only
    for directory in (SLOT_DIR, STATE_DIR):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            if directory.is_dir() and not directory.is_symlink() and directory.stat().st_uid == os.getuid():
                os.chmod(directory, 0o700)
        except OSError as error:       # a read-only or foreign-format mount: keep going with what the folder allows
            log(f"Could not restrict {directory} to this user: {error}")
    sweep_slot_leftovers()
    recover_orphans_at_start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    threading.Thread(target=monitor, daemon=True).start()
    server = BoundedServer((G["gatewayHost"], G["gatewayPort"]), guarded_handler(Handler, web_policy, G["clientIdleSeconds"]), G["maxConnections"])
    log(f"Gateway listening on http://{G['gatewayHost']}:{G['gatewayPort']} -> llama-server :{G['llamaPort']} "
        f"(config {CFG_DIR}, override={st.override}, idle unload {G['idleUnloadSeconds']}s)")
    server.serve_forever()


if __name__ == "__main__":
    main()
