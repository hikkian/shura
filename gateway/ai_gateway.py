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
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import http.client
import importlib.util
import json
import math
import os
import re
import resource
import shutil
import signal
import secrets
import socket
import stat
import subprocess
import sys
import tempfile
import sqlite3
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_nvml import NVML
from gpu_backend import open_device
from response_recovery import forward as forward_recovery, wait_for_process_exit
from checkpoint_access import private_checkpoint_handler
from crash_report import write_crash_report
from game_mode import DEFAULTS as GAME_DEFAULTS, GamePolicy, GameSignals
from polite_model import DEFAULTS as POLITE_DEFAULTS, Controller as PoliteController
from slot_gears import DEFAULTS as GEAR_DEFAULTS, Budget, Ratchet, RollbackFailed, apply_slots as apply_slot_transaction
import cache_reclaim
from vram_valve import DEFAULTS as VALVE_DEFAULTS, SlotCostEstimator, Valve
import user_modes
import lang_guard
import script_guard
import prefill_progress
import layout_estimate
import ram_valve
import ram_budget
import prefill_yield
import vram_budget
import session_cache
import desk_need
import gpu_reset
import desk_runtime
import desk_sources
from monitor import Monitor, _read_zram, gateway_snapshot, monitor_authorized, monitor_route, read_expert_cache_metrics
from monitor_sources import sample_gpu
from web_guard import BadFraming, BoundedServer, DEFAULTS as WEB_DEFAULTS, Policy as WebPolicy, content_length, guarded_handler
from proxy_core import DEFAULTS as PROXY_DEFAULTS, error_body, forward_request, inspect_body
import service_requests
import context_fit
import gateway_common as common
import lifelong_request
import lifelong_store   # LL1: the knowledge store (used only when memoryEnabled is true)
import experience_ledger
import switch_counter      # SHADOW1: switch counters for shadow candidates (off unless switchCounters is set)

CFG_DIR = Path(os.environ.get("AI_GATEWAY_CONFIG", Path(__file__).resolve().parent.parent / "config"))
STATE_DIR = Path(os.path.expanduser(os.environ.get("AI_GATEWAY_STATE", "~/.local/state/ai-gateway")))


PATH_KEYS = {"exePath", "modelPath", "mmprojPath", "moeCacheProfile", "slotSaveDir", "vramPressureSlotDir"}
PRESSURE_GATE_OVERRIDE = ContextVar("shura_pressure_gate_override", default=True)


@contextmanager
def memory_pressure_gate(enabled):
    token = PRESSURE_GATE_OVERRIDE.set(bool(enabled))
    try:
        yield
    finally:
        PRESSURE_GATE_OVERRIDE.reset(token)


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
G_RAW = dict(G)
M = load_config("model-launch.json")
M["models"] = {mid: {k: expand(k, v) for k, v in mc.items()} for mid, mc in M["models"].items()}
LIFECYCLE_DEFAULTS = common.LIFECYCLE_DEFAULTS
# VRAMCOR2: the lite layout leaves again only after the short one would keep the card reserve plus this margin for this long
LITE_DEFAULTS = {"vramLiteReturnSeconds": 120, "vramLiteReturnMarginMiB": 600}
# MEMPROF1: a planned restart of the server when its heap + /dev/zero grow over serverMemRestartGiB (0 = off). The session goes to RAM
# first, the next request loads the server again and restores it. A temporary measure for the HIP runtime leak (REPORT-MEMPROF1).
SERVER_MEM_DEFAULTS = {"serverMemRestartGiB": 0, "serverMemRestartIdleSeconds": 20, "serverMemCheckSeconds": 30}
# RAMFIX1: the gateway's own files in RAM (tmpfs) never block a load. shmSpill false = as before. slotSpillDir: the disk folder for
# saved sessions moved out of RAM (null = <state dir>/slots-spill). shmSpillMinMiB: smaller saved sessions always stay in RAM.
# SSD writes are kept to the minimum: a session goes to the disk only when re-reading its prompt would take longer than
# spillMinReprefillSeconds at the measured prefill speed (spillPrefillTps until a long prompt was measured), otherwise it is dropped.
# ramDeskFloorGB: MemAvailable the desktop keeps while the model is loaded (own RAM files go first to keep it); a large new prompt at
# the valve's tight level waits inside the gateway (up to ramValveDeferWaitSeconds) only while the floor would not hold.
SPILL_DEFAULTS = {"shmSpill": True, "slotSpillDir": None, "shmSpillMinMiB": 256, "spillMinReprefillSeconds": 60,
                  "spillPrefillTps": 1200.0, "ramDeskFloorGB": 4.0, "ramValveDeferWaitSeconds": 60}
G = {**SPILL_DEFAULTS,"crashRecovery": False, "vramRetry": False, "vramPolicy": "legacy", "ramPolicy": "off", "ramDeskReserveMiB": None, "ramWaitSeconds": 20, "ramHardMinMiB": ram_budget.RAM_HARD_MIN_MIB, "ramReclaimCache": True, **LITE_DEFAULTS, **SERVER_MEM_DEFAULTS, **POLITE_DEFAULTS, **GEAR_DEFAULTS, **VALVE_DEFAULTS, **GAME_DEFAULTS, **desk_need.DEFAULTS, **gpu_reset.DEFAULTS, **WEB_DEFAULTS, **PROXY_DEFAULTS, **LIFECYCLE_DEFAULTS,
     **ram_valve.DEFAULTS, **session_cache.DEFAULTS, **prefill_yield.DEFAULTS, **prefill_progress.DEFAULTS,
     "vramGuard": False, "vramIdleSeconds": 90,
     "vramEmergencyMiB": 60, "vramEmergencySeconds": 10,
     "vramFreeCriticalMiB": 120, "vramPressureSeconds": 5,
     "vramReloadWaitSeconds": 30, "vramPressureSlotReserveMiB": 512,
     "memoryAdmissionRetryAfterSeconds": 30,
     "layoutSwitchSaveMinTokens": 2000,
     "layoutSwitchRetrySeconds": 60,   # A2PORT1: after a switch to the long layout failed for a reason that is not memory, the next try waits this long
     "vramPressureSlotDir": "/dev/shm",
     "layoutFastStop": False,   # LAYOUT1c: after the slot is saved, a layout switch kills the old server instead of waiting ~5 s for its exit
     "effortBudgets": False,    # EFFORT1: send each level's reasoning budget to the server (needs the budget sampler of shura.cpp)
     "memoryEnabled": False,    # LL1: the knowledge store behind /guardian/memory/* (gateway/lifelong_store.py); off = the product as before
     "memoryPath": None,        # LL1: the SQLite file (default ~/.local/share/shura/lifelong.sqlite)
     "memoryStem": False,       # LL2b: word-form tolerant memory search (cut word endings, prefix match); off = the LL1 search
     "memoryIndexBudgetTokens": None,   # LL2b: the memory index (MEMORY.md) kept inside this many tokens; None = the LL1 index
     "lifelongCapture": False,   # LL6a: explicit client signals become proposed records only
     "experienceLedger": False, # LEDGER1: task/outcome ledger, opt-in and local only
     "memoryInject": False,     # LL6a: append deterministic memory notes to the user's last message
     "switchCounters": False,   # SHADOW1: count where switches (memory note, shadow candidates) fire, per conversation; RAM, ledger with capture
     "shadowCandidates": False, # SHADOW1: {"think-brake": {"soft", "ceiling", "margin"}} or true: server shadow fields on chat requests
     **G}


_MEMORY_STORE = None
_MEMORY_LOCK = threading.Lock()
_MEMORY_INJECT = lifelong_request.FrozenMemory()
_LIFELONG_CONFIRM_TOKEN = secrets.token_urlsafe(32)
_EXPERIENCE_LEDGER = None
_EXPERIENCE_LOCK = threading.Lock()


def experience_ledger_store():
    global _EXPERIENCE_LEDGER
    with _EXPERIENCE_LOCK:
        if _EXPERIENCE_LEDGER is None:
            _EXPERIENCE_LEDGER = experience_ledger.Ledger()
    return _EXPERIENCE_LEDGER


def inject_memory(body):
    if G.get("memoryEnabled") is not True or G.get("memoryInject") is not True or not body:
        return body
    try:
        return _MEMORY_INJECT.apply(body, getattr(REQUEST_CTX, "session", ""), memory_store,
                                    G.get("memoryInjectMinScore", lifelong_request.STEM_THRESHOLD),
                                    report=switch_counter.memory_report(REQUEST_CTX, G))
    except (sqlite3.Error, lifelong_store.StoreError, ValueError, TypeError):
        return body


def memory_store():
    """LL1: one store per gateway process, opened on first use (so a gateway with memoryEnabled false never touches the file). The
    legacy ~/.local/share/shuracode/memory/*.md files are imported once (DESIGN §3.1) and MEMORY.md keeps being regenerated there."""
    global _MEMORY_STORE
    with _MEMORY_LOCK:                      # two first requests on the threaded server must not open two stores
        if _MEMORY_STORE is None:
            path = G.get("memoryPath") or lifelong_store.DEFAULT_DB
            store = lifelong_store.Store(Path(path).expanduser(), stem=G.get("memoryStem") is True,
                                         index_budget_tokens=G.get("memoryIndexBudgetTokens") if type(G.get("memoryIndexBudgetTokens")) is int else None)
            try:
                store.import_legacy()
            except Exception as error:   # the legacy folder is optional and may be unreadable: never a reason to refuse the store
                log.warning("memory: legacy import skipped: %s", error)
            _MEMORY_STORE = store
    return _MEMORY_STORE


def validate_load_pressure_config(config):
    """Validate opt-in memory admission thresholds while the gateway is starting."""
    for key in ("zramMaxPctToLoad", "psiMemoryFullMaxToLoad"):
        if key not in config:
            continue  # missing means the pre-M13 behavior
        value = config[key]
        if value is None or value is False:
            continue  # JSON null/false are explicit disabled values
        finite = False
        if type(value) in (int, float):
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
        if not finite or value > 100:
            raise ValueError(f"Invalid guardian config: {key} must be a finite percentage <= 100; use 0 or a negative value to disable")
    retry_after = config.get("memoryAdmissionRetryAfterSeconds", 30)
    if type(retry_after) is not int or not 1 <= retry_after <= 3600:
        raise ValueError("Invalid guardian config: memoryAdmissionRetryAfterSeconds must be an integer from 1 to 3600")
    switch_retry = config.get("layoutSwitchRetrySeconds", 60)
    if type(switch_retry) is not int or not 1 <= switch_retry <= 3600:
        raise ValueError("Invalid guardian config: layoutSwitchRetrySeconds must be an integer from 1 to 3600")
    switch_min_tokens = config.get("layoutSwitchSaveMinTokens", 2000)
    if type(switch_min_tokens) is not int or switch_min_tokens < 0:
        raise ValueError("Invalid guardian config: layoutSwitchSaveMinTokens must be a non-negative integer")


def pressure_limit(config, key):
    value = config.get(key, 0)
    return 0 if value is None or value is False else value


def legacy_gateway_selected(config, game_enabled):
    """Keep the current gateway split in one testable place without changing its conditions."""
    reserve_value = config.get("vramCardReserveMiB")
    valve_value = config.get("vramValve")
    reserve_path_requested = ("vramCardReserveMiB" in config and reserve_value is not None
                              and reserve_value is not False
                              and not (type(reserve_value) is int and reserve_value == 0))
    valve_path_requested = ("vramValve" in config and valve_value is not False
                            and valve_value is not None and valve_value != 0)
    return (config.get("vramGuard") is not True
            and int(config.get("slotSaveCheckpoints", 0)) == 0
            and not game_enabled and not config.get("politeModel")
            and not config.get("slotGears") and not config.get("crashRecovery")
            and config.get("memoryEnabled") is not True   # LL6a: memory endpoints and injection live in the full gateway
            and not config.get("effortBudgets")   # EFFORT1: the budgets live in the new gateway; asking for them selects it
            and config.get("switchCounters") is not True   # SHADOW1: so do the switch counters
            and config.get("ramPolicy", "off") != "honest"   # RAMLOGIC2: so does the honest RAM policy with its ladder
            and not reserve_path_requested and not valve_path_requested
            and config.get("vramPolicy", "legacy") != "governor")   # VRAMLOGIC1h: the governor runs only on the new entrypoint


def configured_pressure_options(config):
    active, invalid = [], []
    for key in ("zramMaxPctToLoad", "psiMemoryFullMaxToLoad"):
        if key not in config:
            continue
        value = config[key]
        if value is None or value is False:
            continue
        finite = False
        if type(value) in (int, float):
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
        if not finite or value > 100:
            invalid.append(key)
        elif value > 0:
            active.append(key)
    retry_key = "memoryAdmissionRetryAfterSeconds"
    if retry_key in config:
        retry_after = config[retry_key]
        if (type(retry_after) is not int or not 1 <= retry_after <= 3600):
            invalid.append(retry_key)
        elif retry_after != 30:
            active.append(retry_key)
    return active, invalid


def misplaced_pressure_options(model_config):
    misplaced = []
    for key in ("zramMaxPctToLoad", "psiMemoryFullMaxToLoad", "memoryAdmissionRetryAfterSeconds"):
        if key in model_config:
            misplaced.append(key)
        for model_id, settings in model_config.get("models", {}).items():
            if key in settings:
                misplaced.append(f"{key} ({model_id})")
    return misplaced


def warn_misplaced_pressure_options():
    misplaced = misplaced_pressure_options(M)
    if misplaced:
        names = ", ".join(misplaced)
        event("memory_admission_misplaced_config",
              f"{names} in model-launch.json are ignored; memory admission keys only work in guardian.json")


def warn_legacy_pressure_options():
    active, invalid = configured_pressure_options(G_RAW)
    ignored = active + invalid
    if not ignored:
        return
    details = []
    if active:
        details.append(f"enabled keys: {', '.join(active)}")
    if invalid:
        details.append(f"invalid keys: {', '.join(invalid)}")
    event("memory_admission_legacy_ignored",
          f"Memory admission { '; '.join(details) } are ignored by the legacy gateway; enable vramGuard or slotSaveCheckpoints in guardian.json to use the new gateway")
SLOT_DIR = Path(G["slotSaveDir"])
RAM_SLOT_DIR = None
OVERRIDE_FLAG = STATE_DIR / "override.flag"
TERSE_FLAG = STATE_DIR / "terse.flag"      # "ON" / "OFF"; absent = whatever the model config says
CONTEXT_MODE_FLAG = STATE_DIR / "context-mode.flag"   # "extended"; absent = auto (SWITCHUI1)
THINK_FLAG = STATE_DIR / "think.flag"                 # off / low / medium / high; absent = the template default (THINK2)

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
        if getattr(_NVML_DEVICE, "needs_reinit", False) is True:
            try:  # the driver was reset or reloaded: the old handle can never recover
                _NVML_DEVICE.close()
            except (OSError, RuntimeError):
                pass
            _NVML_DEVICE = None
        if _NVML_DEVICE is None:
            _NVML_DEVICE = open_device(NVML)
        return _NVML_DEVICE

def close_nvml():
    global _NVML_DEVICE
    with _NVML_LOCK:
        if _NVML_DEVICE is not None:
            _NVML_DEVICE.close()
            _NVML_DEVICE = None

atexit.register(close_nvml)

TEST_VRAM_FREE_FILE = os.environ.get("AI_GATEWAY_TEST_VRAM_FREE_FILE", "")   # tests only: free VRAM in MiB read from this file


def vram_free_gb():
    if TEST_VRAM_FREE_FILE:
        try:
            return int(Path(TEST_VRAM_FREE_FILE).read_text().strip()) / 1024
        except (OSError, ValueError):
            return -1.0
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


def read_terse_flag():
    try:
        return {"ON": True, "OFF": False}.get(TERSE_FLAG.read_text().strip())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        log(f"State flag ignored: terse.flag ({type(error).__name__}: {error})")
        return None


class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.status = "UNLOADED"  # UNLOADED LOADING READY MEMORY_PRESSURE ERROR
        self.model_id = ""
        self.vision = False
        self.layout = "short"  # LAYOUT1: "short" = the model's own profile, "long" = longContextProfile
        self.proc = None
        self.expected_exit = False
        try:
            override_flag = OVERRIDE_FLAG.read_text().strip()
        except FileNotFoundError:
            override_flag = ""
        except (OSError, ValueError) as error:
            log(f"State flag ignored: override.flag ({type(error).__name__}: {error})")
            override_flag = ""
        self.override = "OFF" if override_flag == "OFF" else "AUTO"
        self.terse = read_terse_flag()
        self.context_mode = user_modes.read_flag(CONTEXT_MODE_FLAG, user_modes.CONTEXT_MODES, log) or "auto"
        self.think = user_modes.read_flag(THINK_FLAG, user_modes.THINK_LEVELS, log)
        self.memory_pressure = False
        self.pressure_sustain = 0
        self.multimedia_lock = False
        self.crash_count = 0
        self.crash_reported = set()
        self.gpu_reset_pids = {}              # GPURESET1: pid of a server killed by a GPU reset -> what the kernel log said
        self.gpu_resets = gpu_reset.ResetLedger(G)
        self.proc_wall_start = 0.0
        self.last_error = ""
        self.last_request = time.time()
        self.busy = 0
        self.busy_lock = threading.Lock()
        self.crash_times = deque()           # monotonic times of recent server failures (see note_crash)
        self.error_since = None              # when the crash limit put the gateway into ERROR (for the cooldown)
        self.loading_since = None
        self.slot_dirty = True               # something was generated since the slot on disk was written or restored
        self.health_failures = 0
        self.last_health_probe = 0.0
        self.shutting_down = threading.Event()
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
        self.events = deque(maxlen=30)
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
        self.slot_session = None          # the session whose snapshot is in the disk slot file (session cache)
        self.checkpoint_checked = False
        self.next_save_attempt = 0.0
        self.mem_checked_at = -math.inf   # MEMPROF1: monotonic time of the last server-memory reading
        self.mem_baseline = (None, 0.0)   # FINAL2: (server pid, heap + /dev/zero GiB at the first reading of that process)
        self.monitor_thread = None
        self.backend_blocked = ""
        self.ram_io_failed = False
        self.load_pressure_telemetry_unavailable = False
        self.load_pressure_refusal_active = {"zram": False, "psi": False}
        self.load_pressure_refusal_episodes = {"zram": 0, "psi": 0}


st = State()
MONITOR = Monitor()
GAME = GamePolicy(G)
GAME_SIGNALS = GameSignals(G)
GAME_REQUESTS = {}


def event(reason, detail, fallback=False):
    """Transition-only logs and in-memory counters; no new persistent event store."""
    with st.lock:
        previous = st.last_event
        st.last_event = {"reason": str(reason)[:100], "detail": str(detail)[:240], "at": time.time()}
        st.events.append(dict(st.last_event))
        st.event_counters[reason] = st.event_counters.get(reason, 0) + 1
        if fallback:
            st.guard_fallback = detail
    try:
        if not previous or (previous["reason"], previous["detail"]) != (reason, detail):
            log(f"Guardian {reason}: {detail}")
    except OSError:
        pass  # A closed journal pipe must not kill the guardian thread.


def crash_snapshot(operation):
    model = st.model_id
    mc = model_config(model, st.vision) if model in M["models"] else {}
    try:
        with open("/proc/pressure/memory") as stream:
            pressure = next((line.strip() for line in stream if line.startswith("full ")), "")
    except OSError:
        pressure = "unavailable"
    return {
        "vram_free_mib": st.vram_free_mib,
        "ram_available_gb": round(mem_available_gb(), 3),
        "memory_psi_full": pressure,
        "gateway_status": st.status,
        "model": model or None,
        "vision": st.vision,
        "mode": "OFF" if st.override == "OFF" else "AUTO",
        "game_mode": GAME.reason if GAME.enabled and GAME.active else None,
        "slots": mc.get("moeCacheSlots"),
        "busy_requests": st.busy,
        "operation": operation,
        "last_gateway_event": st.last_event,
        "checkpoint_saved": st.ram_checkpoint is not None,
    }


def record_server_crash(proc, operation):
    """Persist bounded diagnostics once per child PID; failure must not affect recovery."""
    if proc is None or proc.pid in st.crash_reported:
        return None
    st.crash_reported.add(proc.pid)
    if len(st.crash_reported) > 32:
        st.crash_reported.pop()
    detect_gpu_reset(proc)
    try:
        return write_crash_report(STATE_DIR, proc.args, proc.poll(), operation,
                                  lambda: crash_snapshot(operation))
    except Exception as error:  # noqa: BLE001 - crash reporting is best effort.
        log(f"Crash report failed: {type(error).__name__}")
        return None


def detect_gpu_reset(proc):
    """GPURESET1: after an unexpected exit (not the governor's 75) look for a GPU reset in the kernel log since the server started."""
    code = proc.poll()
    if code in (None, 0, 75) or proc.pid in st.gpu_reset_pids:
        return None
    info = gpu_reset.check_after_exit(gpu_reset.run_command, st.proc_wall_start or time.time() - 900, time.time(), proc.pid)
    if info is None:
        return None
    st.gpu_reset_pids[proc.pid] = info
    if len(st.gpu_reset_pids) > 16:
        st.gpu_reset_pids.pop(next(iter(st.gpu_reset_pids)))
    event("gpu_reset", gpu_reset.label(code, info), fallback=True)
    return info


def killed_by_gpu_reset(proc):
    """gpuResetRetry: asked when an answer breaks before its first token. Did a GPU reset kill this server? Gives the process a
    moment to be reaped, then reads the kernel log (once per process)."""
    if proc is None:
        return False
    wait_for_process_exit(proc)
    return proc.pid in st.gpu_reset_pids or detect_gpu_reset(proc) is not None


def crash_text(proc):
    """last_error for an unexpected exit: a GPU reset says so (the model is not at fault), anything else is a crash."""
    info = st.gpu_reset_pids.get(proc.pid)
    return gpu_reset.label(proc.returncode, info) if info else f"llama-server crashed (exit {proc.returncode})"


def note_crash(now=None, proc=None):
    """Count a server failure towards the restart limit. A GPU reset counts in its own, higher limit (a card that keeps
    resetting stops the restarts too, but one reset is not a strike against the model)."""
    if proc is not None and proc.pid in st.gpu_reset_pids:
        return st.gpu_resets.note(st.gpu_reset_pids[proc.pid], time.monotonic() if now is None else now)
    return common.note_crash(st, G, now)


def clear_error_state():
    common.clear_error_state(st)


def reset_pressure():
    st.vram_pressure_since = st.vram_emergency_since = None
    st.vram_pressure = st.vram_emergency = st.pressure_notified = False


def title_answers_enabled():
    """Chat titles are answered by the gateway (REPORT-CACHE1.md); guardian.json can turn that off."""
    return G.get("titleFromTranscript", True) is not False


PAUSE_WAIT_SECONDS = 600        # PAUSEWAIT1: how long a paused answer waits for the memory the desk holds (pauseResumeWaitSeconds)
PAUSE_WAIT_POLL_SECONDS = 2.0   # PAUSEWAIT1: a fresh admission check this often while waiting (pauseResumePollSeconds)
PAUSE_WAIT = {"state": "idle", "model": "", "since": None, "reason": "", "free_mib": None, "waited_s": 0.0}


def resume_refusal(model_id, vision, layout):
    """PAUSEWAIT1: the admission a load of `layout` would pass right now: "" when it passes, else the refusal (the same check a
    load makes, so the waiting and the load agree on the number)."""
    return preload_check(model_config(model_id, vision, layout), pressure_gate=False, admit=(model_id, layout))


def wait_for_resume_memory(model_id, vision, layout, lease=None, heartbeat=None):
    """PAUSEWAIT1: after a governor pause the desk may still hold the memory the server gave back. Wait for the admission a load
    uses, sampling fresh VRAM, up to pauseResumeWaitSeconds. A cancelled client, a refusal that waiting cannot fix, or the end of
    the wait stops it at once. heartbeat() keeps the client's stream open (an SSE comment) while waiting. Returns "" when the
    resume may load, otherwise the reason."""
    wait_s = float(G.get("pauseResumeWaitSeconds", PAUSE_WAIT_SECONDS))
    poll = float(G.get("pauseResumePollSeconds", PAUSE_WAIT_POLL_SECONDS))
    started = time.monotonic()
    waiting = False
    while True:
        reason = resume_refusal(model_id, vision, layout)
        cancelled = lease is not None and lease["cancel"].is_set()
        if not reason or getattr(reason, "code", None) not in WAIT_FOR_VRAM_CODES or cancelled:
            break
        waited = time.monotonic() - started
        if waited >= wait_s:
            break
        if not waiting:
            waiting = True
            PAUSE_WAIT.update(state="waiting", model=model_id, since=time.time(), reason=str(reason), free_mib=st.vram_free_mib)
            event("vram_pause_resume_waiting", f"model={model_id} layout={layout}: {reason}; waiting up to {int(wait_s)} s for the memory")
        PAUSE_WAIT.update(reason=str(reason), free_mib=st.vram_free_mib, waited_s=round(waited, 1))
        if heartbeat is not None:
            heartbeat()
        time.sleep(poll)
    waited = round(time.monotonic() - started, 1)
    if not reason:
        PAUSE_WAIT.update(state="ready", waited_s=waited, reason="", free_mib=st.vram_free_mib)
        if waiting:
            event("vram_pause_resume_ready", f"model={model_id} layout={layout}: memory back after {waited} s; the answer resumes")
    elif lease is not None and lease["cancel"].is_set():
        PAUSE_WAIT.update(state="cancelled", waited_s=waited, reason=str(reason))
        event("vram_pause_resume_cancelled", f"model={model_id}: the client went away while the answer waited for memory")
    else:
        PAUSE_WAIT.update(state="expired" if waiting and getattr(reason, "code", None) in WAIT_FOR_VRAM_CODES else "refused",
                          waited_s=waited, reason=str(reason))
        if waiting:
            event("vram_pause_resume_expired", f"model={model_id} layout={layout}: {reason}; gave up after {waited} s", fallback=True)
    return reason


def pause_resume_for(proc, model_id, vision, lease, body):
    """VRAMLOGIC1i: the callback a relay uses when the governor paused an answer: restart the backend (no crash accounting),
    restore the checkpoint by name, and return the same request with `resume` (API.md). None = the answer cannot continue."""
    refusal = [""]

    def pause_recover(heartbeat=None):
        """VRAMLOGIC1i: restart after a deliberate governor pause. The exit is expected: no crash accounting."""
        refusal[0] = ""
        if (lease is not None and lease["cancel"].is_set()) or not wait_for_process_exit(proc):
            return False
        with st.lock:
            if st.busy != 1 or st.override == "OFF" or GAME.active or st.status == "ERROR":
                return False
            if st.proc is not None and st.proc is not proc:
                return False
            st.proc = None
            st.status = "UNLOADED"
        if desk_deepen("the server paused while the model slept lightly"):
            return False                  # MEMV2: the game needs more than the cache could give: park, do not come back now
        if not wait_fresh_vram(time.time()):
            return False
        with st.lock:
            if st.busy != 1 or st.override == "OFF" or GAME.active:
                return False
            if governor_on() and not vision and st.layout in LITE_OF:
                governor_mark_too_big(model_id, st.layout)     # the governor paused: this layout does not keep the reserve
            target = governor_layout(model_id, vision, st.layout)
        # PAUSEWAIT1: the desk may still hold the memory the server gave back; wait for the admission the load makes, do not refuse at once
        refusal[0] = wait_for_resume_memory(model_id, vision, target, lease, heartbeat)
        if refusal[0]:
            return False
        with st.lock:
            if st.busy != 1 or st.override == "OFF" or GAME.active:
                return False
            with memory_pressure_gate(False):
                error = start_llama(model_id, vision, layout=target)
        return not error

    def pause_resume(raw_error, tool_call_ids, heartbeat=None):
        """Check the tokens file, restart, restore the checkpoint by name, and return the same request with `resume` (API.md)."""
        PAUSE_STATE["resuming"] += 1        # RAMFIX1: its checkpoint is not a leftover while this runs
        try:
            return resume_paused(raw_error, tool_call_ids, heartbeat)
        finally:
            PAUSE_STATE["resuming"] -= 1

    def resume_paused(raw_error, tool_call_ids, heartbeat):
        try:
            frame = vram_budget.Budget().pause_frame({"error": raw_error})
            if frame is None or not frame["saved"]:
                raise vram_budget.BudgetError("the pause did not save a checkpoint")
            server_dir = SERVER_SLOT["dir"] or SLOT_DIR
            vram_budget.load_pause_tokens(server_dir / frame["tokens_file"], frame_n_past=frame["n_past"])
        except (vram_budget.BudgetError, KeyError, TypeError) as error:
            event("vram_pause_resume_refused", f"model={model_id}: {error}", fallback=True)
            return None
        if not pause_recover(heartbeat):
            why = refusal[0] or "the backend did not restart after the pause"
            event("vram_pause_resume_refused", f"model={model_id}: {why}", fallback=True)
            return None
        code, _res = llama_call("POST", "/slots/0?action=restore", {"filename": frame["checkpoint"]}, timeout=120)
        if code != 200:
            event("vram_pause_resume_refused", f"model={model_id}: slot restore failed (HTTP {code})", fallback=True)
            return None
        payload = json.loads(body)
        payload["resume"] = {"tokens_file": frame["tokens_file"]}
        if tool_call_ids:
            payload["resume"]["tool_call_ids"] = list(tool_call_ids)
        event("vram_pause_resumed", f"model={model_id}: checkpoint {frame['checkpoint']} restored; the answer continues")
        return json.dumps(payload).encode()
    return pause_resume


def legacy_main():
    """Run the preserved original entrypoint with no experimental lifecycle changes."""
    path = Path(__file__).with_name("_gateway_legacy.py")
    spec = importlib.util.spec_from_file_location("shura_legacy_gateway", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.Handler = private_checkpoint_handler(service_requests.with_title_answer(module.Handler, title_answers_enabled, fit_request))
    module.main()


RUNTIME_SLOTS = {}
PREFILL_YIELD = prefill_yield.YieldState()
GEARS = {}
VALVES = {}
SLOT_COSTS = SlotCostEstimator()
SLOT_COST_WARNED = set()
GEARS_UNAVAILABLE = set()   # models whose configuration cannot have gears; see gear_for()
GEAR_BUDGET = Budget()
GEAR_BUDGET_LOADED = False
STARTUP_VRAM_SELECTIONS = {}
POLITE = PoliteController({**G, "politeStateKey": str(STATE_DIR)})
WAKEUP = {"proc": None, "sample_at": None, "p99_ms": None, "started_at": None}


LONG_PROFILE_KEYS = {"ctxSize", "nCpuMoe", "ubatchSize", "batchSize", "specType", "specDraftNMax", "specDraftPMin",
                     "moeCacheSlots", "vramFreeMinGBToLoad", "extraEnv", "extraArgs"}
# VRAMCOR2: a model's liteProfile (the fallback layout under desktop VRAM pressure) may set only what frees VRAM at the same context size
LITE_PROFILE_KEYS = {"nCpuMoe", "ubatchSize", "batchSize", "moeCacheSlots", "vramFreeMinGBToLoad"}
# VRAMCOR4: the long layout has its own lite twin ("longlite" = longContextProfile + longLiteProfile). LITE_OF maps a layout to the layout that
# frees VRAM for it; LITE_BASE is the way back.
LITE_OF = {"short": "lite", "long": "longlite"}
LITE_BASE = {"lite": "short", "longlite": "long"}
LITE_PROFILE_NAME = {"short": "liteProfile", "long": "longLiteProfile"}
# Keys that do not change the serialized KV/draft tensors of a restored snapshot (PREFILL1d: -c, -ncmoe, ub and MTP off
# restore the same answer). Ignored by the checkpoint signature of models that have a longContextProfile.
LAYOUT_NEUTRAL_KEYS = ("ctxSize", "nCpuMoe", "ubatchSize", "batchSize", "specType", "specDraftNMax", "specDraftPMin")


def model_config(model_id, vision, layout="short"):
    mc = dict(M["models"][model_id])
    if model_id in RUNTIME_SLOTS and not vision:
        mc["moeCacheSlots"] = RUNTIME_SLOTS[model_id]
    if vision:
        mc.update(mc.get("visionOverrides", {}))
    elif layout == "long" and mc.get("longContextProfile"):
        profile = dict(mc["longContextProfile"])
        if "moeCacheSlots" not in profile and model_id in RUNTIME_SLOTS:
            profile["moeCacheSlots"] = RUNTIME_SLOTS[model_id]  # the valve owns the slot count unless the profile sets one
        mc.update(profile)
    elif layout == "lite" and mc.get("liteProfile"):
        mc.update(mc["liteProfile"])  # VRAMCOR2: same window as the short layout, more experts on the CPU, a small expert cache
    elif layout == "longlite" and mc.get("longContextProfile"):
        mc = model_config(model_id, vision, "long")
        mc.update(M["models"][model_id].get("longLiteProfile") or {})   # VRAMCOR4: the long window, more experts on the CPU, a small expert cache
    return mc


LONG_CHARS_PER_TOKEN = 2.5   # conservative for Russian text: the 28k test corpus measured 2.48 characters per token
LAYOUT_STATE = {"switched_at": -math.inf, "last_len": 0}
PROGRESS = prefill_progress.Tracker()   # PROG1: the chat's "reading the context" line
TOKENS = layout_estimate.TokenFeedback()  # FABLE1: the server's own token counts, for the calibrated layout estimate
REQUEST_CTX = threading.local()           # the session header of the request being handled: fit_to_context and the layout choice must agree


def token_key(model_id, session):
    """The key of the server's token feedback: the model and the client's session; None (characters only) without a valid session."""
    return f"{model_id}:{session}" if session_cache.valid_session_id(session) else None
LAST_STOP = {"at": -math.inf}           # monotonic time of the last server stop: the driver gives VRAM back about half a second later


def request_size(body):
    """(characters of the message text, max_tokens) of a chat body. Characters, not bytes: the JSON escaping of the client
    (\\uXXXX for every Cyrillic letter with ensure_ascii) and UTF-8 would otherwise overstate the size two- to sixfold."""
    try:
        obj = json.loads(body)
    except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
        return len(body), 0
    if not isinstance(obj, dict):
        return len(body), 0

    def walk(value):
        if isinstance(value, str):
            return len(value)
        if isinstance(value, list):
            return sum(walk(v) for v in value)
        if isinstance(value, dict):
            return sum(walk(v) for v in value.values())
        return 0
    chars = walk(obj.get("messages", []))
    max_tokens = obj.get("max_tokens") or obj.get("max_completion_tokens")
    return chars, (max_tokens if type(max_tokens) is int and max_tokens > 0 else 0)


def request_max_tokens(body):
    return request_size(body)[1]


CONTEXT_MIN_ANSWER_TOKENS = 2048     # context_fit: less room than this for the answer in a window means the request needs a bigger one


def pure_request_layout(model_id, vision, body, max_tokens, session=None):
    """The layout the request needs by size alone: long when its estimated tokens (+ the answer) exceed longContextSwitchTokens.

    longContextEstimate "calibrated" (FABLE1): the messages and the tools are counted at 3.2 characters per token for ASCII and
    2.5 for other text, the server's own token count of the session's last answer is the base for an appended request, and
    the answer counts as longContextAnswerReserveTokens (at most the client's max_tokens). Absent: the legacy estimate."""
    mc = M["models"][model_id]
    threshold = mc.get("longContextSwitchTokens", 0)
    if vision or not threshold or not mc.get("longContextProfile") or not body:
        return "short"
    if mc.get("longContextEstimate") == "calibrated":
        ascii_chars, other_chars = layout_estimate.count_chars(body)
        tokens, how = TOKENS.estimate(session, ascii_chars, other_chars)
        reserve = layout_estimate.answer_reserve(max_tokens, mc.get("longContextAnswerReserveTokens"))
        LAYOUT_STATE["estimate"] = {"tokens": int(tokens), "reserve": reserve, "how": how, "threshold": threshold}
        return "long" if tokens + reserve > threshold else "short"
    chars = request_size(body)[0]
    estimate = chars / LONG_CHARS_PER_TOKEN + max_tokens
    LAYOUT_STATE["estimate"] = {"tokens": int(estimate), "reserve": max_tokens, "how": "legacy", "threshold": threshold}
    return "long" if estimate > threshold else "short"


def choose_request_layout(model_id, vision, body, max_tokens, now, session=None):
    """Layout for one request. Back to short only for a new conversation (the body is not longer than the previous one)
    and after longContextHoldSeconds since the last switch: no ping-pong between the layouts."""
    if not body and st.model_id == model_id and st.vision == vision and st.proc is not None:
        return st.layout              # a request without a conversation (no size) never moves the loaded layout
    chars = request_size(body)[0] if body else 0
    if st.context_mode == "extended" and not vision and M["models"][model_id].get("longContextProfile"):
        if body:
            LAYOUT_STATE["last_len"] = chars
        return lite_twin(model_id, vision, "long", now)   # "Extended context": the long layout from the start, no switches
    wanted = pure_request_layout(model_id, vision, body, max_tokens, session)
    if wanted == "short" and getattr(REQUEST_CTX, "force_long", False) and not vision and M["models"][model_id].get("longContextProfile"):
        wanted = "long"                   # context_fit found the request too big for the short window (careful estimate)
    if wanted == "short" and st.layout in ("long", "longlite") and st.model_id == model_id and body:
        mc = M["models"][model_id]
        continuation = chars > LAYOUT_STATE["last_len"]
        held = now - LAYOUT_STATE["switched_at"] < mc.get("longContextHoldSeconds", 300)
        if continuation or held:
            wanted = "long"
    wanted = lite_twin(model_id, vision, wanted, now)
    if body:
        LAYOUT_STATE["last_len"] = chars
    return wanted


def lite_twin(model_id, vision, wanted, now):
    """VRAMCOR2/VRAMCOR4: the layout to run when `wanted` (short or long) is the size-based choice: its lite twin while the desktop holds
    VRAM, back to the normal one only after the hysteresis (no ping-pong)."""
    if wanted not in LITE_OF or vision or not lite_enabled(model_id, vision, wanted) or st.model_id != model_id or st.proc is None:
        return wanted
    if st.layout == LITE_OF[wanted]:
        relief = governor_lite_relief(model_id, vision, now, wanted) if governor_on() else lite_relief(model_id, vision, now, wanted)
        return wanted if relief else LITE_OF[wanted]
    pressure = governor_lite_pressure if governor_on() else lite_pressure
    if st.layout == wanted and pressure(model_id, vision, wanted):
        LITE_STATE.update(since=now, relief_since=None)
        return LITE_OF[wanted]
    return wanted


def validate_long_context_config(models):
    """LAYOUT1: a second layout for long contexts. Absent keys keep the old behaviour (switch off)."""
    for model_id, settings in models.items():
        profile = settings.get("longContextProfile")
        threshold = settings.get("longContextSwitchTokens", 0)
        hold = settings.get("longContextHoldSeconds", 300)
        if type(threshold) is not int or threshold < 0:
            raise ValueError(f"Invalid model config for {model_id}: longContextSwitchTokens must be a non-negative integer")
        if type(hold) is not int or not 0 <= hold <= 3600:
            raise ValueError(f"Invalid model config for {model_id}: longContextHoldSeconds must be an integer from 0 to 3600")
        if profile is None:
            if threshold:
                raise ValueError(f"Invalid model config for {model_id}: longContextSwitchTokens needs a longContextProfile")
            continue
        if not isinstance(profile, dict) or not set(profile) <= LONG_PROFILE_KEYS:
            raise ValueError(f"Invalid model config for {model_id}: longContextProfile may only set {sorted(LONG_PROFILE_KEYS)}")
        ctx_long = profile.get("ctxSize")
        if type(ctx_long) is not int or ctx_long <= int(settings.get("ctxSize", 0)):
            raise ValueError(f"Invalid model config for {model_id}: longContextProfile.ctxSize must be larger than ctxSize")
        if threshold and not 0 < threshold < int(settings.get("ctxSize", 0)):
            raise ValueError(f"Invalid model config for {model_id}: longContextSwitchTokens must be below ctxSize")
        estimate = settings.get("longContextEstimate")
        if estimate is not None and estimate not in ("legacy", "calibrated"):
            raise ValueError(f"Invalid model config for {model_id}: longContextEstimate must be \"legacy\" or \"calibrated\"")
        reserve = settings.get("longContextAnswerReserveTokens")
        if reserve is not None and (type(reserve) is not int or reserve <= 0 or reserve >= int(settings.get("ctxSize", 0))):
            raise ValueError(f"Invalid model config for {model_id}: longContextAnswerReserveTokens must be a positive integer below ctxSize")


def validate_lite_profile_config(models, config):
    """VRAMCOR2: liteProfile is optional per model; it needs the card reserve, sets only keys that free VRAM, and keeps the window.
    VRAMCOR4: longLiteProfile is the same for the long layout; it needs a longContextProfile and may not set fewer CPU experts than it."""
    for model_id, settings in models.items():
        for name, base in (("liteProfile", settings), ("longLiteProfile", settings.get("longContextProfile") or {})):
            profile = settings.get(name)
            if profile is None:
                continue
            if name == "longLiteProfile" and not isinstance(settings.get("longContextProfile"), dict):
                raise ValueError(f"Invalid model config for {model_id}: longLiteProfile needs a longContextProfile")
            if not isinstance(profile, dict) or not profile or not set(profile) <= LITE_PROFILE_KEYS:
                raise ValueError(f"Invalid model config for {model_id}: {name} may only set {sorted(LITE_PROFILE_KEYS)}")
            if type(config.get("vramCardReserveMiB")) is not int or config["vramCardReserveMiB"] <= 0:
                raise ValueError(f"Invalid model config for {model_id}: {name} needs a positive vramCardReserveMiB")
            for key in ("nCpuMoe", "ubatchSize", "batchSize", "moeCacheSlots"):
                if key in profile and (type(profile[key]) is not int or profile[key] < 0):
                    raise ValueError(f"Invalid model config for {model_id}: {name}.{key} must be a non-negative integer")
            if "nCpuMoe" in profile and int(base.get("nCpuMoe", settings.get("nCpuMoe", 0))) > profile["nCpuMoe"]:
                raise ValueError(f"Invalid model config for {model_id}: {name}.nCpuMoe must not be below nCpuMoe")
            if "vramFreeMinGBToLoad" in profile and (type(profile["vramFreeMinGBToLoad"]) not in (int, float)
                                                     or not 0 <= profile["vramFreeMinGBToLoad"] <= 1000):
                raise ValueError(f"Invalid model config for {model_id}: {name}.vramFreeMinGBToLoad must be a number from 0 to 1000")
    for key, low, high in (("vramLiteReturnSeconds", 1, 3600), ("vramLiteReturnMarginMiB", 0, 8192)):
        value = config.get(key, LITE_DEFAULTS[key])
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid guardian config: {key} must be an integer from {low} to {high}")


def validate_vram_card_reserve_config(config):
    """Validate the opt-in desktop VRAM reserve. Missing/zero keeps the old launch path."""
    if "vramCardReserveMiB" not in config:
        return
    value = config["vramCardReserveMiB"]
    if value is None or value is False or value == 0:
        return
    if type(value) is not int or not 1 <= value <= 1_000_000:
        raise ValueError("Invalid guardian config: vramCardReserveMiB must be an integer from 0 to 1000000")
    per_slot = config.get("slotGearMiBPerSlot", GEAR_DEFAULTS["slotGearMiBPerSlot"])
    if (type(per_slot) not in (int, float) or not math.isfinite(per_slot)
            or per_slot <= 0 or per_slot > 4096):
        raise ValueError("Invalid guardian config: slotGearMiBPerSlot must be finite and greater than 0")


def validate_prefill_yield_config(config):
    prefill_yield.validate(config)
    if config.get("prefillCacheYield") is True and config.get("slotGears"):
        raise ValueError("prefillCacheYield and slotGears cannot run together; both change the expert-cache slot count")


def validate_vram_valve_config(config):
    if "vramValve" not in config:
        return
    enabled = config["vramValve"]
    if type(enabled) is not bool:
        raise ValueError("Invalid guardian config: vramValve must be true or false")
    if not enabled:
        return
    if type(config.get("vramCardReserveMiB")) is not int or config["vramCardReserveMiB"] <= 0:
        raise ValueError("vramValve requires a positive vramCardReserveMiB")
    if config.get("vramGuard") is not True or type(config.get("slotSaveCheckpoints")) is not int \
            or config["slotSaveCheckpoints"] < 1:
        raise ValueError("vramValve requires vramGuard=true and slotSaveCheckpoints >= 1 for safe parking")
    if config.get("slotGears"):
        raise ValueError("vramValve and slotGears cannot run together; they are separate resize policies")
    levels = config.get("vramValveLevels", VALVE_DEFAULTS["vramValveLevels"])
    if (not isinstance(levels, list) or not levels
            or any(type(value) is not int or value < 0 for value in levels)
            or levels != sorted(set(levels), reverse=True)):
        raise ValueError("vramValveLevels must be a descending list of unique non-negative integers")
    for key, low, high in (("vramValveHysteresisMiB", 0, 4096),
                           ("vramValveUpSeconds", 1, 3600),
                           ("vramValveParkSeconds", 1, 30),
                           ("vramValveResizeRetrySeconds", 1, 60)):
        value = config.get(key, VALVE_DEFAULTS[key])
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid guardian config: {key} must be an integer from {low} to {high}")


def validate_vram_valve_model_configs(models):
    """Validate optional per-model ladders; out-of-range gears are clipped at model maximum."""
    for model_id, settings in models.items():
        if "vramValveLevels" not in settings:
            continue
        levels = settings["vramValveLevels"]
        if (not isinstance(levels, list) or not levels
                or any(type(value) is not int or value < 0 for value in levels)
                or levels != sorted(set(levels), reverse=True)):
            raise ValueError(
                f"Invalid model config for {model_id}: vramValveLevels must be a descending list "
                "of unique non-negative integers"
            )


def vram_card_reserve_enabled():
    value = G_RAW.get("vramCardReserveMiB")
    return type(value) is int and value > 0


def startup_slot_levels(model_id, configured):
    """Known supported gears, bounded by this model's configured maximum."""
    if type(configured) is not int or configured < 0:
        raise ValueError("moeCacheSlots must be a non-negative integer")
    if G.get("vramValve"):
        # Live-resize servers require a positive startup cache. Zero remains
        # available only as a runtime emergency state, never as startup sizing.
        levels = {configured}
        for item in G.get("vramValveLevels", VALVE_DEFAULTS["vramValveLevels"]):
            if type(item) is int and 1 <= item <= configured:
                levels.add(item)
        return sorted(levels, reverse=True)
    if configured < 16:
        return [configured]  # e.g. a no-cache model cannot be raised to the gear minimum
    ceiling = configured
    if G.get("slotGears") and model_id in RUNTIME_SLOTS:
        ceiling = min(ceiling, RUNTIME_SLOTS[model_id])  # startup admission must not undo the ratchet
    levels = {ceiling}
    for item in G.get("slotGearLevels", GEAR_DEFAULTS["slotGearLevels"]):
        if type(item) is int and 16 <= item <= ceiling:
            levels.add(item)
    levels.add(16)
    return sorted(levels, reverse=True)


def choose_startup_slots(model_id, vision, mc, layout="short"):
    """Choose the largest supported cache whose estimated post-load card-free VRAM meets reserve.

    vramFreeMinGBToLoad is used as the existing per-model non-cache load-footprint baseline
    (decimal GB); the cache is added using the measured per-slot cost. The current whole-card
    free sample naturally excludes memory already used by the desktop.
    """
    if not vram_card_reserve_enabled() or governor_on():
        # VRAMLOGIC1i: under the governor the server owns the expert cache; admission is preload_check(admit=...) on the learned
        # fixed part, not a legacy estimate that re-runs on every request (it refused a running lite server).
        return ""
    sample_age = time.time() - st.vram_sample_at if st.vram_sample_at is not None else math.inf
    if (st.monitor_error or st.vram_free_mib is None
            or sample_age > G.get("vramTelemetryStaleSeconds", 5)):
        return ""  # preload_check retains its existing fresh-NVML refusal and event
    reserve = G_RAW["vramCardReserveMiB"]
    try:
        maximum = mc.get("moeCacheSlots", 0) if (vision or layout in ("long", "lite", "longlite")) else M["models"][model_id].get("moeCacheSlots", 0)
        configured = maximum
        if not vision and G.get("slotGears") and model_id in RUNTIME_SLOTS:
            configured = min(configured, RUNTIME_SLOTS[model_id])
        levels = startup_slot_levels(model_id, configured)
        cost = (model_slot_cost(model_id)["value_mib"] if G.get("vramValve") else
                float(G.get("slotGearMiBPerSlot", GEAR_DEFAULTS["slotGearMiBPerSlot"])))
        base_mib = max(0.0, float(mc.get("vramFreeMinGBToLoad", 0)) * 1000.0)
        if not math.isfinite(cost) or cost <= 0 or not math.isfinite(base_mib):
            raise ValueError("cache-slot cost and model load estimate must be finite positive values")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        message = Refusal(f"Invalid model VRAM reserve settings: {error}", "config")
        event("vram_card_reserve_config_error", message)
        return message
    free = float(st.vram_free_mib)
    # A model switch unloads the current backend after admission succeeds. Its fresh
    # process allocation is therefore part of the post-switch headroom, like the
    # running_footprint credit used by RAM/VRAM admission below.
    if (st.proc is not None and st.proc.poll() is None
            and (st.model_id != model_id or st.vision != vision or st.layout != layout)):
        current_mib = running_vram_mib()
        if current_mib is not None:
            free += current_mib
    selected = next((slots for slots in levels if free - base_mib - slots * cost >= reserve), None)
    if selected is None:
        minimum = levels[-1]
        projected = free - base_mib - minimum * cost
        message = Refusal(f"Model load deferred: card VRAM reserve {reserve} MiB cannot be kept; "
                          f"free {free:.0f} MiB, minimum cache {minimum} slots estimates {projected:.0f} MiB after load", "vram_reserve")
        event("vram_card_reserve_deferred", message)
        return message

    projected = free - base_mib - selected * cost
    reason = (f"card VRAM free {free:.0f} MiB, reserve {reserve} MiB, slots {selected} of {maximum}; "
              f"estimated free after load {projected:.0f} MiB")
    RUNTIME_SLOTS[model_id] = selected
    STARTUP_VRAM_SELECTIONS[model_id] = {"slots": selected, "configured_slots": maximum,
                                         "free_mib": int(free), "reserve_mib": reserve,
                                         "estimated_after_mib": round(projected), "reason": reason}
    mc["moeCacheSlots"] = selected
    event("vram_card_reserve_selected", reason)
    return ""


LITE_STATE = {"since": -math.inf, "relief_since": None, "relief_checked": -math.inf}


def lite_enabled(model_id, vision=False, layout="short"):
    """VRAMCOR2: this model has a liteProfile (VRAMCOR4: for the long layout a longLiteProfile) and the card reserve is on (both are
    opt-in; nothing changes without them)."""
    key = LITE_PROFILE_NAME.get(layout)
    return (key is not None and not vision and vram_card_reserve_enabled() and bool((M["models"].get(model_id) or {}).get(key)))


def lite_cost_mib(model_id):
    if G.get("vramValve"):
        return model_slot_cost(model_id)["value_mib"]
    return float(G.get("slotGearMiBPerSlot", GEAR_DEFAULTS["slotGearMiBPerSlot"]))


def lite_delta_mib(model_id, layout="short"):
    """VRAM the normal layout (short, or long for the long family) holds over its lite twin: the baseline difference plus the cache
    slots of the configured maximum."""
    normal = model_config(model_id, False, layout) if layout == "long" else M["models"][model_id]
    lite = model_config(model_id, False, LITE_OF[layout])
    base = (float(normal.get("vramFreeMinGBToLoad", 0)) - float(lite.get("vramFreeMinGBToLoad", 0))) * 1000.0
    slots = max(0, int(normal.get("moeCacheSlots", 0)) - int(lite.get("moeCacheSlots", 0)))
    return base + slots * lite_cost_mib(model_id)


def lite_fresh_free_mib():
    sample_age = time.time() - st.vram_sample_at if st.vram_sample_at is not None else math.inf
    if st.monitor_error or st.vram_free_mib is None or sample_age > G.get("vramTelemetryStaleSeconds", 5):
        return None
    return float(st.vram_free_mib)


def lite_fallback(model_id, vision, layout, mc, reason):
    """VRAMCOR2: the short layout cannot keep the card reserve. Instead of answering 503 until the desktop gives VRAM back (40-50 s
    in VRAMCOR1), try the lite layout at once. VRAMCOR4: the same for the long layout (longlite). Returns (layout, mc, reason);
    unchanged when there is no fallback or even the lite layout does not fit."""
    if (not reason or layout not in LITE_OF or refusal_code(reason) != "vram_reserve" or not lite_enabled(model_id, vision, layout)):
        return layout, mc, reason
    lite_layout = LITE_OF[layout]
    lite_mc = model_config(model_id, vision, lite_layout)
    lite_reason = choose_startup_slots(model_id, vision, lite_mc, lite_layout)
    if lite_reason:
        return layout, mc, reason
    LITE_STATE.update(since=time.time(), relief_since=None)
    event("vram_lite_selected", f"model={model_id}: the {layout} layout cannot keep the card reserve, the {lite_layout} layout is loaded "
          f"(nCpuMoe {lite_mc.get('nCpuMoe')}, cache slots {lite_mc.get('moeCacheSlots')})")
    return lite_layout, lite_mc, ""


def lite_pressure(model_id, vision, layout="short"):
    """The normal layout runs with less free VRAM than the reserve, and its lite twin would give enough back to keep it."""
    free = lite_fresh_free_mib()
    if free is None or st.layout != layout or not lite_enabled(model_id, vision, layout):
        return False
    reserve = G_RAW["vramCardReserveMiB"]
    return free < reserve and free + lite_delta_mib(model_id, layout) >= reserve


def lite_relief(model_id, vision, now, layout="short"):
    """The lite twin of `layout` runs and the normal layout would keep the reserve plus the margin, continuously for vramLiteReturnSeconds
    (and not before that long after the lite layout was chosen): the hysteresis against flapping."""
    free = lite_fresh_free_mib()
    if free is None or st.layout != LITE_OF[layout] or not lite_enabled(model_id, vision, layout):
        return False
    held = now - LITE_STATE["since"] >= G["vramLiteReturnSeconds"]
    ok = free - lite_delta_mib(model_id, layout) >= G_RAW["vramCardReserveMiB"] + G["vramLiteReturnMarginMiB"]
    if not ok or now - LITE_STATE["relief_checked"] > 3 * G["vramLiteReturnSeconds"]:
        LITE_STATE["relief_since"] = now if ok else None   # the checks come with requests: a long gap does not count as continuous
    elif LITE_STATE["relief_since"] is None:
        LITE_STATE["relief_since"] = now
    LITE_STATE["relief_checked"] = now
    since = LITE_STATE["relief_since"]
    return bool(held and ok and since is not None and now - since >= G["vramLiteReturnSeconds"])


def slot_filename(model_id, vision):
    return f"{model_id}-{'vision' if vision else 'text'}.bin"


def llama_headers(has_body=False):
    return common.llama_headers(M.get("apiKey"), has_body)


def llama_call(method, path, body=None, timeout=10):
    conn = http.client.HTTPConnection(G["llamaHost"], G["llamaPort"], timeout=timeout)
    try:
        payload = json.dumps(body) if body is not None else None
        conn.request(method, path, body=payload, headers=llama_headers(bool(payload)))
        r = conn.getresponse()
        data = r.read()
        try:
            return r.status, json.loads(data) if data else None
        except ValueError:
            return r.status, None
    finally:
        conn.close()


RUN_EXE = {"path": None}   # GWV1: the binary of the running server when its layout names its own exePath (None = M["exePath"])
ENGINES = ("legacy", "v1")


def exe_for(mc=None, layout=None, check=True):
    """GWV1: the llama-server binary for a layout. An optional "exePath" in the model profile or its layout profile (mc) wins;
    without it the answer is M["exePath"], exactly as before. A named binary that is not an executable file is a ValueError
    that names the layout and the path (check=False skips that test: capability probes of a layout that may not start)."""
    own = mc.get("exePath") if isinstance(mc, dict) else None
    if own is None:
        return M["exePath"]
    if not isinstance(own, str) or not own:
        raise ValueError("exePath must be a non-empty string")
    path = expand("exePath", own)
    if check and path != M["exePath"] and not (os.path.isfile(path) and os.access(path, os.X_OK)):
        name = layout or (mc.get("layoutName") if isinstance(mc, dict) else None) or Path(str(mc.get("modelPath", "?"))).name
        raise ValueError(f"layout {name}: exePath {path} is not an executable file")
    return path


def engine_of(mc):
    """GWV1: "legacy" (no field) or "v1" (shura.cpp 1.0); anything else is refused."""
    engine = mc.get("engine") if isinstance(mc, dict) else None
    if engine is None:
        return "legacy"
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {', '.join(ENGINES)}, not {engine!r}")
    return engine


def bad_layout(mc, name):
    """GWV1: True (st.last_error set, event logged) when the layout's exePath or engine is unusable; the running server is not touched."""
    try:
        exe_for(mc, name)
        engine_of(mc)
    except ValueError as e:
        st.last_error = f"Invalid model configuration: {e}"
        event("layout_exe_invalid", st.last_error, fallback=True)
        return True
    return False


def server_supports(flag, mc=None):
    return common.server_supports(exe_for(mc, check=False) if mc else M.get("exePath"), flag)


def api_key_file():
    try:
        return common.write_api_key_file(STATE_DIR, M["apiKey"])
    except OSError as error:
        log(f"Key file not written ({error}); the key goes on the command line instead")
        return None


def redacted_command(args):
    return common.redacted_command(args)


def model_extra_env(mc):
    """Environment variables the model's llama-server gets on top of the gateway's own (AMD2a: GGML_* and GPU_* switches).

    Names and values are both strings; values are never written to the log, only names are."""
    raw = mc.get("extraEnv", {})
    if not isinstance(raw, dict):
        raise ValueError("extraEnv must be an object of name: string pairs")
    for name, value in raw.items():
        if not isinstance(name, str) or not name or "=" in name or "\0" in name:
            raise ValueError(f"extraEnv name {name!r} is not a valid environment variable name")
        if not isinstance(value, str) or "\0" in value:
            raise ValueError(f"extraEnv value for {name} must be a string")
    return dict(raw)


def validate_model_extra_env(models):
    for model_id, settings in models.items():
        try:
            model_extra_env(settings)
        except ValueError as error:
            raise ValueError(f"Invalid model configuration for {model_id}: {error}") from None


def _ram_cache_budget(mc, ladder=True):
    """(cache-ram MiB, checkpoints per slot) after the RAM cache budget. Budget 0 (default) returns the configured values unchanged;
    after a pressing action the next start uses half of each. RAMLOGIC2: the ladder's light step caps the prompt cache of the next
    start (ladder=False gives the value before that cap)."""
    cache, checkpoints = ram_valve.cache_budget(ram_valve.read_mem_total_gb(), G.get("ramCacheBudgetPct", 0), int(mc["ctxSize"]),
                                                int(G.get("slotSaveCheckpoints", 0)), int(M["cacheRamMB"]), ram_snapshot_mib())
    if RAM_REDUCE["on"] and G.get("ramGuard") is True:
        cache, checkpoints = cache // 2, checkpoints // 2
    if ladder and RAM_LADDER["cache_cap"] is not None:
        cache = min(cache, RAM_LADDER["cache_cap"])
    return cache, checkpoints


def build_args(mc, vision):
    exe = exe_for(mc)
    v1 = engine_of(mc) == "v1"
    own_exe = mc.get("exePath") is not None

    def supports(flag):     # the layout's own binary when it names one; otherwise the plain call, as before
        return server_supports(flag, mc) if own_exe else server_supports(flag)
    a = [exe, "--host", G["llamaHost"], "--port", str(G["llamaPort"]),
         "-m", mc["modelPath"], "-c", str(mc["ctxSize"]), "-np", str(mc["parallelSlots"]),
         "-t", str(mc["threads"]), "-tb", str(mc["threadsBatch"]),
         "-b", str(mc["batchSize"]), "-ub", str(mc["ubatchSize"]),
         "-ngl", str(mc["nGpuLayers"]), "-ncmoe", str(mc["nCpuMoe"]),
         "-ctk", mc["cacheTypeK"], "-ctv", mc["cacheTypeV"], "-fa", mc["flashAttn"],
         "--load-mode", mc["loadMode"],
         "--temp", str(mc["temperature"]), "--top-k", str(mc["topK"]), "--top-p", str(mc["topP"]),
         "--jinja", "--reasoning", mc["reasoning"],
         "--cache-ram", str(_ram_cache_budget(mc)[0]), "--slot-save-path", str(remember_server_slot_dir())]
    # Optional CPU wait policy: absent preserves llama.cpp's default --poll 50.
    poll = mc.get("poll")
    if poll is not None:
        poll = int(poll)
        if not 0 <= poll <= 100:
            raise ValueError("poll must be between 0 and 100")
        a += ["--poll", str(poll)]
    # Sampling the model's authors recommend, beyond temperature/top-k/top-p (optional per model, e.g. Occamy: presence_penalty 1.5).
    for key, flag in (("minP", "--min-p"), ("presencePenalty", "--presence-penalty"), ("repeatPenalty", "--repeat-penalty")):
        if mc.get(key) is not None:
            a += [flag, str(mc[key])]
    if mc.get("imageMaxTokens"):
        # Caps the tokens one image may take: the vision encoder's scratch memory grows with the picture (2560x1600 added ~450 MiB).
        a += ["--image-max-tokens", str(int(mc["imageMaxTokens"]))]
    # llama-server answers any web page by default (Access-Control-Allow-Origin: the caller, with credentials). The gateway strips
    # those headers, but port llamaPort itself is reachable from a browser too: allow only its own origin there.
    a += common.cors_args(exe if mc.get("exePath") is not None else M.get("exePath"), G["llamaPort"])
    extra = mc.get("extraArgs") or []
    if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
        raise ValueError("extraArgs must be a list of strings")
    a += list(extra)             # e.g. ["-ot", "blk\\.40\\.ffn_(up|down|gate)_exps\\.weight=CPU"]: the MTP layer's experts on the CPU
    if not governor_on() and (G.get("vramValve") or G.get("prefillCacheYield")) and "--moe-cache-live-resize" not in a:
        a.append("--moe-cache-live-resize")
    if mc.get("chatTemplateKwargs"):
        # Extra variables for the model's chat template, e.g. {"terse": false} for Tiel-Coder (see docs/INSTALL.md).
        a += ["--chat-template-kwargs", json.dumps(mc["chatTemplateKwargs"], separators=(",", ":"))]
    # v1 (shura.cpp 1.0) has neither --moe-cache-profile nor --moe-cache-slots: its governor sizes the cache in MiB
    if not v1 and mc.get("moeCacheProfile") and (mc.get("moeCacheSlots", 0) > 0 or G.get("vramValve")):
        a += ["--moe-cache-profile", mc["moeCacheProfile"]]
        if not governor_on():
            a += ["--moe-cache-slots", str(mc["moeCacheSlots"])]    # under the governor governor_args() says --moe-cache-slots 0 (the cache grows itself)
    if governor_on():
        a += governor_args(mc)
    if mc.get("specType") and mc["specType"] != "none":
        a += ["--spec-type", mc["specType"], "--spec-draft-n-max", str(mc["specDraftNMax"]),
              "--spec-draft-p-min", str(mc["specDraftPMin"])]
    checkpoints = _ram_cache_budget(mc)[1]
    if GAME.enabled:
        checkpoints = max(1, checkpoints)
    if checkpoints > 0:
        if supports("--slot-save-checkpoints"):
            a += ["--slot-save-checkpoints", str(checkpoints)]
        elif supports("--ctx-checkpoints"):
            # Older supported forks expose the same per-slot checkpoint cap under the
            # upstream name. This fallback is only used when checkpointing is opted in.
            a += ["--ctx-checkpoints", str(checkpoints)]
        else:
            # Keep the previous command construction for unrecognized/fake servers. A
            # vramValve deployment is checked against the server capability at startup.
            a += ["--slot-save-checkpoints", str(checkpoints)]
    a += ["--mmproj", mc["mmprojPath"]] if vision else ["--no-mmproj-auto"]
    if M.get("apiKey"):
        path = api_key_file() if supports("--api-key-file") else None
        a += ["--api-key-file", path] if path else ["--api-key", M["apiKey"]]
    if mc.get("niceLevel"):
        # Lower CPU priority: desktop apps win any contention; no speed is lost while the desktop is idle.
        a = ["nice", "-n", str(mc["niceLevel"])] + a
    return a


SERVER_SLOT = {"dir": None, "owner": None}   # the --slot-save-path the running server was given: pause files are written there;
# owner: (model, vision, layout) of the server started last, to which a pause checkpoint it left behind belongs


def remember_server_slot_dir():
    """VRAMLOGIC1: the directory handed to llama-server as --slot-save-path. Its pause files (the tokens file, the exit marker)
    and the checkpoint are looked up here, never in the gateway's own slot folder."""
    path = server_slot_dir()
    SERVER_SLOT["dir"] = Path(path)
    return path


def server_slot_dir():
    st.ram_backend = False
    if G["vramGuard"] or GAME.enabled or G["slotGears"] or G["crashRecovery"] or G.get("ramGuard") is True:
        try:
            path = prepare_ram_slot_dir()
            st.ram_backend = True
            return path
        except (OSError, RuntimeError) as e:
            event("ram_path_failed", f"RAM sessions unavailable; using ordinary disk slots: {e}", fallback=True)
    return SLOT_DIR


def running_vram_mib():
    """VRAM (MiB) the running llama-server holds. The monitor's per-process sample when the card reports it. The AMD reader
    reports no process for some servers (LAYOUT1 on the real card): then the estimate of the running layout is used, its
    non-cache base plus its cache slots, so that a layout switch can count the memory the old server gives back."""
    proc = st.proc
    if not (proc and proc.poll() is None):
        return None
    fresh = process_vram_mib(proc.pid)  # -1 when the sample is stale or for another process: never credited
    if fresh >= 0:
        return float(fresh)
    if st.proc_vram_mib is not None:
        return None                     # the reader did report this server once, the sample is just old
    if not st.model_id:
        return None
    mc = model_config(st.model_id, st.vision, st.layout)
    try:
        cost = (model_slot_cost(st.model_id)["value_mib"] if G.get("vramValve") else
                float(G.get("slotGearMiBPerSlot", GEAR_DEFAULTS["slotGearMiBPerSlot"])))
        return float(mc.get("vramFreeMinGBToLoad", 0)) * 1000.0 + float(mc.get("moeCacheSlots", 0)) * cost
    except (KeyError, TypeError, ValueError):
        return None


def running_footprint():
    """(RAM GB, VRAM GB) held by the llama-server that is running now. A model switch unloads it before the new load, so the admission
    check for the new model has to count this memory as available (otherwise no switch is possible while a model is loaded)."""
    proc = st.proc
    if not (proc and proc.poll() is None):
        return 0.0, 0.0
    rss = 0.0
    try:
        for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1048576
    except (OSError, ValueError):
        pass
    return rss, (running_vram_mib() or 0) / 1024


def read_load_pressure():
    """Read only the small PSI and zram files needed for load admission.

    Missing or malformed sources are returned as unavailable; admission then falls back to
    the existing RAM/VRAM checks. Tests replace this function or its fixture roots.
    """
    psi = {"full_avg10": None, "full_avg60": None}
    try:
        for line in Path("/proc/pressure/memory").read_text().splitlines():
            fields = line.split()
            if not fields or fields[0] != "full":
                continue
            values = {}
            for field in fields[1:]:
                key, separator, value = field.partition("=")
                if separator and key in ("avg10", "avg60"):
                    values[key] = float(value)
            avg10, avg60 = values.get("avg10"), values.get("avg60")
            if (avg10 is not None and avg60 is not None
                    and math.isfinite(avg10) and math.isfinite(avg60)
                    and 0 <= avg10 <= 100 and 0 <= avg60 <= 100):
                psi = {"full_avg10": avg10, "full_avg60": avg60}
            break
    except (OSError, ValueError):
        pass
    try:
        zram = _read_zram(Path("/sys"))
        zram_pct = zram.get("used_percent") if isinstance(zram, dict) else None
        zram_used_bytes = zram.get("used_bytes") if isinstance(zram, dict) else None
        if (isinstance(zram_pct, (int, float)) and not isinstance(zram_pct, bool)
                and math.isfinite(zram_pct) and 0 <= zram_pct <= 100):
            psi["zram_used_percent"] = float(zram_pct)
        else:
            psi["zram_used_percent"] = None
        if (isinstance(zram_used_bytes, int) and not isinstance(zram_used_bytes, bool)
                and zram_used_bytes >= 0):
            psi["zram_used_bytes"] = zram_used_bytes
        else:
            psi["zram_used_bytes"] = None
    except (OSError, ValueError, TypeError):
        psi["zram_used_percent"] = None
        psi["zram_used_bytes"] = None
    return psi


def record_load_pressure_refusal(kind, message):
    reason = f"load_deferred_{kind}"
    with st.lock:
        if st.load_pressure_refusal_active[kind]:
            st.event_counters[reason] = st.event_counters.get(reason, 0) + 1
            return
        st.load_pressure_refusal_active[kind] = True
        st.load_pressure_refusal_episodes[kind] += 1
        episode = st.load_pressure_refusal_episodes[kind]
    event(reason, f"episode {episode}: {message}")


def reset_load_pressure_refusals():
    with st.lock:
        for kind in st.load_pressure_refusal_active:
            st.load_pressure_refusal_active[kind] = False


def memory_load_pressure_check():
    zram_limit = pressure_limit(G, "zramMaxPctToLoad")
    psi_limit = pressure_limit(G, "psiMemoryFullMaxToLoad")
    check_zram = zram_limit > 0
    check_psi = psi_limit > 0
    if not check_zram and not check_psi:
        return ""

    sample = read_load_pressure()
    if not isinstance(sample, dict):
        sample = {}
    zram_pct = sample.get("zram_used_percent")
    psi_avg10 = sample.get("full_avg10")
    psi_avg60 = sample.get("full_avg60")
    unavailable = ((check_zram and zram_pct is None) or
                   (check_psi and (psi_avg10 is None or psi_avg60 is None)))
    if unavailable:
        if not st.load_pressure_telemetry_unavailable:
            event("load_pressure_telemetry_unavailable",
                  "zram or memory PSI could not be read; using the existing RAM/VRAM admission checks")
        st.load_pressure_telemetry_unavailable = True
        return ""
    if st.load_pressure_telemetry_unavailable:
        st.load_pressure_telemetry_unavailable = False
        event("load_pressure_telemetry_recovered", "zram and memory PSI are readable again")

    if check_zram and zram_pct > zram_limit:
        used_bytes = sample.get("zram_used_bytes")
        used_text = (f"{used_bytes / 1048576:,.0f} MiB RAM used"
                     if isinstance(used_bytes, int) and not isinstance(used_bytes, bool) and used_bytes >= 0
                     else "RAM usage unavailable")
        message = (f"Model load deferred: zram is {zram_pct:.1f}% logically full ({used_text}; "
                   f"limit {zram_limit:.1f}%). "
                   "Wait for memory use to drop below the limit, then retry; set zramMaxPctToLoad to 0 to disable this check.")
        record_load_pressure_refusal("zram", message)
        return message
    if check_psi and psi_avg10 > psi_limit:
        message = (f"Model load deferred: memory PSI full avg10 is {psi_avg10:.1f}% "
                   f"(avg60 {psi_avg60:.1f}%; limit {psi_limit:.1f}%). "
                   "Wait for memory pressure to ease, then retry; set psiMemoryFullMaxToLoad to 0 to disable this check.")
        record_load_pressure_refusal("psi", message)
        return message
    return ""


# VRAMLOGIC1h: the governor path. With vramPolicy "governor" the server's governor owns the expert-cache chunks: the gateway
# never sets slots, admits a layout by the fixed part the server measured after warmup, switches to lite when the governor
# stays in MIN longer than T_lite, and does not run the valve, prefill yield, slot gears or the lite-pressure check.
# With "legacy" (the code default) none of this runs and the old paths are unchanged.
GOV = {"budget": None, "values": None, "checked_at": 0.0, "pushed_pid": None, "snap": None, "state": None, "error_logged": False,
       "too_big": {}}   # (model, layout) -> {"free_mib", "at"}: the server exited 75 (the layout does not keep the reserve)
LAYOUT_TOO_BIG_SECONDS = 600   # a mark expires: the desktop may have given the VRAM back


def governor_mark_too_big(model_id, layout):
    GOV["too_big"][(model_id, layout)] = {"free_mib": st.vram_free_mib, "at": time.time()}
    event("vram_layout_too_big", f"model={model_id} layout={layout} free={st.vram_free_mib} MiB: the layout does not keep the card reserve; "
          f"the lite layout is used for {LAYOUT_TOO_BIG_SECONDS} s")


def governor_layout(model_id, vision, layout):
    """VRAMLOGIC1i: a layout that exited 75 does not fit the card reserve; run its lite twin while the mark lasts."""
    mark = GOV["too_big"].get((model_id, layout))
    if mark is None or time.time() - mark["at"] > LAYOUT_TOO_BIG_SECONDS:
        GOV["too_big"].pop((model_id, layout), None)
        return layout
    if vision or layout not in LITE_OF or not lite_enabled(model_id, vision, layout):
        return layout
    return LITE_OF[layout]



def governor_on():
    return G.get("vramPolicy") == "governor"


def governor_budget():
    if GOV["budget"] is None:
        _policy, values, lite_after = vram_budget.check_policy_config(G)
        GOV["values"] = values
        GOV["budget"] = vram_budget.Budget(reserve_mib=values["reserve_mib"], hysteresis_mib=values["hysteresis_mib"],
                                           buffer_mib=values.get("buffer_mib", 0), lite_after_s=lite_after)
    return GOV["budget"]


def governor_http(method, path, body=None, timeout=2.0):
    conn = http.client.HTTPConnection(G["llamaHost"], G["llamaPort"], timeout=timeout)
    try:
        payload = None if body is None else json.dumps(body)
        conn.request(method, path, body=payload, headers=llama_headers(bool(payload)))   # the server's API key, when set
        response = conn.getresponse()
        raw = response.read()
        return response.status, (json.loads(raw.decode("utf-8") or "null") if raw else None)
    finally:
        conn.close()


def vram_chunk_mib(mc=None):
    """GWV1: the v1 governor's step in MiB: the layout's vramChunkMiB, else guardian's, else 52. Whole numbers only."""
    value = 52
    for source in (G, mc or {}):
        if source.get("vramChunkMiB") is not None:
            value = source["vramChunkMiB"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"vramChunkMiB must be a positive whole number of MiB, not {value!r}")
    return value


def governor_args(mc=None):
    """Server flags for the governor: the server starts with an empty expert cache and grows it itself."""
    governor_budget()
    v = GOV["values"]
    if mc is not None and engine_of(mc) == "v1":
        # shura.cpp 1.0 measures the cache in MiB (--vram-chunk-mib) and has no --moe-cache-chunk-slots / --moe-cache-slots
        args = ["--vram-reserve-mib", str(v["reserve_mib"]), "--vram-hysteresis-mib", str(v["hysteresis_mib"]),
                "--vram-grow-seconds", str(v["grow_seconds"]), "--vram-pause-floor-mib", str(v["pause_floor_mib"]),
                "--vram-min-chunks", str(v["min_chunks"]), "--vram-poll-ms", str(G.get("vramPollMs", 200)),
                "--vram-chunk-mib", str(vram_chunk_mib(mc)), "--vram-sysfs-card", "auto",
                "--warmup-shapes", "prefill_ub,decode_draft,vision"]
        if "buffer_mib" in v:
            args += ["--vram-buffer-mib", str(v["buffer_mib"])]
        if "poll_fast_ms" in v:
            args += ["--vram-poll-fast-ms", str(v["poll_fast_ms"])]
        return args
    args = ["--vram-reserve-mib", str(v["reserve_mib"]), "--vram-hysteresis-mib", str(v["hysteresis_mib"]),
            "--vram-grow-seconds", str(v["grow_seconds"]), "--vram-pause-floor-mib", str(v["pause_floor_mib"]),
            "--vram-min-chunks", str(v["min_chunks"]), "--vram-poll-ms", str(G.get("vramPollMs", 200)),
            "--moe-cache-chunk-slots", str(G.get("vramChunkSlots", 4)), "--vram-sysfs-card", "auto",
            "--warmup-shapes", "prefill_ub,decode_draft,vision", "--moe-cache-slots", "0"]
    # VRAMLOGIC2: only when the config names them (a server from before VRAMLOGIC2 does not know the flags)
    if "buffer_mib" in v:
        args += ["--vram-buffer-mib", str(v["buffer_mib"])]
    if "poll_fast_ms" in v:
        args += ["--vram-poll-fast-ms", str(v["poll_fast_ms"])]
    return args


def governor_tick(now):
    """While READY: read GET /vram about once a second, push the policy once per server process, learn the fixed part."""
    if not governor_on() or st.proc is None or st.proc.poll() is not None or st.status != "READY" or st.vision:
        return
    if now - GOV["checked_at"] < 1.0:
        return
    GOV["checked_at"] = now
    budget = governor_budget()
    try:
        if GOV["pushed_pid"] != st.proc.pid:
            code, out = governor_http("POST", "/vram/policy", GOV["values"])
            if code == 400 and any(key in GOV["values"] for key in vram_budget.VRAMLOGIC2_KEYS):
                # a server from before VRAMLOGIC2: it does not know the buffer, the fast poll and the pause in ms; the rest of the
                # policy still goes, and the stall pause in its whole seconds
                older = vram_budget.older_policy(GOV["values"])
                code, out = governor_http("POST", "/vram/policy", older)
                if code == 200 and not GOV.get("compat_logged"):
                    GOV["compat_logged"] = True
                    event("vram_governor_compat", f"model={st.model_id}: the server does not know {', '.join(vram_budget.VRAMLOGIC2_KEYS)}; "
                          "the floor and the rest of the policy are set, the buffer is the server's own", fallback=True)
            if code != 200:
                raise RuntimeError(f"policy refused HTTP {code}: {out}")
            GOV["pushed_pid"] = st.proc.pid
        code, data = governor_http("GET", "/vram")
        if code != 200:
            raise RuntimeError(f"HTTP {code}")
        snap = vram_budget.parse_snapshot(data)
    except (OSError, RuntimeError, ValueError) as error:
        if not GOV["error_logged"]:
            event("vram_governor_unavailable", f"model={st.model_id}: {error}", fallback=True)
            GOV["error_logged"] = True
        return
    GOV["error_logged"] = False
    GOV["snap"] = snap
    budget.learn_policy(snap.get("policy"))   # VRAMLOGIC2: admission is judged against the target the server really keeps
    if snap["state"] != GOV["state"]:
        event("vram_governor_state", f"model={st.model_id} {GOV['state']}->{snap['state']}: {snap['reason']}")
        GOV["state"] = snap["state"]
    budget.observe(snap["state"], now)
    key = (st.model_id, st.layout)
    known = budget.fixed_learned.get(key)
    # the measured fixed part wobbles by a few MiB from one reading to the next: learn (and log) a change only when it is real
    if snap["ready"] and (known is None or abs(known - snap["fixed_mib"]["total"]) > 16):
        budget.learn(st.model_id, st.layout, snap)
        event("vram_governor_fixed_learned", f"model={st.model_id} layout={st.layout} fixed_mib={snap['fixed_mib']['total']}")


def governor_warm():
    """True once the server reports ready (its warm-up is done). Any error counts as not ready yet."""
    try:
        code, data = governor_http("GET", "/vram")
    except (OSError, ValueError, http.client.HTTPException):
        return False
    return code == 200 and isinstance(data, dict) and data.get("ready") is True


def governor_lite_pressure(model_id, vision, layout="short"):
    return governor_budget().governor_says_lite(GOV["state"], time.time()) if GOV["state"] else False


def governor_lite_relief(model_id, vision, now, layout="short"):
    if GOV["state"] and not governor_budget().lite_relief_ok(GOV["state"], time.time()):
        return False
    # GWSTALL1: the normal layout comes back only when its admission passes right now. A refused return is held (one event), so
    # the sum is not recomputed and re-logged on every request.
    hold = UPGRADE_HOLD.get((model_id, layout))
    if hold is not None and time.time() < hold["until"]:
        return False
    short_by = governor_short_by_mib(model_id, layout)
    if short_by:
        hold_upgrade(model_id, layout, f"Not enough free VRAM for {model_id}/{layout} under the governor ({short_by} MiB short)")
        return False
    return True


def governor_short_by_mib(model_id, layout):
    """MiB the learned fixed part of `layout` lacks to start now: the card's free VRAM plus what the running server gives back against
    fixed + reserve + hysteresis (the same sum preload_check uses). 0 when it fits, or when no fixed part is known yet."""
    budget = governor_budget()
    if budget.fixed_learned.get((model_id, layout)) is None or st.vram_free_mib is None:
        return 0
    free_mib = int(st.vram_free_mib + (running_vram_mib() or 0))
    _ok, short, _source = budget.admission(model_id, layout, free_before_start_mib=free_mib)
    return short


def validate_ram_config(config):
    if config.get("ramPolicy", "off") not in ("off", "honest"):
        raise ValueError("Invalid guardian config: ramPolicy must be \"off\" or \"honest\"")
    reserve = config.get("ramDeskReserveMiB")
    if reserve is not None and (type(reserve) is not int or not 1024 <= reserve <= 16384):
        raise ValueError("Invalid guardian config: ramDeskReserveMiB must be null (follow the desk) or an integer 1024..16384")
    wait = config.get("ramWaitSeconds", 20)
    if type(wait) is not int or not 0 <= wait <= 300:
        raise ValueError("Invalid guardian config: ramWaitSeconds must be an integer 0..300")
    hard = config.get("ramHardMinMiB", ram_budget.RAM_HARD_MIN_MIB)
    if type(hard) is not int or not 512 <= hard <= 8192:
        raise ValueError("Invalid guardian config: ramHardMinMiB must be an integer 512..8192")
    if type(config.get("ramReclaimCache", True)) is not bool:
        raise ValueError("Invalid guardian config: ramReclaimCache must be true or false")


def validate_vram_policy_config(config):
    try:
        vram_budget.check_policy_config(config)
    except vram_budget.BudgetError as error:
        raise ValueError(f"Invalid guardian config: {error}") from None


RAM_REFUSE_SECONDS = 300             # after a load that pushed swap up too much, the next loads are refused this long
RAM_SWAP_REFUSAL = {"until": 0.0, "reason": ""}


def ram_swap_before():
    """RAMGUARD1 in-flight check: swap reading before a load (ramPolicy honest only; None otherwise)."""
    if G.get("ramPolicy") != "honest":
        return None
    return ram_budget.read_meminfo()


def ram_swap_check_after_load(model_id, layout, before):
    """A load that grew swap by more than SWAP_GROWTH_LIMIT_MIB refuses the next loads for RAM_REFUSE_SECONDS."""
    if before is None:
        return
    after = ram_budget.read_meminfo()
    if after is None or not ram_budget.swap_growth_too_big(before, after):
        return
    grew = ram_budget.swap_used_mib(after) - ram_budget.swap_used_mib(before)
    reason = (f"swap grew {grew:.0f} MiB while {model_id} ({layout}) loaded (limit {ram_budget.SWAP_GROWTH_LIMIT_MIB} MiB); "
              f"loads refused for {RAM_REFUSE_SECONDS} s")
    RAM_SWAP_REFUSAL.update(until=time.time() + RAM_REFUSE_SECONDS, reason=reason)
    event("ram_swap_growth", reason)


# RAMLOGIC2: the ladder's state. cache_cap: the prompt cache (MiB) the next start gets after a light or guarded step (None: as
# configured). The wait window opens at the first "wait" and lasts ramWaitSeconds; after it a load is guarded or refused. The
# window is forgotten after a load is admitted, or RAM_WAIT_FORGET_SECONDS after it ended, so a new shortage waits again.
RAM_LADDER = {"cache_cap": None, "state": "idle", "since": 0.0, "until": 0.0, "free_mib": None, "need_mib": None,
              "reserve_mib": None, "reason": ""}
RAM_WAIT_FORGET_SECONDS = 120


def ram_desk_reserve():
    """The desktop's RAM reserve now: the configured ramDeskReserveMiB, else from desk_need (ordinary without it)."""
    if G.get("ramDeskReserveMiB") is not None:
        return int(G["ramDeskReserveMiB"])
    need = getattr(getattr(DESK, "need", None), "need", None)
    return ram_budget.desk_reserve_for(need) if isinstance(need, (int, float)) else ram_budget.DEFAULT_DESK_RESERVE_MIB


def ram_ladder_snapshot():
    return {"policy": G.get("ramPolicy", "off"), "reserve_mib": ram_desk_reserve(), **{k: RAM_LADDER[k] for k in
            ("state", "since", "until", "free_mib", "need_mib", "cache_cap", "reason")}}


RAM_RECLAIM = {}                  # rate limit per caller ("admission"): one reclaim does not use up another's turn
RAM_RECLAIM_MIN_INTERVAL = 10.0


def ram_try_reclaim(shortfall_mib, mem, why, proc_root="/proc", key="admission"):
    """Give clean page cache back (memory.reclaim, file pages only) when it could cover the shortfall. At most once per 10 s per
    caller, behind ramReclaimCache. Honest free counts the inactive file cache by half, so reclaiming X raises it by about X/2: the
    request is twice the shortfall plus a margin, never more than the reclaimable cache (the reclaim takes the whole user cgroup's
    cache, Brave's and VS Code's too, so it asks only for what the shortfall needs). Returns True when a reclaim ran (the caller
    re-reads memory and plans again)."""
    if G.get("ramReclaimCache", True) is not True or shortfall_mib <= 0:
        return False
    reclaimable = cache_reclaim.reclaimable_mib(mem)
    if reclaimable < shortfall_mib:
        return False
    now = time.monotonic()
    if now - RAM_RECLAIM.get(key, -1e9) < RAM_RECLAIM_MIN_INTERVAL:
        return False
    RAM_RECLAIM[key] = now
    want = min(2 * shortfall_mib + cache_reclaim.REPORT_MARGIN_MIB, reclaimable)
    freed, error = cache_reclaim.reclaim_page_cache(want, proc_root)
    took = time.monotonic() - now
    if error:
        event("ram_cache_reclaim", f"{why}: not done ({error})")
        return False
    event("ram_cache_reclaim", f"{why}: freed {freed} MiB of page cache in {took:.1f} s")
    return True


def ram_admission(mc, reclaim_gb=0.0):
    """RAMGUARD1 (R0) + RAMLOGIC2: a load must leave the desktop room to grow (the reserve follows desk_need) once the server's
    un-evictable part is counted. Honest free = MemFree + half of the inactive file cache, minus the swap penalty (not
    MemAvailable). When it does not fit, the ladder: a smaller prompt cache in RAM, a short wait reported in the status
    ("ram_wait" refusals are waited out by acquire_model), then a guarded load above ramHardMinMiB, else an honest refusal with
    the numbers. Nobody is asked to close anything."""
    if time.time() < RAM_SWAP_REFUSAL["until"]:
        event("ram_admission_refused", RAM_SWAP_REFUSAL["reason"])
        return Refusal(f"Model load deferred: RAM {RAM_SWAP_REFUSAL['reason']}", "ram")
    now = time.time()
    if RAM_LADDER["until"] and now - RAM_LADDER["until"] > RAM_WAIT_FORGET_SECONDS:
        RAM_LADDER.update(since=0.0, until=0.0)
    wait_seconds = int(G.get("ramWaitSeconds", 20))
    waited_out = wait_seconds <= 0 or (RAM_LADDER["until"] > 0 and now >= RAM_LADDER["until"])
    settle_pause_leftovers("before a model load")      # RAMFIX1: a pause checkpoint nobody resumes never counts as used memory
    cache_mib = _ram_cache_budget(mc, ladder=False)[0]
    resident = ram_budget.resident_fixed_estimate_mib(mc.get("nCpuMoe", 0), cache_mib)
    reserve = ram_desk_reserve()
    hard_min = int(G.get("ramHardMinMiB", ram_budget.RAM_HARD_MIN_MIB))
    meminfo = ram_budget.read_meminfo()
    pressure = ram_budget.read_pressure()
    plan = ram_budget.plan_load(meminfo, pressure, resident, cache_mib, 0.0, reserve, waited_out, hard_min, reclaim_gb * 1024)
    if plan["step"] != "load" and meminfo is not None and pressure is not None:
        # clean page cache is not freed until somebody allocates: the shortfall to the configured start is covered from it here
        shortfall = resident + reserve - plan["free_mib"]
        if ram_try_reclaim(shortfall, meminfo, f"shortfall {int(shortfall)} MiB for the model"):
            meminfo = ram_budget.read_meminfo()
            plan = ram_budget.plan_load(meminfo, pressure, resident, cache_mib, 0.0, reserve, waited_out, hard_min, reclaim_gb * 1024)
    # RAMFIX1: next the gateway's own files in RAM (a saved session moves to the disk or is dropped), before any wait or refusal;
    # also when the load fits the budget but would leave the desktop less than ramDeskFloorGB of MemAvailable
    deficit = resident + reserve - plan["free_mib"]
    if plan["step"] == "load" and meminfo is not None and meminfo.get("MemAvailable") is not None:
        deficit = float(G.get("ramDeskFloorGB", 4.0)) * 1024 - (meminfo["MemAvailable"] - resident - cache_mib)
    if meminfo is not None and deficit > 0 and G.get("shmSpill") is not False and release_own_tmpfs(deficit) > 0:
        plan = ram_budget.plan_load(ram_budget.read_meminfo(), ram_budget.read_pressure(), resident, cache_mib, 0.0, reserve,
                                    waited_out, hard_min, reclaim_gb * 1024)
    step = plan["step"]
    RAM_LADDER.update(free_mib=plan["free_mib"], need_mib=plan["need_mib"], reserve_mib=reserve, reason=plan["reason"])
    if step in ("load", "light", "guarded"):
        RAM_LADDER.update(cache_cap=None if step == "load" else plan["cache_ram_mib"], state="idle" if step == "load" else step)
        if step != "guarded":
            RAM_LADDER.update(since=0.0, until=0.0)     # a guarded load keeps the spent window: start_llama asks again at once
        if step != "load":
            event("ram_ladder", f"{step}: {plan['reason']}")
        return ""
    if step == "wait":
        if not RAM_LADDER["until"]:
            RAM_LADDER.update(since=now, until=now + wait_seconds)
            event("ram_wait", f"{plan['reason']} (up to {wait_seconds} s)")
        RAM_LADDER["state"] = "waiting"
        return Refusal(f"Model load waiting: RAM {plan['reason']}", "ram_wait")
    RAM_LADDER["state"] = "refused"
    event("ram_admission_refused", plan["reason"])
    return Refusal(f"Model load deferred: RAM {plan['reason']}", "ram")


class Refusal(str):
    """GWSTALL1: a load refusal is the text for the client and the log plus a code for the gateway's own decisions (which refusals
    are worth a wait, which leave a loaded layout answering). Everywhere else it behaves as the plain string."""

    def __new__(cls, text, code):
        value = super().__new__(cls, text)
        value.code = code
        return value


def refusal_code(error):
    return getattr(error, "code", None)


WAIT_FOR_VRAM_CODES = frozenset({"vram", "vram_governor", "telemetry"})    # acquire_model waits vramReloadWaitSeconds for these
RESOURCE_CODES = WAIT_FOR_VRAM_CODES | {"vram_reserve", "ram", "ram_wait", "ram_valve", "pressure"}   # memory, not a mode: a loaded layout answers
VRAM_RETRY_AFTER_SECONDS = 5


def preload_check(mc, reclaim=(0.0, 0.0), pressure_gate=True, admit=None):
    if st.override == "OFF":
        return Refusal("AI is manually disabled (ai-off)", "ai_off")
    honest = G.get("ramPolicy") == "honest"
    if honest:
        ram_refusal = ram_admission(mc, reclaim[0])
        if ram_refusal:
            return ram_refusal
    if st.multimedia_lock:
        return Refusal("GPU is busy with an audio/video job", "multimedia")
    ram = mem_available_gb() + reclaim[0]
    if not honest and ram < G["ramFreeMinGBToLoad"]:     # RAMLOGIC2: under the honest policy the ladder decides, not MemAvailable
        return Refusal(f"Not enough free RAM to load the model ({ram:.1f} GB available, need {G['ramFreeMinGBToLoad']} GB)", "ram")
    if G.get("ramGuard") is True and RAM_VALVE.level in ("pressing", "critical"):
        return Refusal(f"Model load deferred: system memory is {RAM_VALVE.level} (RAM valve). "
                       "The model returns when memory eases; retry shortly.", "ram_valve")
    if pressure_gate:
        pressure = memory_load_pressure_check()
        if pressure:
            return Refusal(pressure, "pressure")
    sample_age = time.time() - st.vram_sample_at if st.vram_sample_at is not None else math.inf
    if sample_age > G.get("vramTelemetryStaleSeconds", 5) or st.monitor_error:
        message = "Fresh NVML admission data unavailable; wait for a fresh VRAM sample"
        if st.monitor_error != message:
            st.monitor_error = message
            event("telemetry_failed", message, fallback=True)
        reset_pressure()
        return Refusal(message, "telemetry")
    vram = (st.vram_free_mib or 0) / 1024 + reclaim[1]
    if governor_on() and admit is not None and governor_budget().fixed_learned.get(admit) is not None:
        free_mib = int((st.vram_free_mib or 0) + reclaim[1] * 1024)
        ok, short, _source = governor_budget().admission(admit[0], admit[1], free_before_start_mib=free_mib)
        if not ok:
            return Refusal(f"Not enough free VRAM for {admit[0]}/{admit[1]} under the governor ({short} MiB short)", "vram_governor")
        st.preload_vram_gb = vram
        reset_load_pressure_refusals()
        return ""
    required = max(mc["vramFreeMinGBToLoad"], st.vram_resume_min_gb)
    if vram < required:
        return Refusal(f"Not enough free VRAM ({vram:.1f} GB free, need {required:.2f} GB)", "vram")
    st.preload_vram_gb = vram
    reset_load_pressure_refusals()
    return ""


def tmpfs_parent(path):
    """The resolved directory, if it lies on tmpfs (RAM)."""
    parent = Path(path).resolve(strict=True)
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        mount = Path(left.split()[4].replace("\\040", " "))
        if parent == mount or mount in parent.parents:
            mounts.append((len(str(mount)), right.split()[0]))
    if not mounts or max(mounts)[1] != "tmpfs":
        raise RuntimeError("vramPressureSlotDir must be on tmpfs (RAM)")
    return parent


def private_slot_root(parent):
    identity = hashlib.sha256(str(STATE_DIR.resolve()).encode()).hexdigest()[:16]
    root = parent / f"shura-slots-{os.getuid()}-{identity}"
    if root.is_symlink():
        raise RuntimeError("RAM slot directory must not be a symlink")
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError("RAM slot directory must be private and owned by this user")
    return root


def prepare_ram_slot_dir():
    """A private verified tmpfs path. Never silently write emergency sessions to SSD."""
    global RAM_SLOT_DIR
    if RAM_SLOT_DIR is not None:
        return RAM_SLOT_DIR
    parent = tmpfs_parent(G["vramPressureSlotDir"])
    try:
        root = private_slot_root(parent)
    except RuntimeError as refused:
        # Another user on this machine can pre-create the predictable name in a shared tmpfs and so switch the feature off. The
        # per-user runtime directory (0700, tmpfs) cannot be occupied by anyone else: use it instead of giving up.
        runtime = os.environ.get("XDG_RUNTIME_DIR")
        if not runtime or "symlink" in str(refused):
            raise
        try:
            root = private_slot_root(tmpfs_parent(runtime))
        except (OSError, RuntimeError):
            raise refused from None
        event("ram_path_moved", f"{parent} is not usable ({refused}); using {root.parent}", fallback=True)
    RAM_SLOT_DIR = root
    for leftover in root.glob(".out-*"):       # RAMFIX1: a release interrupted by a kill leaves its staging file behind
        leftover.unlink(missing_ok=True)
    try:
        recover_orphan(root)
    except (OSError, RuntimeError, ValueError) as e:
        RAM_SLOT_DIR = None
        st.backend_blocked = str(e)
        raise RuntimeError(str(e)) from e
    st.backend_blocked = ""
    try:
        recover_gear_transaction(root)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        st.backend_blocked = f"Gear recovery refused: {error}"
        RAM_SLOT_DIR = None
        raise RuntimeError(st.backend_blocked) from error
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
    (root / "restore.bin").unlink(missing_ok=True)
    return RAM_SLOT_DIR


KV_TYPE_FLAGS = {"-ctk": "k", "--cache-type-k": "k", "-ctv": "v", "--cache-type-v": "v",
                 "-ctkd": "kd", "--cache-type-k-draft": "kd", "-ctvd": "vd", "--cache-type-v-draft": "vd"}


def kv_identity(mc):
    """DEPLOY3: the KV cache types a layout's server really uses (the config fields, then any -ctk/-ctv/-ctkd/-ctvd in extraArgs;
    the last flag wins, as in llama-server). A saved session holds KV in these types and cannot be read with others."""
    kv = {"k": mc.get("cacheTypeK"), "v": mc.get("cacheTypeV"), "kd": None, "vd": None}
    extra = mc.get("extraArgs") or []
    for i, arg in enumerate(extra[:-1]):
        if arg in KV_TYPE_FLAGS:
            kv[KV_TYPE_FLAGS[arg]] = extra[i + 1]
    return kv


def checkpoint_signature(model_id, vision, layout=None):
    # DEPLOY3: the layout's KV types are part of the identity. Layouts with the same KV still share a session (LAYOUT1); a layout with
    # other KV types (the long 262k profile keeps K/V in q5_0) declines the other layout's session instead of failing to read it.
    if layout is None:
        layout = st.layout if (st.model_id == model_id and st.vision == vision and st.layout) else "short"
    kv = kv_identity(model_config(model_id, vision, layout))
    model = model_config(model_id, vision)
    if M["models"][model_id].get("longContextProfile"):
        for key in LAYOUT_NEUTRAL_KEYS:  # a snapshot saved in one layout restores in the other (LAYOUT1)
            model.pop(key, None)
    # CPU execution scheduling cannot change serialized KV/draft state. Keeping these
    # out of A2 identity lets a cold restart switch poll/thread tuning and restore the
    # exact same session without weakening checks on model geometry or sampling layout.
    ignored = ["poll", "threads", "threadsBatch"]
    if G.get("vramValve") or vram_card_reserve_enabled():
        ignored.append("moeCacheSlots")  # cache slots do not change serialized KV/draft tensors
    for key in ignored:
        model.pop(key, None)
    exe = exe_for(model, check=False)
    value = {"model": model, "exe": exe, "kv": kv}
    for key, path in (("exe_stat", exe), ("model_stat", value["model"]["modelPath"])):
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
            or saved["tokens"] > max(model_config(saved["model"], saved["vision"], lay)["ctxSize"] for lay in ("short", "long"))):
        raise ValueError("checkpoint metadata is invalid")
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint file is missing or unsafe")
    if saved["tokens"] < 1:
        raise ValueError("checkpoint token count is invalid")
    actual_bytes = path.stat().st_size
    if actual_bytes != saved["bytes"]:
        raise ValueError(f"checkpoint size mismatch: metadata={saved['bytes']} file={actual_bytes}")
    if file_hash(path) != saved["sha256"]:
        raise ValueError("checkpoint SHA-256 mismatch")


def checkpoint_metadata_path(path):
    path = Path(path)
    return path.with_name(path.name + ".json")


def validate_saved_slot(path, expected_model, expected_vision):
    metadata = checkpoint_metadata_path(path)
    if metadata.is_symlink() or not metadata.is_file() or metadata.stat().st_size > 16384:
        raise ValueError("slot metadata is missing or unsafe")
    saved = json.loads(metadata.read_text())
    if (saved.get("model") != expected_model or saved.get("vision") is not expected_vision
            or saved.get("signature") != checkpoint_signature(expected_model, expected_vision)):
        raise ValueError("slot metadata does not match the requested model and mode")
    validate_checkpoint(path, saved)
    return saved


def write_saved_slot_metadata(path, saved):
    metadata = checkpoint_metadata_path(path)
    temporary = metadata.with_name(metadata.name + ".tmp")
    temporary.write_text(json.dumps(saved, separators=(",", ":")))
    os.replace(temporary, metadata)


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


RAM_VALVE = ram_valve.RamValve(G)
RAM_REDUCE = {"on": False}      # set at pressing: the next start uses half the cache budget
RAM_LAST = {"len": 0}           # size of the last accepted chat body, to judge how much a new request adds
SESSION = {"id": None, "prefix": "", "saved_tokens": 0, "next_try": 0.0}   # current session id from the X-Shura-Session header (session cache)
RAM_NEXT = {"at": 0.0}          # after a refused yield, the next attempt waits (no event every second)


def ram_snapshot_mib():
    """Size of the RAM session snapshot (one per model, in the private tmpfs folder), in MiB."""
    root = RAM_SLOT_DIR
    if root is None:
        return 0.0
    try:
        return (root / "active.bin").stat().st_size / 1048576.0
    except OSError:
        return 0.0


# --- RAMFIX1: the gateway's own files in RAM ---------------------------------------------------------------------------------------
# On 2026-11-16 the live gateway refused the model ("RAM not enough memory") because of its own idle-unload snapshot: 2 GiB in
# /dev/shm (slotSaveDir) that honest free counts as used. Now a load first takes such files back (ram_budget.release_order): copies
# in the session catalogue are evicted, saved sessions and the RAM checkpoint are moved to a disk folder and restored from there.
# And a large ordinary save goes to the disk folder at once when a reload with it in RAM would not fit.

def is_tmpfs(path):
    """True when the path (or its nearest existing parent) lies on tmpfs (RAM)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        tmpfs_parent(p)
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def spill_dir():
    configured = G.get("slotSpillDir")
    return Path(os.path.expanduser(configured)) if configured else STATE_DIR / "slots-spill"


def ready_spill_dir():
    """The disk spill folder, created private; refused when it is in RAM (moving there would free nothing)."""
    folder = spill_dir()
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    if folder.is_symlink() or not folder.is_dir() or folder.stat().st_uid != os.getuid():
        raise RuntimeError(f"spill folder {folder} is not a private directory")
    if is_tmpfs(folder):
        raise RuntimeError(f"spill folder {folder} is in RAM (tmpfs)")
    return folder


def usable_file(path):
    path = Path(path)
    return path.is_file() and not path.is_symlink()


def saved_slot_path(model_id, vision):
    """Where the saved session of a model lies: the slot folder, else the disk spill folder; the slot folder path when neither has it."""
    name = slot_filename(model_id, vision)
    primary = SLOT_DIR / name
    if usable_file(primary):
        return primary
    spilled = spill_dir() / name
    return spilled if usable_file(spilled) else primary


def drop_other_slot_copies(name, keep):
    """A saved session was just written at `keep`: an older copy under the same name in the other folder is stale."""
    for folder in (SLOT_DIR, spill_dir()):
        other = folder / name
        if other != Path(keep) and (other.exists() or checkpoint_metadata_path(other).exists()):
            discard_slot_file(other)


def spill_copy(src, dst):
    """Copy src (in RAM) to dst on the disk atomically; fsync, then drop the written pages from the page cache, so the copy does not
    stay in RAM as file cache."""
    dst = Path(dst)
    tmp = dst.with_name(f".spill-{dst.name}.tmp")
    try:
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            shutil.copyfileobj(fin, fout, 16 * 1048576)
            fout.flush()
            os.fsync(fout.fileno())
            try:
                os.posix_fadvise(fout.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def own_tmpfs_entries():
    """The gateway's own files in RAM that a load may take back: saved sessions in a tmpfs slot folder, the session catalogue when
    it is in tmpfs, and the RAM checkpoint of a pressure unload while no server runs."""
    entries = []

    def add(kind, path, **extra):
        try:
            info = Path(path).lstat()
        except OSError:
            return
        if not stat.S_ISREG(info.st_mode):
            return
        entries.append({"kind": kind, "path": Path(path), "mib": info.st_blocks * 512 / 1048576.0, "bytes": info.st_size,
                        "inode": (info.st_dev, info.st_ino), "mtime": info.st_mtime, **extra})

    try:
        if is_tmpfs(SLOT_DIR) and SLOT_DIR.is_dir():
            for path in SLOT_DIR.glob("*.bin"):
                if usable_file(path) and checkpoint_metadata_path(path).is_file():
                    add("slot", path)
        if session_cache_on():
            catalog = session_catalog()
            if is_tmpfs(catalog.root):
                for meta in catalog._entries():
                    add("cache", catalog.root / f"{meta['session_id']}.bin", session_id=meta["session_id"], catalog=catalog)
        checkpoint = st.ram_checkpoint
        if checkpoint and RAM_SLOT_DIR is not None and not (st.proc and st.proc.poll() is None):
            add("checkpoint", checkpoint["path"], checkpoint=checkpoint)
    except OSError as error:
        event("ram_spill_failed", f"own RAM files not listed: {error}", fallback=True)
    return entries


def prefill_tps():
    """The prefill speed to judge a re-read by: the latest measured long prompt (at least 4000 new tokens) in the monitor's
    request history, else spillPrefillTps (PERF262: about 1200 tok/s at 100k on the 262k mode)."""
    for request in MONITOR.requests():
        rate, total, cached = request.get("prompt_per_second"), request.get("prompt_tokens"), request.get("cached_tokens") or 0
        if (isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0 and isinstance(total, int)
                and total - cached >= 4000):
            return float(rate)
    return float(G.get("spillPrefillTps", SPILL_DEFAULTS["spillPrefillTps"]))


def worth_the_disk(tokens):
    """(write it, seconds a re-read would take): a session is written to the SSD only when re-reading its prompt would take longer
    than spillMinReprefillSeconds; a shorter one is cheaper to re-read than to wear the SSD for."""
    try:
        seconds = max(0, int(tokens)) / prefill_tps()
    except (TypeError, ValueError, ZeroDivisionError):
        return False, 0.0
    return seconds > float(G.get("spillMinReprefillSeconds", 60)), seconds


def take_out(group):
    """Phase 1 of releasing one group of hard links (ram_budget.release_order), under st.lock and instant: the saved session or the RAM
    checkpoint is renamed to a private staging name in its own folder (no listing, restore or save sees it any more), the other links
    go. Returns the job for finish_release; the RAM is given back when finish_release removes the staging file."""
    main = next((e for e in group if e["kind"] == "checkpoint"), None) or next((e for e in group if e["kind"] == "slot"), None)
    job = {"what": "", "staging": None, "saved": None, "name": "", "mib": group[0]["mib"], "bytes": group[0]["bytes"]}
    if main is not None:
        if main["kind"] == "checkpoint":
            saved = {k: v for k, v in main["checkpoint"].items() if k != "path"}
            name = slot_filename(saved["model"], saved["vision"])
        else:
            saved = json.loads(checkpoint_metadata_path(main["path"]).read_text())
            name = main["path"].name
        staging = main["path"].with_name(f".out-{name}")
        os.replace(main["path"], staging)
        job.update(staging=staging, saved=saved, name=name, mib=main["mib"], bytes=main["bytes"], checkpoint=main["kind"] == "checkpoint")
    for entry in group:
        if entry["kind"] == "slot" and entry is not main:
            discard_slot_file(entry["path"])
        elif entry["kind"] == "slot":
            checkpoint_metadata_path(entry["path"]).unlink(missing_ok=True)
        elif entry["kind"] == "cache":
            entry["catalog"].remove(entry["session_id"])
        elif entry["kind"] == "checkpoint":
            st.ram_checkpoint = None
            clear_ram_checkpoint()           # the file itself is already at its staging name
    if main is None:
        job["what"] = f"session catalogue copy {group[0]['path'].name} ({group[0]['mib']:.0f} MiB) evicted"
    return job


def finish_release(job):
    """Phase 2, without st.lock: a session worth the disk (worth_the_disk) is copied to the disk spill folder within the daily SSD
    budget (reserved before the write), then, under st.lock again, put in place with its metadata unless a newer save of the same
    session appeared meanwhile; the staging file in RAM goes in every case. Returns what happened."""
    staging, saved, name = job["staging"], job["saved"], job["name"]
    if staging is None:
        return job["what"]
    tmp = None
    try:
        worth, seconds = worth_the_disk(saved.get("tokens", 0))
        if not worth:
            return (f"saved session {name} ({job['mib']:.0f} MiB) dropped: its {saved.get('tokens', 0)} tokens re-read in about "
                    f"{seconds:.0f} s, cheaper than an SSD write")
        if not ssd_write_budget().reserve(job["bytes"]):
            return f"saved session {name} dropped: the daily SSD write limit is reached; the prompt is re-read"
        folder = ready_spill_dir()
        tmp = folder / f".spill-{name}.part"
        spill_copy(staging, tmp)
        with st.lock:
            if usable_file(SLOT_DIR / name) and checkpoint_metadata_path(SLOT_DIR / name).is_file():
                return f"saved session {name} not kept on the disk: a newer save exists"
            write_saved_slot_metadata(folder / name, saved)
            os.replace(tmp, folder / name)
            tmp = None
            if job.get("checkpoint"):
                st.skip_disk_restore = False
        return f"saved session {name} ({job['mib']:.0f} MiB) moved to the disk"
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        staging.unlink(missing_ok=True)


def release_own_tmpfs(deficit_mib):
    """Free at least deficit_mib of the gateway's own RAM files if it has them, cheapest first. Returns the MiB freed. The renames
    run under st.lock, the copies to the disk do not take it (a caller that holds it, the model load, keeps it: the load waits for
    the memory anyway). Never raises."""
    freed = 0.0
    jobs = []
    try:
        with st.lock:
            for group in ram_budget.release_order(own_tmpfs_entries(), deficit_mib):
                try:
                    jobs.append(take_out(group))
                except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as error:
                    event("ram_spill_failed", f"{group[0]['path'].name}: {error}", fallback=True)
        for job in jobs:
            try:
                what = finish_release(job)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as error:
                what = f"saved session {job['name']} dropped: the move to the disk failed ({error})"
                event("ram_spill_failed", what, fallback=True)
            freed += job["mib"]
            event("ram_spill", f"for the memory: {what}")
    except Exception as error:  # noqa: BLE001 - releasing own files must never fail the load or the valve that asked for it
        event("ram_spill_failed", f"own RAM files not released: {error}", fallback=True)
    return freed


OWN_WORK = {"thread": None}


def release_in_background(deficit_mib):
    """For callers that must not wait on a copy (the monitor, a waiting request): one release at a time in its own thread."""
    worker = OWN_WORK["thread"]
    if worker is not None and worker.is_alive():
        return
    OWN_WORK["thread"] = threading.Thread(target=release_own_tmpfs, args=(deficit_mib,), daemon=True, name="shura-own-ram")
    OWN_WORK["thread"].start()


def slot_folder_for(size_bytes, tokens):
    """Where an ordinary save of size_bytes goes: the slot folder, unless it is in RAM, the file is large (shmSpillMinMiB), its
    prompt is worth the disk (worth_the_disk) and a reload of this model with the file in RAM would not fit (the running server's
    memory counted as free, it leaves first); then the disk spill folder, within the daily SSD budget. A session not worth the disk
    stays in RAM: a load that needs the room drops it."""
    mib = size_bytes / 1048576.0
    if (G.get("shmSpill") is False or mib < float(G.get("shmSpillMinMiB", 256)) or not is_tmpfs(SLOT_DIR)
            or not worth_the_disk(tokens)[0]):
        return SLOT_DIR
    try:
        mc = model_config(st.model_id, st.vision, st.layout)
        cache_mib = _ram_cache_budget(mc, ladder=False)[0]
        resident = ram_budget.resident_fixed_estimate_mib(mc.get("nCpuMoe", 0), cache_mib)
        plan = ram_budget.plan_load(ram_budget.read_meminfo(), ram_budget.read_pressure(), resident, cache_mib, mib,
                                    ram_desk_reserve(), credit_mib=running_footprint()[0] * 1024)
        if plan["step"] == "load":
            return SLOT_DIR
        folder = ready_spill_dir()
        if not ssd_write_budget().reserve(size_bytes):        # reserved before the write; the caller writes, it does not record
            return SLOT_DIR
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as error:
        event("ram_spill_failed", f"saved session stays in RAM: {error}", fallback=True)
        return SLOT_DIR
    event("ram_spill", f"saved session ({mib:.0f} MiB) written to the disk: a reload with it in RAM would not fit ({plan['step']})")
    return folder


# --- RAMFIX1: what a governor pause leaves behind ------------------------------------------------------------------------------
# On 2026-11-16 at 14:43 the governor paused an idle server (exit 75, the 89k-token session saved to vram-pause.bin, 2.7 GiB in
# /dev/shm). No answer was running, so nothing resumed it: the gateway counted a crash and the file stayed in RAM through the next
# start. Now an exit 75 is a pause, and a checkpoint nobody resumes becomes the model's saved session when its owner is known (a
# rename inside tmpfs: no copy, no SSD write; restored by the next load, dropped or moved by a load that needs the room); anything
# else of a pause (a server runs again, the owner is unknown, other files) is deleted. Never written to the disk.

PAUSE_STATE = {"resuming": 0, "owner": None, "recent": []}
PAUSE_PREFIXES = ("vram-pause", "vram-exit")
PAUSE_REPEAT = (3, 300.0)        # this many pauses within this many seconds: a server that keeps pausing is failing, count it


def pause_marker_present():
    root = SERVER_SLOT["dir"] if SERVER_SLOT["dir"] is not None else RAM_SLOT_DIR
    return root is not None and any((Path(root) / name).is_file() for name in ("vram-exit.json", "vram-pause.bin"))


def paused_exit(proc):
    """An exit 75 with the governor on and its marker (vram-exit.json or vram-pause.bin) written is a pause, not a crash; it records
    the owner of the checkpoint (the server that wrote it). A server that pauses again and again counts as failing after all."""
    if proc is None or proc.poll() != 75 or not governor_on() or not pause_marker_present():
        return False
    now = time.monotonic()
    count, window = PAUSE_REPEAT
    PAUSE_STATE["recent"] = [t for t in PAUSE_STATE["recent"] if now - t < window] + [now]
    if len(PAUSE_STATE["recent"]) >= count:
        event("vram_pause_repeated", f"{len(PAUSE_STATE['recent'])} pauses within {int(window)} s: counted as a server failure", fallback=True)
        return False
    if proc is st.proc or PAUSE_STATE["owner"] is None:
        PAUSE_STATE["owner"] = SERVER_SLOT.get("owner")
    return True


def pause_leftovers():
    root = SERVER_SLOT["dir"] if SERVER_SLOT["dir"] is not None else RAM_SLOT_DIR
    if root is None:
        return []
    try:
        return [p for p in Path(root).iterdir() if p.name.startswith(PAUSE_PREFIXES) and p.is_file() and not p.is_symlink()]
    except OSError:
        return []


def adopt_pause_checkpoint(path, owner, background=False):
    """The pause checkpoint becomes the saved session of its model in the slot folder, by renames inside the same tmpfs (no copy, no
    write). First to a private name nobody lists; the SHA-256 is computed (without st.lock when background); then the metadata is
    written and only then the file takes its final name. Raises when it cannot; the private file never outlives a failure."""
    model_id, vision, layout = owner
    info = json.loads(path.with_name(path.name + ".tokens.json").read_text())
    tokens = info.get("n_past")
    if type(tokens) is not int or tokens < 1:
        raise ValueError("the tokens file has no token count")
    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    if path.stat().st_dev != SLOT_DIR.stat().st_dev:
        raise RuntimeError("the slot folder is on another file system (a copy would be needed)")
    name = slot_filename(model_id, vision)
    private = SLOT_DIR / f".adopt-{name}"
    os.replace(path, private)

    def finish(reraise=True):
        try:
            digest = file_hash(private)
            with st.lock:
                saved = {"model": model_id, "vision": vision, "tokens": tokens, "bytes": private.stat().st_size, "sha256": digest,
                         "reload_min_gb": st.vram_resume_min_gb, "signature": checkpoint_signature(model_id, vision, layout),
                         "saved_at": private.stat().st_mtime}
                dest = SLOT_DIR / name
                write_saved_slot_metadata(dest, saved)
                os.replace(private, dest)
                drop_other_slot_copies(name, dest)
                checkpoint = st.ram_checkpoint
                if checkpoint and checkpoint.get("model") == model_id and checkpoint.get("vision") == vision:
                    clear_ram_checkpoint()           # older than the pause: restore would prefer it
                st.skip_disk_restore = False
            event("pause_leftover", f"the pause checkpoint kept as the saved session {name} ({tokens} tokens, {saved['bytes'] / 1048576:.0f} MiB)")
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as error:
            event("pause_leftover_failed", f"pause checkpoint not kept: {error}", fallback=True)
            if reraise:
                raise
        finally:
            private.unlink(missing_ok=True)
    if background:
        threading.Thread(target=finish, args=(False,), daemon=True, name="shura-pause-adopt").start()
    else:
        finish()
    return f"kept as the saved session {name} ({tokens} tokens)"


def settle_pause_leftovers(why, background=False):
    """A pause checkpoint that no answer resumes must not stay in RAM: kept as the saved session (adopt_pause_checkpoint) while the
    server that wrote it is gone and its owner was recorded at its exit 75, else deleted; the other pause files are deleted. Nothing
    is written. background: the hash runs in its own thread (the monitor holds st.lock)."""
    if PAUSE_STATE["resuming"]:
        return
    files = pause_leftovers()
    if not files:
        return
    alive = st.proc is not None and st.proc.poll() is None
    if alive:
        # the live check (2026-11-16): a pausing server writes its files while it still runs. Only files older than its start are
        # spent (an earlier pause it was resumed from); newer ones are its own pause in progress and wait for its exit.
        started = st.proc_wall_start or time.time()
        try:
            files = [p for p in files if p.stat().st_mtime < started - 1]
        except OSError:
            return
        if not files:
            return
    owner = PAUSE_STATE.get("owner")
    PAUSE_STATE["owner"] = None
    checkpoints = [p for p in files if not p.name.endswith(".json")]
    kept = ""
    if not alive and owner and len(checkpoints) == 1 and G.get("shmSpill") is not False:
        try:
            kept = adopt_pause_checkpoint(checkpoints[0], owner, background)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as error:
            event("pause_leftover_failed", f"pause checkpoint not kept: {error}", fallback=True)
    dropped = 0.0
    for path in files:
        try:
            if path.exists():
                dropped += path.stat().st_blocks * 512 / 1048576.0
                path.unlink()
        except OSError as error:
            event("pause_leftover_failed", f"{path.name}: {error}", fallback=True)
    reason = "a server runs again" if alive else "its owner is unknown" if not owner else "no answer resumes it"
    if dropped >= 1 or not kept:
        event("pause_leftover", f"{why}: {dropped:.0f} MiB of pause files deleted" + ("" if kept else f" ({reason})"))


def ram_valve_yield(level):
    """Keep the session in RAM and unload the model (caller holds st.lock). Restores on the next request."""
    if not save_slot_locked(ram_only=True):
        RAM_NEXT["at"] = time.monotonic() + 30
        event("ram_valve_yield_refused", f"level={level}: session not saved to RAM; the model stays loaded, retry in 30 s", fallback=True)
        return False
    if stop_llama_locked(f"ram_valve_{level}", save=False):
        st.last_yield_reason = f"ram_{level}"
        st.last_yield_at = time.time()
        event("ram_valve_yield", f"level={level}: model unloaded, session kept in RAM")
        return True
    return False


def prompt_growth_gb(new_chars):
    """RAM a prompt of new_chars adds to the server: its context checkpoints in host memory (ram_valve: 63 MiB + 2 KiB per token,
    about 4 characters per token)."""
    return (ram_valve.CHECKPOINT_BASE_MIB + new_chars / 4.0 * ram_valve.CHECKPOINT_PER_TOKEN_KIB / 1024.0) / 1024.0


def room_for_prompt(new_chars, release=True):
    """True when MemAvailable stays at or above ramDeskFloorGB after the prompt's growth and memory pressure (PSI full) and the
    zram fill are below the valve's tight thresholds. MemAvailable already counts the page cache the kernel can drop, so a page cache
    reclaim would not change it: when short, the gateway's own RAM files are released instead (in the background, without st.lock)."""
    floor = float(G.get("ramDeskFloorGB", 4.0))
    need = prompt_growth_gb(new_chars)
    mem_gb, psi, zram = ram_valve.read_readings(G["ramGuardProcRoot"], G["ramGuardSysRoot"])
    calm = ((psi is None or psi < float(G["ramGuardPsiFullTight"]))
            and (zram is None or zram < float(G["ramGuardZramPctTight"])))
    if mem_gb is None or (mem_gb - need >= floor and calm):
        return calm
    if release and G.get("shmSpill") is not False and mem_gb - need < floor:
        release_in_background((floor + need - mem_gb) * 1024)
    return False


def wait_room_for_prompt(new_chars):
    """RAMFIX1: at the valve's tight level a large new prompt no longer gets a 503 at once: it goes on when the desktop's floor holds,
    otherwise the gateway itself waits (up to ramValveDeferWaitSeconds, 60: ShuraCode's own header and chunk timeouts are 300 s) and
    goes on as soon as it holds. The waiting request counts as busy (no idle unload meanwhile). False only when it never held."""
    if room_for_prompt(new_chars):
        return True
    deadline = time.monotonic() + float(G.get("ramValveDeferWaitSeconds", 60))
    event("ram_valve_defer_wait", f"new_chars={new_chars}: MemAvailable below the desktop floor or memory pressure; the gateway waits")
    with st.lock:
        st.busy += 1
    try:
        while time.monotonic() < deadline:
            time.sleep(1.0)
            if room_for_prompt(new_chars):
                event("ram_valve_defer_done", f"new_chars={new_chars}: the desktop keeps its floor now; the prompt goes on")
                return True
        return False
    finally:
        with st.lock:
            st.busy -= 1


def ram_valve_tick(now, idle):
    """RAM valve: read the inputs, move the level (an event on each change) and act. Caller holds st.lock. Only with ramGuard true."""
    mem_gb, psi_full, zram_pct = ram_valve.read_readings(G["ramGuardProcRoot"], G["ramGuardSysRoot"])
    floor = float(G.get("ramDeskFloorGB", 4.0))
    if (mem_gb is not None and mem_gb < floor and G.get("shmSpill") is not False and st.proc and st.proc.poll() is None
            and now >= RAM_NEXT.get("own_at", 0.0)):
        # RAMFIX1: below the desktop's floor while the model is loaded: the gateway's own RAM files go before anything else
        RAM_NEXT["own_at"] = now + 30
        release_in_background((floor - mem_gb) * 1024)      # the copies to the disk must not hold the monitor's st.lock
    level, changed = RAM_VALVE.update(mem_gb, psi_full, zram_pct)
    if changed:
        event("ram_guard_level", f"level={level} mem_available_gb={mem_gb} psi_full_avg10={psi_full} zram_used_percent={zram_pct}")
    if level == "ok":
        if RAM_REDUCE["on"]:
            RAM_REDUCE["on"] = False
            event("ram_valve_reduce_off", "memory eased; the next start uses the normal cache budget")
        return
    # tight only defers large new prompts (in handle_any); pressing and critical unload, pressing only when idle
    if level == "critical" or (level == "pressing" and idle >= G["ramGuardIdleSeconds"]):
        if st.status != "READY" or st.busy or (GAME.enabled and GAME.active) or now < RAM_NEXT["at"]:
            return
        if level == "pressing" and not RAM_REDUCE["on"]:
            RAM_REDUCE["on"] = True
            event("ram_valve_reduce_on", "pressing: the next start uses a smaller cache budget")
        ram_valve_yield(level)

def ssd_write_budget():
    """Daily slot-save write budget from ssdWriteDailyLimitGB (no limit when 0)."""
    # The state file sits beside the slot folder, not in it: slot listings must not see it.
    return ram_valve.SsdDailyBudget(G.get("ssdWriteDailyLimitGB", 0), SLOT_DIR.parent / "ssd-write-budget.json")


SESSION_CACHE_DIR_DEFAULT = Path.home() / ".local" / "state" / "ai-gateway" / "sessions"


def session_cache_on():
    return G.get("sessionCacheEnabled") is True


def session_catalog():
    """The NVMe session catalogue (session_cache.py): the quota is the smaller of sessionCacheQuotaGB and sessionCacheQuotaPct."""
    root = Path(G.get("sessionCacheDir") or SESSION_CACHE_DIR_DEFAULT)
    root.mkdir(parents=True, exist_ok=True)
    try:
        fs = os.statvfs(root)
        total = fs.f_blocks * fs.f_frsize
    except OSError:
        total = 0
    quota = session_cache.quota_bytes(total, G["sessionCacheQuotaGB"], G["sessionCacheQuotaPct"])
    return session_cache.SessionCatalog(root, quota, int(G["sessionCacheReserveGB"] * 1024 ** 3), G["sessionCacheTTLDays"])


def session_persist_on():
    """The SSD level (SESS3) is on only with the session cache and a directory set in sessionPersistDir."""
    return session_cache_on() and bool(G.get("sessionPersistDir"))


def session_persist_catalog():
    """The SSD catalogue: one long session at a time, quota sessionPersistMaxGB, written only at a stop or an idle unload."""
    root = Path(os.path.expanduser(G["sessionPersistDir"]))
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    quota = int(G["sessionPersistMaxGB"] * 1024 ** 3)
    return session_cache.SessionCatalog(root, quota, int(G["sessionCacheReserveGB"] * 1024 ** 3), G["sessionCacheTTLDays"])


def persist_write_budget():
    """Daily SSD write budget of the SSD level (ssdWriteDailyLimitGB). Its state file sits on the SSD beside the catalogue, not in
    /dev/shm, so the count survives a reboot: a reboot is exactly when the SSD level writes."""
    root = Path(os.path.expanduser(G["sessionPersistDir"]))
    return session_cache.DailyWriteBudget(G.get("ssdWriteDailyLimitGB", 0), root.parent / "sessions-persist-budget.json")


def session_persist_on_stop(reason):
    """SSD level (SESS3): after the slot is saved at a gateway stop or an idle unload, copy it to the SSD if the session is long
    (session_cache.persist_should_write), then keep only the newest snapshot. Called with the server still alive and the slot
    already saved. Returns True when a snapshot was written."""
    if not session_persist_on() or SESSION["id"] is None:
        return False
    slot = saved_slot_path(st.model_id, st.vision)
    if not slot.is_file() or slot.is_symlink():
        return False
    try:
        code, slots = llama_call("GET", "/slots")
        tokens = int(slots[0].get("n_prompt_tokens", 0)) if code == 200 and isinstance(slots, list) and slots else 0
        saved = json.loads(checkpoint_metadata_path(slot).read_text())
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        event("session_persist_skipped", f"slot record unreadable: {error}", fallback=True)
        return False
    size = slot.stat().st_size
    catalog = session_persist_catalog()
    write, why = session_cache.persist_should_write(tokens, reason, cfg=G, budget_ok=persist_write_budget().allowed(size),
                                                    disk_ok=catalog.fits_after_eviction(size))
    if not write:
        event("session_persist_skipped", why)
        return False
    tmp = catalog.root / f".{SESSION['id']}.tmp"
    catalog.make_room(size, keep=SESSION["id"])
    try:
        subprocess.run(session_cache.idle_copy_command(slot, tmp), check=True, timeout=900)
    except (OSError, subprocess.SubprocessError) as error:
        tmp.unlink(missing_ok=True)
        event("session_persist_failed", f"copy to the SSD failed: {error}", fallback=True)
        return False
    ok, why = catalog.adopt_copied(SESSION["id"], session_model_key(), tokens, SESSION.get("prefix", ""), saved, tmp, now=time.time())
    if not ok:
        tmp.unlink(missing_ok=True)
        event("session_persist_failed", why, fallback=True)
        return False
    persist_write_budget().record(size)
    removed = catalog.keep_newest(G["sessionPersistKeep"], keep_id=SESSION["id"])
    event("session_persist_saved", f"session {SESSION['id'][:8]}: {tokens} tokens, {size // 1048576} MiB to the SSD ({reason}); "
          f"removed {len(removed)} older SSD snapshot(s)")
    return True


def io_pressure_some_avg10():
    """PSI I/O `some avg10` in percent; None when the kernel does not report it."""
    try:
        for line in Path("/proc/pressure/io").read_text().splitlines():
            if line.startswith("some "):
                for part in line.split():
                    if part.startswith("avg10="):
                        return float(part.split("=", 1)[1])
    except (OSError, ValueError):
        pass
    return None


def session_model_key(model_id=None, vision=None):
    """Compatibility key: the model file identity (path, size, mtime), mode, chat template file and KV types.
    The model of the request is passed in before the model is loaded; afterwards the loaded one is used."""
    model_id = st.model_id if model_id is None else model_id
    vision = st.vision if vision is None else vision
    mc = model_config(model_id, vision)
    try:
        info = os.stat(mc["modelPath"])
        model_identity = f"{mc['modelPath']}|{info.st_size}|{int(info.st_mtime)}"
    except (OSError, KeyError):
        model_identity = str(st.model_id)
    template = str(mc.get("chatTemplateFile") or M.get("chatTemplateFile") or "")
    return session_cache.compatibility_key(model_identity, vision, template, [mc.get("cacheTypeK"), mc.get("cacheTypeV")])


def session_cache_checkpoints_ok():
    """A snapshot without context checkpoints restores as 'legacy': the server then re-reads the whole prompt (measured in SESS2c:
    38 s instead of 3.5 s), so the session cache needs slotSaveCheckpoints > 0 to be worth anything."""
    return int(G.get("slotSaveCheckpoints", 0)) > 0


def session_write_behind(now, idle):
    """Write-behind, only when idle and quiet (session_cache.should_write). The slot is saved to the SSD slot file, then copied to the
    catalogue in the idle I/O class. Never during work; never for an emergency save."""
    if not session_cache_on() or SESSION["id"] is None or st.status != "READY" or st.busy or now < SESSION["next_try"]:
        return
    if not session_cache_checkpoints_ok():
        SESSION["next_try"] = now + 300
        event("session_cache_needs_checkpoints", "sessionCacheEnabled needs slotSaveCheckpoints > 0; nothing written", fallback=True)
        return
    try:
        code, slots = llama_call("GET", "/slots")
        tokens = int(slots[0].get("n_prompt_tokens", 0)) if code == 200 and isinstance(slots, list) and slots else 0
    except Exception as error:
        event("session_cache_probe_failed", f"slot probe failed: {error}", fallback=True)
        return
    growth = tokens - SESSION["saved_tokens"]
    estimate = int(63 * 1024 ** 2 + 8 * 1024 * max(tokens, 0))
    catalog = session_catalog()
    game = bool(GAME.enabled and GAME.active) or bool(st.multimedia_lock)
    ram_level = RAM_VALVE.level if G.get("ramGuard") is True else "ok"
    io = io_pressure_some_avg10()
    write, reason = session_cache.should_write(idle, growth, tokens, io if io is not None else 100.0, game, ram_level, cfg=G,
                                               budget_ok=ssd_write_budget().allowed(estimate), disk_ok=catalog.fits_after_eviction(estimate))
    if not write:
        return
    name = slot_filename(st.model_id, st.vision)
    before = {folder: (checkpoint_metadata_path(folder / name).stat().st_mtime_ns if checkpoint_metadata_path(folder / name).exists() else 0)
              for folder in (SLOT_DIR, spill_dir())}
    if not save_slot_locked():
        SESSION["next_try"] = now + 60
        event("session_cache_save_refused", f"slot save did not complete: {st.last_error or 'see the checkpoint events'}", fallback=True)
        return
    # The ordinary save can return success without writing (below its own slotSaveMinTokens): only a new record counts.
    try:
        slot = saved_slot_path(st.model_id, st.vision)     # RAMFIX1: the save may have gone to the disk spill folder
        meta_path = checkpoint_metadata_path(slot)
        if meta_path.stat().st_mtime_ns == before.get(slot.parent, 0):
            SESSION["next_try"] = now + 60
            event("session_cache_save_refused", "slot save wrote no new record (below slotSaveMinTokens)")
            return
        saved = json.loads(meta_path.read_text())
    except (OSError, ValueError) as error:
        SESSION["next_try"] = now + 60
        event("session_cache_save_refused", f"slot metadata unreadable: {error}", fallback=True)
        return
    tmp = catalog.root / f".{SESSION['id']}.tmp"
    # Quota before the copy: evict the least recently used snapshots (a full quota must not freeze the catalogue until TTL).
    catalog.make_room(slot.stat().st_size if slot.exists() else estimate, keep=SESSION["id"])
    try:
        subprocess.run(session_cache.idle_copy_command(slot, tmp), check=True, timeout=900)
    except (OSError, subprocess.SubprocessError):
        tmp.unlink(missing_ok=True)
        SESSION["next_try"] = now + 60
        return
    size = tmp.stat().st_size if tmp.exists() else 0
    ok, why = catalog.adopt_copied(SESSION["id"], session_model_key(), tokens, SESSION.get("prefix", ""), saved, tmp, now=time.time())
    if ok:
        ssd_write_budget().record(size)
        removed = catalog.prune(now=time.time())             # TTL and quota after every new snapshot
        SESSION["saved_tokens"] = tokens
        st.slot_session = SESSION["id"]
        event("session_cache_saved", f"session {SESSION['id'][:8]}: {tokens} tokens, {size // 1048576} MiB ({reason})")
        if removed:
            event("session_cache_evicted", f"removed {len(removed)} old session snapshot(s) (TTL or quota)")
    else:
        tmp.unlink(missing_ok=True)
        SESSION["next_try"] = now + 60
        event("session_cache_save_refused", why, fallback=True)


def session_prepare_restore(model_id, vision):
    """Before a model load: if this session has a catalogue snapshot and restoring it is much cheaper than recomputing, place it where the
    disk restore already reads it. Otherwise the prompt is re-read (no change). Off unless sessionCacheEnabled."""
    if not session_cache_on() or SESSION["id"] is None or st.status == "READY" or st.ram_checkpoint is not None:
        return
    if not session_cache_checkpoints_ok():
        return
    if st.slot_session == SESSION["id"]:
        return
    catalog = session_catalog()
    entry = catalog.find(SESSION["id"], session_model_key(model_id, vision))
    if entry is None and session_persist_on():
        # SESS3: after a reboot the RAM catalogue is empty; the SSD copy is the second place to look.
        catalog = session_persist_catalog()
        entry = catalog.find(SESSION["id"], session_model_key(model_id, vision))
    if entry is None or not isinstance(entry.get("saved"), dict):
        return
    restore, t_restore, t_recompute = session_cache.should_restore(
        entry["size"], G["sessionCacheReadGBps"] * 1e9, G["sessionCacheLoadGBps"] * 1e9, G["sessionCachePrefillTps"],
        entry["tokens"], G["sessionCacheRestoreRatio"])
    if not restore:
        event("session_cache_recompute", f"restore {t_restore:.1f} s is not much cheaper than recompute {t_recompute:.1f} s; prompt re-read")
        return
    dst = SLOT_DIR / slot_filename(model_id, vision)
    try:
        # RAMFIX1: a hard link when the catalogue shares the file system (both in /dev/shm): no second copy of the session in RAM
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.link(entry["path"], dst)
        except OSError:
            shutil.copyfile(entry["path"], dst)
        write_saved_slot_metadata(dst, entry["saved"])
        drop_other_slot_copies(dst.name, dst)
    except OSError as error:
        event("session_cache_restore_failed", str(error), fallback=True)
        return
    st.skip_disk_restore = False
    st.slot_session = SESSION["id"]
    SESSION["saved_tokens"] = int(entry["tokens"])
    catalog.touch(SESSION["id"])
    event("session_cache_restore", f"session {SESSION['id'][:8]}: {entry['tokens']} tokens placed for the disk restore")


def warm_refusal():
    """Why the warm route must not load the model now (the valves decide), or '' when it may."""
    if GAME.enabled and GAME.active:
        return "a game is running"
    if st.multimedia_lock:
        return "an audio or video job holds the GPU"
    if G.get("ramGuard") is True and RAM_VALVE.level != "ok":
        return f"system memory is {RAM_VALVE.level} (RAM valve)"
    if st.vram_pressure:
        return "the desktop needs VRAM"
    return ""


def save_slot_locked(ram_only=False, min_tokens=None):
    """Pressure saves are atomic in RAM; ordinary saves retain the original SSD policy."""
    root = None
    target = None
    try:
        code, slots = llama_call("GET", "/slots")
        if code != 200 or not isinstance(slots, list) or not slots:
            raise RuntimeError("unable to inspect active slot")
        n = slots[0].get("n_prompt_tokens", 0)
        if ram_only and min_tokens is not None and n < min_tokens:
            clear_ram_checkpoint()
            st.skip_disk_restore = True
            event("layout_checkpoint_skipped", f"active context {n} tokens is below layout switch minimum {min_tokens}; next layout starts empty")
            return True
        if n == 0:
            if ram_only:
                # A restored backend may report no newly prompted tokens before the
                # first continuation. Keep its verified disk A2 available for crash
                # recovery instead of marking the next process as empty.
                if st.ram_checkpoint is not None:
                    st.skip_disk_restore = False
                else:
                    disk = saved_slot_path(st.model_id, st.vision)
                    if disk.is_file() and not disk.is_symlink():
                        st.skip_disk_restore = False
                    else:
                        clear_ram_checkpoint()
                        st.skip_disk_restore = True
            return True
        if not ram_only and n < G["slotSaveMinTokens"]:
            return True
        if not ram_only and not st.slot_dirty:
            existing = saved_slot_path(st.model_id, st.vision)
            if existing.is_file() and not existing.is_symlink():
                log("Slot unchanged since it was restored; not written to disk again")
                return True
        if not st.ram_backend:
            if ram_only:
                raise RuntimeError("server has no RAM slot backend")
            name = slot_filename(st.model_id, st.vision)
            target = SLOT_DIR / f"{name}.pending"
            # Daily SSD write budget (ssdWriteDailyLimitGB, 0 = no limit): over the limit the save is skipped and logged;
            # the session is simply re-read later. The estimate is the previous file's size (0 when there is none).
            previous = SLOT_DIR / name
            estimate = previous.stat().st_size if previous.is_file() and not previous.is_symlink() else 0
            if not ssd_write_budget().allowed(estimate):
                log("Slot save skipped: daily SSD write limit reached (ssdWriteDailyLimitGB); the session is re-read later")
                return True
            code, res = llama_call("POST", "/slots/0?action=save",
                                   {"filename": target.name}, timeout=120)
            if code != 200 or not res or res.get("n_saved") != n or not target.is_file():
                raise RuntimeError(f"slot save failed: HTTP {code}")
            os.replace(target, SLOT_DIR / name)
            ssd_write_budget().record((SLOT_DIR / name).stat().st_size)
            saved = {"model": st.model_id, "vision": st.vision, "tokens": n,
                     "bytes": (SLOT_DIR / name).stat().st_size,
                     "sha256": file_hash(SLOT_DIR / name), "reload_min_gb": st.vram_resume_min_gb,
                     "signature": checkpoint_signature(st.model_id, st.vision), "saved_at": time.time()}
            write_saved_slot_metadata(SLOT_DIR / name, saved)
            drop_other_slot_copies(name, SLOT_DIR / name)
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
            folder = slot_folder_for(target.stat().st_size, n)     # RAMFIX1: the disk spill folder when a reload with it in RAM would not fit
            fd, tmp = tempfile.mkstemp(prefix=".slot-", dir=folder)
            os.close(fd)
            try:
                if folder == SLOT_DIR:
                    shutil.copyfile(target, tmp)
                    os.replace(tmp, SLOT_DIR / name)
                else:
                    spill_copy(target, folder / name)           # the SSD budget was reserved by slot_folder_for
                saved = {"model": st.model_id, "vision": st.vision, "tokens": n,
                         "bytes": (folder / name).stat().st_size,
                         "sha256": file_hash(folder / name), "reload_min_gb": st.vram_resume_min_gb,
                         "signature": checkpoint_signature(st.model_id, st.vision), "saved_at": time.time()}
                write_saved_slot_metadata(folder / name, saved)
                drop_other_slot_copies(name, folder / name)
                st.skip_disk_restore = False
                ram_checkpoint = st.ram_checkpoint
                if (ram_checkpoint and ram_checkpoint.get("model") == st.model_id
                        and ram_checkpoint.get("vision") == st.vision):
                    # The live slot just produced a newer disk snapshot. Do not let an older RAM
                    # snapshot take precedence on the next load; restore_slot_locked prefers RAM.
                    clear_ram_checkpoint()
            finally:
                Path(tmp).unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            log(f"Saved slot ({n} tokens) to {'the disk spill folder' if folder != SLOT_DIR else 'disk'}")
        return True
    except Exception as e:  # noqa: BLE001 - this is the checkpoint integrity/fallback boundary.
        st.last_error = f"Session checkpoint failed: {e}"
        event("checkpoint_save_failed", st.last_error, fallback=True)
        if ram_only:
            clear_ram_checkpoint()
        elif target is not None:
            target.unlink(missing_ok=True)
        return False


def discard_slot_file(path):
    """A saved session that cannot be used is deleted with its metadata: left in place it would be rejected (and, before, would
    even restart a healthy server) on every cold start."""
    metadata = checkpoint_metadata_path(path)
    for name in (path, metadata, metadata.with_name(metadata.name + ".tmp")):
        try:
            Path(name).unlink(missing_ok=True)
        except OSError:
            pass


def stage_for_restore(disk, target):
    """RAMFIX1: put a saved session where the server reads it (its private RAM folder) without a second copy in RAM. A hard link when
    both lie on the same file system (the slot folder and the server's folder are both in /dev/shm). From the disk spill folder a
    symbolic link: the engine checks only the file name and opens the path, so it reads the disk file through the page cache (which
    the kernel may drop) and nothing is staged. The disk file was fully validated (size, SHA-256) just before. A copy only when
    neither link can be made, and only while the desktop keeps ramDeskFloorGB of MemAvailable; otherwise the prompt is re-read."""
    target.unlink(missing_ok=True)
    try:
        os.link(disk, target)
        return
    except OSError:
        pass
    if G.get("restoreViaSymlink", True) and not is_tmpfs(disk):
        try:
            os.symlink(disk.resolve(), target)
            return
        except OSError:
            pass
    stage_copy(disk, target)


def stage_copy(disk, target):
    """A copy of a saved session into RAM for the server, only within the RAM slot space and while the desktop keeps
    ramDeskFloorGB of MemAvailable (a target outside tmpfs costs no RAM and is not checked); otherwise RuntimeError: re-read."""
    if is_tmpfs(target):
        size = disk.stat().st_size
        if not ram_slot_space_ok(size):
            raise RuntimeError("not enough RAM for disk checkpoint staging")
        mem_gb = ram_valve.read_readings(G["ramGuardProcRoot"], G["ramGuardSysRoot"])[0]
        if mem_gb is not None and mem_gb - size / 1024 ** 3 < float(G.get("ramDeskFloorGB", 4.0)):
            raise RuntimeError(f"staging {size / 1048576:.0f} MiB would take the desktop below its floor; the prompt is re-read")
    target.unlink(missing_ok=True)
    shutil.copyfile(disk, target)


class StagingRefused(RuntimeError):
    """RAMFIX1: the copy for a second restore try could not be made; the session on the SSD is not at fault and stays."""


class LayoutWindowError(RuntimeError):
    """A saved session longer than the window of the layout being loaded. The session is kept (it fits the other layout),
    the server is not touched, and nothing is restarted for it."""


def restore_slot_locked(timeout=120):
    """"" = continue with whatever the server holds now; "restart_empty" = the server was told to restore and may hold corrupt
    state, so it must be restarted empty. A session that fails validation never reaches the server, so it costs no restart."""
    st.slot_dirty = True
    checkpoint = st.ram_checkpoint
    from_ram = bool(checkpoint and checkpoint["model"] == st.model_id and checkpoint["vision"] == st.vision)
    if st.skip_disk_restore and not from_ram:
        st.skip_disk_restore = False
        return ""
    disk = saved_slot_path(st.model_id, st.vision)
    if not from_ram and not disk.exists():
        return ""
    from_spill = not from_ram and disk.parent == spill_dir()
    spilled = None                     # the SSD copy behind a session staged into the slot folder: deleted only after a restore
    root = RAM_SLOT_DIR if st.ram_backend else SLOT_DIR
    target = checkpoint["path"] if from_ram else root / "restore.bin" if st.ram_backend else disk
    try:
        try:
            if from_ram:
                if checkpoint["signature"] != checkpoint_signature(st.model_id, st.vision):
                    raise ValueError("RAM session configuration changed")
                validate_checkpoint(target, checkpoint)
            else:
                saved = validate_saved_slot(disk, st.model_id, st.vision)
                if not st.ram_backend and disk.parent != SLOT_DIR:
                    # RAMFIX1: the server reads only its own slot folder: the session moved to the disk spill folder goes back first
                    staged = SLOT_DIR / disk.name
                    stage_copy(disk, staged)
                    write_saved_slot_metadata(staged, saved)
                    spilled, disk, target = disk, staged, staged
                if st.ram_backend:
                    stage_for_restore(disk, target)
                    if not target.is_symlink():      # a symbolic link points at the disk file validated above
                        validate_checkpoint(target, saved)
            tokens = checkpoint["tokens"] if from_ram else saved["tokens"]
            window = model_config(st.model_id, st.vision, st.layout)["ctxSize"]
            if tokens > window:
                raise LayoutWindowError(f"{tokens} tokens do not fit the {st.layout} layout ({window})")
        except LayoutWindowError as e:
            event("checkpoint_kept_other_layout", f"Saved session kept for the other layout: {e}")
            return ""
        except (ValueError, KeyError, TypeError) as e:
            # Nothing was sent to the server: it is untouched. The session is unusable, so drop it and start empty.
            event("checkpoint_rejected", f"Saved session discarded; the next prompt will be re-read: {e}", fallback=True)
            if from_ram:
                clear_ram_checkpoint()          # the server is untouched; a valid disk session saved earlier stays usable next time
            else:
                discard_slot_file(disk)
            return ""
        except (OSError, RuntimeError) as e:
            event("checkpoint_unavailable", f"Saved session not restored now; the next prompt will be re-read: {e}", fallback=True)
            if from_ram:
                clear_ram_checkpoint()
            return ""

        def ask_restore():
            code, res = llama_call("POST", "/slots/0?action=restore", {"filename": target.name}, timeout=timeout)
            if (code != 200 or not res or res.get("n_restored", 0) <= 0
                    or (from_ram and res["n_restored"] != checkpoint["tokens"])):
                raise RuntimeError(f"slot restore failed or token count differs: HTTP {code}")
            return res
        try:
            try:
                res = ask_restore()
            except Exception as first:  # a link the engine could not follow gets one more try with a copy
                if from_ram or not target.is_symlink():
                    raise
                event("checkpoint_restore_retry", f"restore through the link to the disk failed ({first}); one more try with a copy")
                try:
                    stage_copy(disk, target)
                except (OSError, RuntimeError) as no_room:
                    raise StagingRefused(str(no_room)) from first
                validate_checkpoint(target, saved)
                res = ask_restore()
            if from_ram:
                cleaned = clear_ram_checkpoint()
                event("checkpoint_restored", f"Restored {res['n_restored']} RAM tokens; " +
                      ("checkpoint removed" if cleaned else "checkpoint cleanup failed"))
            else:
                st.slot_dirty = False       # the file on disk is exactly what the server holds now
                log(f"Restored slot from disk ({res['n_restored']} tokens)")
                if from_spill and disk.parent == spill_dir():
                    # RAMFIX1: read back from the SSD: the server holds it now; the next unload decides again where it goes
                    discard_slot_file(disk)
                    st.slot_dirty = True
                if spilled is not None:
                    discard_slot_file(spilled)  # staged into the slot folder and restored: the SSD copy is not needed any more
            return ""
        except Exception as e:  # noqa: BLE001 - partially restored backends must be discarded safely.
            event("checkpoint_restore_failed", f"RAM/disk session unusable; prompt will be re-read: {e}", fallback=True)
            # The server was told to restore and may hold corrupt state. Force a fresh process before proxying.
            if from_ram:
                clear_ram_checkpoint()
            elif isinstance(e, StagingRefused):
                pass                            # the retry copy did not fit (RAM space or the desktop's floor): the SSD session stays
            else:
                discard_slot_file(disk)
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


def lease_root():
    """Where the owned-server lease lives: the private RAM directory when it is in use, else the state directory."""
    return RAM_SLOT_DIR if st.ram_backend and RAM_SLOT_DIR is not None else STATE_DIR


def stop_llama_locked(reason, save=True, fast=False, ram_only=False, min_tokens=None):
    """fast: once the slot is saved (or there was nothing to save) kill the server instead of waiting for its exit; a
    llama-server with MTP takes about 5 s to exit and gives its VRAM back only then (LAYOUT1b). Without a save, stop as before."""
    crashed = False
    if st.proc and st.proc.poll() is not None and not st.expected_exit:
        # Dead before we got here and nobody had noticed: that is a crash, not a clean stop.
        failed = st.proc
        if paused_exit(failed):
            event("vram_pause_exit", f"the server had paused for the desk's VRAM (exit 75) before the stop ({reason})")
        else:
            record_server_crash(failed, f"server found dead while stopping ({reason})")
            st.last_error = crash_text(failed)
            event("server_crashed", st.last_error, fallback=True)
            crashed = note_crash(proc=failed)
    elif st.proc and st.proc.poll() is None:
        saved = False
        if save and st.status == "READY":
            saved = save_slot_locked(ram_only=ram_only, min_tokens=min_tokens)
            if not saved and (GAME.enabled or ram_only):
                event("layout_checkpoint_save_failed" if ram_only else "game_save_failed",
                      "RAM-only slot save failed; server stays loaded" if ram_only else "Save failed; server stays loaded",
                      fallback=True)
                return False
            if saved and not ram_only:
                session_persist_on_stop(reason)     # SESS3: the SSD copy only at a stop or an idle unload (decided inside)
        log(f"Stopping llama-server (pid {st.proc.pid}): {reason}")
        st.expected_exit = True
        try:
            if fast and saved:
                log(f"Fast stop: slot saved, killing llama-server (pid {st.proc.pid}) instead of waiting for its exit")
                st.proc.kill()
                st.proc.wait(timeout=5)
                # exePath may be a wrapper (a test stand runs the server inside a memory-limited scope): the kill reached only
                # the wrapper. Kill the server on our port by its exact PID too (LAYOUT1c).
                for pid in llama_server_pids(G["llamaPort"]):
                    try:
                        os.kill(pid, signal.SIGKILL)
                        log(f"Fast stop: killed llama-server pid {pid} left behind by a wrapper")
                    except OSError:
                        pass
            else:
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
    LAST_STOP["at"] = time.monotonic()
    for root in {lease_root(), STATE_DIR}:
        try:
            (root / "server.json").unlink(missing_ok=True)
        except OSError:
            pass
    st.proc = None
    # ERROR is left only by `ai-on` or the cooldown, never by an unrelated stop.
    st.status = "ERROR" if (crashed or st.status == "ERROR") else "UNLOADED"
    st.model_id = ""
    st.vision = False
    st.layout = "short"
    return True


def process_identity(pid):
    return common.process_identity(pid)


def server_exec_ready(pid):
    return common.server_exec_ready(pid, RUN_EXE["path"] or M["exePath"])


def write_server_lease():
    common.write_lease(lease_root(), st.proc, RUN_EXE["path"] or M["exePath"])


def recover_orphan(root):
    # Late-bound lookups, so a caller (or a test) that replaces process_identity, llama_call or event is honoured.
    common.stop_owned_orphan(root, lambda *a, **k: llama_call(*a, **k), lambda *a, **k: event(*a, **k),
                             lambda pid: process_identity(pid))


def backend_port_in_use():
    try:
        with socket.create_connection((G["llamaHost"], G["llamaPort"]), timeout=0.3):
            return True
    except OSError:
        return False


def light_admission():
    """The part of the admission check that does not depend on telemetry (used when the same model has just been running)."""
    if st.override == "OFF":
        return Refusal("AI is manually disabled (ai-off)", "ai_off")
    if st.multimedia_lock:
        return Refusal("GPU is busy with an audio/video job", "multimedia")
    return ""


def recover_orphans_at_start():
    """A server left behind by a gateway that died keeps holding its RAM and VRAM. Stop it now, before the first request: that request
    would otherwise be refused by the memory check (the orphan's own memory counts against it) and never reach the point where the
    orphan is dealt with. Only a process whose identity matches the lease is touched. (The RAM backend keeps its lease in its own
    directory, which `acquire_model` prepares, and recovers the orphan from, before any memory check.)"""
    try:
        if (STATE_DIR / "server.json").exists():
            recover_orphan(STATE_DIR)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, http.client.HTTPException) as error:
        log(f"Orphaned server left alone: {error}")


def claim_port():
    """True when the backend port is free, or held by an idle server this gateway started before it died (that one is stopped)."""
    if not backend_port_in_use():
        return True
    try:
        recover_orphan(lease_root())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, http.client.HTTPException) as e:
        st.last_error = f"Backend port is occupied; refusing a second server ({e})"
        return False
    for _ in range(30):               # the stopped server's socket takes a moment to disappear
        if not backend_port_in_use():
            return True
        time.sleep(0.1)
    return False


def llama_server_pids(port):
    """PIDs of llama-server processes started with `--port port` (by their command line, not by who listens)."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            args = Path(f"/proc/{entry}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if args and args[0].endswith(b"llama-server") and any(
                args[i] == b"--port" and args[i + 1] == str(port).encode() for i in range(len(args) - 1)):
            pids.append(int(entry))
    return pids


def wait_for_llama_exit(port, timeout=30.0):
    """True once no llama-server listens on `port`. A stopped server gives its VRAM back only when it has exited: a new load
    started before that finds the card full (LAYOUT1 on the real card: a long-context load failed with cudaMalloc)."""
    deadline = time.monotonic() + timeout
    while True:
        if not llama_server_pids(port):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def wait_for_vram_settle(timeout=3.0, quiet=0.5, interval=0.1):
    """After the old server exits the driver gives its VRAM back over about half a second (LAYOUT1b probe on the card: 90 %
    released at 0.53 s, plateau at 0.53 to 1.39 s). Wait until the free memory has not risen for `quiet` seconds, at most
    `timeout`. Returns the free MiB that was settled on, or None when the card cannot be read or the cap was reached."""
    deadline = time.monotonic() + timeout
    last = None
    still_since = time.monotonic()
    while True:
        free = vram_free_gb() * 1024
        if free < 0:
            return None
        now = time.monotonic()
        if last is None or free > last + 1:  # rising (or first sample): the quiet period starts again
            last, still_since = free, now
        elif now - still_since >= quiet:
            return int(free)
        if now >= deadline:
            event("vram_settle_timeout", f"free {free:.0f} MiB still rising after {timeout:.1f} s; loading anyway", fallback=True)
            return None
        time.sleep(interval)


def start_llama(model_id, vision, fresh=False, health_timeout=None, restore_timeout=120, admitted=False,
                pressure_gate=True, layout="short"):
    load_deadline = None if health_timeout is None else time.monotonic() + health_timeout
    with st.lock:
        if st.shutting_down.is_set():
            return "Gateway is shutting down"
        if st.proc and st.proc.poll() is not None and not st.expected_exit:
            failed = st.proc
            st.proc = None
            if paused_exit(failed):
                st.status = "UNLOADED"
                event("vram_pause_exit", "the server paused for the desk's VRAM (exit 75); this request loads it again")
            else:
                record_server_crash(failed, "request arrived after unexpected server exit")
                st.last_error = crash_text(failed)
                st.status = "ERROR" if note_crash(proc=failed) else "UNLOADED"
        if (st.proc and st.proc.poll() is None and st.status == "READY"
                and st.model_id == model_id and st.vision == vision and st.layout == layout):
            return ""
        if (st.proc and st.proc.poll() is None and st.busy > 1
                and (st.model_id != model_id or st.vision != vision or st.layout != layout)):
            return "Another response is in progress; retry before switching model, vision mode or context layout"
        if st.status == "ERROR":
            return st.last_error
        mc = model_config(model_id, vision, layout)
        if bad_layout(mc, f"{model_id}/{layout}"):      # GWV1: refused before the running server is touched
            return st.last_error
        if vision and not mc.get("mmprojPath"):
            vision = False
        if not admitted and load_deadline is None:
            reason = choose_startup_slots(model_id, vision, mc, layout)
            if reason:
                layout, mc, reason = lite_fallback(model_id, vision, layout, mc, reason)
            if reason:
                log_load_blocked(reason)
                st.last_error = reason
                return reason
        # Admission is decided before the running model is touched: a refused switch must leave it loaded. The memory the old
        # server holds is credited, because the monitor's sample still shows it.
        pressure_gate = pressure_gate and PRESSURE_GATE_OVERRIDE.get()
        if admitted:
            reason = light_admission()
        elif pressure_gate:
            reason = preload_check(mc, reclaim=running_footprint(), admit=(model_id, layout))
        else:
            reason = preload_check(mc, reclaim=running_footprint(), pressure_gate=False, admit=(model_id, layout))
        if reason:
            log_load_blocked(reason)
            st.last_error = reason
            return reason
        load_blocked_cleared()
        t_stop = time.monotonic()
        layout_only = st.model_id == model_id and st.vision == vision and st.layout != layout
        if st.proc and st.proc.poll() is None and not stop_llama_locked(
                f"switching to model={model_id} vision={vision}", save=True,
                fast=layout_only and G.get("layoutFastStop") is True, ram_only=layout_only,
                min_tokens=G["layoutSwitchSaveMinTokens"] if layout_only else None):
            return st.last_error
        LAYOUT_STATE["timing"] = {"stop_s": round(time.monotonic() - t_stop, 2)}
        t_exit = time.monotonic()
        if not wait_for_llama_exit(G["llamaPort"]):
            st.last_error = "The previous llama-server did not exit in time; its memory is not free yet"
            event("llama_exit_timeout", st.last_error, fallback=True)
            return st.last_error
        LAYOUT_STATE["timing"]["exit_s"] = round(time.monotonic() - t_exit, 2)
        t_settle = time.monotonic()
        settled = wait_for_vram_settle()
        LAYOUT_STATE["settle_s"] = round(time.monotonic() - t_settle, 2)
        if settled is not None:
            event("vram_settled", f"free {settled} MiB after {time.monotonic() - t_settle:.2f} s")
        st.status, st.model_id, st.vision, st.layout = "LOADING", model_id, vision, layout
        t_load = time.monotonic()
        st.loading_since = time.time()
        st.vram_pressure_since = None
        try:
            result = load_server_locked(model_id, vision, mc, fresh, load_deadline, restore_timeout)
            if LAYOUT_STATE.get("timing") is not None:
                LAYOUT_STATE["timing"]["load_s"] = round(time.monotonic() - t_load, 2)
            return result
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


def load_server_locked(model_id, vision, mc, fresh, load_deadline, restore_timeout):
    layout = st.layout   # DEPLOY3: the layout of this start; stop_llama_locked() resets st.layout to "short" before a restart
    try:
        args = build_args(mc, vision)
        RUN_EXE["path"] = exe_for(mc, check=False) if mc.get("exePath") is not None else None
        extra_env = model_extra_env(mc)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as e:
        st.status = "UNLOADED"
        st.last_error = f"Invalid model configuration: {type(e).__name__}: {e}"
        return st.last_error
    if st.backend_blocked or not claim_port():
        st.status = "UNLOADED"
        st.last_error = st.backend_blocked or st.last_error or "Backend port is occupied; refusing a second server"
        event("backend_blocked", st.last_error, fallback=True)
        return st.last_error
    settle_pause_leftovers("before a server start")    # RAMFIX1: also when ramPolicy is not honest (no ram_admission)
    log("Starting llama-server: " + redacted_command(args))
    if extra_env:
        log("llama-server extra environment (names only): " + ", ".join(sorted(extra_env)))
    st.expected_exit = False
    try:
        # no extraEnv: the child inherits the gateway's environment exactly as before
        st.proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, env={**os.environ, **extra_env} if extra_env else None)
        SERVER_SLOT["owner"] = (model_id, vision, layout)
        st.proc_wall_start = time.time()
        POLITE.attach(st.proc.pid)
        # RAM valve: the model is the first thing the kernel may kill. Off unless llamaOomScoreAdj is set; a failed write is only logged.
        oom_adj = G.get("llamaOomScoreAdj", 0)
        if oom_adj:
            _oom_ok, oom_error = ram_valve.oom_score_adj_write(st.proc.pid, oom_adj)
            if oom_error:
                event("llama_oom_score_adj_failed", f"pid={st.proc.pid}: {oom_error}")
    except OSError as e:
        stop_llama_locked("spawn failed", save=False)
        st.last_error = f"Could not spawn llama-server: {e}"
        return st.last_error
    window = G["healthCheckTimeoutSeconds"] if load_deadline is None else max(0, load_deadline - time.monotonic())
    deadline = time.monotonic() + window
    while time.monotonic() < deadline:
        if st.shutting_down.wait(0.5):
            stop_llama_locked("gateway shutdown during load", save=False)
            st.last_error = "Gateway is shutting down"
            return st.last_error
        if st.override == "OFF" or st.multimedia_lock:
            stop_llama_locked("switched off during load", save=False)
            st.last_error = light_admission()
            return st.last_error
        if st.proc.poll() is not None:
            failed = st.proc
            if governor_on() and failed.returncode == 75 and not vision:
                # VRAMLOGIC1i: the governor found the layout does not keep the reserve during warm-up: not a crash
                st.proc = None
                st.status = "UNLOADED"
                governor_mark_too_big(model_id, st.layout)
                st.last_error = f"layout {st.layout} does not keep the card reserve (server exit 75 during load)"
                log(st.last_error)
                return st.last_error
            record_server_crash(failed, "loading llama-server")
            st.last_error = f"llama-server exited during load (code {failed.returncode})"
            st.proc = None
            st.status = "ERROR" if note_crash() else "UNLOADED"
            log(st.last_error)
            return st.last_error
        try:
            code, _ = llama_call("GET", "/health", timeout=3)
        except (OSError, ValueError, http.client.HTTPException):
            continue
        if code != 200:
            continue
        if governor_on() and not governor_warm():
            continue                # VRAMLOGIC1h: under the governor, READY waits for the server's warm-up (/vram.ready)
        st.status, st.last_error = "READY", ""
        st.health_failures, st.last_health_probe = 0, time.monotonic()
        try:
            # Written now, not at spawn: a launcher script or `nice` has finished exec'ing, so the identity recorded is the final
            # one, and the process that answered /health on our port is provably ours.
            write_server_lease()
        except OSError as lease_error:
            if st.ram_backend:
                st.last_error = f"Could not record the server lease: {lease_error}"
                stop_llama_locked("lease failed", save=False)
                return st.last_error          # the RAM backend must be able to prove which process is its own
            # Disk mode: the lease only helps a later gateway to find its orphan. Without one the port check still protects.
            log(f"Server lease not written ({lease_error}); an orphaned server would have to be stopped by hand")
        log(f"llama-server READY (pid {st.proc.pid}, vision={vision})")
        event("server_started", f"model={model_id} vision={vision}")
        if load_deadline is not None and time.monotonic() >= load_deadline:
            st.last_error = "Gear restart deadline exceeded before restore"
            return st.last_error
        error = "" if fresh else (restore_slot_locked() if load_deadline is None else
                restore_slot_locked(timeout=min(restore_timeout, max(0.1, load_deadline - time.monotonic()))))
        if error == "restart_empty":
            if not stop_llama_locked("checkpoint restore failed; starting empty", save=False):
                return st.last_error
            st.skip_disk_restore = False
            st.vram_resume_min_gb = 0.0
            # The same model was running a moment ago, so the telemetry-based admission is not repeated: after a stop it could only
            # fail on a sample that predates the stop.
            # DEPLOY3: the layout of this start, not st.layout (stop_llama_locked has just reset it to "short")
            return start_llama(model_id, vision, fresh=True, admitted=True, restore_timeout=restore_timeout, layout=layout,
                               health_timeout=None if load_deadline is None else max(0.1, load_deadline - time.monotonic()))
        if not error:
            st.vram_resume_min_gb = 0.0
        return error
    st.last_error = f"Health check timed out after {G['healthCheckTimeoutSeconds']}s"
    if note_crash():
        st.status = "ERROR"
    stop_llama_locked("load timeout", save=False)
    return st.last_error


def parse_smaps_gib(text):
    """heap + /dev/zero resident size in GiB from a process's smaps text: the two areas that grow by about 3.3 GiB/h on the HIP runtime leak."""
    kib, name = 0, ""
    for line in text.splitlines():
        parts = line.split()
        if parts and re.match(r"^[0-9a-f]+-[0-9a-f]+$", parts[0]) and len(parts) >= 5:
            name = parts[5] if len(parts) > 5 else ""
        elif line.startswith("Rss:") and name in ("[heap]", "/dev/zero"):
            kib += int(parts[1])
    return kib / 1048576


def server_mem_gib(pid, proc="/proc"):
    try:
        with open(f"{proc}/{pid}/smaps", encoding="utf-8", errors="replace") as handle:
            return parse_smaps_gib(handle.read())
    except OSError:
        return None


def validate_server_mem_config(config):
    value = config.get("serverMemRestartGiB", SERVER_MEM_DEFAULTS["serverMemRestartGiB"])
    if type(value) not in (int, float) or isinstance(value, bool) or not (value == 0 or 1 <= value <= 256):
        raise ValueError("Invalid guardian config: serverMemRestartGiB must be 0 (off) or a number from 1 to 256")
    for key, low, high in (("serverMemRestartIdleSeconds", 1, 3600), ("serverMemCheckSeconds", 1, 3600)):
        number = config.get(key, SERVER_MEM_DEFAULTS[key])
        if type(number) is not int or not low <= number <= high:
            raise ValueError(f"Invalid guardian config: {key} must be an integer from {low} to {high}")
    if value and (config.get("vramGuard") is not True or type(config.get("slotSaveCheckpoints")) is not int
                  or config["slotSaveCheckpoints"] < 1):
        raise ValueError("serverMemRestartGiB needs vramGuard=true and slotSaveCheckpoints >= 1 to keep the session")


def server_mem_restart_tick(now, idle):
    """MEMPROF1: between turns (no request, idle for serverMemRestartIdleSeconds) a server whose heap + /dev/zero are over the limit
    is stopped after its session was saved to RAM; the next request loads it again and restores the session. Called with st.lock held."""
    limit = G["serverMemRestartGiB"]
    if (not limit or st.status != "READY" or st.busy != 0 or st.proc is None or idle < G["serverMemRestartIdleSeconds"]
            or now - st.mem_checked_at < G["serverMemCheckSeconds"] or (GAME.enabled and GAME.active)):
        return
    st.mem_checked_at = now
    used = server_mem_gib(st.proc.pid)
    if used is None:
        return
    # FINAL2: the limit is the GROWTH of heap + /dev/zero since the first reading of this server process, not the absolute size: a server with
    # the pinned expert cache holds 6-7 GiB in /dev/zero from the start (the absolute rule restarted it after every request).
    base_pid, base = st.mem_baseline
    if base_pid != st.proc.pid:
        st.mem_baseline = (st.proc.pid, used)
        return
    grown = used - base
    if grown <= limit:
        return
    saved = save_slot_locked(ram_only=True)
    if not saved:
        event("server_mem_restart_skipped", f"heap + /dev/zero grew by {grown:.2f} GiB over {limit} GiB, but the session could not be saved to RAM; "
              "the server stays loaded, the next check follows", fallback=True)
        return
    if stop_llama_locked("server memory growth", save=False):
        st.last_yield_reason, st.last_yield_at = "server_mem", time.time()
        event("server_mem_restart", f"heap + /dev/zero grew by {grown:.2f} GiB (now {used:.2f}, at the first reading {base:.2f}) over {limit} GiB: "
              "the session is in RAM, the server is stopped; the next request loads it again")


def monitor_tick(now, free_mib, ram, wall_now=None, process_mib=None, proc=None):
    """Monotonic pressure timers; idle means time since the last completed/admitted request. `proc` is the server the sample
    was taken for: a sample of a process that has been replaced meanwhile is not stored."""
    with st.lock:
        st.monitor_last_tick = time.time()
        if math.isfinite(free_mib) and 0 <= free_mib <= 2097152:
            st.vram_free_mib = int(free_mib)
            st.vram_sample_at = st.monitor_last_tick
        if isinstance(process_mib, int) and process_mib >= 0 and (proc is None or st.proc is proc):
            st.proc_vram_mib = process_mib
            st.proc_vram_pid = st.proc.pid if st.proc else None
            st.proc_vram_sample_at = st.monitor_last_tick
        if (st.status == "ERROR" and st.error_since is not None and G["errorCooldownSeconds"] > 0
                and now - st.error_since >= G["errorCooldownSeconds"]):
            clear_error_state()
            st.status = "UNLOADED"
            event("error_cooldown_over", "Restart limit cooldown is over; the next request starts the model again")
        if st.proc and st.proc.poll() is not None and not st.expected_exit:
            failed = st.proc
            st.proc = None
            st.model_id, st.vision, st.layout = "", False, "short"
            if paused_exit(failed):
                st.status = "UNLOADED"
                event("vram_pause_exit", "the server paused for the desk's VRAM (exit 75); the next request loads it again")
            else:
                record_server_crash(failed, "monitor observed unexpected server exit")
                st.last_error = crash_text(failed)
                event("server_crashed", st.last_error, fallback=True)
                st.status = "ERROR" if note_crash(proc=failed) else "UNLOADED"
        if st.busy == 0 and not (st.proc and st.proc.poll() is None and st.status != "READY"):
            settle_pause_leftovers("no answer resumes it", background=True)

        valid = math.isfinite(free_mib) and 0 <= free_mib <= 2097152
        if valid and st.monitor_error:
            event("telemetry_recovered", "VRAM readings available again")
            st.monitor_error = ""
            st.guard_fallback = ""
        wall = time.time() if wall_now is None else wall_now
        if wall_now is None:
            common.compensate_wall_clock(st, wall, now)
        idle = wall - st.last_request
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
                        st.vram_resume_min_gb = max(model_config(st.model_id, st.vision, st.layout)["vramFreeMinGBToLoad"], required)
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

        server_mem_restart_tick(now, idle)
        if st.status == "READY" and st.busy == 0 and idle > G["idleUnloadSeconds"] and not (GAME.enabled and GAME.active):
            stop_llama_locked(f"idle for {int(idle)}s", save=True)

        if now - st.last_ram_poll >= G["pollIntervalSeconds"]:
            st.last_ram_poll = now
            if G.get("ramGuard") is True:
                ram_valve_tick(now, idle)
            if session_cache_on():
                session_write_behind(now, idle)
            if st.status == "READY":
                st.pressure_sustain = st.pressure_sustain + 1 if ram < G["ramFreeMinGB"] else 0
                if st.pressure_sustain >= G["memoryPressureSustainPolls"] and st.busy == 0 and not (GAME.enabled and GAME.active):
                    # Freeing memory comes first: if the session cannot be saved (which also needs memory), unload without it.
                    if (stop_llama_locked("memory pressure", save=True)
                            or stop_llama_locked("memory pressure (session not saved)", save=False)):
                        st.memory_pressure = True
                        st.status = "MEMORY_PRESSURE"
            elif st.memory_pressure and ram >= G["ramFreeMinGBToLoad"]:
                st.memory_pressure = False
                if st.status == "MEMORY_PRESSURE":
                    st.status = "UNLOADED"       # ERROR and the others are left to their own ways out
        st.monitor_healthy = True


def stop_wakeup_probe():
    proc = WAKEUP["proc"]
    if proc is not None:
        unit = WAKEUP.get("unit")
        if unit:
            try:
                pid = int(subprocess.check_output(["systemctl", "--user", "show", unit, "-p", "MainPID", "--value"], text=True, timeout=2).strip())
                if pid > 1:
                    expected = subprocess.check_output(["systemctl", "--user", "show", unit, "-p", "ControlGroup", "--value"], text=True, timeout=2).strip()
                    group = Path(f"/proc/{pid}/cgroup").read_text().strip().split("::", 1)[1]
                    if expected and group == expected:
                        os.kill(pid, signal.SIGTERM)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        proc.stdout.close()
        WAKEUP["proc"] = None


def start_wakeup_probe():
    if not POLITE.recoverable or G["politeMode"] != "dynamic":
        return
    unit = f"shura-wakeup-{os.getpid()}.service"
    proc = subprocess.Popen(["systemd-run", "--user", "--quiet", "--collect", "--pipe", "--wait", "--unit="+unit,
                             "--property=CPUWeight=100", sys.executable, "-u", str(Path(__file__).with_name("wakeup_probe.py"))],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
    WAKEUP["proc"], WAKEUP["started_at"], WAKEUP["unit"] = proc, time.monotonic(), unit

    def read_probe():
        for line in proc.stdout:
            try:
                data = json.loads(line)
                value = float(data["p99_ms"])
                if math.isfinite(value) and value >= 0:
                    WAKEUP["p99_ms"], WAKEUP["sample_at"] = value, time.monotonic()
            except (ValueError, KeyError, TypeError):
                continue
    threading.Thread(target=read_probe, daemon=True, name="shura-wakeup-reader").start()


def ensure_wakeup_probe(now):
    """The probe is a child process: if it died the dynamic controller would stay blind for good."""
    proc = WAKEUP["proc"]
    if proc is None or proc.poll() is None or now - WAKEUP.get("restarted_at", -math.inf) < 30:
        return
    WAKEUP["restarted_at"] = now
    try:
        stop_wakeup_probe()
        WAKEUP["sample_at"] = None
        start_wakeup_probe()
        event("wakeup_probe_restarted", "Wakeup probe exited; started a new one")
    except (OSError, subprocess.SubprocessError) as error:
        event("wakeup_probe_failed", str(error), fallback=True)


atexit.register(stop_wakeup_probe)

def close_cpu_policy():
    problems = POLITE.shutdown()   # also stops the monitor thread from re-applying a weight afterwards
    if problems:
        event("cpu_restore_failed", problems, fallback=True)

atexit.register(close_cpu_policy)


class GearBackend:
    """Only called under st.lock, between requests. Snapshot is a tmpfs hardlink, not a second copy."""
    def __init__(self):
        self.model, self.vision, self.layout = st.model_id, st.vision, st.layout
        self.original_signature = checkpoint_signature(self.model, self.vision)
        self.original_slots = self.current_slots()

    def current_slots(self):
        return model_config(self.model, self.vision)["moeCacheSlots"]

    def save(self):
        if st.busy or self.vision or int(G.get("slotSaveCheckpoints", 0)) < 1:
            return None
        if not save_slot_locked(ram_only=True):
            return None
        saved = st.ram_checkpoint
        if saved is None:
            self.journal({"empty": True})
            return {"empty": True}
        validate_checkpoint(saved["path"], saved)
        if saved["signature"] != checkpoint_signature(self.model, self.vision):
            raise ValueError("gear snapshot configuration mismatch")
        # Private root is verified by prepare_ram_slot_dir. Refuse stale transaction files.
        backup = prepare_ram_slot_dir() / "gear-rollback.bin"
        if backup.exists() or backup.is_symlink():
            raise RuntimeError("unfinished gear transaction; refusing another restart")
        self.journal(saved)
        os.link(saved["path"], backup)
        return {**saved, "path": backup}

    def journal(self, snapshot):
        root = prepare_ram_slot_dir()
        record = {"version": 1, "model": self.model, "slots": self.original_slots,
                  "signature": self.original_signature,
                  "snapshot": {key: value for key, value in snapshot.items() if key != "path"}}
        temporary = root / "gear-transaction.json.tmp"
        temporary.write_text(json.dumps(record))
        os.replace(temporary, root / "gear-transaction.json")

    def restart(self, slots, snapshot):
        # Putting the old gear back is the safety path: it gets its own, longer limits, so a slow
        # load that defeated the forward step cannot also defeat the rollback.
        rollback = slots == self.original_slots
        deadline = time.monotonic() + (G.get("slotGearRollbackSeconds", 120) if rollback else G["slotGearRestartSeconds"])
        # Recompute the original signature to prove that only slot count changed.
        present = RUNTIME_SLOTS.get(self.model)
        try:
            RUNTIME_SLOTS[self.model] = self.original_slots
            if checkpoint_signature(self.model, self.vision) != self.original_signature:
                raise RuntimeError("model/binary/config changed during gear transaction")
        finally:
            if present is None:
                RUNTIME_SLOTS.pop(self.model, None)
            else:
                RUNTIME_SLOTS[self.model] = present
        if not stop_llama_locked("slot gear change", save=False):
            return False
        stopped_at = time.time()
        RUNTIME_SLOTS[self.model] = slots
        root = prepare_ram_slot_dir()
        if not snapshot.get("empty"):
            # Model, vision, binary and all other settings remain fixed within this locked transaction.
            active = root / "active.bin"
            active.unlink(missing_ok=True)
            os.link(snapshot["path"], active)
            saved = {**snapshot, "path": active, "signature": checkpoint_signature(self.model, self.vision)}
            st.ram_checkpoint = saved
            metadata = {key: value for key, value in saved.items() if key != "path"}
            temporary = root / "checkpoint.json.tmp"
            temporary.write_text(json.dumps(metadata))
            os.replace(temporary, root / "checkpoint.json")
        before = st.event_counters.get("checkpoint_restored", 0)
        st.status = "UNLOADED"
        if not wait_fresh_vram(stopped_at, timeout=min(5, max(0, deadline-time.monotonic()))):
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        with memory_pressure_gate(False):
            error = start_llama(self.model, self.vision, layout=self.layout, fresh=snapshot.get("empty", False),
                                health_timeout=remaining, restore_timeout=min(30 if rollback else 5, remaining))
        return not error and (snapshot.get("empty") or st.event_counters.get("checkpoint_restored", 0) > before)

    def release(self, snapshot):
        root = prepare_ram_slot_dir()
        # Mark the completed transaction first; an orphaned backup alone never authorizes recovery.
        (root / "gear-transaction.json").unlink(missing_ok=True)
        if not snapshot.get("empty"):
            snapshot["path"].unlink(missing_ok=True)


def apply_slots(n):
    with st.lock:
        if st.busy or st.status != "READY" or st.vision or not st.ram_backend:
            return None   # refused before anything was touched: not a failed change
        if type(n) is not int or n < 16 or n > M["models"][st.model_id]["moeCacheSlots"]:
            return None
        return apply_slot_transaction(n, GearBackend())


def recover_gear_transaction(root):
    journal = root / "gear-transaction.json"
    if not journal.exists():
        return
    if journal.is_symlink() or journal.stat().st_size > 16384:
        raise ValueError("unsafe gear journal")
    record = json.loads(journal.read_text())
    mid, slots = record["model"], record["slots"]
    if (record.get("version") != 1 or mid not in M["models"] or type(slots) is not int
            or not 16 <= slots <= M["models"][mid]["moeCacheSlots"]):
        raise ValueError("invalid gear journal model/version/slots")
    prior = RUNTIME_SLOTS.get(mid)
    RUNTIME_SLOTS[mid] = slots
    try:
        if checkpoint_signature(mid, False) != record["signature"]:
            raise ValueError("gear journal binary/model/config changed")
        saved = record["snapshot"]
        if not isinstance(saved, dict):
            raise ValueError("invalid gear snapshot")
        if saved.get("empty") is not True:
            if saved["model"] != mid or saved["vision"] is not False or saved["signature"] != record["signature"]:
                raise ValueError("gear snapshot ownership mismatch")
            candidate = root / "gear-rollback.bin"
            if not candidate.exists():
                candidate = root / "active.bin"  # journal written before the hardlink / shutdown
            validate_checkpoint(candidate, saved)
            active = root / "active.bin"
            if candidate != active:
                active.unlink(missing_ok=True)
                os.link(candidate, active)
            if not G["slotGears"]:
                RUNTIME_SLOTS.pop(mid, None)  # disabled means the configured static baseline
            saved = {**saved, "signature": checkpoint_signature(mid, False)}
            temporary = root / "checkpoint.json.tmp"
            temporary.write_text(json.dumps(saved))
            os.replace(temporary, root / "checkpoint.json")
        else:
            st.skip_disk_restore = True
            if not G["slotGears"]:
                RUNTIME_SLOTS.pop(mid, None)
        journal.unlink()
        (root / "gear-rollback.bin").unlink(missing_ok=True)
        event("gear_recovered", f"Incomplete gear transaction rolled back for {mid}; A2 available")
    except BaseException:
        if prior is None:
            RUNTIME_SLOTS.pop(mid, None)
        else:
            RUNTIME_SLOTS[mid] = prior
        raise


def wait_fresh_vram(after, timeout=5):
    """A gear restart needs a post-stop sample, not a potentially double-counted memory credit."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if st.vram_sample_at is not None and st.vram_sample_at > after and st.vram_free_mib is not None:
            if st.monitor_error:
                if st.guard_fallback == st.monitor_error:
                    st.guard_fallback = ""
                st.monitor_error = ""
                event("telemetry_recovered", "Fresh post-stop VRAM sample available")
            return True
        time.sleep(0.05)
    event("gear_telemetry_timeout", "No fresh post-stop VRAM sample; refusing load", fallback=True)
    return False


def gear_budget_root():
    global GEAR_BUDGET_LOADED
    root = prepare_ram_slot_dir()
    if not GEAR_BUDGET_LOADED:
        ledger = root / "gear-budget.json"
        if ledger.exists():
            if ledger.is_symlink() or ledger.stat().st_size > 16384:
                raise RuntimeError("invalid gear budget ledger")
            history = json.loads(ledger.read_text())
            if (not isinstance(history, list) or len(history) > G["slotGearChangesPerDay"]
                    or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0
                           or t > time.monotonic() for t in history)):
                raise RuntimeError("invalid gear budget timestamps")
            GEAR_BUDGET.history = history
        GEAR_BUDGET_LOADED = True
    return root


def persist_gear_budget():
    root = gear_budget_root()
    temporary = root / "gear-budget.json.tmp"
    temporary.write_text(json.dumps(GEAR_BUDGET.history))
    os.replace(temporary, root / "gear-budget.json")


def gear_for(mid):
    """The ratchet of one model, or None when its configuration cannot have gears.

    A bad value must disable gears for that model only: raising here every second would wipe the
    VRAM-guard timers and block admission through monitor_error."""
    if mid in GEARS_UNAVAILABLE:
        return None
    if mid not in GEARS:
        try:
            ratchet = Ratchet(G, M["models"][mid]["moeCacheSlots"], GEAR_BUDGET)
            ratchet.current = model_config(mid, False)["moeCacheSlots"]
        except (KeyError, TypeError, ValueError) as error:
            GEARS_UNAVAILABLE.add(mid)
            event("gears_unavailable", f"model={mid}: {type(error).__name__}: {error}")
            return None
        GEARS[mid] = ratchet
    return GEARS[mid]


def model_slot_cost(mid):
    """Resolve a model's valve price in config > learned > legacy-default order."""
    settings = M["models"].get(mid) or {}
    info = SLOT_COSTS.resolve(
        mid, settings, G.get("slotGearMiBPerSlot", GEAR_DEFAULTS["slotGearMiBPerSlot"])
    )
    if info.get("error"):
        warning = (mid, info["error"])
        if warning not in SLOT_COST_WARNED:
            SLOT_COST_WARNED.add(warning)
            event("vram_valve_slot_cost_config_invalid", f"model={mid}: {info['error']}")
    return info


def apply_slot_cost_info(valve, info):
    valve.slot_cost_mib = info["value_mib"]
    valve.slot_cost_source = info["source"]
    valve.slot_cost_samples = info["samples"]


def _valve_vram_probe():
    """Capture whole-card free VRAM and per-process use for uncontended learning."""
    device = nvml_device()
    free = device.memory_info().get("free_mib")
    processes = device.processes()
    model_pid = st.proc.pid if st.proc and st.proc.poll() is None else None
    if type(free) is not int or free < 0 or not isinstance(processes, dict):
        return None
    foreign = {pid: used for pid, used in processes.items() if pid != model_pid}
    return {"free_mib": free, "foreign": foreign}


def valve_vram_probe_window(timeout=3.0, interval=0.1):
    """Wait for two stable whole-card probes; preserve the full window for noise checks."""
    deadline = time.monotonic() + timeout
    samples, foreign_samples = [], []
    while time.monotonic() <= deadline:
        try:
            sample = _valve_vram_probe()
        except (OSError, RuntimeError, ValueError, OverflowError):
            sample = None
        if sample is not None:
            samples.append(sample["free_mib"])
            foreign_samples.append(sample["foreign"])
            if len(samples) >= 2 and abs(samples[-1] - samples[-2]) <= 16:
                foreign = foreign_samples[-1] if all(row == foreign_samples[0] for row in foreign_samples) else None
                return {"free_samples": samples, "foreign": foreign}
        time.sleep(interval)
    return {"free_samples": samples, "foreign": None}


def learn_slot_cost_after_resize(mid, old_slots, new_slots, before, after):
    if before is None or after is None:
        result = {"accepted": False, "reason": "stable VRAM window unavailable"}
    else:
        result = SLOT_COSTS.observe_resize(
            mid, old_slots, new_slots,
            before.get("free_samples"), after.get("free_samples"),
            before.get("foreign"), after.get("foreign"),
        )
    if result.get("accepted"):
        event("vram_valve_slot_cost_measured",
              f"model={mid} value={result['measurement_mib']:.2f} MiB/slot; "
              f"rolling={result['value_mib']:.2f}; samples={result['samples']}")
    else:
        event("vram_valve_slot_cost_measurement_skipped",
              f"model={mid}: {result.get('reason', 'unknown measurement error')}")
    valve = VALVES.get(mid)
    if valve:
        apply_slot_cost_info(valve, model_slot_cost(mid))
    return result


def valve_for(mid):
    if st.model_id == mid and st.vision:
        return None
    if mid not in VALVES:
        maximum = M["models"][mid].get("moeCacheSlots", 0)
        initial = model_config(mid, False).get("moeCacheSlots", maximum)
        config = dict(G)
        config["vramValveLevels"] = M["models"][mid].get(
            "vramValveLevels", G.get("vramValveLevels", VALVE_DEFAULTS["vramValveLevels"])
        )
        try:
            VALVES[mid] = Valve(config, maximum, initial)
        except (KeyError, TypeError, ValueError) as error:
            event("vram_valve_config_error", f"model={mid}: {type(error).__name__}: {error}")
            return None
    apply_slot_cost_info(VALVES[mid], model_slot_cost(mid))
    return VALVES[mid]


def apply_valve_resize_locked(mid, target, now):
    """Queue an in-place resize only while the server has no active request."""
    valve = valve_for(mid)
    if (valve is None or st.busy or st.status != "READY" or st.vision
            or GAME.active or target not in valve.levels or target > valve.maximum):
        if valve:
            valve.reason = "resize deferred: waiting for an idle request boundary"
        return None
    old_slots = valve.current
    decision = valve.reason
    cost_before = valve_vram_probe_window() if G.get("vramValve") else None
    try:
        code, result = llama_call("POST", "/moe-cache/slots", {"slots": target}, timeout=5)
    except (OSError, ValueError, http.client.HTTPException, RuntimeError) as error:
        valve.completed(now, target, False)
        event("vram_valve_resize_failed", f"model={mid} slots={valve.current}->{target}: {error}")
        return False
    reported = result.get("new_slots") if isinstance(result, dict) else None
    if (code == 200 and isinstance(result, dict) and result.get("success") is True
            and type(reported) is int and reported == target and 0 <= reported <= valve.maximum):
        RUNTIME_SLOTS[mid] = reported
        valve.completed(now, reported, True)
        cost_after = valve_vram_probe_window() if G.get("vramValve") else None
        learn_slot_cost_after_resize(mid, old_slots, reported, cost_before, cost_after)
        event("vram_valve_resized", f"model={mid} slots={reported}/{valve.maximum}; "
              f"price={valve.slot_cost_mib:.2f} MiB/slot source={valve.slot_cost_source}; decision={decision}")
        return True
    if type(reported) is int and 0 <= reported <= valve.maximum and reported != valve.current:
        # The server's double-failure recovery can truthfully report zero even with HTTP 500.
        valve.current = reported
        RUNTIME_SLOTS[mid] = reported
        valve.reason = f"server reports {reported} slots after failed resize"
    else:
        valve.completed(now, target, False)
    detail = result.get("message", "unexpected resize response") if isinstance(result, dict) else "empty resize response"
    event("vram_valve_resize_failed", f"model={mid} HTTP {code}: {detail}")
    return False


def prefill_yield_restore(model_id, slots):
    """Set the cache back to slots. On failure keep the debt; prefill_yield_tick retries when the server is idle."""
    try:
        code, result = llama_call("POST", "/moe-cache/slots", {"slots": slots}, timeout=5)
    except (OSError, ValueError, http.client.HTTPException, RuntimeError) as error:
        code, result = 0, {"message": str(error)}
    if code == 200 and isinstance(result, dict) and result.get("success") is True and result.get("new_slots") == slots:
        RUNTIME_SLOTS[model_id] = slots
        PREFILL_YIELD.pending = None
        PREFILL_YIELD.count("restored")
        event("prefill_yield_restored", f"model={model_id} slots 0->{slots}")
        return True
    PREFILL_YIELD.pending = {"model": model_id, "slots": slots}
    PREFILL_YIELD.count("restore_failed")
    detail = result.get("message", "unexpected response") if isinstance(result, dict) else "empty response"
    event("prefill_yield_restore_failed", f"model={model_id} slots 0->{slots} HTTP {code}: {detail}; retry when idle")
    return False


def prefill_yield_begin(model_id, vision, body):
    """Shrink the expert cache to 0 for one large new prefill. Returns the level to restore, or None (request goes as it is)."""
    if not G.get("prefillCacheYield") or not body or not model_id or governor_on():
        return None
    body_len = len(body)
    previous = PREFILL_YIELD.remember_len(body_len)
    tokens = prefill_yield.new_tokens_estimate(body_len, previous)
    slots_now = RUNTIME_SLOTS.get(model_id, model_config(model_id, vision).get("moeCacheSlots", 0))
    with st.lock:
        ready = (st.status == "READY" and st.model_id == model_id and st.vision == vision
                 and st.proc is not None and st.proc.poll() is None)
        busy = st.busy
    ok, _reason = prefill_yield.should_yield(G["prefillCacheYield"], tokens, G["prefillCacheYieldMinTokens"], slots_now,
                                             ready, busy, vision, GAME.active, G["slotGears"])
    if not ok or PREFILL_YIELD.pending is not None:
        PREFILL_YIELD.count("skipped")
        return None
    try:
        code, result = llama_call("POST", "/moe-cache/slots", {"slots": 0}, timeout=5)
    except (OSError, ValueError, http.client.HTTPException, RuntimeError) as error:
        PREFILL_YIELD.count("resize_failed")
        event("prefill_yield_failed", f"model={model_id} slots {slots_now}->0: {error}; request goes as it is")
        return None
    if code == 200 and isinstance(result, dict) and result.get("success") is True and result.get("new_slots") == 0:
        RUNTIME_SLOTS[model_id] = 0
        PREFILL_YIELD.count("yielded")
        event("prefill_yield_started", f"model={model_id} slots {slots_now}->0 new_tokens~{tokens}")
        return slots_now
    PREFILL_YIELD.count("resize_failed")
    detail = result.get("message", "unexpected response") if isinstance(result, dict) else "empty response"
    event("prefill_yield_failed", f"model={model_id} slots {slots_now}->0 HTTP {code}: {detail}; request goes as it is")
    return None


def prefill_yield_end(restore_to, model_id):
    if restore_to is not None:
        prefill_yield_restore(model_id, restore_to)


def prefill_yield_tick():
    pending = PREFILL_YIELD.pending
    if pending is None:
        return
    if st.model_id != pending["model"] or st.status != "READY":
        # the server was replaced: the new one starts with its configured slots, so the debt is void
        PREFILL_YIELD.pending = None
        event("prefill_yield_restore_void", f"model={pending['model']} server changed before the restore")
        return
    with st.lock:
        idle = st.busy == 0
    if idle:
        prefill_yield_restore(pending["model"], pending["slots"])


def park_vram_valve_model(mid, valve, now):
    if st.busy or st.status != "READY" or not G["vramGuard"]:
        return
    if not save_slot_locked(ram_only=True):
        valve.retry_at = now + 30
        valve.reason = f"minimum {valve.levels[-1]} slots still below reserve; A2 save failed, model remains loaded"
        event("vram_valve_park_save_failed", valve.reason)
        return
    minimum = valve.levels[-1]
    if stop_llama_locked("VRAM valve minimum reached; vramGuard parking", save=False):
        st.last_yield_reason, st.last_yield_at = "vram_valve_exhausted", time.time()
        event("vram_valve_parked", f"model={mid}: minimum {minimum} slots did not restore reserve; A2 saved")
        notify_vram_yield("Видеопамяти всё ещё мало. Модель сохранена и припаркована.")
    else:
        valve.retry_at = now + 30
        event("vram_valve_park_failed", st.last_error)


def vram_valve_tick(now):
    if not G["vramValve"] or not st.model_id:
        return
    if PREFILL_YIELD.pending is not None:
        return  # a prefill yield still owes its restore; the valve must not move the cache under it
    mid = st.model_id
    valve = valve_for(mid)
    if valve is None:
        return
    cost = model_slot_cost(mid)
    apply_slot_cost_info(valve, cost)
    age = time.time() - st.vram_sample_at if st.vram_sample_at is not None else math.inf
    fresh = age <= G.get("vramTelemetryStaleSeconds", 5) and not st.monitor_error
    free = st.vram_free_mib if fresh else None
    previous_reason = valve.reason
    target = valve.target(now, free, G["vramCardReserveMiB"],
                          per_slot_mib=cost["value_mib"],
                          # The server refuses live tensor changes while a slot is processing;
                          # resize only after the gateway reaches a request boundary.
                          busy=st.busy > 0,
                          ready=st.status == "READY" and not st.vision and not GAME.active)
    if not fresh and previous_reason != valve.reason:
        event("vram_valve_telemetry_unavailable", "Fresh NVML sample missing; cache size unchanged")
    if target is not None:
        event("vram_valve_decision", f"model={mid} slot_cost={cost['value_mib']:.2f} MiB/slot "
              f"source={cost['source']}: {valve.reason}")
        apply_valve_resize_locked(mid, target, now)
        return
    if fresh and valve.park_due(now):
        park_vram_valve_model(mid, valve, now)


def reliability_tick(now):
    with st.lock:
        proc = st.proc
        if POLITE.enabled and G["politeMode"] == "dynamic":
            ensure_wakeup_probe(now)
        polite_on = POLITE.enabled and (POLITE.available or POLITE.retry_due(now))
        if polite_on and G["politeMode"] == "static":
            before = POLITE.snapshot()
            POLITE.tick(now, proc.pid if proc and proc.poll() is None else None, 0.0, (os.getpid(),))
            after = POLITE.snapshot()
            if before["available"] != after["available"]:
                event("cpu_policy_changed", f"mode={after['mode']} available={after['available']} error={after['error']}")
        if polite_on and G["politeMode"] == "dynamic":
            age = now - WAKEUP["sample_at"] if WAKEUP["sample_at"] is not None else math.inf
            if age <= 5:
                helper = WAKEUP["proc"]
                before = POLITE.snapshot()
                POLITE.tick(now, proc.pid if proc and proc.poll() is None else None,
                            WAKEUP["p99_ms"], (os.getpid(), helper.pid if helper else None))
                after = POLITE.snapshot()
                if (before["level"], before["weight"], before["available"]) != (after["level"], after["weight"], after["available"]):
                    event("cpu_policy_changed", f"mode={after['mode']} level={after['level']} weight={after['weight']} ms={after['change_ms']} error={after['error']}")
            elif (WAKEUP["sample_at"] is not None or
                  (WAKEUP["started_at"] is not None and now - WAKEUP["started_at"] > 5)):
                POLITE.stale(proc.pid if proc and proc.poll() is None else None)
        if governor_on() and st.model_id:
            try:
                governor_tick(now)
            except Exception as error:  # noqa: BLE001 - the governor read never takes the gateway down
                event("vram_governor_failed", f"{type(error).__name__}: {error}")
        if G["vramValve"] and st.model_id and not governor_on():
            try:
                vram_valve_tick(now)
            except Exception as error:  # noqa: BLE001 - controller faults never take the gateway down
                event("vram_valve_failed", f"{type(error).__name__}: {error}")
        if G.get("prefillCacheYield") and st.model_id and not governor_on():
            try:
                prefill_yield_tick()
            except Exception as error:  # noqa: BLE001 - the yield never takes the gateway down
                event("prefill_yield_failed", f"{type(error).__name__}: {error}")
        if not G["slotGears"] or not st.model_id or governor_on():
            return
        try:
            gear_budget_root()
        except (OSError, ValueError, RuntimeError) as error:
            G["slotGears"] = False
            event("gears_disabled", f"Budget cannot be verified: {error}", fallback=True)
            return
        mid = st.model_id
        gear = gear_for(mid)
        if gear is None:
            return
        age = time.time() - st.vram_sample_at if st.vram_sample_at is not None else math.inf
        free = st.vram_free_mib if age <= G.get("vramTelemetryStaleSeconds", 5) and not st.monitor_error else None
        target = gear.target(now, free, busy=st.busy > 0,
                             ready=st.status == "READY" and not st.vision and not GAME.active)
        if target is not None:
            fatal = False
            try:
                ok = apply_slots(target)
            except RollbackFailed as error:
                ok, fatal = False, True
                G["slotGears"] = False
                st.last_error = str(error)
                if not (st.proc and st.proc.poll() is None):
                    # No server is running: leave the door open for a normal load instead of a dead
                    # end that only an operator command or a restart clears. A2 stays retained.
                    st.status = "UNLOADED"
                event("gears_disabled", str(error), fallback=True)
            except Exception as error:  # noqa: BLE001 - an unexpected failure must still be charged once
                ok = False
                event("gear_failed", f"{type(error).__name__}: {error}", fallback=True)
            if ok is None:
                gear.deferred(time.monotonic(), "change deferred: server busy or not ready")
                return
            gear.completed(time.monotonic(), target, ok)
            if fatal:
                gear.reason = "rollback failed; A2 snapshot retained; controller disabled"
            try:
                persist_gear_budget()
            except OSError as error:
                G["slotGears"] = False
                event("gears_disabled", f"Cannot persist change budget: {error}", fallback=True)
            event("gear_changed" if ok else "gear_rolled_back", f"model={mid} slots={gear.current}: {gear.reason}")


def publish_vram_sample(free_mib, proc, process_mib):
    """Sensor publication cannot wait behind a 15-second lifecycle transaction.

    These are observational fields only; pressure decisions still require st.lock.
    Publish the timestamp last, after the value it certifies.
    """
    wall = time.time()
    if math.isfinite(free_mib) and 0 <= free_mib <= 2097152:
        st.vram_free_mib = int(free_mib)
        st.vram_sample_at = wall
    if proc is not None and st.proc is proc and isinstance(process_mib, int) and process_mib >= 0:
        st.proc_vram_mib = process_mib
        st.proc_vram_pid = proc.pid
        st.proc_vram_sample_at = wall


def declare_hung(proc, reason):
    """The server is alive but does not answer: stop it, count it as a failure, and let the next request start a fresh one."""
    acquired = st.lock.acquire(timeout=G["shutdownLockSeconds"])
    try:
        if not acquired:
            return False                      # a load is still running; the next probe or stall tries again
        if st.proc is not proc or proc.poll() is not None:
            return False
        record_server_crash(proc, reason)
        st.last_error = f"llama-server hung ({reason})"
        event("server_hung", st.last_error, fallback=True)
        st.expected_exit = True
        try:
            proc.kill()                       # a hung server cannot save a session or answer SIGTERM (a stopped process never even sees it)
            proc.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            if proc.poll() is None:
                return False
        st.proc = None
        st.model_id, st.vision, st.layout = "", False, "short"
        st.status = "ERROR" if note_crash() else "UNLOADED"
        return True
    finally:
        if acquired:
            st.lock.release()


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
    event("health_probe_failed", f"llama-server /health failed ({st.health_failures} in a row)", fallback=True)
    if st.health_failures >= G["healthProbeFailures"]:
        st.health_failures = 0
        declare_hung(proc, f"/health failed {G['healthProbeFailures']} times in a row")


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
            # Samples are published whatever the lifecycle lock is doing: admission needs them fresh, and a model load holds the
            # lock for up to minutes. The decisions below still need the lock and simply wait for the next second.
            publish_vram_sample(free_mib, proc, proc_mib)
            health_probe(time.monotonic())
            if st.lock.acquire(blocking=False):
                try:
                    monitor_tick(time.monotonic(), free_mib, mem_available_gb(), process_mib=proc_mib, proc=proc)
                    reliability_tick(time.monotonic())
                finally:
                    st.lock.release()
        except Exception as e:  # noqa: BLE001 - a failed sensor tick must be recorded, not kill the monitor.
            with try_lock(2) as got:    # a load holding the lock must not hang the monitor thread as well
                if got:
                    st.monitor_healthy = False
                    st.monitor_error = str(e)
                    reset_pressure()
                    event("monitor_failed", f"Guard disabled for this tick; original lifecycle retried next tick: {e}", fallback=True)


def settle_dead_backend():
    """A killed server closes its socket first and goes away only when the driver has released its GPU memory, which takes seconds for
    a 10 GB model. In that time `proc.poll()` still says "running" although nothing listens, and a request would be sent into a refused
    connection. Wait for the process to finish (the crash is then counted and the model restarted by the normal path); if it never
    finishes, treat it as hung."""
    proc = st.proc
    if proc is None or st.status != "READY" or proc.poll() is not None:
        return
    if common.backend_listening(G["llamaHost"], G["llamaPort"]):
        return
    if wait_for_process_exit(proc, timeout=G["deadBackendWaitSeconds"]):
        return
    declare_hung(proc, "stopped listening but did not exit")


LAYOUT_UPGRADE_HOLD_SECONDS = 120   # after a refused upgrade the loaded layout keeps answering this long before admission is asked again
UPGRADE_HOLD = {}                   # (model, wanted layout) -> {"until", "reason"}


def hold_upgrade(model_id, wanted, reason):
    """Remember that the switch to `wanted` was refused; one event per hold, the next try after LAYOUT_UPGRADE_HOLD_SECONDS."""
    hold = UPGRADE_HOLD.get((model_id, wanted))
    if hold is None or time.time() >= hold["until"]:
        event("layout_upgrade_held", f"model={model_id}: the {wanted} layout is refused ({reason}); the loaded {st.layout} layout "
              f"answers, the switch is tried again in {LAYOUT_UPGRADE_HOLD_SECONDS} s")
    UPGRADE_HOLD[(model_id, wanted)] = {"until": time.time() + LAYOUT_UPGRADE_HOLD_SECONDS, "reason": str(reason)}


def loaded_layout_fits(model_id, vision, wanted):
    """The same model runs READY in another layout whose context window is at least as large as `wanted`'s: it can answer the
    request (lite for short, longlite for long, long for short). Caller holds st.lock."""
    if not (st.proc is not None and st.proc.poll() is None and st.status == "READY" and st.model_id == model_id
            and st.vision == vision and st.layout != wanted):
        return False
    return model_config(model_id, vision, st.layout).get("ctxSize", 0) >= model_config(model_id, vision, wanted).get("ctxSize", 0)


def upgrade_on_hold(model_id, vision, wanted):
    hold = UPGRADE_HOLD.get((model_id, wanted))
    return hold is not None and time.time() < hold["until"] and loaded_layout_fits(model_id, vision, wanted)


def serve_on_loaded_layout(model_id, vision, wanted, error):
    """GWSTALL1: the switch to `wanted` was refused for lack of memory (SAGE1: short was 248 MiB short while lite ran and had just
    answered). A loaded layout that fits answers at once; the upgrade is asked again after LAYOUT_UPGRADE_HOLD_SECONDS, on a
    request boundary, never by holding a request. Caller holds st.lock. Returns True when the request is counted on the loaded server."""
    if refusal_code(error) not in RESOURCE_CODES or not loaded_layout_fits(model_id, vision, wanted):
        return False
    hold_upgrade(model_id, wanted, error)
    st.busy += 1
    st.last_request = time.time()
    return True


LOAD_BLOCKED = {}                      # reason (digits folded) -> {"count", "first", "last_log"}
LOAD_BLOCKED_REPEAT_SECONDS = 300      # the same refusal is written again, with its count, this often at most


def log_load_blocked(reason):
    """GWSTALL2: a refused load is logged once per distinct reason (the MiB figure wobbles between retries and does not count
    as a new reason); repeats are counted and written at most every LOAD_BLOCKED_REPEAT_SECONDS."""
    key = re.sub(r"\d+", "#", str(reason))
    now = time.time()
    seen = LOAD_BLOCKED.get(key)
    if seen is None or now - seen["last_log"] >= LOAD_BLOCKED_REPEAT_SECONDS:
        count = (seen["count"] + 1) if seen else 1
        since = f" (refused {count} times in the last {int(now - seen['first'])} s)" if seen else ""
        LOAD_BLOCKED[key] = {"count": count, "first": seen["first"] if seen else now, "last_log": now}
        log(f"Load blocked: {reason}{since}")
    else:
        seen["count"] += 1


def load_blocked_cleared():
    """A load was admitted: one summary line for the refusals that were counted silently, then a clean slate."""
    quiet = sum(max(0, item["count"] - 1) for item in LOAD_BLOCKED.values())
    if quiet:
        oldest = min(item["first"] for item in LOAD_BLOCKED.values())
        log(f"Load admitted after {quiet} further refusals over {int(time.time() - oldest)} s")
    LOAD_BLOCKED.clear()


def lighter_layout(model_id, vision, layout):
    """GWSTALL2: the lightest layout of the same model with the same window: the lite twin of short or long, when the model has
    one and the request is not vision. None when there is nothing lighter."""
    twin = LITE_OF.get(layout)
    if twin is None or vision or not lite_enabled(model_id, vision, layout):
        return None
    return twin


def acquire_model(model_id, vision, layout="short"):
    """Reserve the request before checking/loading, closing the ready-to-proxy unload race.

    GWSTALL2: when the wanted layout is refused for lack of memory and no loaded layout can answer, the lite twin (the same
    window, less VRAM) is tried before the client sees an error or the gateway waits; the refused layout is held as an upgrade
    (GWSTALL1), so later requests go straight to the twin until the hold ends."""
    settle_dead_backend()
    deadline = time.monotonic() + G["vramReloadWaitSeconds"]
    ram_deadline = time.monotonic() + int(G.get("ramWaitSeconds", 20)) + 5     # a bound in case the window was opened earlier
    busy_deadline = time.monotonic() + 1.0
    while True:
        # A switch (model, vision, layout) waits a moment for the previous answer to be counted out: an agent sends its next
        # request as soon as the last one ended, a hair before the gateway takes it off the busy count (LAYOUT1c saw a 503).
        # An answer still streaming after that moment is never cut: the switch is refused as before.
        with st.lock:
            other_busy = (st.proc is not None and st.proc.poll() is None and st.busy > 0
                          and (st.model_id != model_id or st.vision != vision or st.layout != layout)
                          and not upgrade_on_hold(model_id, vision, layout))
        if other_busy and time.monotonic() < busy_deadline:
            time.sleep(0.1)
            continue
        with st.lock:
            if GAME.enabled and (GAME.active or not GAME.ready):
                return "Model paused because of a game: " + (GAME.reason or "detector initializing")
            if (G["vramGuard"] or G["vramValve"] or GAME.enabled or G["slotGears"] or G["crashRecovery"]) and RAM_SLOT_DIR is None:
                try:
                    prepare_ram_slot_dir()
                except (OSError, RuntimeError) as e:
                    event("ram_path_failed", f"RAM sessions unavailable; ordinary lifecycle retained: {e}", fallback=True)
            ready = (st.proc and st.proc.poll() is None and st.status == "READY"
                     and st.model_id == model_id and st.vision == vision and st.layout == layout)
            if ready or upgrade_on_hold(model_id, vision, layout):
                st.busy += 1
                st.last_request = time.time()
                return ""
            wanted = governor_layout(model_id, vision, layout)
            twin = lighter_layout(model_id, vision, wanted)
            error, started = "", False
            for attempt, candidate in enumerate([wanted] + ([twin] if twin else [])):
                refused = error
                error, started = acquire_attempt(model_id, vision, candidate)
                if not error:
                    if attempt:
                        hold_upgrade(model_id, wanted, refused)   # the wanted layout did not fit, its twin runs: no retry per request
                    return ""
                if refusal_code(error) not in RESOURCE_CODES:
                    break
            # as before GWSTALL2: only a refusal of the first check waits for VRAM; a refusal inside start_llama comes back at once
            wait_for_vram = (not started and refusal_code(error) in WAIT_FOR_VRAM_CODES and st.override != "OFF" and not st.multimedia_lock)
            # RAMLOGIC2: the RAM ladder's short wait; ram_admission ends it itself (a guarded load or a refusal after ramWaitSeconds)
            wait_for_ram = not started and refusal_code(error) == "ram_wait"
        if wait_for_ram and time.monotonic() < ram_deadline:
            time.sleep(1)
            continue
        if not wait_for_vram or time.monotonic() >= deadline:
            return error
        time.sleep(min(1, max(0, deadline - time.monotonic())))


def layout_switch_cooling(model_id, vision, layout):
    """A2PORT1: a switch to the long layout failed a moment ago for a reason that is not memory (the A2 checkpoint, the start itself): another
    try would repeat it and, worse, would stop and restart the short layout on every retry of the client. Within layoutSwitchRetrySeconds the
    answer is immediate. Returns a Refusal (code layout_switch) or ""."""
    if vision or layout not in ("long", "longlite"):
        return ""
    failed = (LAYOUT_STATE.get("switch_failed") or {}).get((model_id, layout))
    if not failed:
        return ""
    left = failed["until"] - time.time()
    if left <= 0:
        LAYOUT_STATE["switch_failed"].pop((model_id, layout), None)
        return ""
    return layout_switch_refusal(f"the switch failed {int(time.time() - failed['at'])} s ago ({failed['error']}) and is not tried again for {int(left) + 1} s; "
                                 "the short layout stays loaded", left)


def layout_switch_refusal(reason, retry_after):
    refusal = Refusal(f"Could not switch to the long context: {reason}", "layout_switch")
    refusal.retry_after = max(1, int(retry_after + 0.999))
    return refusal


def acquire_attempt(model_id, vision, layout):
    """One admission and start of `layout` under st.lock. Returns (error, started): "" when the request is counted on a running
    server (started now, or a loaded layout that fits answered a refusal), else the refusal; `started` says whether start_llama
    was reached (its refusals never wait, the first check's do)."""
    cooling = layout_switch_cooling(model_id, vision, layout)
    if cooling:
        return cooling, False
    mc = model_config(model_id, vision, layout)
    error = choose_startup_slots(model_id, vision, mc, layout)
    if error:
        layout, mc, error = lite_fallback(model_id, vision, layout, mc, error)
    if not error:
        error = preload_check(mc, reclaim=running_footprint())
    if error and serve_on_loaded_layout(model_id, vision, layout, error):
        return "", False
    if not error:
        st.busy += 1
        previous_layout = st.layout
        switching = previous_layout != layout and st.proc is not None and st.proc.poll() is None
        swap_before = ram_swap_before()
        if switching:
            eta = LAYOUT_STATE.setdefault("durations", {}).get(layout, 5)
            LAYOUT_STATE["switch"] = {"state": "switching", "to": layout, "started_at": time.time(), "eta_s": eta}
        try:
            with memory_pressure_gate(False):
                error = start_llama(model_id, vision, layout=layout)
            if not error:
                ram_swap_check_after_load(model_id, layout, swap_before)
            if error and governor_on() and layout in LITE_OF and (model_id, layout) in GOV["too_big"]:
                layout = governor_layout(model_id, vision, layout)     # the 75 just marked it: start its lite twin
                error = start_llama(model_id, vision, layout=layout) if layout in LITE_OF.values() else error
            if switching:
                ended = time.time()
                LAYOUT_STATE["switch"] = {"state": "failed" if error else "done", "to": layout,
                                          "started_at": LAYOUT_STATE["switch"]["started_at"], "ended_at": ended}
                if not error:
                    LAYOUT_STATE["durations"][layout] = max(1, round(ended - LAYOUT_STATE["switch"]["started_at"]))
            if error and layout in ("long", "longlite") and not vision:
                # LAYOUT1: the long layout did not start. Put the short one back, so the model is still there.
                back = start_llama(model_id, vision, layout="short")
                reason = (f"{error}. " + ("The short layout is loaded again." if not back else f"The short layout did not come back: {back}"))
                if refusal_code(error) in RESOURCE_CODES:
                    # memory: the code stays, the client gets the same Retry-After as for a plain refusal and the next request may try again
                    error = Refusal(f"Could not switch to the long context: {reason}", refusal_code(error))
                else:
                    # A2PORT1: anything else (the checkpoint before the switch, the long start itself) is answered as a switch failure with its
                    # own error code and a Retry-After, and is not repeated for layoutSwitchRetrySeconds
                    error = layout_switch_refusal(reason, G["layoutSwitchRetrySeconds"])
                    LAYOUT_STATE.setdefault("switch_failed", {})[(model_id, layout)] = {
                        "until": time.time() + G["layoutSwitchRetrySeconds"], "at": time.time(), "error": reason[:300]}
                if switching:
                    LAYOUT_STATE["switch"]["error"] = reason[:300]
                event("layout_switch_failed", error)
            elif not error and layout != previous_layout:
                (LAYOUT_STATE.get("switch_failed") or {}).pop((model_id, layout), None)
                LAYOUT_STATE["switched_at"] = time.time()
                timing = LAYOUT_STATE.get("timing") or {}
                event("layout_switched", f"model={model_id} layout={layout} stop={timing.get('stop_s')}s "
                      f"exit={timing.get('exit_s')}s settle={LAYOUT_STATE.get('settle_s')}s load={timing.get('load_s')}s")
        except BaseException:
            st.busy -= 1
            raise
        if error:
            st.busy -= 1
            if serve_on_loaded_layout(model_id, vision, layout, error):   # start_llama refused the switch before touching it
                return "", True
        return error, True
    return error, False


def desk_deep_reasons():
    """MEMV2: desk_need's deep sleep (L2) as a game-mode reason; it carries its own hysteresis."""
    need = getattr(DESK, "need", None)
    if need is None or need.sleep_level < 2:
        return []
    return [f"desk: {need.deep_reason or need.reason}"]


def game_tick(now, reasons, errors=(), settled=()):
    notifications = []
    with st.lock:
        changed = GAME.update(now, reasons, errors, settled)
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
                    conn = lease.get("conn")
                    sock = lease.get("sock") or (conn.sock if conn is not None else None)
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
        if st.status == "ERROR":
            return notifications               # the crash limit stands; the cooldown or `ai-on` ends it, not a game
        st.status = "PARKED_BY_GAME"
        st.last_error = ""
        if st.guard_fallback.startswith(("A2 save failed", "Session checkpoint failed")):
            st.guard_fallback = ""
        GAME.next_save = 0
        GAME.parked = True
        event("game_parked", GAME.reason)
        notifications.append("Модель на паузе из-за игры. Контекст сохранён в RAM.")
    return notifications


# MEMV2: the desk's need drives the governor's reserve and sleep level (governor policy and deskNeed only)
DESK = None


def desk_deepen(why):
    """MEMV2: during light sleep a governor pause means the desk needs the whole card: deep sleep (L2) now, and park the model
    through game mode at once instead of restarting it into the game's way. True when the model is parked for that."""
    need = getattr(DESK, "need", None)
    if need is None or not GAME.enabled or not need.deepen(why):
        return False
    event("desk_deep_sleep", why)
    for message in game_tick(time.monotonic(), desk_deep_reasons(), (), desk_deep_reasons()):
        notify_vram_yield(message)
    return True


def desk_http(method, path, body=None):
    return governor_http(method, path, body, timeout=2.0)


def desk_monitor(stop=None):
    """Own thread: reading the desk (a few subprocesses and /proc files) must not delay the VRAM guard's loop."""
    global DESK
    stop = stop or threading.Event()
    DESK = desk_runtime.DeskRuntime(
        G, desk_http, desk_sources.DeskSources(G), STATE_DIR / "desk-ledger.json",
        lambda kind, text: event(kind, text))
    while not stop.is_set():
        try:
            proc = st.proc
            alive = proc is not None and proc.poll() is None
            ready = alive and st.status == "READY" and not st.vision
            DESK.step(time.monotonic(), proc.pid if alive else None, ready, (proc.pid,) if alive else ())
        except Exception as error:  # noqa: BLE001 - the desk reader never takes the gateway down
            event("desk_need_failed", f"{type(error).__name__}: {error}", fallback=True)
        stop.wait(1.0)
    DESK.flush()


def game_monitor(stop=None):
    stop = stop or threading.Event()
    while not stop.is_set():
        if not GAME.enabled:
            stop.wait(G["gamePollSeconds"])
            continue
        try:
            proc = st.proc
            pid = proc.pid if proc else None
            nvml = nvml_error = None
            if G["gameDetectNvml"]:
                try:
                    nvml = nvml_device()
                except Exception as error:  # noqa: BLE001 - a missing driver is an error sample, not "no sample"
                    nvml_error = getattr(error, "backend", "NVML") + ": " + str(error)
            try:
                reasons, errors = GAME_SIGNALS.collect(pid, nvml, nvml_error=nvml_error)
            except Exception as error:  # noqa: BLE001 - the policy must still see the failure
                reasons, errors = [], ["detector: " + str(error)]
            settled = desk_deep_reasons()
            for message in game_tick(time.monotonic(), reasons + settled, errors, settled):
                notify_vram_yield(message)
        except Exception as error:
            with st.lock:
                GAME.errors = [str(error)]
                event("game_detector_failed", str(error), fallback=True)
        stop.wait(G["gamePollSeconds"])


def context_mode_snapshot():
    model_id = st.model_id or M["defaultModel"]
    return user_modes.context_snapshot(st.context_mode, model_id, M["models"].get(model_id) or {}, st.layout)


def think_status():
    """EFFORT1: the level, the budget it sends and whether budgets are on, for /guardian/status and the think/effort endpoints."""
    model_id = st.model_id or M["defaultModel"]
    return user_modes.think_snapshot(model_id, M["models"].get(model_id) or {}, st.think, budgets=G.get("effortBudgets") is True)


def terse_effective(model_id=None):
    """Is the model's 'be concise' system prompt on? The runtime switch wins; else the model config; else the template default (on)."""
    if st.terse is not None:
        return st.terse
    kw = (M["models"].get(model_id or M["defaultModel"], {}).get("chatTemplateKwargs") or {})
    return bool(kw.get("terse", True))


def supports_terse(model_id):
    """A model has the terse switch when its config says so, or declares the `terse` template variable (Tiel-Coder's template has it, Occamy's does not)."""
    mc = M["models"].get(model_id) or {}
    if "terseSwitch" in mc:
        return bool(mc["terseSwitch"])
    return "terse" in (mc.get("chatTemplateKwargs") or {})


def apply_terse(body, path, model_id=None):
    """With a runtime terse switch set, put it into the request's chat_template_kwargs (the request wins over the server flag), for models that have it."""
    if st.terse is None or not path.endswith("/chat/completions"):
        return body
    if model_id is not None and not supports_terse(model_id):
        return body
    try:
        obj = json.loads(body)
        kwargs = obj.get("chat_template_kwargs")
        obj["chat_template_kwargs"] = {**(kwargs if isinstance(kwargs, dict) else {}), "terse": st.terse}
        return json.dumps(obj).encode()
    except (ValueError, AttributeError, RecursionError):
        return body


def fit_to_context(body, path, model_id, vision):
    """Shorten old tool results so that a chat request fits the model's context window (context_fit.py, CACHE3). Without
    this the server answers 400 and ShuraCode starts a new session with another system message."""
    if not path.endswith("/chat/completions") or G.get("contextFit", True) is False or not body:
        return body
    if not isinstance(body, (bytes, bytearray, str)):
        return body                       # a request already parsed by someone else: not ours to shorten
    try:
        obj = json.loads(body)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return body
    if not isinstance(obj, dict):
        return body
    max_tokens = obj.get("max_tokens") or obj.get("max_completion_tokens")
    session = token_key(model_id, getattr(REQUEST_CTX, "session", ""))
    layout = pure_request_layout(model_id, vision, body, max_tokens or 0, session)
    mc = M["models"].get(model_id) or {}
    can_go_long = bool(mc.get("longContextProfile")) and not vision
    # the answer room is the one the layout choice counts (longContextAnswerReserveTokens), not the client's full max_tokens
    reserve = max_tokens
    if can_go_long and mc.get("longContextAnswerReserveTokens"):
        reserve = layout_estimate.answer_reserve(max_tokens or 0, mc["longContextAnswerReserveTokens"])
    ctx = model_config(model_id, vision, layout).get("ctxSize")
    new, info = context_fit.fit_body(body, ctx, reserve)
    if info and not info["fits"] and ctx and layout == "short" and can_go_long and ctx - info["before"] < CONTEXT_MIN_ANSWER_TOKENS:
        # not even the prompt and a minimal answer fit the short window by the careful estimate: the long layout has the room
        long_ctx = model_config(model_id, vision, "long").get("ctxSize")
        new, long_info = context_fit.fit_body(body, long_ctx, reserve)
        REQUEST_CTX.force_long = True
        event("context_fit", f"request {info['before']} est. tokens leaves no room for an answer in the short window ({ctx}): "
              f"the long layout is used (window {long_ctx})", fallback=False)
        layout, ctx, info = "long", long_ctx, long_info
        if info is None:
            return new
    if info:
        room = (ctx - info["before"]) if ctx else None
        event("context_fit", f"request {info['before']} est. tokens, window budget {info['budget']}: "
              f"{info['trimmed']} tool result(s) shortened, now {info['after']}"
              + ("" if info["fits"] else f" (cannot be shortened; the {ctx} window still leaves ~{room} tokens for the answer)"),
              fallback=not info["fits"] and room is not None and room < CONTEXT_MIN_ANSWER_TOKENS)
        if not info["fits"] and ctx:
            # Only the prompt itself decides whether the server can answer at all. Refuse when both the careful estimate and the
            # calibrated one say the prompt alone is larger than the biggest window we have: an honest error, not a send-and-see.
            ascii_chars, other_chars = layout_estimate.count_chars(body)
            calibrated = TOKENS.estimate(session, ascii_chars, other_chars)[0]
            if info["before"] > ctx and calibrated > ctx:
                event("context_fit_refused", f"prompt ~{int(calibrated)} tokens does not fit the window of {ctx}", fallback=True)
                raise service_requests.RequestRejected(400, {"error": {
                    "code": 400, "type": "exceed_context_size_error", "n_prompt_tokens": int(calibrated), "n_ctx": int(ctx),
                    "message": f"the request exceeds the available context size ({int(calibrated)} tokens, window {int(ctx)}); "
                               "the gateway did not send it"}})
    return new


def fit_request(path, body):
    """The request-level form of fit_to_context, for the shared handler wrapper: the model is the one the body names."""
    if not body or not path.endswith("/chat/completions") or not isinstance(body, (bytes, bytearray, str)):
        return body
    try:
        obj = json.loads(body)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return body
    if not isinstance(obj, dict):
        return body
    name = obj.get("model")
    model_id = name if isinstance(name, str) and name in M["models"] else M["defaultModel"]
    body = switch_counter.prepare(path, body, REQUEST_CTX, G,   # SHADOW1; a fresh ledger per write (sqlite connections are per thread)
                                  lambda: experience_ledger.Ledger() if G.get("experienceLedger") is True else None,
                                  M["models"].get(model_id) or {})
    return fit_to_context(inject_memory(body), path, model_id, False)


def try_lock(seconds):
    return common.try_lock(st.lock, seconds)


def status_snapshot():
    return common.snapshot_without_waiting(st, _status_snapshot_locked)


def _status_snapshot_locked():
    proc = st.proc
    slot_status_models = {mid: gear.snapshot() for mid, gear in GEARS.items()}
    for mid, selection in STARTUP_VRAM_SELECTIONS.items():
        status = slot_status_models.setdefault(mid, {"slots": selection["slots"], "reason": "startup reserve"})
        status["startup_reason"] = selection["reason"]
        if mid in GEARS:
            status["reason"] += "; startup: " + selection["reason"]
        else:
            status["reason"] = "startup reserve: " + selection["reason"]
    valve_models = {mid: valve.snapshot() for mid, valve in VALVES.items()}
    if st.model_id and st.vision:
        actual_slots = model_config(st.model_id, True).get("moeCacheSlots", 0)
        valve_models[st.model_id] = {
            "slots": actual_slots, "configured_max": actual_slots, "levels": [],
            "reason": "vision mode; expert-cache valve inactive", "active": False,
            "slot_cost_mib": None, "slot_cost_source": "inactive", "slot_cost_samples": 0,
        }
    elif st.model_id in valve_models:
        valve_models[st.model_id]["active"] = bool(G["vramValve"] and st.status == "READY" and not governor_on())
    return {
        "terse": terse_effective(st.model_id or None), "terse_override": st.terse,
        "terse_supported": supports_terse(st.model_id or M["defaultModel"]),
        "terse_models": [m for m in M["models"] if supports_terse(m)],
        "game_mode": GAME.snapshot(),
        "desk_need": DESK.snapshot() if DESK is not None else {"enabled": False},
        "crash_recovery_enabled": G["crashRecovery"],
        "gpu_resets": st.gpu_resets.snapshot(),
        "polite_model": POLITE.snapshot(),
        "slot_gears": {"enabled": G["slotGears"] or bool(STARTUP_VRAM_SELECTIONS),
                       "controller_enabled": G["slotGears"], "models": slot_status_models},
        "vram_card_reserve": {"enabled": vram_card_reserve_enabled(),
                              "reserve_mib": G_RAW.get("vramCardReserveMiB", 0),
                              "models": dict(STARTUP_VRAM_SELECTIONS)},
        "prefill_yield": PREFILL_YIELD.snapshot(G.get("prefillCacheYield")),
        "vram_valve": {"enabled": G["vramValve"] and not governor_on(),   # the governor owns the cache: the valve stays out
                       "active": bool(G["vramValve"] and st.model_id and not st.vision and st.status == "READY" and not governor_on()),
                       "models": valve_models},
        "status": st.status, "model": st.model_id, "vision": st.vision, "layout": st.layout,
        "lite": {"available": (lite_enabled(st.model_id, st.vision, "short") or lite_enabled(st.model_id, st.vision, "long")) if st.model_id else False, "since": LITE_STATE["since"] if LITE_STATE["since"] > 0 else None},
        "layout_switch": LAYOUT_STATE.get("switch") or {"state": "idle"}, "layout_estimate": LAYOUT_STATE.get("estimate") or {}, "override": st.override,
        "pause_resume": dict(PAUSE_WAIT),
        "context_mode": context_mode_snapshot(), "think": think_status(), "prompt_progress": PROGRESS.snapshot(),
        "loading_seconds": round(time.time() - st.loading_since, 1) if st.loading_since else None,
        "memory_pressure": st.memory_pressure, "multimedia_lock": st.multimedia_lock, "ram_ladder": ram_ladder_snapshot(),
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
        "default_model": M["defaultModel"], "events": list(st.events),
        "slot_dir": str(RAM_SLOT_DIR if st.ram_backend and RAM_SLOT_DIR is not None else SLOT_DIR),
    }


def _monitor_authorized(handler):
    return monitor_authorized(handler.headers, {**G, "apiKey": G.get("apiKey") or M.get("apiKey")})


def _monitor_gpu():
    pid = st.proc.pid if st.proc and st.proc.poll() is None else None
    return sample_gpu("/sys", lambda: nvml_device().monitor_info(pid))


def _monitor_slots():
    if not (st.proc and st.proc.poll() is None and st.status == "READY"):
        return None
    try:
        code, value = llama_call("GET", "/slots", timeout=0.5)
        return value if code == 200 and isinstance(value, list) else None
    except Exception:
        return None


def _monitor_state_payload():
    def status_provider():
        status = status_snapshot()
        checkpoint = st.ram_checkpoint
        if checkpoint and checkpoint.get("model") == st.model_id and checkpoint.get("vision") == st.vision:
            status["saved_slot"] = True
        return status
    return gateway_snapshot(MONITOR, status_provider, model_config, lambda _s: _monitor_gpu(),
                            lambda _s: _monitor_slots(),
                            lambda status: read_expert_cache_metrics(G["llamaHost"], G["llamaPort"], M.get("apiKey"))
                            if status.get("llama_pid") else None,
                            POLITE.snapshot)


def _is_monitor_request(path):
    return path in ("/monitor", "/monitor/", "/monitor/api/state", "/monitor/api/requests", "/monitor/api/stream")


def _is_api_request(path):
    return path.startswith("/v1/") or path in ("/completion", "/infill", "/completions")


def _monitor_request_model(body):
    try:
        payload = json.loads(body) if body else {}
        mid = payload.get("model") if isinstance(payload, dict) else None
        return mid if mid in M["models"] else M["defaultModel"]
    except Exception:
        return M["defaultModel"]


def _deliberate_refusal(code):
    """503s the gateway sends on purpose (game pause, AI off, GPU job, memory pressure) are not failures."""
    return code == 503 and bool(
        st.override == "OFF" or st.multimedia_lock or st.memory_pressure
        or (GAME.enabled and (GAME.active or not GAME.ready)))


def _finish_monitor_request(handler, status=None, error=False):
    ident = getattr(handler, "_monitor_request_id", None)
    if ident is not None:
        try:
            MONITOR.finish_request(ident, status=status, error=error,
                                   expected=status is not None and _deliberate_refusal(status))
        except Exception:
            pass
        handler._monitor_request_id = None


def _monitor_call(method, *args, **kwargs):
    """The monitor is a side channel: any collector exception is swallowed."""
    try:
        return getattr(MONITOR, method)(*args, **kwargs)
    except Exception:
        return None


RECOVERY_GATE = threading.Lock()
RECOVERABLE_PATHS = ("/v1/chat/completions", "/chat/completions", "/completion")
PASSIVE_METHODS = ("GET", "HEAD")


def gateway_key():
    return G.get("gatewayApiKey") or G.get("monitorApiKey") or G.get("apiKey") or M.get("apiKey") or None


def web_policy():
    policy = WebPolicy(G, G.get("gatewayHost", "127.0.0.1"), gateway_key())
    if G.get("switchCounters") is True:     # SHADOW1: the switch counters are plain reads (numbers and labels only)
        policy.guardian_reads = frozenset(policy.guardian_reads) | {"/guardian/switches", "/guardian/switches.jsonl"}
    return policy


def model_is_ready():
    """Is a model running that a read-only request can be passed to? Never starts one (a health probe, a model list, a favicon
    must not load 20 GB onto the GPU). Such requests are not counted as busy: they must not block an unload or a model switch,
    and one that loses its server meanwhile just gets a clean 502."""
    return bool(st.proc and st.proc.poll() is None and st.status == "READY" and st.override != "OFF"
                and not (GAME.enabled and (GAME.active or not GAME.ready)))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def send_json(self, code, obj, retry_after=None):
        request_id = getattr(self, "_monitor_request_id", None)
        if request_id is not None:
            _monitor_call("set_status", request_id, code)
            if code >= 400:
                _finish_monitor_request(self, status=code, error=True)
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

    def guardian_endpoint(self, path, body=None):
        busy_reply = {"error": "The gateway is loading a model; retry in a moment"}
        if path in ("/guardian/switches", "/guardian/switches.jsonl") and G.get("switchCounters") is True:   # SHADOW1
            return switch_counter.guardian(self, path, G)
        if path.startswith("/guardian/experience/") and G.get("experienceLedger") is True:
            return self.experience_endpoint(path[len("/guardian/experience/"):], body)
        if path.startswith("/guardian/memory/") and G.get("memoryEnabled") is True:
            return self.memory_endpoint(path[len("/guardian/memory/"):], body)
        if path == "/guardian/status":
            self.send_json(200, status_snapshot())
        elif path == "/guardian/prepare-rollback":
            with try_lock(5) as got:
                if not got:
                    return self.send_json(503, busy_reply)
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
            with try_lock(5) as got:
                if not got:
                    return self.send_json(503, busy_reply)
                was_active = GAME.active
                GAME.manual_switch(path.endswith("pause"), time.monotonic())
                if was_active and not GAME.active:
                    if st.status == "PARKED_BY_GAME":
                        st.status = "UNLOADED"
                    GAME.parked = False
                    notify_vram_yield("Игра окончена. Модель загрузится по следующему запросу.")
            self.send_json(200, {"ok": True, "game_mode": GAME.snapshot()})
        elif path == "/guardian/ai-on":
            try:
                OVERRIDE_FLAG.write_text("ON")
            except OSError as e:
                return self.send_json(503, {"error": f"Cannot persist the switch: {e}"})
            st.override = "AUTO"
            with try_lock(5) as got:
                if got and st.status == "ERROR":
                    st.status = "UNLOADED"
                    clear_error_state()
            self.send_json(200, {"ok": True, "override": "AUTO"})
        elif path == "/guardian/ai-off":
            try:
                OVERRIDE_FLAG.write_text("OFF")
            except OSError as e:
                return self.send_json(503, {"error": f"Cannot persist the switch, so nothing was changed: {e}"})
            st.override = "OFF"          # a model load in progress notices this and stops
            with try_lock(30) as got:
                if not got:
                    return self.send_json(503, {"error": "AI is off, but the model could not be stopped yet (a load is still running)"})
                if not stop_llama_locked("manual ai-off", save=True):
                    return self.send_json(503, {"error": f"AI is off, but the server could not be stopped: {st.last_error}"})
            self.send_json(200, {"ok": True, "override": "OFF"})
        elif path in ("/guardian/terse", "/guardian/terse-on", "/guardian/terse-off", "/guardian/terse-default"):
            if path != "/guardian/terse":
                value = {"on": True, "off": False, "default": None}[path.rsplit("-", 1)[1]]
                try:
                    if value is None:
                        TERSE_FLAG.unlink(missing_ok=True)
                    else:
                        TERSE_FLAG.write_text("ON" if value else "OFF")
                except OSError as e:
                    return self.send_json(503, {"error": f"Cannot persist the terse switch: {e}"})
                st.terse = value
            self.send_json(200, {"ok": True, "terse": terse_effective(), "override": st.terse})
        elif path in ("/guardian/context-mode", "/guardian/context-mode-auto", "/guardian/context-mode-extended"):
            if path != "/guardian/context-mode":
                mode = path.rsplit("-", 1)[1]
                try:
                    user_modes.write_flag(CONTEXT_MODE_FLAG, None if mode == "auto" else mode)
                except OSError as e:
                    return self.send_json(503, {"error": f"Cannot persist the context mode: {e}"})
                if mode != st.context_mode:
                    event("context_mode", f"mode={mode}")
                st.context_mode = mode
            self.send_json(200, {"ok": True, **context_mode_snapshot()})
        elif path in ("/guardian/think", "/guardian/effort") or path.startswith(("/guardian/think-", "/guardian/effort-")):
            # THINK2 / EFFORT1: /guardian/effort-<level> is the same switch under the name the slider uses
            if path not in ("/guardian/think", "/guardian/effort"):
                level = path.split("-", 1)[1]
                if level not in user_modes.THINK_LEVELS + ("default",):
                    return self.send_json(404, {"error": f"Unknown effort level {level!r}; use {', '.join(user_modes.THINK_LEVELS)} or default"})
                level = None if level == "default" else level
                try:
                    user_modes.write_flag(THINK_FLAG, level)
                except OSError as e:
                    return self.send_json(503, {"error": f"Cannot persist the thinking level: {e}"})
                if level != st.think:
                    event("think_level", f"level={level or 'default'}")
                st.think = level
            self.send_json(200, {"ok": True, **think_status()})
        elif path == "/guardian/warm":
            # Loads the model and restores the session only when the valves allow it; never loads a model on its own otherwise.
            if not session_cache_on():
                return self.send_json(404, {"error": "session cache is off (sessionCacheEnabled)"})
            sid = self.headers.get("X-Shura-Session", "")
            if session_cache.valid_session_id(sid):
                SESSION["id"] = sid
            reason = warm_refusal()
            if reason:
                return self.send_json(200, {"warm": False, "reason": reason})
            if st.status == "READY":
                return self.send_json(200, {"warm": True, "status": "READY", "restored": st.slot_session == SESSION["id"]})
            session_prepare_restore(M["defaultModel"], False)
            err = acquire_model(M["defaultModel"], False)
            if err:
                return self.send_json(200, {"warm": False, "reason": err})
            self.send_json(200, {"warm": True, "status": st.status, "restored": st.slot_session == SESSION["id"]})
        elif path == "/guardian/unload":
            force = "force=1" in getattr(self, "_query", "").split("&")
            end = time.monotonic() + 2           # an answer that has just been delivered may still be closing its books
            while st.busy and not force and time.monotonic() < end:
                time.sleep(0.05)
            with try_lock(5) as got:
                if not got:
                    return self.send_json(503, busy_reply)
                if st.busy and not force:
                    return self.send_json(409, {"error": "Response in progress; unload would cut it (add ?force=1 to do it anyway)",
                                                "busy_requests": st.busy})
                if not stop_llama_locked("manual unload", save=True):
                    return self.send_json(503, {"error": st.last_error})
            # Answer once the memory is really free: a request right after an unload otherwise found the card still full (AMD2b)
            wait_for_llama_exit(G["llamaPort"], timeout=10)
            settled = wait_for_vram_settle()
            self.send_json(200, {"ok": True, "vram_free_mib": settled})
        elif path == "/guardian/multimedia-lock":
            st.multimedia_lock = True    # no new load starts, and one in progress stops
            with try_lock(30) as got:
                if not got:
                    return self.send_json(503, {"error": "GPU is reserved, but a model load is still running; retry"})
                if not stop_llama_locked("multimedia pipeline acquiring GPU", save=True):
                    return self.send_json(503, {"error": f"GPU is reserved, but the server could not be stopped: {st.last_error}"})
            self.send_json(200, {"ok": True})
        elif path == "/guardian/multimedia-unlock":
            st.multimedia_lock = False
            self.send_json(200, {"ok": True})
        else:
            self.send_json(404, {"error": f"unknown guardian endpoint {path}"})

    def memory_endpoint(self, op, body):
        """LL1: /guardian/memory/<op> over gateway/lifelong_store.py. Reached only with memoryEnabled true (otherwise these paths are
        unknown guardian endpoints, as before). GET: search?q=&scope=&limit=, read?key=, list, export, stats, summary; POST (JSON body): save, fix, capture,
        forget, restore, import, resolve. Errors of the store are 400 with the store's message."""
        if op == "session-token":
            if self.command != "POST":
                return self.send_json(405, {"error": "memory/session-token needs POST"})
            # Loopback only: the client plugin keeps this capability in its closure, never in config/env.
            if self.client_address[0] not in ("127.0.0.1", "::1"):
                return self.send_json(403, {"error": "local client only"})
            return self.send_json(200, {"token": _LIFELONG_CONFIRM_TOKEN})
        if op == "capture" and G.get("lifelongCapture") is not True:
            return self.send_json(404, {"error": "lifelongCapture is off"})
        import urllib.parse
        q = {k: v[0] for k, v in urllib.parse.parse_qs(getattr(self, "_query", "") or "").items()}
        try:
            store = memory_store()
        except Exception as error:   # noqa: BLE001 - an unreadable database is a 500, never a dropped connection
            log.error("memory: store unavailable: %s", error)
            return self.send_json(500, {"error": f"memory store unavailable: {error}"})
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except (ValueError, UnicodeDecodeError):
            return self.send_json(400, {"error": "memory: the body must be JSON"})
        if not isinstance(data, dict):
            return self.send_json(400, {"error": "memory: the body must be a JSON object"})
        def text_of(value, limit=4000):
            if value is None:
                return ""
            if not isinstance(value, (str, int, float)):
                raise ValueError("text fields must be strings")
            return str(value)[:limit]
        try:
            authorized = secrets.compare_digest(self.headers.get("X-Shura-Confirmation", ""), _LIFELONG_CONFIRM_TOKEN)
            actor = data.get("actor", "user")
            if actor not in ("user", "shura", "code"):
                raise ValueError("actor must be user, shura or code")
            tags = data.get("tags") or []
            if not isinstance(tags, list):
                raise ValueError("tags must be a list")
            if op == "list":
                return self.send_json(200, {"results": store.list_all(scope=text_of(q.get("scope"), 80) or None)})
            if op == "search":
                args = {**q, **data}
                limit = max(1, min(int(args.get("limit") or 8), 100))
                return self.send_json(200, {"results": store.search(text_of(args.get("q") or args.get("query")), limit=limit, scope=text_of(args.get("scope"), 80) or None)})
            if op == "read":
                return self.send_json(200, store.read(text_of(q.get("key") or data.get("key"), 100)))
            if op == "export":
                text = store.export().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Content-Length", str(len(text)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(text)
                return None
            if op == "summary":
                return self.send_json(200, store.summary(text_of(q.get("model"), 60) or "tiel"))
            if op == "capture":
                if self.command != "POST":
                    return self.send_json(405, {"error": "memory/capture needs POST"})
                return self.send_json(200, store.capture(data.get("signal"), data.get("text"), data.get("event_id")))
            if op == "stats":
                return self.send_json(200, store.stats())
            if self.command != "POST":
                return self.send_json(405, {"error": f"memory/{op} needs POST"})
            if op == "save":
                if not authorized:
                    data["trust"] = "proposed"
                    candidate_key = text_of(data.get("key") or data.get("name"), 100)
                    if candidate_key:
                        try:
                            store.read(candidate_key)
                        except lifelong_store.StoreError:
                            pass
                        else:
                            return self.send_json(403, {"error": "updating an existing record requires user confirmation"})
                elif data.get("trust", "confirmed") == "confirmed" and data.get("confirm") is not True:
                    data["trust"] = "proposed"
                title = text_of(data.get("title") or data.get("description"), 1000)
                return self.send_json(200, store.save(text_of(data.get("key") or data.get("name"), 100) or title[:40], title,
                                                      text_of(data.get("body") or data.get("content"), 20000), kind=text_of(data.get("kind"), 20) or "fact",
                                                      scope=text_of(data.get("scope"), 80) or "global", source=text_of(data.get("source"), 200) or "user_command",
                                                      trust=text_of(data.get("trust"), 20) or "proposed", tags=[text_of(t, 60) for t in tags], actor=actor,
                                                      confirmed_by="user" if authorized else "shura"))
            if op == "fix":
                if not authorized:
                    original = store.read(text_of(data.get("key"), 100))
                    proposed_key = (original["key"] + "-proposal-" + hashlib.sha256(text_of(data.get("body"), 20000).encode()).hexdigest()[:8])[:80]
                    proposed = store.save(proposed_key, text_of(data.get("title"), 1000) or original["title"],
                                          text_of(data.get("body"), 20000) or original["body"], kind=original["kind"],
                                          scope=original["scope"], source="capture", trust="proposed", actor="shura", confirmed_by="shura")
                    return self.send_json(200, {"record": proposed["record"], "proposal_for": original["key"]})
                return self.send_json(200, store.fix(text_of(data.get("key"), 100), title=text_of(data.get("title"), 1000) or None,
                                                     body=text_of(data.get("body"), 20000) or None, kind=text_of(data.get("kind"), 20) or None,
                                                     actor=actor, relation=text_of(data.get("relation"), 20) or "replace"))
            if op == "forget":
                if data.get("purge") and not authorized:
                    return self.send_json(403, {"error": "user confirmation token required for purge"})
                # the purge export always lands next to the database: a client cannot choose where on disk the record is written
                return self.send_json(200, store.forget(text_of(data.get("key"), 100), actor=actor, purge=bool(data.get("purge"))))
            if op == "restore":
                if not authorized:
                    return self.send_json(403, {"error": "user confirmation token required"})
                return self.send_json(200, store.restore(text_of(data.get("key"), 100), actor=actor))
            if op == "confirm":
                if not authorized:
                    return self.send_json(403, {"error": "user confirmation token required"})
                return self.send_json(200, store.confirm(text_of(data.get("key"), 100), actor="user"))
            if op == "import":
                if not authorized:
                    return self.send_json(200, store.import_jsonl(text_of(data.get("text"), 50_000_000), actor="shura", external_source="mcp-import"))
                if data.get("confirm") is not True:
                    return self.send_json(400, {"error": "confirmed import requires explicit confirm"})
                return self.send_json(200, store.import_jsonl(text_of(data.get("text"), 50_000_000), actor="user", external_source=text_of(data.get("external_source"), 200) or None))
            if op == "import-legacy":
                if not authorized:
                    return self.send_json(403, {"error": "user confirmation token required"})
                return self.send_json(200, store.import_legacy())      # only the configured legacy folder, never a client-chosen path
            if op == "resolve":
                if not authorized:
                    return self.send_json(403, {"error": "user confirmation token required"})
                return self.send_json(200, store.resolve(text_of(data.get("a"), 100), text_of(data.get("b"), 100), text_of(data.get("status"), 20) or "both_true", actor=actor))
        except (lifelong_store.StoreError, ValueError, TypeError) as error:
            return self.send_json(400, {"error": str(error)})
        except (sqlite3.Error, OSError) as error:
            log.error("memory: %s", error)
            return self.send_json(500, {"error": f"memory store: {error}"})
        return self.send_json(404, {"error": f"unknown memory endpoint {op}"})

    def experience_endpoint(self, op, body):
        if self.command != "POST":
            return self.send_json(405, {"error": "experience endpoints need POST"})
        if op == "session-token":
            if self.client_address[0] not in ("127.0.0.1", "::1"):
                return self.send_json(403, {"error": "local client only"})
            return self.send_json(200, {"token": _LIFELONG_CONFIRM_TOKEN})
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
            if not isinstance(data, dict): raise ValueError("body must be a JSON object")
            ledger = experience_ledger_store()
            privacy = data.get("privacy", "work")
            if op == "consent": result = ledger.set_consent(data.get("name"), data.get("enabled"))
            elif op == "task": result = ledger.add_task(data.get("request"), data.get("repo_snapshot"), data.get("tests"), data.get("repo"), data.get("block"), privacy, data.get("author", "unknown"))
            elif op == "outcome": result = ledger.add_outcome(data.get("task_id"), data.get("answer"), data.get("status"), data.get("test_results"), privacy, data.get("context"), data.get("author", "unknown"))
            elif op == "correction": result = ledger.add_correction(data.get("task_id"), data.get("answer"), data.get("correction"), privacy, data.get("author", "unknown"))
            elif op == "gate": result = ledger.add_gate_decision(data.get("decision"), data.get("details"), privacy)
            elif op == "growth-anchor": result = ledger.set_growth_anchor(data.get("baseline"), data.get("candidate"), data.get("ci_low"), data.get("ci_high"), data.get("task_count"), privacy)
            elif op == "growth": result = ledger.growth()
            elif op == "exam-answer": result = ledger.add_exam_answer(data.get("exam_id"), data.get("task_id"), data.get("answer"), privacy)
            elif op == "pack-stack": result = ledger.add_pack_stack(data.get("states"), privacy)
            elif op == "list": result = {"records": ledger.list()}
            elif op == "export":
                raw = ledger.export().encode("utf-8")
                self.send_response(200); self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw); return None
            elif op == "stats": result = ledger.stats()
            elif op == "forget":
                if not secrets.compare_digest(self.headers.get("X-Shura-Confirmation", ""), _LIFELONG_CONFIRM_TOKEN):
                    return self.send_json(403, {"error": "user confirmation token required"})
                result = ledger.forget(data.get("id"))
            elif op == "migrate-capture":
                if G.get("memoryEnabled") is not True: raise ValueError("memoryEnabled is required for capture migration")
                result = ledger.migrate_capture(G.get("memoryPath") or lifelong_store.DEFAULT_DB)
            else: return self.send_json(404, {"error": f"unknown experience endpoint {op}"})
            return self.send_json(200, result)
        except (experience_ledger.LedgerError, ValueError, TypeError, KeyError, UnicodeDecodeError) as error:
            return self.send_json(400, {"error": str(error)})
        except (sqlite3.Error, OSError) as error:
            return self.send_json(500, {"error": f"experience ledger: {error}"})

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
        if path.startswith("/monitor") and not _is_monitor_request(path):
            return self.send_json(404, {"error": "Unknown monitor endpoint"})
        if _is_monitor_request(path):
            return monitor_route(self, path, _monitor_state_payload, MONITOR.requests,
                                 authorized=_monitor_authorized(self), monitor_instance=MONITOR)
        body = self.read_body()
        if body is None:
            return None

        if path.startswith("/guardian/"):
            return self.guardian_endpoint(path, body)

        if self.command in PASSIVE_METHODS:
            return self.handle_passive(path)
        REQUEST_CTX.session = self.headers.get("X-Shura-Session", "")
        REQUEST_CTX.force_long = False
        REQUEST_CTX.project = self.headers.get("X-Shura-Project", "")      # SHADOW1: optional slice labels from the client
        REQUEST_CTX.task_kind = self.headers.get("X-Shura-Task-Kind", "")
        REQUEST_CTX.switch = None
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
            # Always the configured ids, loaded or not: they are the names a request can use, and the list must not change
            # (llama-server's own list names the model file) depending on whether something happens to be loaded.
            return self.send_json(200, {"object": "list", "data": [{"id": mid, "object": "model", "owned_by": "shura"} for mid in M["models"]]})
        if model_is_ready():
            return self.proxy_game(b"") if GAME.enabled else self.proxy(b"")
        return self.send_api_error(503, "The model is not loaded. Send a generation request to load it.", "service_unavailable",
                                   "SHURA_MODEL_UNLOADED")

    def handle_generation(self, path, body):
        if _is_api_request(path):
            self._monitor_request_id = _monitor_call("begin_request", _monitor_request_model(body), path, self.command)

        st.last_request = time.time()
        pause = G["gameRetrySeconds"]
        if GAME.enabled and (GAME.active or not GAME.ready):
            return self.send_api_error(503, "Модель на паузе из-за игры", "service_unavailable", "SHURA_GAME_PAUSE", pause,
                                       reason=GAME.reason or "detector initializing")
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
            body = apply_terse(body, path, model_id)
            if path.endswith("/chat/completions"):
                body = user_modes.apply_think(body, st.think, M["models"].get(model_id) or {}, budgets=G.get("effortBudgets") is True)
                body = prefill_progress.ask_for_progress(body, G.get("promptProgress") is True)

        if G.get("ramGuard") is True and body:
            grown = len(body) - RAM_LAST["len"]
            if RAM_VALVE.level != "ok" and grown > G["ramValveDeferChars"] and not wait_room_for_prompt(grown):
                event("ram_valve_defer_prefill", f"level={RAM_VALVE.level} new_chars={grown}: large new prompt waits", fallback=False)
                return self.send_api_error(503, "System memory is tight: a large new prompt waits until it eases; retry.",
                                           "service_unavailable", "SHURA_RAM_TIGHT", G.get("memoryAdmissionRetryAfterSeconds", 30))
            RAM_LAST["len"] = len(body)
        recovery = (G["crashRecovery"] or G["vramRetry"] or G["gpuResetRetry"]) and self.command == "POST" and path in RECOVERABLE_PATHS
        if recovery and not RECOVERY_GATE.acquire(timeout=G["recoveryQueueSeconds"]):
            # One protected generation at a time, the others wait their turn (llama-server would queue them too).
            return self.send_api_error(503, "Another response is taking too long; retry", "service_unavailable", "SHURA_QUEUE_TIMEOUT", 5)
        try:
            if session_cache_on() and body:
                sid = self.headers.get("X-Shura-Session", "")
                if session_cache.valid_session_id(sid):
                    if sid != SESSION["id"]:
                        SESSION["saved_tokens"] = 0
                    SESSION["id"] = sid
                    text = body.decode("utf-8", "replace") if isinstance(body, (bytes, bytearray)) else str(body)
                    SESSION["prefix"] = session_cache.prefix_hash(text)
            session_prepare_restore(model_id, vision)
            # the server's token count is reused only inside one session (X-Shura-Session): other clients are sized by characters
            feedback_key = token_key(model_id, self.headers.get("X-Shura-Session", ""))
            self._usage_seen = None
            layout = choose_request_layout(model_id, vision, body, request_max_tokens(body) if body else 0, time.time(), feedback_key)
            err = acquire_model(model_id, vision, layout)
            if err:
                retry_after = (getattr(err, "retry_after", G["layoutSwitchRetrySeconds"]) if refusal_code(err) == "layout_switch" else
                               G.get("memoryAdmissionRetryAfterSeconds", 30)
                               if err.startswith("Model load deferred:") or refusal_code(err) in ("ram", "ram_wait", "ram_valve", "pressure") else
                               G["gameRetrySeconds"] if GAME.enabled and GAME.active else
                               VRAM_RETRY_AFTER_SECONDS if refusal_code(err) in WAIT_FOR_VRAM_CODES else None)
                return self.send_api_error(503, f"Could not start model '{model_id}': {err}", "service_unavailable",
                                           "SHURA_LAYOUT_SWITCH_FAILED" if refusal_code(err) == "layout_switch" else "SHURA_MODEL_UNAVAILABLE",
                                           retry_after, reason=err)
            st.slot_dirty = True
            yield_back = prefill_yield_begin(model_id, vision, body)
            if G.get("promptProgress") is True and path.endswith("/chat/completions"):
                PROGRESS.start()
            try:
                if recovery:
                    if G["crashRecovery"] and not self.protect_session():   # vramRetry alone replays without a checkpoint: no per-request save
                        return None
                    self.proxy_recoverable(body, model_id, vision)
                elif GAME.enabled:
                    self.proxy_game(body)
                else:
                    self.proxy(body)
            finally:
                prefill_yield_end(yield_back, model_id)  # restore while still counted busy, so the valve stays out
                PROGRESS.finish()
                if body and getattr(self, "_usage_seen", None):
                    TOKENS.commit(feedback_key, sum(layout_estimate.count_chars(body)), self._usage_seen)
                with st.lock:
                    st.busy -= 1
                    st.last_request = time.time()
                _finish_monitor_request(self, status=500, error=True)
        finally:
            if recovery:
                RECOVERY_GATE.release()
        return None

    def protect_session(self):
        """crashRecovery: a checkpoint is saved before the answer starts. It needs the slot idle, so wait for any other request that
        is still in flight (a title, a model list) instead of refusing this one."""
        deadline = time.monotonic() + G["recoveryQueueSeconds"]
        while True:
            with st.lock:
                if st.busy == 1:
                    if save_slot_locked(ram_only=True):
                        return True
                    break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        self.send_api_error(503, "Cannot protect the session before generation", "service_unavailable", "SHURA_CHECKPOINT_NOT_READY", 5)
        return False

    hop_by_hop = HOP_BY_HOP

    def recovery_event(self, reason, detail):
        event(reason, detail, fallback=True)

    def observe(self, kind, value):
        request_id = getattr(self, "_monitor_request_id", None)
        if kind == "status":
            _monitor_call("set_status", request_id, value)
        elif kind == "chunk":
            _monitor_call("observe_sse", request_id, value)
            PROGRESS.observe(value)
            self._usage_seen = layout_estimate.parse_usage(value) or getattr(self, "_usage_seen", None)
            switch_counter.observe(REQUEST_CTX, value)      # SHADOW1
        elif kind == "json":
            _monitor_call("observe_json", request_id, value)
            self._usage_seen = layout_estimate.parse_usage(value) or getattr(self, "_usage_seen", None)
            switch_counter.observe(REQUEST_CTX, value)
        elif kind == "empty_turn":
            event(f"empty_turn_{value}", "a turn ended at the output limit with no text and no tool call (emptyTurnContinue)")

    def stalled(self):
        proc = st.proc
        if proc is not None:
            threading.Thread(target=declare_hung, args=(proc, f"no data for {G['upstreamIdleSeconds']} s in the middle of an answer"),
                             daemon=True, name="shura-hung-server").start()

    def proxy_recoverable(self, body, model_id, vision):
        proc = st.proc
        request_id = getattr(self, "_monitor_request_id", None)
        monitor_status = {"code": None}
        lease = {"cancel": threading.Event(), "conn": None}
        key = id(self)
        if GAME.enabled:
            with st.lock:
                GAME_REQUESTS[key] = lease
        def register(conn):
            lease["conn"] = conn
        def recover():
            # The socket can close just before waitpid publishes SIGKILL/exit status.
            # A short bounded wait avoids returning a false 503 in that race.
            if lease["cancel"].is_set():
                event("response_recovery_refused", "Request was cancelled before replay", fallback=True)
                return False
            if not wait_for_process_exit(proc):
                event("response_recovery_refused", "Backend process did not exit after the upstream disconnect", fallback=True)
                return False
            with st.lock:
                if st.busy != 1 or st.override == "OFF" or GAME.active or st.status == "ERROR":
                    event("response_recovery_refused",
                          f"Lifecycle changed before replay (busy={st.busy}, mode={st.override}, game={GAME.active}, status={st.status})",
                          fallback=True)
                    return False
                if st.proc is not None and st.proc is not proc:
                    event("response_recovery_refused", "A different backend process replaced the failed child", fallback=True)
                    return False
                limit_reached = False
                if st.proc is proc:
                    record_server_crash(proc, "request interrupted before first client byte")
                    limit_reached = note_crash(proc=proc)
                if limit_reached or (proc.pid not in st.gpu_reset_pids and st.crash_count >= G["crashRestartMaxAttempts"]):
                    st.status = "ERROR"
                    st.last_error = "Backend crash restart limit reached"
                    event("response_recovery_refused", st.last_error, fallback=True)
                    return False
                st.proc = None
                st.status = "UNLOADED"
            # Release lifecycle lock so the monitor can publish post-crash admission data.
            if not wait_fresh_vram(time.time()):
                event("response_recovery_refused", "No fresh VRAM sample after backend exit", fallback=True)
                return False
            with st.lock:
                if st.busy != 1 or st.override == "OFF" or GAME.active:
                    event("response_recovery_refused",
                          f"Lifecycle changed while waiting for VRAM (busy={st.busy}, mode={st.override}, game={GAME.active})",
                          fallback=True)
                    return False
                with memory_pressure_gate(False):
                    error = start_llama(model_id, vision, layout=st.layout)
                if not error:
                    event("response_restarted", "Owned backend restored through A2; replay before first byte")
                else:
                    event("response_recovery_refused", f"Backend restart failed: {error}", fallback=True)
                return not error
        pause_resume = pause_resume_for(proc, model_id, vision, lease, body)
        try:
            def observe(kind, value, is_stream):
                if kind == "status":
                    monitor_status["code"] = value
                    _monitor_call("set_status", request_id, value)
                elif kind == "chunk" and is_stream:
                    _monitor_call("observe_sse", request_id, value)
                    PROGRESS.observe(value)
                    self._usage_seen = layout_estimate.parse_usage(value) or getattr(self, "_usage_seen", None)
                    switch_counter.observe(REQUEST_CTX, value)      # SHADOW1
                elif kind == "body" and not is_stream:
                    _monitor_call("observe_json", request_id, value)
                    self._usage_seen = layout_estimate.parse_usage(value) or getattr(self, "_usage_seen", None)
                    switch_counter.observe(REQUEST_CTX, value)

            forward_recovery(self, body,
                             lambda: http.client.HTTPConnection(G["llamaHost"], G["llamaPort"], timeout=G["upstreamFirstByteSeconds"]),
                             recover, lease["cancel"], register, observe=observe,
                             idle_seconds=G["upstreamIdleSeconds"], on_stall=self.stalled, watch_client=G["watchClientLeave"],
                             retry_before_content=(True if G["vramRetry"] else (lambda: killed_by_gpu_reset(proc)) if G["gpuResetRetry"] else False), pause_resume=pause_resume if governor_on() else None)
        finally:
            code = monitor_status["code"]
            _finish_monitor_request(self, status=code or 503, error=not code or code >= 400)
            if GAME.enabled:
                with st.lock:
                    GAME_REQUESTS.pop(key, None)

    def proxy_game(self, body):
        """Same relay, but the game monitor may cut it: after the drain time the client gets an explicit error."""
        key = id(self)
        lease = {"conn": None, "cancel": threading.Event(), "retry_after": G["gameRetrySeconds"]}
        with st.lock:
            GAME_REQUESTS[key] = lease
        try:
            self.proxy(body, lease)
        finally:
            with st.lock:
                GAME_REQUESTS.pop(key, None)

    def proxy(self, body, lease=None):
        guard_lang = ""
        try:
            request_obj = json.loads(body) if body else {}
            settings = M["models"].get(st.model_id, {})
            kwargs = request_obj.get("chat_template_kwargs") or {}
            effort = request_obj.get("reasoning_effort") or kwargs.get("reasoning_effort")
            if not effort:
                session_level = user_modes.effective_think_level(settings, st.think)
                effort = (settings.get("thinkLevels") or {}).get(session_level, {}).get("reasoning_effort")
            is_high = isinstance(effort, str) and effort.lower() in ("high", "xhigh", "max")
            if request_obj.get("stream") is True and is_high:
                guard_lang = lang_guard.guard_for(request_obj, self.path, settings)
        except (ValueError, TypeError, AttributeError):
            pass
        body = script_guard.apply(body, self.path, M["models"].get(st.model_id, {}))   # ORNFIX0: after all size accounting, so the long bias list never counts as prompt
        result = forward_request(self, body, G["llamaHost"], G["llamaPort"], G, lease=lease, observe=self.observe, on_stall=self.stalled,
                                 pause_resume=pause_resume_for(st.proc, st.model_id, st.vision, lease, body) if governor_on() else None,
                                 lang_guard_lang=guard_lang)
        _finish_monitor_request(self, status=result["status"], error=result["error"])

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = handle_any


def sweep_slot_leftovers():
    """A save interrupted by a kill leaves a half-written file behind; nothing else would ever remove it."""
    try:
        for folder in (SLOT_DIR, spill_dir()):
            for pattern in ("*.pending", ".slot-*", ".spill-*.tmp", ".spill-*.part", ".out-*", ".adopt-*", "*.json.tmp"):
                for leftover in folder.glob(pattern):
                    if leftover.is_file() and not leftover.is_symlink():
                        leftover.unlink(missing_ok=True)
    except OSError:
        pass


def kill_server(proc):
    common.kill_server(proc)


def shutdown(signum, _frame):
    log(f"Signal {signum}: saving active slot and stopping llama-server")
    st.shutting_down.set()            # no new load starts, and one in progress gives up within a second
    got = st.lock.acquire(timeout=G["shutdownLockSeconds"])
    try:
        if got:
            if not stop_llama_locked("gateway shutdown", save=True):
                stop_llama_locked("gateway shutdown (session not saved)", save=False)
        else:
            kill_server(st.proc)      # a loading thread still holds the lock: the child must not outlive the gateway
    finally:
        if got:
            st.lock.release()
    sys.exit(0)


def disable_core_dumps():
    """A crashing llama-server would otherwise leave a core dump of its whole memory (10 GB seen on 2026-10-03) in /var/lib/systemd/coredump:
    SSD wear for nothing. The soft limit is inherited by the servers this gateway starts; the hard limit stays, so nothing is locked."""
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_CORE)
        resource.setrlimit(resource.RLIMIT_CORE, (0, hard))
        return True
    except (ValueError, OSError):
        return False


def main():
    disable_core_dumps()
    use_legacy = legacy_gateway_selected(G, GAME.enabled)
    warn_misplaced_pressure_options()
    if use_legacy:
        warn_legacy_pressure_options()
    else:
        validate_load_pressure_config(G)
        validate_vram_card_reserve_config(G_RAW)
        validate_vram_policy_config(G_RAW)
        validate_ram_config(G_RAW)
        validate_vram_valve_config(G_RAW)
        validate_prefill_yield_config(G_RAW)
        validate_model_extra_env(M["models"])
        validate_long_context_config(M["models"])
        validate_lite_profile_config(M["models"], G_RAW)
        validate_server_mem_config(G_RAW)
        desk_runtime.validate_desk_config(G_RAW)
        gpu_reset.validate_config(G_RAW)
        user_modes.validate_think_config(M["models"])
        if type(G_RAW.get("promptProgress", False)) is not bool:
            raise ValueError("Invalid guardian config: promptProgress must be true or false")
        if type(G_RAW.get("layoutFastStop", False)) is not bool:
            raise ValueError("Invalid guardian config: layoutFastStop must be true or false")
        if G.get("vramValve"):
            validate_vram_valve_model_configs(M["models"])
        if type(G.get("vramRetry")) is not bool:
            raise ValueError("Invalid guardian config: vramRetry must be true or false")
    try:
        POLITE.scheduler.recover()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        POLITE.available = False
        POLITE.error = str(error)
        event("cpu_recovery_failed", str(error), fallback=True)
    if use_legacy:
        return legacy_main()
    if (G["slotGears"] or G["crashRecovery"]) and int(G.get("slotSaveCheckpoints", 0)) < 1:
        raise RuntimeError("slotGears requires slotSaveCheckpoints >= 1")
    if G.get("vramValve"):
        if not server_supports("--moe-cache-live-resize"):
            raise RuntimeError("vramValve requires a llama-server that supports --moe-cache-live-resize")
        if not server_supports("--slot-save-checkpoints") and not server_supports("--ctx-checkpoints"):
            raise RuntimeError("vramValve requires a llama-server with slot/context checkpoint support")
        for model_id, settings in M["models"].items():
            if type(settings.get("moeCacheSlots")) is not int or settings["moeCacheSlots"] < 1 \
                    or not settings.get("moeCacheProfile"):
                raise RuntimeError(f"vramValve requires an expert-cache profile and positive slot count for {model_id}")
    if G.get("prefillCacheYield") and not G.get("vramValve"):
        if not server_supports("--moe-cache-live-resize"):
            raise RuntimeError("prefillCacheYield requires a llama-server that supports --moe-cache-live-resize")
        for model_id, settings in M["models"].items():
            if type(settings.get("moeCacheSlots")) is not int or settings["moeCacheSlots"] < 1 \
                    or not settings.get("moeCacheProfile"):
                raise RuntimeError(f"prefillCacheYield requires an expert-cache profile and positive slot count for {model_id}")
    if POLITE.enabled and POLITE.available:
        try:
            POLITE.scheduler.change(G["politeStaticWeight"] if G["politeMode"] == "static" else G["politeWeights"][0])
            event("cpu_policy_started", f"mode={G['politeMode']} weight={POLITE.scheduler.weight} ms={POLITE.scheduler.last_ms}")
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            POLITE.available = False
            POLITE.error = str(error)
            event("cpu_policy_failed", str(error), fallback=True)
    start_wakeup_probe()
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
    st.monitor_thread = threading.Thread(target=monitor, daemon=True, name="shura-vram-guard")
    st.monitor_thread.start()
    if GAME.enabled:
        threading.Thread(target=game_monitor, daemon=True, name="shura-game-mode").start()
    if governor_on() and G.get("deskNeed") is True:
        threading.Thread(target=desk_monitor, daemon=True, name="shura-desk-need").start()
    server = BoundedServer((G["gatewayHost"], G["gatewayPort"]),
                           guarded_handler(private_checkpoint_handler(service_requests.with_title_answer(Handler, title_answers_enabled, fit_request)),
                                           web_policy, G["clientIdleSeconds"]),
                           G["maxConnections"])
    log(f"Gateway listening on http://{G['gatewayHost']}:{G['gatewayPort']} -> llama-server :{G['llamaPort']} "
        f"(config {CFG_DIR}, override={st.override}, idle unload {G['idleUnloadSeconds']}s)")
    server.serve_forever()


if __name__ == "__main__":
    main()
