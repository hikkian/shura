"""RSI1: the stand behind the gateway. The executor talks to a gateway of its own (port 8091, its llama-server on 8092), started from a
git worktree pinned to one commit, with the candidate configuration (reports/amd2c/guardian.candidate.json and
reports/layout1/model-launch.candidate.json: governor, honest RAM ladder, desk_need, Tiel and Sage). This is the stand the future live
Shura runs on, so the model's effort level (thinkLevels, EFFORT1) and its sampling (SAGECFG1) act the way they act for the user.
GpuStand (llama-server alone, the pilot's stand) stays for the old runs.

Rules kept from GpuStand (RSIL1 fixes): one VRAM guard for the whole stand, its reason written down (RSIL0f); every stop by a PID this
stand started (the gateway's wrapper and the llama-server below it); a llama-server on the port that is not below this stand's gateway is
never used and never stopped; SIGINT/SIGTERM stop the stand before the driver exits; a planned restart by the server's memory.

The pin: the worktree's HEAD must be the given commit and the worktree must be clean, at every start and restart. The candidate files are
copied once, when the stand is made, and every start uses the copies. The agent's side (the ShuraCode engine binary and the plugin
folders the harness copies) is fingerprinted then too and checked at every start. Anything that moved stops the run: the results of one
night never mix two gateways, two layouts or two engines.

    git -C ~/Projects/shura-integration worktree add --detach /dev/shm/RSI1/gw <commit>    # once, before the night
"""
import http.client
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from . import executor
from .executor import (
    GUARD_SECONDS,
    KEEP_FREE_GIB,
    RELEASE_MARGIN_GIB,
    RELEASE_TIMEOUT_S,
    SAFE_MAX_GB,
    SAFE_MIN_FREE_GB,
    SAFE_RUN,
    guard_step,
    guard_stop_record,
    last_prompt_tokens,
    listener_pids,
    over_memory_limit,
    pid_running,
    server_memory_gib,
    stop_own,
    stop_pid,
    used_mib,
    wait_mem,
)

GUARDIAN_CANDIDATE = Path("/home/alisher/Desktop/AI/reports/amd2c/guardian.candidate.json")
LAUNCH_CANDIDATE = Path("/home/alisher/Desktop/AI/reports/layout1/model-launch.candidate.json")
GW_PORT, LLAMA_PORT = 8091, 8092
DESK_RESERVE_MIB = 4096     # stand only: the user's word 2026-10-28 (as in SAGE1occ); the live gateway's default is not changed
IDLE_UNLOAD_S = 3600        # the editors think for minutes between runs: the model must not unload in between
RAM_STATE = Path("/dev/shm/RSI1/state")   # the stand's slot dumps and RAM sessions (tmpfs; the gateway refuses anything else)
READY_TIMEOUT_S = 120       # the gateway answers /guardian/status (the model loads with the first request, not here)
EFFORT_LEVELS = ("off", "low", "medium", "high", "max")
SAMPLING_KEYS = {"temperature": "temperature", "topP": "top_p", "topK": "top_k", "minP": "min_p", "presencePenalty": "presence_penalty"}


def git(worktree, *args):
    r = subprocess.run(["git", "-C", str(worktree), *args], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} in {worktree}: {r.stderr.strip()}")
    return r.stdout.strip()


def check_pin(worktree, commit):
    """The worktree is at `commit` and has no changes. Returns the full hash; anything else stops the run."""
    worktree = Path(worktree)
    if not (worktree / "gateway" / "ai_gateway.py").is_file():
        raise SystemExit(f"{worktree} is not a gateway worktree (no gateway/ai_gateway.py)")
    head, want = git(worktree, "rev-parse", "HEAD"), git(worktree, "rev-parse", f"{commit}^{{commit}}")
    if head != want:
        raise SystemExit(f"the stand's gateway moved: {worktree} is at {head[:9]}, the run is pinned to {want[:9]}")
    dirty = git(worktree, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit(f"the stand's gateway worktree has changes, refusing to start:\n{dirty}")
    return head


def model_sampling(model_id, launch=None):
    """The engine's request options for `model_id` from the candidate layout: the client's options win over the server's defaults, so
    the executor must send the model's own sampling (Sage: temperature 1.0, presence 1.5), not one set for every model."""
    launch = json.loads(Path(launch or LAUNCH_CANDIDATE).read_text())
    mc = launch["models"][model_id]
    return {opt: mc[key] for key, opt in SAMPLING_KEYS.items() if mc.get(key) is not None}


def build_config(cfg_dir, state_dir, model_id, worktree, gw_port=GW_PORT, llama_port=LLAMA_PORT, guardian=None, launch=None,
                 model_overrides=None, guardian_file=None):
    """The stand's AI_GATEWAY_CONFIG folder: the candidate files with the stand's ports and its own state folders. Nothing live is read
    or written. `guardian` and `model_overrides` are for tests (the fake server)."""
    cfg_dir, state_dir = Path(cfg_dir), Path(state_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    g = json.loads(Path(guardian_file or GUARDIAN_CANDIDATE).read_text())
    g.update({"gatewayPort": gw_port, "llamaPort": llama_port, "idleUnloadSeconds": IDLE_UNLOAD_S,
              "ramGuard": False,                       # safe-run's scope guards the stand's memory (as in SAGE1 and QLEV1)
              "ramDeskReserveMiB": DESK_RESERVE_MIB})
    # The gateway keeps its RAM sessions and slot dumps only on tmpfs ("vramPressureSlotDir must be on tmpfs (RAM)"), and the A2 session
    # checkpoint behind every switch to the long layout needs them: with the state on the run's disk folder every such switch failed
    # (RSI1 night 1, 2026-10-31: 40 minutes on one run). The dumps are also hundreds of MiB each: not for the SSD. So the stand's state
    # folders live in /dev/shm unless the given state folder is already there.
    if str(state_dir.resolve()).startswith("/dev/shm/"):
        ram_dir = state_dir
    else:
        ram_dir = RAM_STATE / "_".join(state_dir.resolve().parts[-3:-1])
    for key, sub in (("slotSaveDir", "slots"), ("vramPressureSlotDir", "pressure"), ("sessionCacheDir", "sessions"),
                     ("sessionPersistDir", "persist")):
        g[key] = str(ram_dir / sub)
        Path(g[key]).mkdir(parents=True, exist_ok=True)
    g.update(guardian or {})
    m = json.loads(Path(launch or LAUNCH_CANDIDATE).read_text())
    if model_id not in m["models"]:
        raise SystemExit(f"{model_id} is not in the candidate layout")
    m["defaultModel"] = model_id
    for mc in m["models"].values():
        mc.update(model_overrides or {})
    (cfg_dir / "guardian.json").write_text(json.dumps(g, ensure_ascii=False, indent=1))
    (cfg_dir / "model-launch.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    traces = Path(worktree) / "config" / "moe-trace"      # the expert-cache profiles, from the pinned commit
    if traces.is_dir():
        shutil.copytree(traces, cfg_dir / "moe-trace", dirs_exist_ok=True)
    return cfg_dir


def agent_side():
    """The files of the agent's side that are not in the pinned worktree: the engine binary and the plugin folders (harness.py)."""
    from harness import ENGINE, PLUGIN_SRC, RELIABILITY_PLUGIN_SRC
    return {"engine": Path(ENGINE), "plugins": Path(PLUGIN_SRC), "reliability": Path(RELIABILITY_PLUGIN_SRC),
            "model_harness": executor.MODEL_HARNESS_PLUGIN}


def fingerprints(paths):
    return {k: executor.fingerprint(p) if p.exists() else "missing" for k, p in paths.items()}


def ancestors(pid, proc="/proc"):
    out = []
    while pid > 1:
        try:
            pid = int(Path(proc, str(pid), "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        out.append(pid)
    return out


def http_json(port, method, path, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(method, path, body=b"" if method == "POST" else None)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read() or b"{}")
    finally:
        conn.close()


class GatewayStand:
    """The gateway of the pinned worktree under safe-run, the proxy in front of it, one VRAM guard. Same interface as GpuStand."""
    fake = False
    client_strict = True     # a run counts only when every request got 200 and a finished answer (executor.client_verdict)

    def __init__(self, work, model_id, worktree, commit, effort="medium", gw_port=GW_PORT, llama_port=LLAMA_PORT, wrapper=True,
                 guard=True, guardian=None, launch=None, model_overrides=None, pin=True):
        if effort not in EFFORT_LEVELS:
            raise SystemExit(f"unknown effort {effort!r}; use one of {', '.join(EFFORT_LEVELS)}")
        self.work, self.model_id, self.worktree, self.commit = Path(work), model_id, Path(worktree), commit
        self.effort, self.gw_port, self.port = effort, gw_port, llama_port
        self.wrapper, self.guard, self.pin = wrapper, guard, pin
        # the candidate files as they are now, for the whole run (a layout edited during the night does not reach this stand)
        pinned = self.work / "pinned"
        pinned.mkdir(parents=True, exist_ok=True)
        for src, name in ((GUARDIAN_CANDIDATE, "guardian.candidate.json"), (Path(launch or LAUNCH_CANDIDATE), "model-launch.candidate.json")):
            if not (pinned / name).exists():
                shutil.copy2(src, pinned / name)
        self.guardian_file, self.launch_file = pinned / "guardian.candidate.json", pinned / "model-launch.candidate.json"
        self.test_cfg = {"guardian": guardian, "launch": self.launch_file, "guardian_file": self.guardian_file,
                         "model_overrides": model_overrides}
        self.sampling = model_sampling(model_id, self.launch_file)
        self.agent_pins = fingerprints(agent_side())
        self.pins = {"guardian": executor.fingerprint(self.guardian_file), "launch": executor.fingerprint(self.launch_file),
                     **self.agent_pins}
        self.proc = None
        self.restarts, self.guard_stops, self.guard_dips = [], [], []   # guard_dips: VRAMDIP1, every dip below the keep level, however short
        self.stop_reason = None
        self.broken = None               # set by the guard: the stand is restarted before the next run
        self.lock = threading.RLock()
        self.desktop_mib_before = None
        self.current_run = ""
        self._done = threading.Event()
        self._old_handlers = {}

    # ---- the stand's life ----
    def __enter__(self):
        self.work.mkdir(parents=True, exist_ok=True)
        if self.guard and not (executor.VRAM_USED.exists() and executor.VRAM_TOTAL.exists()):
            raise RuntimeError("no VRAM counters under /sys/class/drm/card1: refuse to start the real stand")
        self.free_before = self._free_gib() if self.guard else None
        self._old_handlers = {sig: signal.signal(sig, self._on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            (self.work / "gateway.log").write_text("", encoding="utf-8")
            with self.lock:
                self._start("start")
            self.proxy_log = self.work / "proxy.jsonl"
            self.proxy_port = executor.free_port()
            self.srv = executor.proxy.serve(self.proxy_port, f"http://127.0.0.1:{self.gw_port}", str(self.proxy_log), extra={})
            threading.Thread(target=self.srv.serve_forever, daemon=True).start()
            if self.guard:
                threading.Thread(target=self._guard, daemon=True).start()
        except BaseException:
            self.stop_server()
            self._restore_handlers()
            raise
        return self

    def __exit__(self, *exc):
        self._done.set()
        self.srv.shutdown()
        self.srv.server_close()
        self.stop_server()
        self._restore_handlers()
        if not self.guard:
            return
        self.free_after_stop = self._free_gib()
        self.release_target = max(KEEP_FREE_GIB + RELEASE_MARGIN_GIB, self.free_before - 1.0)
        t0 = time.time()
        while self._free_gib() < self.release_target and time.time() - t0 < RELEASE_TIMEOUT_S:
            time.sleep(0.5)
        self.free_released = self._free_gib()
        self.release_wait_s = round(time.time() - t0, 1)
        self.released = self.free_released >= self.release_target

    def alive(self) -> bool:
        """The stand's gateway runs and answers /guardian/status (a GET that never loads a model), and no guard stop is pending."""
        if self.broken or self.proc is None or self.proc.poll() is not None:
            return False
        try:
            return http_json(self.gw_port, "GET", "/guardian/status", timeout=5)[0] == 200
        except OSError:
            return False

    def crashes(self) -> int:
        """How many times the gateway has logged that the llama-server crashed since this stand started (`Guardian server_crashed`). The
        log is read from where the last call stopped, so the run's abort check (every second) stays cheap."""
        log = self.work / "gateway.log"
        try:
            with open(log, "rb") as f:
                f.seek(0, 2)
                if f.tell() < getattr(self, "_crash_pos", 0):      # the log was cut: read it from the start
                    self._crash_pos = 0
                f.seek(getattr(self, "_crash_pos", 0))
                chunk = f.read()
                self._crash_pos = f.tell()
        except OSError:
            return getattr(self, "_crash_n", 0)
        self._crash_n = getattr(self, "_crash_n", 0) + chunk.count(b"Guardian server_crashed")
        return self._crash_n

    def restart(self, reason=""):
        """The whole stand again: gateway and server stopped by their PIDs, the pin checked, the same config, the effort set again."""
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
        """The llama-servers below this stand's gateway (by PID, found before the stop), then the wrapper's process group."""
        with self.lock:
            own = self.own_server_pids()
            if self.proc is not None:
                stop_own(self.proc, grace=30.0)      # the gateway saves nothing at exit; 30 s covers its llama-server's own stop
            for pid in own:
                if pid_running(pid):
                    stop_pid(pid)
            self.proc = None

    def own_server_pids(self):
        """llama-server processes listening on the stand's llama port whose ancestors include this stand's wrapper."""
        if self.proc is None:
            return []
        return sorted(p for p in listener_pids(self.port) if self.proc.pid in ancestors(p) and pid_running(p))

    def server_memory(self):
        pids = self.own_server_pids()
        if len(pids) != 1:
            return None
        try:
            return server_memory_gib(pids[0])
        except OSError:
            return None

    def memory_over_limit(self) -> bool:
        mem = self.server_memory()
        return mem is not None and over_memory_limit(*mem)

    # ---- inside ----
    def _start(self, label):
        if self.pin:
            self.head = check_pin(self.worktree, self.commit)
            now = fingerprints(agent_side())
            moved = {k: (self.agent_pins[k], v) for k, v in now.items() if v != self.agent_pins[k]}
            if moved:
                raise SystemExit(f"the agent's side changed since the stand was made: {moved}")
        self.broken = None
        for port in (self.gw_port, self.port):
            self._wait_port_free(port)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cfg = build_config(self.work / "cfg", self.work / "state" / stamp, self.model_id, self.worktree, self.gw_port, self.port,
                           **self.test_cfg)
        env = dict(os.environ, AI_GATEWAY_CONFIG=str(cfg), AI_GATEWAY_STATE=str(self.work / "state" / stamp / "gateway"),
                   PYTHONUNBUFFERED="1")
        Path(env["AI_GATEWAY_STATE"]).mkdir(parents=True, exist_ok=True)
        cmd = ["python3", "-B", str(self.worktree / "gateway" / "ai_gateway.py")]
        if self.wrapper:
            cmd = ["bash", str(SAFE_RUN), SAFE_MAX_GB, SAFE_MIN_FREE_GB, "--", *cmd]
        self.desktop_mib_before = used_mib() if self.guard else None
        with open(self.work / "gateway.log", "a") as log:
            waited = wait_mem() if self.wrapper else 0
            log.write(f"\n--- {label}; pin {getattr(self, 'head', 'off')[:9]}; waited {waited} s for MemAvailable\n")
            log.flush()
            self.proc = subprocess.Popen(cmd, env=env, cwd=str(self.work), start_new_session=True, stdout=log,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        self._wait_ready()
        status, body = http_json(self.gw_port, "POST", f"/guardian/effort-{self.effort}")
        if status != 200:
            raise RuntimeError(f"the stand's gateway refused effort {self.effort}: {status} {body}")
        self.think = body

    def _wait_ready(self):
        deadline = time.time() + READY_TIMEOUT_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = (self.work / "gateway.log").read_text(errors="replace")[-1500:]
                raise RuntimeError(f"the stand's gateway exited early ({self.work / 'gateway.log'}):\n{tail}")
            try:
                if http_json(self.gw_port, "GET", "/guardian/status", timeout=3)[0] == 200:
                    return
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError("the stand's gateway did not answer /guardian/status")

    def _wait_port_free(self, port, timeout=executor.PORT_FREE_TIMEOUT_S):
        deadline = time.time() + timeout
        while listener_pids(port):
            if time.time() > deadline:
                raise SystemExit(f"port {port} is held by pid {sorted(listener_pids(port))}, not by this stand: nothing is stopped")
            time.sleep(0.5)

    def _on_signal(self, signum, frame):
        self.stop_server()
        raise SystemExit(128 + signum)

    def _restore_handlers(self):
        for sig, handler in self._old_handlers.items():
            signal.signal(sig, handler)

    def _free_gib(self) -> float:
        return (int(executor.VRAM_TOTAL.read_text()) - int(executor.VRAM_USED.read_text())) / 2 ** 30

    def _guard(self):
        """VRAM below the desktop's share for longer than GUARD_SECONDS: the server is stopped by its PID and the stand is marked broken,
        so the run in progress is discarded and the next one starts on a fresh stand."""
        low_since = low_min = None
        dips = executor.DipRecorder(keep=KEEP_FREE_GIB, snapshot=executor.vram_by_process)
        while not self._done.is_set():
            pids = self.own_server_pids()
            if not pids:
                low_since = low_min = None
                for d in dips.flush(time.time()):
                    self._record_dip(d)
            else:
                now, free = time.time(), self._free_gib()
                for d in dips.step(free, now, time.strftime("%Y-%m-%dT%H:%M:%S") + f".{int(now * 1000) % 1000:03d}"):
                    self._record_dip(d)
                low_since, low_min, trigger = guard_step(low_since, low_min, free, now)
                if trigger:
                    with self.lock:
                        self._record_guard_stop(trigger[1], trigger[0])
                        for pid in self.own_server_pids():
                            stop_pid(pid)
                        self.broken = self.stop_reason
                    low_since = low_min = None
            time.sleep(executor.GUARD_POLL_S)

    def _record_dip(self, dip):
        """VRAMDIP1: a dip below the keep level, however short: written to guard_dips.jsonl with the run, the context and who held the memory."""
        dip = dict(dip, run=self.current_run or "(no run)",
                   context_tokens=last_prompt_tokens(getattr(self, "proxy_log", self.work / "proxy.jsonl")))
        self.guard_dips.append(dip)
        with open(self.work / "guard_dips.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(dip, ensure_ascii=False) + "\n")

    def _record_guard_stop(self, min_free, low_s):
        rec = guard_stop_record(t=time.strftime("%Y-%m-%dT%H:%M:%S"), min_free_gib=min_free, low_s=low_s,
                                desktop_mib=self.desktop_mib_before,
                                context_tokens=last_prompt_tokens(getattr(self, "proxy_log", self.work / "proxy.jsonl")),
                                run=self.current_run or "(no run)", threshold=KEEP_FREE_GIB, window=GUARD_SECONDS)
        self.guard_stops.append(rec)
        self.stop_reason = (f"VRAM guard: free {rec['min_free_GiB']} GiB below {rec['threshold_GiB']} GiB for {rec['low_s']} s, "
                            f"desktop {rec['desktop_used_MiB_before_start']} MiB, context {rec['context_tokens']} tokens, run {rec['run']}")
        with open(self.work / "guard_stops.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(self.work / "gateway.log", "a", encoding="utf-8") as log:
            log.write(f"\n--- VRAM guard stop: {self.stop_reason}\n")

