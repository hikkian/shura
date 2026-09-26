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
import re
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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


st = State()


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
         "--cache-ram", str(M["cacheRamMB"]), "--slot-save-path", str(SLOT_DIR)]
    if mc.get("moeCacheProfile") and mc.get("moeCacheSlots", 0) > 0:
        a += ["--moe-cache-profile", mc["moeCacheProfile"], "--moe-cache-slots", str(mc["moeCacheSlots"])]
    if mc.get("specType") and mc["specType"] != "none":
        a += ["--spec-type", mc["specType"], "--spec-draft-n-max", str(mc["specDraftNMax"]),
              "--spec-draft-p-min", str(mc["specDraftPMin"])]
    a += ["--mmproj", mc["mmprojPath"]] if vision else ["--no-mmproj-auto"]
    if M.get("apiKey"):
        a += ["--api-key", M["apiKey"]]
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
    try:
        _, slots = llama_call("GET", "/slots")
        n = (slots or [{}])[0].get("n_prompt_tokens", 0)
        if n < G["slotSaveMinTokens"]:
            log(f"Slot has {n} tokens (< {G['slotSaveMinTokens']}), not saving to disk")
            return
        name = slot_filename(st.model_id, st.vision)
        code, res = llama_call("POST", "/slots/0?action=save", {"filename": name}, timeout=120)
        log(f"Saved slot ({n} prompt tokens) -> {name}: HTTP {code} {res and res.get('n_saved')} tokens")
    except Exception as e:
        log(f"Slot save failed (non-fatal): {e}")


def restore_slot_locked():
    name = slot_filename(st.model_id, st.vision)
    if not (SLOT_DIR / name).exists():
        return
    try:
        code, res = llama_call("POST", "/slots/0?action=restore", {"filename": name}, timeout=120)
        log(f"Restored slot {name}: HTTP {code} {res and res.get('n_restored')} tokens")
    except Exception as e:
        log(f"Slot restore failed (non-fatal, falls back to normal prefill): {e}")


def stop_llama_locked(reason, save=True):
    if st.proc and st.proc.poll() is None:
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
    st.proc = None
    st.status = "UNLOADED"
    st.model_id = ""
    st.vision = False


def start_llama(model_id, vision):
    with st.lock:
        if (st.proc and st.proc.poll() is None and st.status == "READY"
                and st.model_id == model_id and st.vision == vision):
            return ""
        if st.status == "ERROR":
            return st.last_error
        if st.proc and st.proc.poll() is None:
            stop_llama_locked(f"switching to model={model_id} vision={vision}", save=True)
        mc = model_config(model_id, vision)
        if vision and not mc.get("mmprojPath"):
            vision = False
        reason = preload_check(mc)
        if reason:
            log(f"Load blocked: {reason}")
            st.last_error = reason
            return reason
        st.status, st.model_id, st.vision = "LOADING", model_id, vision
        args = build_args(mc, vision)
        log("Starting llama-server: " + " ".join(args))
        st.expected_exit = False
        st.proc = subprocess.Popen(args, stdin=subprocess.DEVNULL)
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
                    restore_slot_locked()
                    return ""
            except OSError:
                pass
        st.last_error = f"Health check timed out after {G['healthCheckTimeoutSeconds']}s"
        stop_llama_locked("load timeout", save=False)
        return st.last_error


def monitor():
    while True:
        time.sleep(G["pollIntervalSeconds"])
        try:
            with st.lock:
                if st.proc and st.proc.poll() is not None and not st.expected_exit:
                    st.crash_count += 1
                    st.last_error = f"llama-server crashed (exit {st.proc.returncode})"
                    log(st.last_error)
                    st.proc = None
                    st.status = "ERROR" if st.crash_count >= G["crashRestartMaxAttempts"] else "UNLOADED"

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
                    st.status = "UNLOADED"
                    log(f"Memory pressure cleared ({ram:.1f} GB available)")
        except Exception as e:
            log(f"Monitor error: {e}")


def status_snapshot():
    return {
        "status": st.status, "model": st.model_id, "vision": st.vision, "override": st.override,
        "memory_pressure": st.memory_pressure, "multimedia_lock": st.multimedia_lock,
        "busy_requests": st.busy, "crash_count": st.crash_count, "last_error": st.last_error,
        "ram_available_gb": round(mem_available_gb(), 2), "vram_free_gb": round(vram_free_gb(), 2),
        "idle_seconds": int(time.time() - st.last_request),
        "idle_unload_after_seconds": G["idleUnloadSeconds"],
        "llama_pid": st.proc.pid if st.proc and st.proc.poll() is None else None,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def guardian_endpoint(self, path):
        if path == "/guardian/status":
            self.send_json(200, status_snapshot())
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
            # OpenCode's AI-SDK client sends camelCase "reasoningEffort"; llama-server only reads "reasoning_effort".
            if '"reasoningEffort"' in text:
                body = re.sub(r'"reasoningEffort"\s*:', '"reasoning_effort":', text).encode()

        err = start_llama(model_id, vision)
        if err:
            return self.send_json(503, {"error": f"Could not start model '{model_id}'", "reason": err})

        with st.busy_lock:
            st.busy += 1
        try:
            self.proxy(body)
        finally:
            with st.busy_lock:
                st.busy -= 1
            st.last_request = time.time()

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
    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    threading.Thread(target=monitor, daemon=True).start()
    server = ThreadingHTTPServer((G["gatewayHost"], G["gatewayPort"]), Handler)
    server.daemon_threads = True
    log(f"Gateway listening on http://{G['gatewayHost']}:{G['gatewayPort']} -> llama-server :{G['llamaPort']} "
        f"(config {CFG_DIR}, override={st.override}, idle unload {G['idleUnloadSeconds']}s)")
    server.serve_forever()


if __name__ == "__main__":
    main()
