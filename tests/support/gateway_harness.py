"""Run the real Shura gateway against the fake llama-server in a throw-away directory.

    with GatewayHarness() as gw:
        code, body = gw.request("POST", "/v1/chat/completions", {"messages": [...]})
        print(gw.status())

The gateway code comes from SHURA_SRC (default: this repository). Nothing outside a temp folder is touched, no GPU is
used (NVML is replaced by a stub in run_gateway.py), and no model is loaded. Standard library only.
"""
import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKE_BIN = HERE / "fake-llama-bin"
RUNNER = HERE / "run_gateway.py"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def shura_src():
    return Path(os.environ.get("SHURA_SRC", HERE.parent.parent))


class GatewayHarness:
    def __init__(self, behaviour=None, guardian=None, model=None, env=None, launch=None, model_file=None):
        self.src = shura_src()
        self.tmp = Path(tempfile.mkdtemp(prefix="shura-qa-", dir="/dev/shm" if os.path.isdir("/dev/shm") else None))
        self.cfg = self.tmp / "config"
        self.cfg.mkdir()
        self.gw_port, self.llama_port = free_port(), free_port()
        self.fake_cfg = self.tmp / "fake.json"
        self.set_behaviour(**(behaviour or {}))
        g = json.loads((self.src / "config" / "guardian.example.json").read_text())
        g.update({"gatewayPort": self.gw_port, "llamaPort": self.llama_port, "healthCheckTimeoutSeconds": 30, "idleUnloadSeconds": 3600,
                  "ramFreeMinGBToLoad": 0, "ramFreeMinGB": 0.1, "slotSaveDir": str(self.tmp / "slots"), "vramPressureSlotDir": str(self.tmp),
                  "pollIntervalSeconds": 1, "vramGuard": False, "crashRestartMaxAttempts": 3})
        # VRAMLOGIC1h: SHURA_SUITE_VRAM_POLICY=governor runs the whole suite on the governor path (default: legacy)
        if os.environ.get("SHURA_SUITE_VRAM_POLICY"):
            g["vramPolicy"] = os.environ["SHURA_SUITE_VRAM_POLICY"]
        g.update(guardian or {})
        (self.cfg / "guardian.json").write_text(json.dumps(g))
        m = json.loads((self.src / "config" / "model-launch.example.json").read_text())
        # model_file: reuse one dummy model across two gateway processes (a reboot keeps the same model file, and the
        # session key includes its path and size); by default each harness writes its own model.gguf in its folder.
        dummy = Path(model_file) if model_file else self.tmp / "model.gguf"
        if not dummy.exists():
            dummy.write_bytes(b"GGUF")
        m["exePath"] = str(FAKE_BIN)
        for mc in m["models"].values():
            mc["modelPath"] = str(dummy)
            mc["vramFreeMinGBToLoad"] = 0
            mc.pop("mmprojPath", None)
            mc.pop("visionOverrides", None)
            mc.update(model or {})
            if os.environ.get("SHURA_SUITE_VRAM_POLICY") == "governor":
                # the fake's warm-up is short under the governor suite; tests do not wait for a real 4 s warm-up
                extra = list(mc.get("extraArgs") or [])
                if "--vram-warmup-seconds" not in extra:
                    mc["extraArgs"] = extra + ["--vram-warmup-seconds", "0.1"]
        m.update(launch or {})
        (self.cfg / "model-launch.json").write_text(json.dumps(m))
        # The gateway child gets a temporary HOME, and a stand-in nvidia-smi that reports no GPU, so a test never reads the
        # real driver or the user's files (the gateway's legacy path asks nvidia-smi for free VRAM).
        home, stand_in = self.tmp / "home", self.tmp / "bin"
        home.mkdir()
        stand_in.mkdir()
        (stand_in / "nvidia-smi").write_text("#!/bin/sh\nexit 1\n")
        (stand_in / "nvidia-smi").chmod(0o755)
        self.env = dict(os.environ, AI_GATEWAY_KERNEL_LOG="off", HOME=str(home), PATH=f"{stand_in}{os.pathsep}{os.environ.get('PATH', '')}",
                        AI_GATEWAY_CONFIG=str(self.cfg), AI_GATEWAY_STATE=str(self.tmp / "state"), FAKE_LLAMA_CONFIG=str(self.fake_cfg),
                        PYTHONUNBUFFERED="1", **(env or {}))
        self.proc = None
        self.log_path = self.tmp / "gateway.log"

    # ----- control
    def set_behaviour(self, **kw):
        cur = {}
        if self.fake_cfg.is_file():
            cur = json.loads(self.fake_cfg.read_text())
        cur.update(kw)
        self.fake_cfg.write_text(json.dumps(cur))

    def start(self, timeout=25):
        if self.proc is not None and self.proc.poll() is None:
            return self      # `with GatewayHarness(...) as gw:` already started it; a second gw.start() used to leak a gateway per test
        self.logf = open(self.log_path, "w")
        self.proc = subprocess.Popen([sys.executable, str(RUNNER), str(self.src / "gateway" / "ai_gateway.py")], env=self.env, cwd=str(self.tmp),
                                     stdout=self.logf, stderr=subprocess.STDOUT, start_new_session=True)
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                raise RuntimeError("the gateway exited at start-up:\n" + self.log())
            try:
                socket.create_connection(("127.0.0.1", self.gw_port), timeout=0.5).close()
                return self
            except OSError:
                time.sleep(0.2)
        raise RuntimeError("the gateway did not listen in time:\n" + self.log())

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        # the fake may outlive a killed gateway: stop whatever still listens on its port
        for pid in self.pids_on(self.llama_port):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            self.logf.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def pids_on(port):
        out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True).stdout
        pids = []
        for part in out.replace(",", " ").split():
            if part.startswith("pid="):
                pids.append(int(part[4:]))
        return pids

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()

    def log(self):
        try:
            return self.log_path.read_text(errors="replace")[-4000:]
        except OSError:
            return ""

    def full_log(self):
        """The whole gateway log (log() keeps only the last 4000 characters; the governor adds lines)."""
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""

    # ----- requests
    def request(self, method, path, body=None, timeout=60, port=None, stream=False, headers=None):
        """Returns (status, parsed JSON or text); with stream=True the body is the list of SSE data lines (and an `error` entry if cut short)."""
        conn = http.client.HTTPConnection("127.0.0.1", port or self.gw_port, timeout=timeout)
        try:
            data = json.dumps(body).encode() if body is not None else None
            conn.request(method, path, body=data, headers={**({"Content-Type": "application/json"} if data else {}), **(headers or {})})
            resp = conn.getresponse()
            if stream:
                lines, error = [], None
                try:
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        line = line.decode(errors="replace").strip()
                        if line.startswith("data:"):
                            lines.append(line[5:].strip())
                except (http.client.IncompleteRead, ConnectionError, socket.timeout, OSError) as e:
                    error = repr(e)
                return resp.status, {"lines": lines, "error": error, "complete": bool(lines) and lines[-1] == "[DONE]"}
            raw = resp.read()
            try:
                return resp.status, json.loads(raw.decode() or "null")
            except ValueError:
                return resp.status, raw.decode(errors="replace")
        finally:
            conn.close()

    def chat(self, content="hello", stream=False, **kw):
        body = {"model": "fake", "messages": [{"role": "user", "content": content}], "stream": stream}
        body.update(kw)
        return self.request("POST", "/v1/chat/completions", body, stream=stream)

    def status(self):
        return self.request("GET", "/guardian/status", timeout=10)[1]

    def fake_log(self):
        return self.request("GET", "/__fake/log", port=self.llama_port, timeout=5)[1]

    def wait_for(self, predicate, timeout=20, what="condition"):
        end = time.time() + timeout
        last = None
        while time.time() < end:
            try:
                last = predicate()
                if last:
                    return last
            except Exception as e:  # the gateway may be busy for a moment
                last = repr(e)
            time.sleep(0.3)
        raise AssertionError(f"timed out waiting for {what} (last: {last!r})\n--- gateway log ---\n{self.log()}")
