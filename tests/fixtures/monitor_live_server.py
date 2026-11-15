#!/usr/bin/env python3
"""Isolated local gateway + deterministic llama stub for manual UI smoke tests."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gateway"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

tmp = tempfile.TemporaryDirectory(prefix="shura-monitor-", dir="/dev/shm")
config = Path(tmp.name)
(config / "guardian.json").write_text(json.dumps({"gatewayHost": "127.0.0.1", "gatewayPort": 18080,
    "llamaHost": "127.0.0.1", "llamaPort": 18081, "slotSaveDir": str(config / "slots"),
    "crashRecovery": False}))
(config / "model-launch.json").write_text(json.dumps({"defaultModel": "tiel-coder", "models": {
    "tiel-coder": {"ctxSize": 200000, "apiKey": None}}}))
os.environ["AI_GATEWAY_CONFIG"] = str(config)

import _gateway_legacy as gateway  # noqa: E402
import monitor_stub_openai as stub  # noqa: E402


# Requests use the real legacy Handler and a deterministic llama stub. This server
# never starts a model and deliberately disables every GPU adapter for no-source UI checks.
gateway.st.status = "UNLOADED"
gateway.st.model_id = ""
gateway.st.proc = None
gateway.st.override = "AUTO"
gateway.st.memory_pressure = False
gateway.start_llama = lambda *_args, **_kwargs: None
gateway._monitor_gpu = lambda _status: {}
gateway._monitor_slots = lambda _status: None
if os.environ.get("SHURA_TEST_NO_MONITOR_STREAM") == "1":
    route = gateway.monitor_route
    def no_stream(handler, path, *args, **kwargs):
        if path == "/monitor/api/stream":
            return handler.send_json(404, {"error": "SSE disabled for fallback check"})
        return route(handler, path, *args, **kwargs)
    gateway.monitor_route = no_stream

stub.profile = "text"
stub.log_path = None
llama = ThreadingHTTPServer(("127.0.0.1", 18081), stub.Handler)
server = ThreadingHTTPServer(("127.0.0.1", 18080), gateway.Handler)
threading.Thread(target=llama.serve_forever, daemon=True, name="monitor-stub").start()
stopped = threading.Event()


def stop(_sig, _frame):
    stopped.set()
    threading.Thread(target=server.shutdown, daemon=True).start()


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
print(f"monitor test gateway ready pid={os.getpid()} ports=18080,18081", flush=True)
threading.Thread(target=server.serve_forever, daemon=True, name="monitor-gateway").start()
stopped.wait()
server.server_close()
llama.shutdown()
llama.server_close()
tmp.cleanup()
