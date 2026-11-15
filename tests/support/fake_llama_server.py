#!/usr/bin/env python3
"""A stand-in for llama-server, for testing the Shura gateway and harness without a GPU or a model.

It accepts the same command line (unknown flags are ignored), answers /health, /slots, slot save and restore, /props, /v1/models
and /v1/chat/completions (streaming and not), and can misbehave on demand: load slowly, answer 503, stall, drop the connection
or die in the middle of an answer. Behaviour comes from a JSON file (FAKE_LLAMA_CONFIG, re-read on every request) and can be
changed while it runs with POST /__fake/behavior. GET /__fake/log returns what it received, so a test can check what the
gateway really sent. Standard library only.
"""
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DEFAULTS = {
    "load_seconds": 0.5,        # /health answers 503 "Loading model" until then
    "token_delay_ms": 5,        # pause between streamed tokens
    "n_tokens": 12,             # tokens in an answer
    "prompt_tokens": 100,       # usage.prompt_tokens (a real server counts the prompt)
    "text": "ok",               # every token is this word
    "reasoning_tokens": 0,      # how many tokens come first as reasoning_content
    "empty_length_turns": 0,    # the next N client turns end by the output limit with reasoning only (finish "length");
                                # a continuation (last message from the assistant) is answered normally
    "status_for_next": [],      # e.g. [503, 503]: the next chat requests get these statuses, one each
    "crash_after_tokens": None, # os._exit(137) after this many streamed tokens (a killed server)
    "drop_after_tokens": None,  # close the connection after this many streamed tokens, server stays up
    "hang_seconds": 0,          # sleep before answering a chat request (an unresponsive server)
    "progress_steps": 0,        # with "return_progress": true in a streamed request, this many prompt_progress chunks first ...
    "progress_step_ms": 100,    # ... this far apart (llama-server sends one per batch while it reads the prompt)
    "health_status": None,      # force /health to answer this status (e.g. 500)
    "exit_after_seconds": None, # the process exits by itself after this long (a crash while idle)
    "cors": False,              # answer like llama-server's default CORS: Access-Control-Allow-Origin = the caller, with credentials
    "stall_after_tokens": None, # go silent (connection stays open) after this many streamed tokens ...
    "stall_seconds": 3600,      # ... for this long
    "rst_after_tokens": None,   # reset the connection (RST, SO_LINGER 0) after this many streamed tokens
    "restore_status": None,     # force the slot-restore call to answer this status (e.g. 500)
    "no_slot_estimate": False,  # /slots without n_slot_save_estimated_bytes (a server built without the A2 estimate; A2PORT1)
    "prompt_per_second": 800.0,
    "predicted_per_second": 40.0,
}
STARTED = time.time()
LOG = []
LOCK = threading.Lock()
RUNTIME = {}
STATE = {"active": 0, "prompt_tokens": 0}
ARGS = {"host": "127.0.0.1", "port": 8090, "slot_path": None, "ctx": 200000}


# VRAMLOGIC1g: the governor contract (reports/vramlogic1/API.md) is served from fake_vram.FakeVram. The state advances
# with wall-clock time on every /vram request, so tests can drive it with POST /vram/simulate (needs --vram-test-hooks).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fake_vram import FakeVram, PolicyError  # noqa: E402
VRAM = {"gov": None, "last": None, "hooks": False}


def vram_gov():
    now = time.monotonic()
    if VRAM["gov"] is None:
        argv = ARGS.get("argv", [])
        warmup = float(argv[argv.index("--vram-warmup-seconds") + 1]) if "--vram-warmup-seconds" in argv else 4.2
        VRAM["gov"] = FakeVram(policy={"grow_seconds": 0}, warmup_seconds=warmup)
        VRAM["gov"].legacy_api = "--vram-legacy-api" in argv      # behave like a server before MEMV2
        VRAM["last"] = now
    gov = VRAM["gov"]
    gov.tick(max(0.0, now - VRAM["last"]))
    VRAM["last"] = now
    return gov


def behaviour():
    cfg = dict(DEFAULTS)
    path = os.environ.get("FAKE_LLAMA_CONFIG")
    if path and Path(path).is_file():
        try:
            cfg.update(json.loads(Path(path).read_text()))
        except (OSError, ValueError):
            pass
    cfg.update(RUNTIME)
    return cfg


def parse_args(argv):
    i = 0
    while i < len(argv):
        a = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if a == "--host" and nxt:
            ARGS["host"] = nxt
        elif a == "--port" and nxt:
            ARGS["port"] = int(nxt)
        elif a == "--slot-save-path" and nxt:
            ARGS["slot_path"] = nxt
        elif a == "--api-key" and nxt:
            ARGS["key"] = nxt
        elif a == "--api-key-file" and nxt:
            ARGS["key"] = Path(nxt).read_text().strip().splitlines()[0]
        elif a in ("-c", "--ctx-size") and nxt:
            ARGS["ctx"] = int(nxt)
        i += 1


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def cors(self):
        if behaviour().get("cors"):  # what llama-server does by default: allow whatever origin asks
            origin = self.headers.get("Origin")
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Credentials", "true")

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", self.headers.get("Access-Control-Request-Headers", "*"))
        self.send_header("Content-Length", "0")
        self.end_headers()

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        self.raw_body = raw
        try:
            return json.loads(raw.decode() or "{}")
        except ValueError:
            return {}

    def loaded(self):
        return time.time() - STARTED >= float(behaviour()["load_seconds"])

    def authorized(self, path):
        """llama-server with --api-key asks for it on everything except /health (and our own test endpoints)."""
        key = ARGS.get("key")
        if not key or path in ("/health", "/v1/health") or path.startswith("/__fake/"):
            return True
        if self.headers.get("Authorization") == "Bearer " + key:
            return True
        self.send_json(401, {"error": {"message": "Invalid API Key", "type": "authentication_error"}})
        return False

    def do_GET(self):
        u = urlparse(self.path)
        with LOCK:
            LOG.append({"method": "GET", "path": self.path})
        if not self.authorized(u.path):
            return
        cfg = behaviour()
        if u.path == "/vram":
            return self.send_json(200, vram_gov().snapshot())
        if u.path == "/vram/policy":
            return self.send_json(200, vram_gov().get_policy())
        if u.path == "/vram/journal" and vram_gov().legacy_api:
            return self.send_json(404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})
        if u.path == "/vram/journal":
            q = parse_qs(u.query)
            return self.send_json(200, vram_gov().journal_json(int((q.get("since") or ["0"])[0]), int((q.get("limit") or ["200"])[0])))
        if u.path == "/health":
            if cfg["health_status"]:
                return self.send_json(int(cfg["health_status"]), {"error": {"message": "forced by the test"}})
            if not self.loaded():
                return self.send_json(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
            return self.send_json(200, {"status": "ok"})
        if u.path == "/slots":
            slot = {"id": 0, "n_ctx": ARGS["ctx"], "is_processing": STATE["active"] > 0, "id_task": 0, "speculative": True,
                    "n_prompt_tokens": STATE["prompt_tokens"], "n_slot_save_estimated_bytes": 1400}
            if behaviour().get("no_slot_estimate"):
                del slot["n_slot_save_estimated_bytes"]
            return self.send_json(200, [slot])
        if u.path == "/props":
            return self.send_json(200, {"model_alias": "fake", "total_slots": 1, "default_generation_settings": {"n_ctx": ARGS["ctx"]}})
        if u.path in ("/v1/models", "/models"):
            return self.send_json(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})
        if u.path == "/__fake/env":
            # AMD2a: what the gateway put into this process's environment (only SHURA_TEST_* names are reported)
            return self.send_json(200, {k: v for k, v in os.environ.items() if k.startswith("SHURA_TEST_")})
        if u.path == "/__fake/log":
            with LOCK:
                return self.send_json(200, LOG[-200:])
        if u.path == "/__fake/args":
            return self.send_json(200, ARGS.get("argv", []))
        if u.path == "/__fake/state":
            with LOCK:
                return self.send_json(200, {"active": STATE["active"], "client_closed": sum(1 for e in LOG if e.get("event") == "client_closed")})
        return self.send_json(404, {"error": {"message": "not found: " + u.path}})

    def do_POST(self):
        u = urlparse(self.path)
        body = self.read_json()
        with LOCK:
            entry = {"method": "POST", "path": self.path, "body": body}
            if behaviour().get("record_prompts") and u.path.endswith("/chat/completions"):
                # CACHE1: deterministic chat-template stand-in. The generation header is the next assistant turn's prefix.
                prompt = json.dumps(body.get("tools", []), ensure_ascii=False, sort_keys=True) + "\n"
                for msg in body.get("messages", []):
                    content = msg.get("content", "")
                    if not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False, sort_keys=True)
                    extra = {k: v for k, v in msg.items() if k not in ("role", "content")}
                    prompt += "<|im_start|>" + msg.get("role", "") + "\n" + content
                    if extra:
                        prompt += json.dumps(extra, ensure_ascii=False, sort_keys=True)
                    prompt += "<|im_end|>\n"
                entry["prompt"] = prompt + "<|im_start|>assistant\n"
                entry["raw_body"] = self.raw_body.decode("utf-8")
            LOG.append(entry)
        if not self.authorized(u.path):
            return
        cfg = behaviour()
        if u.path == "/__fake/behavior":
            RUNTIME.update(body)
            return self.send_json(200, {"ok": True, "behaviour": behaviour()})
        if u.path == "/__fake/vanish":
            # what a killed server with a big GPU footprint does: the listening socket closes at once, the process lingers while the
            # driver releases memory, then it exits
            STATE["linger"] = float(body.get("seconds", 3))
            srv = self.server
            self.send_json(200, {"ok": True})
            threading.Thread(target=lambda: (srv.shutdown(), srv.server_close()), daemon=True).start()
            return None
        if u.path.startswith("/slots/"):
            return self.slot(u, body)
        if u.path in ("/v1/chat/completions", "/chat/completions", "/completion", "/completions"):
            return self.chat(u, body, cfg)
        if u.path == "/vram/policy":
            code, out = vram_gov().post_policy(body)
            return self.send_json(code, out)
        if u.path == "/vram/pause":
            code, out = vram_gov().post_pause(body)
            return self.send_json(code, out)
        if u.path == "/moe-cache/chunks":
            code, out = vram_gov().post_chunks(body)
            return self.send_json(code, out)
        if u.path == "/vram/simulate":
            if not VRAM["hooks"]:
                return self.send_json(404, {"error": {"message": "start the fake with --vram-test-hooks"}})
            code, out = vram_gov().simulate(body)
            return self.send_json(code, out)
        if u.path == "/moe-cache/slots":
            # PREFILL2c: live expert-cache resize; behaviour {"resize_status": 500} makes it refuse like a failed rollback
            if behaviour().get("resize_status"):
                return self.send_json(behaviour()["resize_status"], {"success": False, "message": "resize refused by fake server"})
            STATE["moe_slots"] = body.get("slots")
            return self.send_json(200, {"success": True, "new_slots": body.get("slots")})
        if u.path == "/tokenize":
            return self.send_json(200, {"tokens": list(range(len(str(body.get("content", "")).split())))})
        return self.send_json(404, {"error": {"message": "not found: " + u.path}})

    def slot(self, u, body):
        action = parse_qs(u.query).get("action", [""])[0]
        name = body.get("filename") or "slot.bin"
        base = Path(ARGS["slot_path"] or ".")
        if action == "save":
            base.mkdir(parents=True, exist_ok=True)
            (base / name).write_bytes(b"fake slot data" * 100)
            return self.send_json(200, {"id_slot": 0, "filename": name, "n_saved": STATE["prompt_tokens"] or 123, "n_written": 1400,
                                        "timings": {"save_ms": 1.0}})
        if action == "restore":
            if behaviour()["restore_status"]:
                return self.send_json(int(behaviour()["restore_status"]), {"error": {"message": "forced by the test"}})
            if not (base / name).is_file():
                return self.send_json(404, {"error": {"message": "slot file not found", "type": "not_found_error"}})
            return self.send_json(200, {"id_slot": 0, "filename": name, "n_restored": int(behaviour().get("restore_tokens", 123)),
                                        "n_read": 1400, "timings": {"restore_ms": 1.0}})
        if action == "erase":
            return self.send_json(200, {"id_slot": 0, "n_erased": 1})
        return self.send_json(400, {"error": {"message": "unknown slot action"}})

    def chat(self, u, body, cfg):
        STATE["active"] += 1
        try:
            return self.chat_inner(u, body, cfg)
        finally:
            STATE["active"] -= 1
            STATE["prompt_tokens"] = int(cfg["prompt_tokens"])

    def vram_pause(self, cfg, sent, total, chunk):
        """VRAMLOGIC1i fake: the governor pauses after `sent` tokens. Writes the checkpoint and <checkpoint>.tokens.json next to it
        (API.md), sends the vram_pause error frame as the last frame, then exits with 75. The config drops the pause so the
        restarted server does not pause again."""
        argv = ARGS.get("argv", [])
        slot_dir = Path(argv[argv.index("--slot-save-path") + 1]) if "--slot-save-path" in argv else Path(".")
        slot_dir.mkdir(parents=True, exist_ok=True)
        name = "vram-pause.bin"
        (slot_dir / name).write_bytes(b"fake-checkpoint")
        prompt = int(cfg["prompt_tokens"])
        tokens = list(range(1, prompt + sent + 1))
        left = total - sent
        (slot_dir / (name + ".tokens.json")).write_text(json.dumps({
            "slot": 0, "reason": "pause_floor", "n_past": len(tokens), "tokens": tokens, "multimodal": False, "processing": True,
            "generated_tokens": sent, "n_predict": total, "n_predict_left": left,
            "params": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "seed": 7}}))
        frame = {"error": {"code": 503, "type": "vram_pause", "message": json.dumps({
            "checkpoint": name, "tokens_file": name + ".tokens.json", "slot": 0, "reason": "pause_floor",
            "n_past": len(tokens), "generated_tokens": sent, "saved": True})}}
        data = ("data: " + json.dumps(frame) + "\n\n").encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n0\r\n\r\n")
        self.wfile.flush()
        path = os.environ.get("FAKE_LLAMA_CONFIG")
        if path and Path(path).is_file():
            cur = json.loads(Path(path).read_text())
            cur.pop("pause_after_tokens", None)
            Path(path).write_text(json.dumps(cur))
        RUNTIME.pop("pause_after_tokens", None)
        time.sleep(0.5)
        os._exit(75)

    def chat_inner(self, u, body, cfg):
        if not self.loaded():
            return self.send_json(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
        if cfg["hang_seconds"]:
            time.sleep(float(cfg["hang_seconds"]))
        queued = list(RUNTIME.get("status_for_next", []) or cfg["status_for_next"])
        if queued:
            code = queued.pop(0)
            RUNTIME["status_for_next"] = queued
            return self.send_json(int(code), {"error": {"code": int(code), "message": "forced by the test"}})
        stream = bool(body.get("stream"))
        n = int(cfg["n_tokens"])
        reasoning = int(cfg["reasoning_tokens"])
        # EFFORT1: the reasoning budget of shura.cpp: reasoning stops at the budget, the wrap-up message is the last reasoning text
        budget = body.get("reasoning_budget_tokens")
        wrap_up = ""
        if isinstance(budget, int) and budget >= 0 and reasoning > budget:
            reasoning, wrap_up = budget, str(body.get("reasoning_budget_message") or "")
        finish_reason = "stop"
        last_role = ((body.get("messages") or [{}])[-1] or {}).get("role") if isinstance(body.get("messages"), list) and body.get("messages") else None
        if last_role != "assistant":
            if "empty_left" not in STATE:
                STATE["empty_left"] = int(cfg["empty_length_turns"])
            if STATE["empty_left"] > 0:
                STATE["empty_left"] -= 1
                n, reasoning, finish_reason = 0, max(reasoning, 4), "length"
        last_message = (body.get("messages") or [{}])[-1] if body.get("messages") else {}
        if isinstance(last_message, dict) and isinstance(last_message.get("content"), str) and last_message["content"].startswith(("Перепиши своё", "Соңғы хабарламаңды")):
            text = cfg.get("rewrite_text", cfg["text"])
        else:
            text = cfg["text"]
        words = [text] * n
        usage = {"prompt_tokens": int(cfg["prompt_tokens"]), "completion_tokens": n + reasoning, "total_tokens": int(cfg["prompt_tokens"]) + n + reasoning}
        timings = {"prompt_n": usage["prompt_tokens"], "prompt_per_second": cfg["prompt_per_second"], "predicted_n": n + reasoning, "predicted_per_second": cfg["predicted_per_second"]}
        if not stream:
            msg = {"role": "assistant", "content": " ".join(words)}
            if reasoning or wrap_up:
                msg["reasoning_content"] = " ".join(["think"] * reasoning + ([wrap_up] if wrap_up else []))
            return self.send_json(200, {"id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()), "model": "fake",
                                        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": usage, "timings": timings})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(delta, finish=None, extra=None):
            obj = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": int(time.time()), "model": "fake",
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if extra:
                obj.update(extra)
            data = ("data: " + json.dumps(obj) + "\n\n").encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        try:
            steps = int(cfg.get("progress_steps") or 0) if body.get("return_progress") else 0
            total = int(cfg["prompt_tokens"])
            for k in range(1, steps + 1):
                time.sleep(float(cfg.get("progress_step_ms", 100)) / 1000)
                chunk({"role": "assistant", "content": None}, extra={"prompt_progress": {
                    "total": total, "cache": 0, "processed": total * k // steps, "time_ms": k * float(cfg.get("progress_step_ms", 100))}})
            chunk({"role": "assistant", "content": ""})
            sent = 0
            resume = body.get("resume") if isinstance(body.get("resume"), dict) else None
            done_before = 0
            if resume is not None:
                # VRAMLOGIC1i: resume (API.md «Продолжение ответа»): the tokens file says what was generated and what is left.
                # Only the rest is streamed; the role chunk above is the one the gateway drops.
                argv = ARGS.get("argv", [])
                slot_dir = Path(argv[argv.index("--slot-save-path") + 1]) if "--slot-save-path" in argv else Path(".")
                info = json.loads((slot_dir / str(resume["tokens_file"])).read_text())
                done_before = int(info["generated_tokens"])
                reasoning_total = int(cfg.get("reasoning_tokens", 0))
                reasoning = max(0, reasoning_total - done_before)   # reasoning not yet streamed
                n = max(0, int(info["n_predict_left"]) - reasoning)  # the rest is content
            for k in range(reasoning + n):
                if cfg.get("pause_after_tokens") is not None and resume is None and sent >= int(cfg["pause_after_tokens"]):
                    return self.vram_pause(cfg, sent, reasoning + n, chunk)
                if cfg["crash_after_tokens"] is not None and sent >= int(cfg["crash_after_tokens"]):
                    os._exit(137)
                if cfg["stall_after_tokens"] is not None and sent >= int(cfg["stall_after_tokens"]):
                    time.sleep(float(cfg["stall_seconds"]))
                if cfg["rst_after_tokens"] is not None and sent >= int(cfg["rst_after_tokens"]):
                    import struct
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    os.close(self.connection.fileno())      # closing the descriptor itself sends the RST
                    self.close_connection = True
                    return
                if cfg["drop_after_tokens"] is not None and sent >= int(cfg["drop_after_tokens"]):
                    # a real FIN/RST: socket.close() alone does not close the TCP connection while rfile/wfile still hold it
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    self.close_connection = True
                    return
                time.sleep(float(cfg["token_delay_ms"]) / 1000)
                if k == reasoning and wrap_up:
                    chunk({"reasoning_content": wrap_up + " "})
                chunk({"reasoning_content": "think "} if k < reasoning else {"content": text + " "})
                sent += 1
            if n == 0 and wrap_up:
                chunk({"reasoning_content": wrap_up + " "})
            chunk({}, finish_reason, {"usage": usage, "timings": timings})
            end = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(end):x}\r\n".encode() + end + b"\r\n0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            with LOCK:
                LOG.append({"event": "client_closed"})


HELP = """usage: llama-server [options]
--cors-origins ORIGINS   comma-separated list of allowed origins for CORS (default: *)
--cors-credentials, --no-cors-credentials   whether to allow credentials for CORS
--api-key KEY            API key to use for authentication
--api-key-file FNAME     path to file containing API keys, one per line
--moe-cache-live-resize  allow live resizing of the expert cache (POST /moe-cache/slots)
"""


def main(argv):
    if "--help" in argv or "-h" in argv:
        print(HELP)
        return
    if "--fake-exit-at-start" in argv:  # LAYOUT1: a server that dies while loading (a failed long-context start)
        raise SystemExit(1)
    ARGS["argv"] = list(argv)
    VRAM["hooks"] = "--vram-test-hooks" in argv
    # VRAMLOGIC1i: a layout that does not keep the card reserve exits 75 during warm-up (the governor's own answer)
    if behaviour().get("exit75_when_ncmoe_below") is not None and "-ncmoe" in argv:
        if int(argv[argv.index("-ncmoe") + 1]) < int(behaviour()["exit75_when_ncmoe_below"]):
            os._exit(75)
    parse_args(argv)
    srv = ThreadingHTTPServer((ARGS["host"], ARGS["port"]), Handler)
    srv.daemon_threads = True
    exit_after = behaviour()["exit_after_seconds"]
    if exit_after:
        threading.Timer(float(exit_after), lambda: os._exit(1)).start()
    print(f"fake llama-server listening on http://{ARGS['host']}:{ARGS['port']}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    if STATE.get("linger"):
        time.sleep(STATE["linger"])
        os._exit(137)


if __name__ == "__main__":
    main(sys.argv[1:])
