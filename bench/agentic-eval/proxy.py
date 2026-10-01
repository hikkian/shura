"""Measuring proxy between the agent (ShuraCode) and llama-server for the model comparison.

Why it exists: every arm must be measured by the same instrument, independent of what the agent prints.
  * counts tokens (prompt, completion) and wall time per request from llama-server's own `usage`/`timings`;
  * does what our gateway does for the agent: `reasoningEffort` (the AI SDK's spelling) becomes `reasoning_effort`;
  * adds per-arm request fields, e.g. `chat_template_kwargs: {"terse": false}` for the Tiel arm without terseness;
  * streams (SSE) without buffering, so the agent behaves exactly as with a direct connection;
  * writes one JSON line per request to the log file (no prompts or code, only numbers).

    python3 proxy.py --listen 8097 --upstream http://127.0.0.1:8099 --log run.jsonl --extra '{"chat_template_kwargs": {"terse": false}}'
"""
import argparse
import http.client
import json
import re
import socketserver
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade",
       "content-length", "host"}


def rewrite_body(raw, extra):
    """Request body -> request body: spelling fix + per-arm fields + ask for usage in streams. Returns bytes."""
    text = raw.decode("utf-8", errors="replace")
    text = re.sub(r'"reasoningEffort"\s*:', '"reasoning_effort":', text)
    try:
        body = json.loads(text)
    except ValueError:
        return raw
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(body.get(k), dict):
            body[k] = {**body[k], **v}
        else:
            body[k] = v
    if body.get("stream"):
        body.setdefault("stream_options", {})["include_usage"] = True
    return json.dumps(body).encode()


def scan_usage(payload):
    """Pull usage and timings out of a (JSON or SSE) response body; {} when absent."""
    found = {}
    chunks = payload.split(b"\n") if payload.lstrip().startswith(b"data:") or b"\ndata:" in payload else [payload]
    for line in chunks:
        line = line.strip()
        if line.startswith(b"data:"):
            line = line[5:].strip()
        if not line or line == b"[DONE]" or not line.startswith(b"{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj.get("usage"), dict):
            found["usage"] = obj["usage"]
        if isinstance(obj.get("timings"), dict):
            found["timings"] = obj["timings"]
    return found


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = None
    extra = None
    log_path = None
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _forward(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if self.command == "POST" and raw:
            raw = rewrite_body(raw, self.extra)
        u = urllib.parse.urlparse(self.upstream)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers["Content-Length"] = str(len(raw))
        t0 = time.monotonic()
        try:
            conn.request(self.command, self.path, body=raw if raw else None, headers=headers)
            resp = conn.getresponse()
        except OSError as e:
            self.send_error(502, f"upstream: {e}")
            return
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        tail = b""
        first = None
        try:
            while True:
                chunk = resp.read1(8192) if hasattr(resp, "read1") else resp.read(8192)
                if not chunk:
                    break
                if first is None:
                    first = time.monotonic() - t0
                tail = (tail + chunk)[-65536:]
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass                                            # the agent hung up (its own timeout or abort): still record the request
        finally:
            conn.close()
        if self.command == "POST" and "/chat/completions" in self.path:
            info = scan_usage(tail)
            rec = {"t": time.time(), "path": self.path, "status": resp.status, "seconds": round(time.monotonic() - t0, 3),
                   "first_byte_s": round(first or 0.0, 3), **info}
            with self.lock, open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    do_GET = do_POST = do_PUT = do_DELETE = _forward


def totals(log_path):
    """Sum a request log: {requests, prompt_tokens, completion_tokens, seconds, errors}."""
    out = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "errors": 0}
    try:
        lines = open(log_path).read().splitlines()
    except OSError:
        return out
    for line in lines:
        r = json.loads(line)
        out["requests"] += 1
        out["errors"] += 1 if r["status"] >= 400 else 0
        out["seconds"] += r["seconds"]
        u = r.get("usage") or {}
        out["prompt_tokens"] += u.get("prompt_tokens", 0)
        out["completion_tokens"] += u.get("completion_tokens", 0)
    return out


class Quick(ThreadingHTTPServer):
    def server_bind(self):                  # skip the reverse DNS lookup of HTTPServer (it stalls on some systems)
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


def serve(listen, upstream, log_path, extra=None):
    handler = type("H", (Handler,), {"upstream": upstream, "extra": extra, "log_path": log_path})
    srv = Quick(("127.0.0.1", listen), handler)
    srv.daemon_threads = True
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--extra", default="{}", help="JSON object merged into every request body")
    a = ap.parse_args()
    serve(a.listen, a.upstream, a.log, json.loads(a.extra)).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
