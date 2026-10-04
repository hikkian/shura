"""Deterministic local OpenAI-compatible endpoint for offline pipeline checks."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time


PROVERB = '''proverb.py
```
def proverb(*words, qualifier=None):
    if not words:
        return []
    lines = [f"For want of a {left} the {right} was lost." for left, right in zip(words, words[1:])]
    first = f"{qualifier} {words[0]}" if qualifier else words[0]
    lines.append(f"And all for the want of a {first}.")
    return lines
```
'''


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    profile = "text"
    log_path: Path | None = None

    def log_message(self, *_args):
        pass

    def _send(self, status, body, content_type="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.endswith("/models"):
            return self._send(200, {"object": "list", "data": [{"id": "shura-stub", "object": "model"}]})
        return self._send(200, {"ok": True})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        tools = request.get("tools") or []
        messages = request.get("messages") or []
        text = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        if self.log_path:
            rec = {k: request.get(k) for k in ("model", "temperature", "top_p", "top_k", "max_tokens", "presence_penalty", "stream")}
            rec["tool_count"] = len(tools)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        if self.profile == "aider" and "proverb" in text.lower():
            content = PROVERB
        elif self.profile == "bfcl" and tools:
            # Multi-turn smoke: make one valid call, then return a final answer on
            # later turns so a broken loop cannot hide behind the fixture.
            prior_tool_results = sum(1 for message in messages if isinstance(message, dict) and message.get("role") in {"tool", "function"})
            if prior_tool_results:
                return self._send(200, {"id": "chatcmpl-stub", "object": "chat.completion", "created": int(time.time()),
                                        "model": request.get("model", "shura-stub"),
                                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Finished the available steps."}, "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 48, "completion_tokens": 6, "total_tokens": 54}})
            tool = tools[0]
            fn = tool.get("function", tool)
            name = fn.get("name", "unknown")
            args = {"base": 10, "height": 5} if name == "calculate_triangle_area" else {}
            call = {"id": "call-stub", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            self._send(200, {"id": "chatcmpl-stub", "object": "chat.completion", "created": int(time.time()),
                             "model": request.get("model", "shura-stub"),
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [call]}, "finish_reason": "tool_calls"}],
                             "usage": {"prompt_tokens": 32, "completion_tokens": 12, "total_tokens": 44}})
            return
        else:
            content = "Offline stub response: request reached the local evaluation endpoint."

        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = [
                {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "created": int(time.time()),
                 "model": request.get("model", "shura-stub"), "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}]},
                {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "created": int(time.time()),
                 "model": request.get("model", "shura-stub"), "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "created": int(time.time()),
                 "model": request.get("model", "shura-stub"), "choices": [], "usage": {"prompt_tokens": 32, "completion_tokens": 8, "total_tokens": 40}},
            ]
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self._send(200, {"id": "chatcmpl-stub", "object": "chat.completion", "created": int(time.time()),
                         "model": request.get("model", "shura-stub"),
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 32, "completion_tokens": 8, "total_tokens": 40}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19090)
    parser.add_argument("--profile", choices=("text", "aider", "bfcl"), default="text")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    Handler.profile = args.profile
    Handler.log_path = args.log
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
