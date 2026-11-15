#!/usr/bin/env python3
"""EXAM200e: a stand-in OpenAI-compatible server for testing run_micro.py without a model.

For each /v1/chat/completions request it finds the task whose start/solution.py text is inside the user message,
and answers with the file ANSWERS/<id>.py in a ```python block (or an empty reply with --empty).
Standard library only. Binds to 127.0.0.1 only. Refuses to run on the live Shura ports (8080, 18090).

Usage: fake_answer_server.py --tasks DIR --answers DIR --port 18555 [--empty]
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REFUSED = {8080, 18090}


def make_handler(index, answers, empty):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
            user = next(m["content"] for m in body["messages"] if m["role"] == "user")
            tid = next((t for key, t in index.items() if key in user), None)
            if empty or tid is None:
                text = ""
            else:
                text = "```python\n" + answers[tid] + "```\n"
            out = {"id": "fake", "object": "chat.completion", "model": body.get("model", "fake"),
                   "choices": [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant", "content": text}}],
                   "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": len(text) // 4,
                             "total_tokens": len(user) // 4 + len(text) // 4}}
            data = json.dumps(out, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--empty", action="store_true")
    a = ap.parse_args()
    if a.port in REFUSED:
        print("refused: live Shura port", file=sys.stderr)
        return 2
    index, answers = {}, {}
    for d in sorted(a.tasks.iterdir()):
        s = d / "start" / "solution.py"
        f = a.answers / f"{d.name}.py"
        if s.exists() and f.exists():
            index[s.read_text(encoding="utf-8").strip()] = d.name
            answers[d.name] = f.read_text(encoding="utf-8")
    # the user message holds the start text; match on the full start text
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(index, answers, a.empty))
    print("listening", a.port, len(index), flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
