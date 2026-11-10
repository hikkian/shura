#!/usr/bin/env python3
"""Tiny OpenAI-compatible stub: logs every chat request body to dummy_requests.jsonl and answers 'OK' (streamed). Used to test plugin hooks without a model."""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
LOG = sys.argv[1]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0)); body = self.rfile.read(n)
        try: req = json.loads(body)
        except Exception: req = {"raw": body[:200].decode('utf8', 'replace')}
        open(LOG, 'a').write(json.dumps(req, ensure_ascii=False) + "\n")
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        for c in [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}}]},
                  {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}]:
            self.wfile.write(("data: " + json.dumps(dict(c, id="x", object="chat.completion.chunk", model="m")) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
HTTPServer(("127.0.0.1", 9999), H).serve_forever()
