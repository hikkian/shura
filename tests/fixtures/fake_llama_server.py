#!/usr/bin/env python3
"""Stand-in for llama-server in tests: serves /health and /completion on --port. FAKE_MODE=crash exits at once,
FAKE_MODE=empty answers with an empty text, FAKE_TOK_S sets the reported speed."""
import json
import os
import socketserver
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

mode = os.environ.get("FAKE_MODE", "ok")
if mode == "crash":
    print("fake server: cannot load the model", file=sys.stderr)
    sys.exit(3)
port = int(sys.argv[sys.argv.index("--port") + 1])


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"status": "ok"})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._send({"content": "" if mode == "empty" else " there was a little girl",
                    "timings": {"predicted_n": 24, "predicted_per_second": float(os.environ.get("FAKE_TOK_S", "50")),
                                "prompt_per_second": 400.0}})


class Server(HTTPServer):
    def server_bind(self):                      # HTTPServer resolves the host name here, which stalls for ages on some runners
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


Server(("127.0.0.1", port), H).serve_forever()
