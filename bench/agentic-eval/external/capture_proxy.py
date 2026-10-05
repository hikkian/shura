"""Body-capturing wrapper around the unchanged v2 measurement proxy.

Raw request JSON is gzip-compressed in the run's tmpfs output directory. It may
contain the task prompt, so the caller must keep that directory under /dev/shm and
remove it after the experiment.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import threading
import time

import proxy


class _TeeReader:
    def __init__(self, source, capture: bytearray):
        self.source = source
        self.capture = capture

    def read(self, size=-1):
        value = self.source.read(size)
        self.capture.extend(value)
        return value

    def __getattr__(self, name):
        return getattr(self.source, name)


class CaptureHandler(proxy.Handler):
    capture_path: Path | None = None
    capture_lock = threading.Lock()

    def _forward(self):
        original = self.rfile
        captured = bytearray()
        self.rfile = _TeeReader(original, captured)
        try:
            super()._forward()
        finally:
            self.rfile = original
            if self.command == "POST" and captured and self.capture_path:
                try:
                    body = json.loads(captured)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    body = {"raw_body_utf8": captured.decode("utf-8", errors="replace")}
                try:
                    effective = json.loads(proxy.rewrite_body(json.dumps(body).encode(), self.extra))
                except (TypeError, ValueError, UnicodeDecodeError):
                    effective = None
                record = {"ts": time.time(), "path": self.path, "request": body,
                          "effective_request": effective}
                base = Path(self.capture_path)
                suffix = ".jsonl.gz"
                directory = base.parent / (base.name[:-len(suffix)] if base.name.endswith(suffix) else base.stem)
                directory.mkdir(parents=True, exist_ok=True)
                with self.capture_lock:
                    sequence = getattr(type(self), "capture_sequence", 0)
                    type(self).capture_sequence = sequence + 1
                filename = f"{time.time_ns()}-{os.getpid()}-{sequence}.json.gz"
                destination = directory / filename
                temporary = directory / f"{filename}.tmp"
                with gzip.open(temporary, "wb") as stream:
                    stream.write(json.dumps(record, separators=(",", ":")).encode() + b"\n")
                os.replace(temporary, destination)


def serve(listen: int, upstream: str, usage_log: str, body_log: str, extra=None):
    handler = type("H", (CaptureHandler,), {
        "upstream": upstream,
        "extra": extra or {},
        "log_path": usage_log,
        "capture_path": Path(body_log),
        "capture_lock": threading.Lock(),
        "capture_sequence": 0,
        "do_GET": CaptureHandler._forward,
        "do_POST": CaptureHandler._forward,
        "do_PUT": CaptureHandler._forward,
        "do_DELETE": CaptureHandler._forward,
    })
    server = proxy.Quick(("127.0.0.1", listen), handler)
    server.daemon_threads = True
    return server
