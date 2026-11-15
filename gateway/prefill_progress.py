"""PROG1: how far the model has read a long prompt, for the chat's progress line ("reading the context 45 000 / 100 000").

llama-server, asked with "return_progress": true, sends chunks with a "prompt_progress" object {total, cache, processed,
time_ms} while it reads the prompt. The gateway asks for them on streamed chat requests (promptProgress), reads them from
the stream it relays and publishes the state in GET /guardian/status. The chunks themselves are ordinary role-only deltas
that clients ignore."""
import json
import re
import threading
import time

DEFAULTS = {"promptProgress": False}
_PROGRESS = re.compile(rb'"prompt_progress"\s*:\s*(\{[^{}]*\})')


def ask_for_progress(body, enabled):
    """Add "return_progress": true to a streamed chat request that does not say otherwise; anything else stays as it is."""
    if not enabled or not body:
        return body
    try:
        obj = json.loads(body)
    except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
        return body
    if not isinstance(obj, dict) or obj.get("stream") is not True or "return_progress" in obj:
        return body
    obj["return_progress"] = True
    return json.dumps(obj).encode()


class Tracker:
    def __init__(self, clock=time.time):
        self.clock = clock
        self.lock = threading.Lock()
        self.state = {"state": "idle"}

    def start(self):
        with self.lock:
            self.state = {"state": "waiting", "started_at": self.clock()}

    def observe(self, data):
        """Read the last progress object in a relayed piece of the stream (a piece that splits one is simply skipped)."""
        last = None
        for match in _PROGRESS.finditer(data if isinstance(data, (bytes, bytearray)) else bytes(data)):
            last = match
        if last is None:
            return
        try:
            p = json.loads(last.group(1))
            total, cache, processed, ms = (int(p["total"]), int(p.get("cache", 0)), int(p["processed"]), float(p.get("time_ms", 0)))
        except (ValueError, KeyError, TypeError):
            return
        todo, done = max(0, total - cache), max(0, processed - cache)
        rate = done / (ms / 1000) if ms > 0 and done > 0 else None
        left = round((todo - done) / rate, 1) if rate and todo > done else 0.0
        with self.lock:
            if self.state.get("state") == "idle":
                return                                  # an answer that already finished
            self.state = {"state": "done" if processed >= total else "reading", "started_at": self.state.get("started_at"),
                          "total": total, "cache": cache, "processed": processed,
                          "rate_tps": round(rate) if rate else None, "eta_s": left}

    def finish(self):
        with self.lock:
            self.state = {"state": "idle"}

    def snapshot(self):
        with self.lock:
            return dict(self.state)
