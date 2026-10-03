"""Never replay an HTTP response once even its headers reached the client."""

import http.client
import json
import time


def wait_for_process_exit(proc, timeout=6.0, interval=0.02):
    """Confirm child exit after a disconnected socket before deciding whether to replay."""
    if proc is None:
        return False
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        if proc.poll() is not None:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(interval, remaining))


class StreamState:
    def __init__(self):
        self.tail = b""
        self.complete = False

    def feed(self, data):
        self.tail += data
        if len(self.tail) > 1024 * 1024:
            raise ValueError("Oversized SSE frame")
        while b"\n" in self.tail:
            line, self.tail = self.tail.split(b"\n", 1)
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                self.complete = True
                continue
            try:
                value = json.loads(payload)
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("stop") is True:
                self.complete = True


def interrupted_event(game=False):
    message = (
        "Генерация остановлена для игры. Ответ неполный; продолжите вручную."
        if game
        else "Генерация прервана. Ответ неполный; автоповтор отключён. Продолжите вручную."
    )
    return (
        "data: "
        + json.dumps(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "SHURA_GAME_PAUSED"
                    if game
                    else "SHURA_RESPONSE_INTERRUPTED",
                }
            }
        )
        + "\n\n"
    ).encode()


def forward(handler, body, open_connection, recover, cancel=None, register=None):
    """First body read is buffered before headers; all later failures are non-retryable."""
    sent = False
    conn = None
    state = StreamState()
    stream = False
    try:
        for attempt in range(2):
            conn = open_connection()
            if register:
                register(conn)
            try:
                headers = {
                    k: v
                    for k, v in handler.headers.items()
                    if k.lower() not in handler.hop_by_hop
                }
                conn.request(
                    handler.command, handler.path, body=body or None, headers=headers
                )
                reply = conn.getresponse()
                stream = "text/event-stream" in (reply.getheader("Content-Type") or "")
                first = reply.read1(65536) if stream else reply.read()
                if cancel and cancel.is_set():
                    raise ConnectionAbortedError("Game requested pause")
                if stream and not first:
                    raise http.client.RemoteDisconnected("Empty upstream stream")
                break
            except (OSError, http.client.HTTPException):
                conn.close()
                if not attempt and not (cancel and cancel.is_set()) and recover():
                    continue
                return handler.send_json(
                    503,
                    {
                        "error": "Backend unavailable before response",
                        "code": "SHURA_BACKEND_NOT_READY",
                    },
                    5,
                )
        handler.send_response(reply.status)
        for key, value in reply.getheaders():
            if key.lower() not in handler.hop_by_hop:
                handler.send_header(key, value)
        handler.send_header("Transfer-Encoding", "chunked")
        # end_headers itself writes bytes: never recover beyond this boundary.
        sent = True
        handler.end_headers()

        def write(data):
            if data:
                handler.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                handler.wfile.flush()

        write(first)
        if stream:
            state.feed(first)
            while True:
                try:
                    data = reply.read1(65536)
                except (OSError, http.client.HTTPException):
                    break
                if cancel and cancel.is_set():
                    break
                if not data:
                    break
                state.feed(data)
                write(data)
            if reply.status == 200 and not state.complete:
                write(interrupted_event(bool(cancel and cancel.is_set())))
                handler.recovery_event(
                    "response_interrupted",
                    "Non-retryable stream error; no gateway replay",
                )
        handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        handler.close_connection = (
            True  # client departed: not grounds for backend restart
        )
    except (OSError, http.client.HTTPException, ValueError):
        if sent and stream:
            try:
                data = interrupted_event()
                handler.wfile.write(
                    f"{len(data):X}\r\n".encode() + data + b"\r\n0\r\n\r\n"
                )
                handler.wfile.flush()
            except OSError:
                handler.close_connection = True
        elif not sent:
            handler.send_json(
                503,
                {"error": "Backend not ready", "code": "SHURA_BACKEND_NOT_READY"},
                5,
            )
        else:
            handler.close_connection = True
    finally:
        if conn:
            conn.close()
