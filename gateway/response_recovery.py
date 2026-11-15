"""Never replay an HTTP response once even its headers reached the client."""

import http.client
import json
import time

from proxy_core import ClientWatch, NOT_FORWARDED_DOWN, NOT_FORWARDED_UP, _close_upstream, leave_detection_applies


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
        self.content = False     # VRAMCOR4: a real token (text, reasoning, tool call) went to the client; progress frames do not count
        self.pause = None        # VRAMLOGIC1i: the error object of a vram_pause frame (the governor paused this answer)
        self.tool_call_ids = []  # ids of tool calls already sent to the client: a resumed answer must keep them

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
            if isinstance(value, dict) and isinstance(value.get("error"), dict) and value["error"].get("type") == "vram_pause":
                self.pause = value["error"]
                continue
            if isinstance(value, dict):
                for choice in value.get("choices") or []:
                    for call in ((choice.get("delta") or {}) if isinstance(choice, dict) else {}).get("tool_calls") or []:
                        if isinstance(call, dict) and call.get("id") and call["id"] not in self.tool_call_ids:
                            self.tool_call_ids.append(call["id"])
                if value.get("content"):
                    self.content = True                       # the native /completion stream
                for choice in value.get("choices") or []:
                    delta = choice.get("delta") or {} if isinstance(choice, dict) else {}
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                        self.content = True


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
                    "type": "api_error",             # not a client mistake: clients must not show "bad request" or refuse to go on
                    "code": "SHURA_GAME_PAUSED"
                    if game
                    else "SHURA_RESPONSE_INTERRUPTED",
                }
            }
        )
        + "\n\n"
    ).encode()


def is_role_chunk(event):
    """True for the first chunk of a stream: {"choices": [{"delta": {"role": "assistant"}}]} (the client already has it)."""
    if not event.startswith(b"data:"):
        return False
    try:
        value = json.loads(event[5:].strip())
    except ValueError:
        return False
    choices = value.get("choices") if isinstance(value, dict) else None
    delta = choices[0].get("delta") if choices and isinstance(choices[0], dict) else None
    return isinstance(delta, dict) and delta.get("role") == "assistant" and not delta.get("content") and not delta.get("reasoning_content")


def forward(handler, body, open_connection, recover, cancel=None, register=None, observe=None, idle_seconds=None, on_stall=None, watch_client=True,
            retry_before_content=False, pause_resume=None):
    """First body read is buffered before headers; all later failures are non-retryable.

    A client that leaves while the model is still thinking closes the upstream connection (the slot is freed) and is never a
    reason to restart the backend. With `idle_seconds`, an upstream that goes silent mid-answer is given up on and reported.

    retry_before_content (VRAMCOR4): a stream that ends or breaks before any real token reached the client (only the gateway's own
    progress frames did) is replayed once on the restarted backend, into the same open response. After the first token it is never replayed.
    It may be a callable, asked when the stream breaks (GPURESET1: replay only when the break was a GPU reset)."""
    sent = False
    conn = None
    state = StreamState()
    stream = False
    watch = None

    def notify(kind, value, is_stream):
        if observe:
            try:
                observe(kind, value, is_stream)
            except Exception:
                pass  # Telemetry is best effort and can never interrupt a response.
    try:
        for attempt in range(2):
            conn = open_connection()
            if register:
                register(conn)
            try:
                headers = {
                    k: v
                    for k, v in handler.headers.items()
                    if k.lower() not in handler.hop_by_hop and k.lower() not in NOT_FORWARDED_UP
                }
                conn.request(
                    handler.command, handler.path, body=body or None, headers=headers
                )
                if watch is not None:
                    watch.stop()
                client = getattr(handler, "connection", None)
                watch = (ClientWatch(client, lambda c=conn: _close_upstream(c))
                         if client is not None and leave_detection_applies(handler, watch_client) else None)
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
                if watch is not None and watch.gone:
                    handler.close_connection = True        # nobody is listening: neither an answer nor a restart is wanted
                    return None
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
        notify("status", reply.status, stream)
        if idle_seconds and conn.sock is not None:
            conn.sock.settimeout(float(idle_seconds))
        handler.send_response(reply.status)
        for key, value in reply.getheaders():
            lower = key.lower()
            if lower not in handler.hop_by_hop and lower not in NOT_FORWARDED_DOWN and not lower.startswith("access-control-"):
                handler.send_header(key, value)
        handler.send_header("Transfer-Encoding", "chunked")
        # end_headers itself writes bytes: never recover beyond this boundary.
        sent = True
        handler.end_headers()

        def write(data):
            if data:
                handler.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                handler.wfile.flush()

        pending = [b""]      # VRAMLOGIC1i: pause-aware output works on whole SSE events, so the pause frame can be dropped
        drop_role = [False]  # the first chunk of a resumed answer is the role chunk the client already has

        def emit(data):
            if pause_resume is None:
                write(data)
                return
            pending[0] += data
            while b"\n\n" in pending[0]:
                event, pending[0] = pending[0].split(b"\n\n", 1)
                event += b"\n\n"
                if b'"vram_pause"' in event:
                    continue
                if drop_role[0]:
                    drop_role[0] = False
                    if is_role_chunk(event):
                        continue
                write(event)

        (emit if stream else write)(first)
        notify("chunk" if stream else "body", first, stream)
        if stream:
            state.feed(first)
            replays = 1 if retry_before_content else 0
            resumes = 1 if pause_resume is not None else 0

            def resume_after_pause():
                """VRAMLOGIC1i: the governor paused the answer. pause_resume restarts and restores the slot and returns the same
                request with `resume`; the new reply continues the same response (its role chunk is dropped)."""
                nonlocal conn, reply, state, watch, body
                ids = list(state.tool_call_ids)
                try:
                    conn.close()
                except OSError:
                    pass
                # PAUSEWAIT1: while the resume waits for memory the client's stream gets an SSE comment, not silence
                new_body = pause_resume(state.pause, ids, heartbeat=lambda: write(b": keep-alive\n\n"))
                if new_body is None:
                    return False
                body = new_body
                try:
                    conn = open_connection()
                    if register:
                        register(conn)
                    headers = {k: v for k, v in handler.headers.items()
                               if k.lower() not in handler.hop_by_hop and k.lower() not in NOT_FORWARDED_UP}
                    conn.request(handler.command, handler.path, body=body, headers=headers)
                    if watch is not None:
                        watch.stop()
                    client = getattr(handler, "connection", None)
                    watch = (ClientWatch(client, lambda c=conn: _close_upstream(c))
                             if client is not None and leave_detection_applies(handler, watch_client) else None)
                    reply = conn.getresponse()
                    if reply.status != 200 or "text/event-stream" not in (reply.getheader("Content-Type") or ""):
                        return False
                    if idle_seconds and conn.sock is not None:
                        conn.sock.settimeout(float(idle_seconds))
                except (OSError, http.client.HTTPException):
                    return False
                state = StreamState()
                state.tool_call_ids = ids
                drop_role[0] = True
                handler.recovery_event("response_resumed_after_pause", "The governor paused the answer; it continues on the restored slot")
                return True

            def replay():
                """Restart the backend and open the same request again; the new reply must be a 200 stream, else the old failure stands."""
                nonlocal conn, reply, state, watch
                if cancel and cancel.is_set():
                    return False
                try:
                    conn.close()
                except OSError:
                    pass
                if not recover():
                    return False
                try:
                    conn = open_connection()
                    if register:
                        register(conn)
                    headers = {k: v for k, v in handler.headers.items()
                               if k.lower() not in handler.hop_by_hop and k.lower() not in NOT_FORWARDED_UP}
                    conn.request(handler.command, handler.path, body=body or None, headers=headers)
                    if watch is not None:
                        watch.stop()
                    client = getattr(handler, "connection", None)
                    watch = (ClientWatch(client, lambda c=conn: _close_upstream(c))
                             if client is not None and leave_detection_applies(handler, watch_client) else None)
                    reply = conn.getresponse()
                    if reply.status != 200 or "text/event-stream" not in (reply.getheader("Content-Type") or ""):
                        return False
                    if idle_seconds and conn.sock is not None:
                        conn.sock.settimeout(float(idle_seconds))
                except (OSError, http.client.HTTPException):
                    return False
                state = StreamState()
                return True

            while True:
                failed = False
                try:
                    data = reply.read1(65536)
                except (OSError, http.client.HTTPException) as error:
                    if watch is not None and watch.gone:
                        handler.close_connection = True      # the client left: nothing to tell it
                        return None
                    if isinstance(error, TimeoutError) and on_stall is not None:
                        on_stall()
                        break                                # a silent server is declared hung, not retried
                    failed = True
                    data = b""
                if cancel and cancel.is_set():
                    break
                if not data:
                    if state.pause is not None and resumes and reply.status == 200:
                        resumes -= 1
                        if resume_after_pause():
                            continue
                        break
                    if (replays and not state.complete and not state.content and reply.status == 200
                            and (retry_before_content() if callable(retry_before_content) else True) and replay()):
                        replays -= 1
                        handler.recovery_event("response_replayed_before_content",
                                               "The backend stopped before the first token; the request was replayed once")
                        continue
                    break
                state.feed(data)
                emit(data)
                notify("chunk", data, True)
            if pending[0] and state.pause is None:
                write(pending[0])
            pending[0] = b""
            if reply.status == 200 and not state.complete:
                closing = interrupted_event(bool(cancel and cancel.is_set()))
                if pause_resume is not None:
                    closing += b"data: [DONE]\n\n"   # the plain relay (proxy_core) closes the same way under the governor
                write(closing)
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
        if watch is not None:
            watch.stop()
        if conn:
            conn.close()
