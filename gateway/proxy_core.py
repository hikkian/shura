"""Forwarding a request to llama-server and relaying the answer, for both gateway entry points.

What this adds over a plain copy loop:

* a failure of llama-server after the answer has started is told to the client (an SSE `error` event and a clean end for a
  stream, a cut connection for anything else) instead of leaving a 200 that just stops;
* a client that goes away while the gateway waits for the model makes the gateway close its upstream connection, so the
  single slot is not kept busy by an answer nobody reads;
* a silent upstream is given up on after `upstreamIdleSeconds` once the answer has begun (before it begins a long prompt may
  legitimately take minutes, bounded by `upstreamFirstByteSeconds`);
* the vision flag and the model name come from the parsed JSON, not from a regular expression over the raw text;
* the browser's CORS headers from llama-server never reach the client (the gateway decides who may call it).

Standard library only.
"""
import http.client
import json
import select
import socket
import threading

import empty_turn
import lang_guard
from pause_stream import PauseTracker

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade", "content-length", "host"}
NOT_FORWARDED_UP = HOP_BY_HOP | {"expect", "origin", "referer", "cookie"}
NOT_FORWARDED_DOWN = HOP_BY_HOP | {"server", "date"}
DEFAULTS = {"upstreamFirstByteSeconds": 3600, "upstreamIdleSeconds": 600,
            "watchClientLeave": True,      # close the upstream when the client goes away while the model is thinking
            "emptyTurnContinue": False}    # continue a turn that ended empty by the output limit (empty_turn.py); off until checked live
BUFFER_LIMIT = 64 * 1024 * 1024
IMAGE_PART_TYPES = {"image_url", "image", "input_image"}


def error_body(message, kind="server_error", code=None, **extra):
    """The error shape OpenAI-style clients read (`error.message`), so the user sees the reason and not a bare status."""
    error = {"message": message, "type": kind}
    if code:
        error["code"] = code
    return {"error": error, **extra}


# --- what the request asks for ----------------------------------------------------------------------------------------

def _has_image(value, depth=0):
    if depth > 4 or not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, dict):
            continue
        if item.get("type") in IMAGE_PART_TYPES or "image_url" in item:
            return True
        if _has_image(item.get("content"), depth + 1):
            return True
    return False


def inspect_body(body, models, default):
    """Return (vision, model_id, body). Reads the real JSON fields, so a tool schema that merely has a property called
    `image_url` does not switch the server into vision mode. The AI-SDK spelling `reasoningEffort` becomes `reasoning_effort`."""
    if not body:
        return False, default, body
    try:
        obj = json.loads(body)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return False, default, body          # llama-server answers the 400 itself
    if not isinstance(obj, dict):
        return False, default, body
    name = obj.get("model")
    model = name if isinstance(name, str) and name in models else default
    vision = _has_image(obj.get("messages")) or _has_image(obj.get("input"))
    if "reasoningEffort" in obj:
        value = obj.pop("reasoningEffort")
        obj.setdefault("reasoning_effort", value)
        body = json.dumps(obj).encode()
    return vision, model, body


# --- noticing that the client left ------------------------------------------------------------------------------------

def leave_detection_applies(handler, enabled=True):
    """A client that half-closes after sending (HTTP/1.0, `Connection: close`, nc -N) still waits for the answer: it is not watched.
    (A half-close is indistinguishable from a close until the answer is written, so an HTTP/1.1 client that half-closes and keeps
    waiting is cut off: none of the usual clients does that; `watchClientLeave: false` switches the watching off.)"""
    return (enabled and getattr(handler, "request_version", "HTTP/1.1") != "HTTP/1.0"
            and "close" not in (handler.headers.get("Connection") or "").lower())


class _NoWatch:
    gone = False

    def stop(self):
        pass


class ClientWatch:
    """While the gateway waits for the model, look at the client's socket: end-of-file there means nobody is listening, and
    `on_gone` closes the upstream connection (llama-server then aborts the generation). Data on the socket (a pipelined
    request) only ends the watch."""

    def __init__(self, client_socket, on_gone, interval=0.25):
        self.sock, self.on_gone, self.interval = client_socket, on_gone, interval
        self.gone = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="shura-client-watch")
        self._thread.start()

    def _run(self):
        try:
            poller = select.poll()                 # poll, not select: select() fails for descriptors above 1023
            poller.register(self.sock, select.POLLIN | select.POLLRDHUP | select.POLLERR | select.POLLHUP)
        except (OSError, ValueError, TypeError):
            return                                 # not a real socket (a test double): nothing to watch
        while not self._stop.is_set():
            try:
                if not poller.poll(self.interval * 1000):
                    continue
                if self.sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b"":
                    self.gone = True
                    self.on_gone()
                return
            except BlockingIOError:
                continue
            except OSError:
                self.gone = True
                self.on_gone()
                return
            except Exception:  # noqa: BLE001 - never let a watcher thread die noisily
                return

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval * 4)


# --- the relay --------------------------------------------------------------------------------------------------------

def _chunk(data):
    return f"{len(data):X}\r\n".encode() + data + b"\r\n"


def _sse_error(message, kind, code):
    payload = json.dumps(error_body(message, kind, code))
    return f"data: {payload}\n\ndata: [DONE]\n\n".encode()


def _abort(handler):
    """End the client connection without a valid terminator, so the client sees an incomplete body and not a finished one."""
    handler.close_connection = True
    try:
        handler.connection.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def forward_request(handler, body, host, port, config=None, lease=None, observe=None, on_stall=None, pause_resume=None,
                    lang_guard_lang=""):
    """Send the handler's request to llama-server and relay the answer.

    `lease` (game mode) is a dict with a "cancel" Event and slots the game monitor uses to cut the request; `observe(kind,
    value)` receives "status", "chunk" (SSE bytes), "json" (the beginning of a JSON body); `on_stall()` is called when the
    upstream went silent mid-answer. Returns {"status": int, "error": bool}.
    """
    cfg = {**DEFAULTS, **(config or {})}
    seen = observe or (lambda kind, value: None)
    conn = http.client.HTTPConnection(host, port, timeout=float(cfg["upstreamFirstByteSeconds"]))
    if lease is not None:
        lease["conn"] = conn
    result = {"status": 502, "error": True}
    watch = None
    cancelled = lambda: bool(lease is not None and lease["cancel"].is_set())  # noqa: E731
    try:
        headers = {k: v for k, v in handler.headers.items() if k.lower() not in NOT_FORWARDED_UP}
        if body or handler.command in ("POST", "PUT", "PATCH"):
            headers["Content-Length"] = str(len(body))
        try:
            conn.request(handler.command, handler.path, body=body or None, headers=headers)
        except (ValueError, http.client.InvalidURL) as error:          # a path or header http.client refuses (control characters)
            return _reply_error(handler, result, 400, error_body(f"Invalid request: {error}", "invalid_request_error", "SHURA_BAD_REQUEST"))
        except (OSError, http.client.HTTPException) as error:
            return _reply_error(handler, result, 502, error_body(f"llama-server unreachable: {error}", "api_error", "SHURA_UPSTREAM_UNREACHABLE"))
        if lease is not None:
            lease["sock"] = conn.sock
        watch = (ClientWatch(handler.connection, lambda: _close_upstream(conn))
                 if leave_detection_applies(handler, cfg.get("watchClientLeave", True)) else _NoWatch())
        try:
            reply = conn.getresponse()
        except (OSError, http.client.HTTPException) as error:
            if watch.gone:
                result.update(status=499, error=False)          # nobody is listening any more
                return result
            if cancelled():
                return _reply_error(handler, result, 503, error_body(
                    "Генерация остановлена для игрового режима; повторите запрос после игры.", "game_pause_timeout", "SHURA_GAME_PAUSE"),
                    retry_after=lease.get("retry_after"))
            return _reply_error(handler, result, 502, error_body(f"llama-server did not answer: {error}", "api_error", "SHURA_UPSTREAM_FAILED"))
        result.update(status=reply.status, error=reply.status >= 400)
        seen("status", reply.status)
        if cancelled():
            return _reply_error(handler, result, 503, error_body(
                "Генерация остановлена для игрового режима; повторите запрос после игры.", "game_pause_timeout", "SHURA_GAME_PAUSE"),
                retry_after=lease.get("retry_after"))
        content_type = (reply.getheader("Content-Type") or "").lower()
        is_sse = "text/event-stream" in content_type
        length = reply.getheader("Content-Length")
        try:
            length_value = int(length) if length is not None else None
        except ValueError:
            length_value = None
        if conn.sock is not None:
            conn.sock.settimeout(float(cfg["upstreamIdleSeconds"]))

        def send_headers(chunked, length_header=None):
            handler.send_response(reply.status)
            for name, value in reply.getheaders():
                lower = name.lower()
                if lower not in NOT_FORWARDED_DOWN and not lower.startswith("access-control-"):
                    handler.send_header(name, value)
            if chunked:
                handler.send_header("Transfer-Encoding", "chunked")
            elif length_header is not None:
                handler.send_header("Content-Length", length_header)
            handler.end_headers()

        # Language correction needs the complete final SSE event before deciding whether to rewrite. Keep this opt-in path
        # separate from pause/empty-turn relays, whose multi-request streams have their own framing semantics.
        if (lang_guard_lang and is_sse and handler.command == "POST" and reply.status < 400
                and handler.path.split("?")[0].endswith("/chat/completions")
                and not cfg.get("emptyTurnContinue") and pause_resume is None):
            try:
                guard = lang_guard.StreamGuard(lang_guard_lang,
                    lambda answer: _rewrite_language(host, port, handler, body, answer, lang_guard_lang, cfg))
                pieces = []
                while True:
                    data = reply.read1(65536)
                    if not data:
                        break
                    pieces.append(guard.feed(data))
                payload = b"".join(pieces) + guard.finish_bytes()
            except (OSError, http.client.HTTPException, TimeoutError):
                return _reply_error(handler, result, 502,
                    error_body("llama-server answer was cut before language check completed", "api_error", "SHURA_UPSTREAM_INTERRUPTED"))
            send_headers(True)
            try:
                if payload:
                    handler.wfile.write(_chunk(payload))
                    handler.wfile.flush()
                    seen("chunk", payload)
                handler.wfile.write(b"0\r\n\r\n")
            except OSError:
                result.update(status=499, error=False)
                handler.close_connection = True
            return result

        # A whole JSON answer of reasonable size is read before anything is sent: a failure then still becomes a clean 502.
        if length_value is not None and not is_sse and length_value <= BUFFER_LIMIT:
            try:
                payload = reply.read() if handler.command != "HEAD" else b""
            except (OSError, http.client.HTTPException) as error:
                if watch.gone:
                    result.update(status=499, error=False)
                    return result
                if cancelled():
                    return _reply_error(handler, result, 503, error_body(
                        "Генерация остановлена для игрового режима; повторите запрос после игры.", "game_pause_timeout", "SHURA_GAME_PAUSE"),
                        retry_after=lease.get("retry_after"))
                return _reply_error(handler, result, 502, error_body(f"llama-server answer was cut: {error}", "api_error", "SHURA_UPSTREAM_INTERRUPTED"))
            send_headers(False, str(len(payload)) if handler.command != "HEAD" else length)
            if payload:
                handler.wfile.write(payload)
                seen("json", payload)
            return result

        if handler.command == "HEAD" or reply.status in (204, 304) or 100 <= reply.status < 200:
            send_headers(False, length if handler.command == "HEAD" else None)      # these answers have no body, so no framing either
            return result
        chunked = length_value is None or is_sse
        send_headers(chunked, None if chunked else length)
        if chunked and is_sse and cfg.get("emptyTurnContinue") and handler.command == "POST" \
                and handler.path.split("?")[0].endswith("/chat/completions"):
            return _relay_stream_continuing(handler, reply, conn, body, host, port, cfg, seen, result)
        if chunked and is_sse and pause_resume is not None and handler.command == "POST" \
                and handler.path.split("?")[0].endswith("/chat/completions"):
            # VRAMLOGIC1i: a governor pause is continued here. The empty-turn relay (above, when enabled) does not continue pauses.
            return _relay_stream_pausable(handler, reply, conn, body, host, port, cfg, seen, result, watch, lease, on_stall,
                                          cancelled, pause_resume)
        metadata = bytearray()
        sent_length = 0
        while handler.command != "HEAD":
            try:
                data = reply.read1(65536)
            except (OSError, http.client.HTTPException) as error:
                result["error"] = True
                if watch.gone:
                    result.update(status=499, error=False)
                    _abort(handler)
                    return result
                if isinstance(error, TimeoutError) and on_stall is not None:
                    on_stall()
                if cancelled():
                    message, kind, code = ("Генерация остановлена для игрового режима; повторите запрос после игры.",
                                           "game_pause_timeout", "SHURA_GAME_PAUSE")
                elif isinstance(error, TimeoutError):
                    message, kind, code = ("llama-server stopped answering mid-response; the answer is incomplete.", "api_error", "SHURA_UPSTREAM_STALLED")
                else:
                    message, kind, code = ("llama-server stopped mid-response; the answer is incomplete.", "api_error", "SHURA_RESPONSE_INTERRUPTED")
                _finish_failed(handler, chunked, is_sse, message, kind, code)
                return result
            if not data:
                break
            try:
                handler.wfile.write(_chunk(data) if chunked else data)
                handler.wfile.flush()
            except OSError:
                result.update(status=499, error=False)          # the client went away; closing upstream (finally) aborts the generation
                handler.close_connection = True
                return result
            sent_length += len(data)
            if is_sse:
                seen("chunk", data)
            elif len(metadata) < 65536:
                metadata.extend(data[:65536 - len(metadata)])
        if not chunked and sent_length < length_value and handler.command != "HEAD":
            result["error"] = True
            _abort(handler)                     # promised more bytes than arrived: do not leave the client waiting for them
            return result
        if chunked:
            try:
                handler.wfile.write(b"0\r\n\r\n")
            except OSError:
                handler.close_connection = True
        if not is_sse and metadata:
            seen("json", bytes(metadata))
        return result
    except OSError:
        result["error"] = True                  # writing to a client that already left
        handler.close_connection = True
        return result
    finally:
        if watch is not None:
            watch.stop()
        conn.close()


def _rewrite_language(host, port, handler, body, answer, language, cfg):
    """Ask the same model to rewrite only this final answer; any failure leaves the original stream intact."""
    try:
        req = json.loads(body)
        rewritten = lang_guard.rewrite_request(req, answer, language)
        payload = json.dumps(rewritten, ensure_ascii=False).encode()
        headers = {k: v for k, v in handler.headers.items() if k.lower() not in NOT_FORWARDED_UP}
        headers["Content-Length"] = str(len(payload))
        conn = http.client.HTTPConnection(host, port, timeout=float(cfg["upstreamFirstByteSeconds"]))
        try:
            conn.request("POST", handler.path, body=payload, headers=headers)
            reply = conn.getresponse()
            if reply.status >= 400:
                return None
            value = json.loads(reply.read())
            return value["choices"][0]["message"].get("content")
        finally:
            conn.close()
    except (ValueError, KeyError, TypeError, OSError, http.client.HTTPException, TimeoutError):
        return None


def _relay_stream_pausable(handler, reply, conn, body, host, port, cfg, seen, result, watch, lease, on_stall, cancelled, pause_resume):
    """The standard relay loop, plus the governor's pause: the pause frame is held back; when the answer ends with one, the gateway
    restores the slot (pause_resume) and the same request continues on a new connection. The client sees one stream. Stall,
    interruption, client-gone and the closing frames are the standard loop's."""
    tracker = PauseTracker()
    opened = []

    def send(event):
        handler.wfile.write(_chunk(event))
        handler.wfile.flush()

    def fail(message, kind, code):
        result["error"] = True
        _finish_failed(handler, True, True, message, kind, code)
        return result

    try:
        while handler.command != "HEAD":
            try:
                data = reply.read1(65536)
            except (OSError, http.client.HTTPException) as error:
                result["error"] = True
                if watch.gone:
                    result.update(status=499, error=False)
                    _abort(handler)
                    return result
                if isinstance(error, TimeoutError) and on_stall is not None:
                    on_stall()
                if cancelled():
                    return fail("Генерация остановлена для игрового режима; повторите запрос после игры.",
                                "game_pause_timeout", "SHURA_GAME_PAUSE")
                if isinstance(error, TimeoutError):
                    return fail("llama-server stopped answering mid-response; the answer is incomplete.", "api_error", "SHURA_UPSTREAM_STALLED")
                return fail("llama-server stopped mid-response; the answer is incomplete.", "api_error", "SHURA_RESPONSE_INTERRUPTED")
            if not data:
                if tracker.pause is None:
                    break
                # PAUSEWAIT1: while the resume waits for memory the client's stream gets an SSE comment, not silence
                new_body = pause_resume(tracker.pause, list(tracker.tool_call_ids),
                                        heartbeat=lambda: (handler.wfile.write(_chunk(b": keep-alive\n\n")), handler.wfile.flush()))
                if new_body is None:
                    return fail("llama-server paused the answer and it could not be continued; the answer is incomplete.",
                                "api_error", "SHURA_RESPONSE_INTERRUPTED")
                try:
                    nxt = http.client.HTTPConnection(host, port, timeout=float(cfg["upstreamFirstByteSeconds"]))
                    headers = {k: v for k, v in handler.headers.items() if k.lower() not in NOT_FORWARDED_UP}
                    headers["Content-Length"] = str(len(new_body))
                    nxt.request(handler.command, handler.path, body=new_body, headers=headers)
                    reply2 = nxt.getresponse()
                except (OSError, http.client.HTTPException):
                    return fail("llama-server did not continue the paused answer; the answer is incomplete.",
                                "api_error", "SHURA_RESPONSE_INTERRUPTED")
                if reply2.status != 200 or "text/event-stream" not in (reply2.getheader("Content-Type") or "").lower():
                    nxt.close()
                    return fail("llama-server refused to continue the paused answer; the answer is incomplete.",
                                "api_error", "SHURA_RESPONSE_INTERRUPTED")
                if nxt.sock is not None:
                    nxt.sock.settimeout(float(cfg["upstreamIdleSeconds"]))
                opened.append(nxt)
                reply = reply2
                if lease is not None:
                    lease["conn"] = nxt
                    lease["sock"] = nxt.sock
                tracker = PauseTracker(tracker.tool_call_ids, drop_role=True)
                continue
            seen("chunk", data)
            for event in tracker.feed(data):
                send(event)
        rest = tracker.remainder()
        if rest:
            handler.wfile.write(_chunk(rest))
        handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()
        return result
    except OSError:
        result["error"] = True
        handler.close_connection = True
        return result
    finally:
        for extra in opened:
            try:
                extra.close()
            except OSError:
                pass


def _relay_stream_continuing(handler, reply, conn, body, host, port, cfg, seen, result):
    """Relay a chat stream. A turn that ends empty by the output limit is continued (see empty_turn.py).

    The closing frames of each stream are held until it is known whether the turn ends, so the client never receives a
    finish reason that is about to be replaced. The reasoning of the empty stream goes back to the model as an unfinished
    assistant message, and the continuation's frames follow on the same answer."""
    reasoning, continuations, current, owned = [], 0, reply, None
    try:
        while True:
            frames, state, held = empty_turn.SSEFrames(), empty_turn.TurnState(), []
            while True:
                try:
                    data = current.read1(65536)
                except (OSError, http.client.HTTPException) as error:
                    result["error"] = True
                    if isinstance(error, TimeoutError):
                        message, kind, code = ("llama-server stopped answering mid-response; the answer is incomplete.",
                                               "api_error", "SHURA_UPSTREAM_STALLED")
                    else:
                        message, kind, code = ("llama-server stopped mid-response; the answer is incomplete.",
                                               "api_error", "SHURA_RESPONSE_INTERRUPTED")
                    _finish_failed(handler, True, True, message, kind, code)
                    return result
                if not data:
                    break
                seen("chunk", data)
                for frame in frames.feed(data):
                    state.feed(frame)
                    if held or state.ends_turn(frame):
                        held.append(frame)
                        continue
                    try:
                        handler.wfile.write(_chunk(frame))
                        handler.wfile.flush()
                    except OSError:
                        result.update(status=499, error=False)
                        handler.close_connection = True
                        return result
            reasoning.extend(state.reasoning)
            following = None
            if state.empty_length():
                if continuations >= empty_turn.MAX_CONTINUATIONS:
                    empty_turn.STATS["exhausted"] += 1
                    seen("empty_turn", "exhausted")
                else:
                    body2 = empty_turn.continuation_body(body, "".join(reasoning))
                    following = _open_continuation(handler, body2, host, port, cfg) if body2 is not None else None
                    if following is None:
                        empty_turn.STATS["gave_up"] += 1
                        seen("empty_turn", "gave_up")
            if following is None:
                try:
                    for frame in held:
                        handler.wfile.write(_chunk(frame))
                    handler.wfile.write(b"0\r\n\r\n")
                    handler.wfile.flush()
                except OSError:
                    handler.close_connection = True
                return result
            # The empty stream's held frames carried its finish reason: they are dropped, the continuation replaces them.
            empty_turn.STATS["continued"] += 1
            seen("empty_turn", "continued")
            continuations += 1
            if owned is not None:
                owned.close()
            owned, current = following
    finally:
        if owned is not None:
            owned.close()


def _open_continuation(handler, body2, host, port, cfg):
    """(connection, reply) of the continuation request, or None when it cannot be made or is not a stream."""
    conn2 = http.client.HTTPConnection(host, port, timeout=float(cfg["upstreamFirstByteSeconds"]))
    try:
        headers = {k: v for k, v in handler.headers.items() if k.lower() not in NOT_FORWARDED_UP}
        headers["Content-Length"] = str(len(body2))
        conn2.request("POST", handler.path, body=body2, headers=headers)
        reply2 = conn2.getresponse()
        if reply2.status != 200 or "text/event-stream" not in (reply2.getheader("Content-Type") or "").lower():
            conn2.close()
            return None
        if conn2.sock is not None:
            conn2.sock.settimeout(float(cfg["upstreamIdleSeconds"]))
        return conn2, reply2
    except (OSError, http.client.HTTPException, ValueError):
        conn2.close()
        return None


def _close_upstream(conn):
    sock = conn.sock
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _reply_error(handler, result, status, body, retry_after=None):
    result.update(status=status, error=True)
    payload = json.dumps(body).encode()
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        if retry_after is not None:
            handler.send_header("Retry-After", str(retry_after))
        handler.end_headers()
        if handler.command != "HEAD":
            handler.wfile.write(payload)
    except OSError:
        handler.close_connection = True
    return result


def _finish_failed(handler, chunked, is_sse, message, kind, code):
    """The answer has begun and cannot be completed. A stream gets an error event and a proper end, so the client reports
    the reason; any other body is cut so that it cannot pass for a complete one."""
    if chunked and is_sse:
        try:
            handler.wfile.write(_chunk(_sse_error(message, kind, code)) + b"0\r\n\r\n")
            handler.wfile.flush()
        except OSError:
            handler.close_connection = True
        return
    _abort(handler)
