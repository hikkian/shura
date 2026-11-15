"""Requests the engine makes for itself, answered without the model (see reports/REPORT-CACHE1.md and REPORT-CACHE2.md).

A chat title used to be a model request. With one llama-server slot it took the slot from the conversation, so the next turn
read the whole conversation again (about 13 000 tokens). The title is now made from the user's first message: no model is
loaded, the slot is untouched, and the answer has the OpenAI shape the engine expects. Only titles are answered here;
summaries and the real chat stay with the model.
"""
import json
import time

TITLE_SYSTEM = "You are a title generator."
TITLE_ASK = "Generate a title for this conversation:"
MAX_CHARS = 60
FALLBACK = "Chat"


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def parse_title_request(path, raw):
    """The decoded body when `raw` is a title request, else None. Anything that is not clearly a title request is left alone."""
    if not path.endswith("/chat/completions") or not raw:
        return None
    try:
        obj = json.loads(raw)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None
    if not isinstance(obj, dict) or obj.get("tools"):
        return None
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or not msgs or not isinstance(msgs[0], dict) or msgs[0].get("role") != "system":
        return None
    return obj if _text(msgs[0].get("content")).lstrip().startswith(TITLE_SYSTEM) else None


def title_from(obj):
    """A short title from the user's first real message (the title request quotes it after its own question)."""
    users = [_text(m.get("content")).strip() for m in obj.get("messages", []) if isinstance(m, dict) and m.get("role") == "user"]
    users = [u for u in users if u and not u.startswith(TITLE_ASK)]
    text = " ".join(users[-1].split()) if users else ""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if len(text) > MAX_CHARS:
        space = text.rfind(" ", 0, MAX_CHARS)
        text = text[:space] if space > 0 else text[:MAX_CHARS]
    text = text.rstrip(" .,;:!?-")
    return text or FALLBACK


def _usage():
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def response_bytes(obj, title, stream):
    """The body the engine expects: a plain completion, or the same events a streamed completion sends."""
    model = obj.get("model", "")
    created = int(time.time())
    if not stream:
        body = {"id": "chatcmpl-title", "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": title}, "finish_reason": "stop"}],
                "usage": _usage()}
        return "application/json", json.dumps(body, ensure_ascii=False).encode()

    def event(delta, finish=None, extra=None):
        chunk = {"id": "chatcmpl-title", "object": "chat.completion.chunk", "created": created, "model": model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if extra:
            chunk.update(extra)
        return ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode()

    payload = (event({"role": "assistant", "content": ""})
               + event({"content": title})
               + event({}, "stop", {"usage": _usage()})
               + b"data: [DONE]\n\n")
    return "text/event-stream", payload


def serve_title(handler, obj):
    """Writes the title answer on `handler` (a BaseHTTPRequestHandler). The model is never asked and never started."""
    stream = bool(obj.get("stream"))
    content_type, payload = response_bytes(obj, title_from(obj), stream)
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    handler.wfile.write(payload)


class RequestRejected(Exception):
    """Raised by a fit hook when a request cannot be served at all: the handler answers with this status and body instead of
    forwarding it (CACHE3: an honest error, never a silent "send as is")."""

    def __init__(self, status, payload):
        super().__init__(str(payload))
        self.status, self.payload = status, payload


def with_title_answer(base, enabled, fit=None):
    """A subclass of a request handler that answers chat titles itself. Both gateway paths use it, so the preserved legacy
    handler (_gateway_legacy.py, pinned by a hash test) stays unchanged. `fit(path, body) -> body`, when given, shortens a
    chat request that does not fit the model's window (CACHE3) before the handler sees it."""

    class TitleAnswering(base):
        def handle_generation(self, path, body):
            if enabled():
                title = parse_title_request(path, body)
                if title is not None:
                    return serve_title(self, title)
            if fit is not None:
                try:
                    body = fit(path, body)
                except RequestRejected as rejected:
                    return self.send_json(rejected.status, rejected.payload)
            return super().handle_generation(path, body)

    TitleAnswering.__name__ = base.__name__
    TitleAnswering.__qualname__ = base.__qualname__
    return TitleAnswering
