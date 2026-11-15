"""Pause-aware SSE pieces shared by the plain relay (proxy_core) and the recovering relay (response_recovery). VRAMLOGIC1i.

The governor may pause an answer: the server sends one error event of type vram_pause as the last frame and exits with 75
(reports/vramlogic1/API.md). The gateway must not show that frame to the client; it restores the slot and continues the same
response (`resume`), so the client sees one stream. PauseTracker splits the bytes into whole SSE events, drops the pause frame
and the role chunk of a resumed answer, and remembers the tool-call ids the client already has.
"""

import json


def _payload(event):
    if not event.startswith(b"data:"):
        return None
    try:
        return json.loads(event[5:].strip())
    except ValueError:
        return None


def is_role_chunk(event):
    """True for the first chunk of a stream: {"choices": [{"delta": {"role": "assistant"}}]} (the client already has it)."""
    value = _payload(event)
    choices = value.get("choices") if isinstance(value, dict) else None
    delta = choices[0].get("delta") if choices and isinstance(choices[0], dict) else None
    return isinstance(delta, dict) and delta.get("role") == "assistant" and not delta.get("content") and not delta.get("reasoning_content")


def pause_error_of(event):
    """The raw error object of a vram_pause frame, or None."""
    value = _payload(event)
    if isinstance(value, dict) and isinstance(value.get("error"), dict) and value["error"].get("type") == "vram_pause":
        return value["error"]
    return None


class PauseTracker:
    def __init__(self, tool_call_ids=None, drop_role=False):
        self.buf = b""
        self.pause = None
        self.tool_call_ids = list(tool_call_ids or [])
        self.drop_role = drop_role

    def feed(self, data):
        """Return the complete events to send to the client. Records a pause frame instead of returning it."""
        self.buf += data
        out = []
        while b"\n\n" in self.buf:
            event, self.buf = self.buf.split(b"\n\n", 1)
            event += b"\n\n"
            raw = pause_error_of(event)
            if raw is not None:
                self.pause = raw
                continue
            if self.pause is not None:
                continue                        # nothing after the pause frame is the client's
            value = _payload(event)
            for choice in (value.get("choices") or []) if isinstance(value, dict) else []:
                for call in ((choice.get("delta") or {}) if isinstance(choice, dict) else {}).get("tool_calls") or []:
                    if isinstance(call, dict) and call.get("id") and call["id"] not in self.tool_call_ids:
                        self.tool_call_ids.append(call["id"])
            if self.drop_role:
                self.drop_role = False
                if is_role_chunk(event):
                    continue
            out.append(event)
        return out

    def remainder(self):
        """Bytes that never formed a whole event (the stream ended mid-event). Empty after a pause."""
        if self.pause is not None:
            return b""
        rest, self.buf = self.buf, b""
        return rest

    def resumed(self):
        """A tracker for the continuation: same tool-call ids, the role chunk of the new reply is dropped."""
        return PauseTracker(self.tool_call_ids, drop_role=True)
