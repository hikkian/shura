"""Empty turn: the answer hit the output limit while the model was still thinking (finish_reason=length), with no text and no tool call.

Such a turn used to be the end of the task: the client got a closed answer with nothing to act on. With `emptyTurnContinue`
on, `proxy_core.forward_request` holds back the closing frames of that turn and asks the model to go on from its own
reasoning, at most `MAX_CONTINUATIONS` times per turn. The reasoning is not shortened; the client sees one continuous stream.

Off by default: the continuation is checked only with a fake upstream (tests/test_gateway_empty_turn.py). How the real model
renders a prefilled reasoning in the Sharp template is not verified here, so the switch stays off until a live check passes."""

import json

MAX_CONTINUATIONS = 2
STATS = {"continued": 0, "exhausted": 0, "gave_up": 0}      # counters for the gateway journal


class SSEFrames:
    """Cuts SSE bytes into frames. A frame is the bytes up to and including the blank line that ends it."""

    def __init__(self):
        self.tail = b""

    def feed(self, data):
        self.tail += data
        frames = []
        while b"\n\n" in self.tail:
            frame, self.tail = self.tail.split(b"\n\n", 1)
            frames.append(frame + b"\n\n")
        return frames


def frame_payload(frame):
    for line in frame.split(b"\n"):
        if line.startswith(b"data:"):
            return line[5:].strip()
    return None


class TurnState:
    """What one upstream stream said: reasoning, text, tool calls and the finish reason."""

    def __init__(self):
        self.reasoning = []
        self.text = False
        self.tool = False
        self.finish = None

    def feed(self, frame):
        payload = frame_payload(frame)
        if payload is None or payload == b"[DONE]":
            return
        try:
            value = json.loads(payload)
        except ValueError:
            return
        if not isinstance(value, dict):
            return
        for choice in value.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self.text = True
            if delta.get("tool_calls"):
                self.tool = True
            if delta.get("reasoning_content"):
                self.reasoning.append(delta["reasoning_content"])
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]

    def ends_turn(self, frame):
        """True for the frame that carries the finish reason, and for the [DONE] that follows it."""
        payload = frame_payload(frame)
        if payload == b"[DONE]":
            return True
        try:
            value = json.loads(payload) if payload else None
        except ValueError:
            return False
        if not isinstance(value, dict):
            return False
        return any(isinstance(c, dict) and c.get("finish_reason") for c in value.get("choices") or [])

    def empty_length(self):
        return self.finish == "length" and not self.text and not self.tool


def continuation_body(body, reasoning):
    """The same request with the reasoning so far as an unfinished assistant message. None when the body is not a chat request."""
    try:
        request = json.loads(body)
    except (TypeError, ValueError):
        return None
    messages = request.get("messages") if isinstance(request, dict) else None
    if not isinstance(messages, list) or not messages:
        return None
    request["messages"] = messages + [{"role": "assistant", "content": "", "reasoning_content": reasoning}]
    request["continue_final_message"] = True
    request["add_generation_prompt"] = False
    return json.dumps(request).encode()
