"""FABLE1: a calibrated size estimate for the layout switch (short profile A, long profile B).

The legacy estimate (`chars of the messages / 2.5 + max_tokens`) was calibrated on Russian prose and counts the whole
output limit of the client (ShuraCode: 16 384) as the answer. For the real client the prompt is mostly English text, code
and the JSON of the tools (about 3.2 characters per token on a cold request, CTX1; the growth of a conversation is
estimated from the server's own count), the tools are not counted at all (about 40 000 characters, 10 000 tokens, with the live tool set), and a
typical answer is a few hundred tokens (p95 under 3 000). The two errors do not cancel: the switch to the slow long profile
came after about 6 500 characters of conversation, that is after two or three tool calls.

This module gives the gateway three things:
  * `count_chars(body)`: characters of the messages AND the tools, ASCII and other (Cyrillic) separately;
  * `base_estimate(...)`: tokens from those counts with one calibration per script;
  * `TokenFeedback`: what the server said about the last answered request of a session (usage: prompt and completion
    tokens, read from the relayed stream), so that the next, appended request is estimated as "known tokens + new text",
    which is exact for an append-only conversation.
Everything here is a pure function of bytes and numbers; nothing is sent anywhere.
"""
import json
import re
import threading

ASCII_CHARS_PER_TOKEN = 3.2    # English text, code and tool JSON on a cold request (CTX1: 3.8 underestimated it by 10.5–15.9 %;
                               # 3.2 fits the 14 cold samples in 2.96–3.36). Warm requests use TokenFeedback instead
OTHER_CHARS_PER_TOKEN = 2.5    # Cyrillic and other non-ASCII text (LAYOUT1 corpus: 2.48)
GROWTH_CHARS_PER_TOKEN = 3.5   # text appended after a known point: a little below 3.76 on purpose (tool outputs, reasoning)
TEMPLATE_OVERHEAD_TOKENS = 64  # the chat template's own tokens around one turn (role markers, tool wrappers)

_USAGE = re.compile(rb'"usage"\s*:\s*(\{[^{}]*\})')
_TIMINGS = re.compile(rb'"timings"\s*:\s*(\{[^{}]*\})')


def _walk(value, out):
    if isinstance(value, str):
        other = sum(1 for ch in value if ord(ch) > 127)
        out[0] += len(value) - other
        out[1] += other
    elif isinstance(value, list):
        for v in value:
            _walk(v, out)
    elif isinstance(value, dict):
        for v in value.values():
            _walk(v, out)


def count_chars(body):
    """(ascii_chars, other_chars) of the text of a chat body: the messages and the tools (their names, descriptions and
    schemas are read by the model too). A body that is not a chat object counts as empty."""
    try:
        obj = json.loads(body) if not isinstance(body, dict) else body
    except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
        return 0, 0
    if not isinstance(obj, dict):
        return 0, 0
    out = [0, 0]
    _walk(obj.get("messages", []), out)
    _walk(obj.get("tools", []), out)
    return out[0], out[1]


def base_estimate(ascii_chars, other_chars):
    """Tokens of a prompt from its characters, with no feedback from the server."""
    return ascii_chars / ASCII_CHARS_PER_TOKEN + other_chars / OTHER_CHARS_PER_TOKEN


def answer_reserve(max_tokens, configured):
    """Tokens kept for the answer: the client's output limit, capped by longContextAnswerReserveTokens when it is set."""
    if type(configured) is int and configured > 0:
        return min(max_tokens, configured) if max_tokens else configured
    return max_tokens or 0


def parse_usage(data):
    """The last usage (and timings) object in a relayed piece of the answer: {"prompt": N, "completion": M} or None.
    llama-server sends usage at the end of a stream and in a plain JSON answer; timings carry the same counts."""
    raw = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    found = None
    for match in _USAGE.finditer(raw):
        try:
            u = json.loads(match.group(1))
            found = {"prompt": int(u["prompt_tokens"]), "completion": int(u.get("completion_tokens", 0))}
        except (ValueError, KeyError, TypeError):
            continue
    if found is None:
        for match in _TIMINGS.finditer(raw):
            try:
                t = json.loads(match.group(1))
                found = {"prompt": int(t.get("prompt_n", 0)) + int(t.get("cache_n", 0)), "completion": int(t.get("predicted_n", 0))}
            except (ValueError, KeyError, TypeError):
                continue
    return found if found and found["prompt"] > 0 else None


class TokenFeedback:
    """Per session: the characters of the last answered request and the tokens the server counted for it (prompt + answer).
    The next request of the same session that only appends text is estimated from that point."""

    def __init__(self):
        self.lock = threading.Lock()
        self.known = {}
        self.last = {}

    def commit(self, key, chars, usage):
        if not key or not usage or chars <= 0:
            return
        with self.lock:
            self.known[key] = {"chars": chars, "tokens": usage["prompt"] + usage["completion"]}
            if len(self.known) > 64:                      # sessions come and go; keep the tracker small
                oldest = next(iter(self.known))
                self.known.pop(oldest, None)

    def estimate(self, key, ascii_chars, other_chars):
        """(tokens, how) for a request: from the session's known point when the request is at least as long, else from the
        characters alone."""
        chars = ascii_chars + other_chars
        with self.lock:
            point = self.known.get(key) if key else None
        if point and chars >= point["chars"]:
            tokens = point["tokens"] + (chars - point["chars"]) / GROWTH_CHARS_PER_TOKEN + TEMPLATE_OVERHEAD_TOKENS
            how = "feedback"
        else:
            tokens = base_estimate(ascii_chars, other_chars)
            how = "chars"
        with self.lock:
            self.last = {"tokens": int(tokens), "how": how, "chars": chars}
        return tokens, how

    def snapshot(self):
        with self.lock:
            return dict(self.last)
