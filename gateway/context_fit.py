"""Keep a chat request inside the server's context window before it is sent (CACHE3).

When a request is larger than the context the model was loaded with, llama-server answers HTTP 400, and ShuraCode then
starts a new session with another system message: the whole prompt is read again and the work of the session is lost.
This module shrinks the request instead, so the server never sees an overflow:

  * only the text of TOOL results is shortened, largest first (equal sizes: oldest first), and only when the request does not fit;
  * a tool result keeps its head and its tail and a marker says how much was removed; the tool_call_id stays;
  * the system message, the user messages, the assistant messages and the tool calls are never touched;
  * the same request always gets the same shortened text, so the server's cache of the shortened prefix stays valid
    for the following turns (only the first trimmed request pays a re-read).

The token count is an estimate (characters / CHARS_PER_TOKEN, deliberately low so that the margin is on the safe side).
Nothing here is sent anywhere; it is a pure function of the request bytes.
"""
import json

CHARS_PER_TOKEN = 3.0       # low on purpose: the estimate is above the real token count for prose and code
DEFAULT_RESERVE = 16384     # room for the answer when the request does not say max_tokens
KEEP_HEAD = 1500            # characters of a shortened tool result kept at its start
KEEP_TAIL = 500             # and at its end
MIN_TRIM = KEEP_HEAD + KEEP_TAIL + 200   # shorter tool results are not worth touching


def estimate_tokens(obj):
    return int(len(json.dumps(obj, ensure_ascii=False)) / CHARS_PER_TOKEN)


def _marker(removed):
    return f"\n[... {removed} characters of this tool result were removed by the gateway to fit the context window ...]\n"


def _shorten(text):
    removed = len(text) - KEEP_HEAD - KEEP_TAIL
    return text[:KEEP_HEAD] + _marker(removed) + text[-KEEP_TAIL:]


def _tool_text_slots(msg):
    """(container, key) pairs holding the text of a tool result: a string content or the text parts of a list content."""
    content = msg.get("content")
    if isinstance(content, str):
        return [(msg, "content")]
    if isinstance(content, list):
        return [(part, "text") for part in content if isinstance(part, dict) and part.get("type") == "text"
                and isinstance(part.get("text"), str)]
    return []


def fit_body(raw, context_size, max_tokens=None):
    """(new_raw, info). `raw` is the request body (bytes). `info` is None when nothing was done, else a dict.

    Returns the input unchanged when the body is not a chat request, has no context size, already fits, or cannot be made
    to fit by shortening tool results (the server then answers as before)."""
    if not raw or not context_size:
        return raw, None
    try:
        obj = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return raw, None
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
        return raw, None
    reserve = max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else DEFAULT_RESERVE
    budget = int(context_size) - reserve
    before = estimate_tokens(obj)
    if before <= budget:
        return raw, None
    messages = obj["messages"]
    slots = []
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            for container, key in _tool_text_slots(msg):
                text = container.get(key)
                if len(text) >= MIN_TRIM:
                    slots.append((len(text), i, container, key))
    # largest first, and among equal sizes the oldest first: the same request always trims the same way
    slots.sort(key=lambda s: (-s[0], s[1]))
    trimmed = 0
    for length, _, container, key in slots:
        if estimate_tokens(obj) <= budget:
            break
        container[key] = _shorten(container[key])
        trimmed += 1
    after = estimate_tokens(obj)
    if after > budget:
        return raw, {"fits": False, "trimmed": 0, "before": before, "after": before, "budget": budget}
    out = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return out, {"fits": True, "trimmed": trimmed, "before": before, "after": after, "budget": budget}
