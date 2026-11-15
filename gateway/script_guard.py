"""Script guard: keep CJK and other unexpected scripts out of Russian / English / Kazakh answers (ORNFIX0).

The model sometimes drops a Chinese word into a Russian sentence ("...рост覆盖率, сокращение..."). The guard adds a strong negative
logit bias to every vocabulary token written in a foreign script (CJK ideographs, kana, hangul, Arabic, Hebrew, Thai, Indic, ...), but only when
nothing in the conversation uses such a script: if the system prompt, a user message, a tool result or an earlier assistant message already contains one
(the user writes Chinese, the file being edited has Unicode strings), the request is left untouched. Cyrillic, Latin, Greek, digits, punctuation, emoji,
box drawing and arrows are never biased, so Kazakh and code are unaffected. A request that carries its own "logit_bias" keeps its entries.

Enabled per model: "scriptGuard": true (or {"bias": -10, "idsFile": "script_guard_ids_qwen36.json"}) in model-launch.json. Pure stdlib.
"""
import json
import os

DEFAULT_BIAS = -10.0
DEFAULT_IDS = "script_guard_ids_qwen36.json"
_CACHE = {}

# ranges of the scripts the guard keeps out (must match reports/ornfix/build_script_bias.py)
_RANGES = ((0x2E80, 0x2FDF), (0x3040, 0x30FF), (0x3100, 0x312F), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7AF), (0x1100, 0x11FF),
           (0xF900, 0xFAFF), (0x20000, 0x2FA1F), (0x0600, 0x06FF), (0x0750, 0x077F), (0x0590, 0x05FF), (0x0E00, 0x0EFF), (0x0900, 0x097F),
           (0x0980, 0x09FF), (0x0B80, 0x0BFF), (0x0C00, 0x0C7F), (0x1780, 0x17FF), (0x1000, 0x109F), (0x10A0, 0x10FF), (0x0530, 0x058F))


def has_foreign_script(text):
    for ch in text:
        c = ord(ch)
        if c < 0x0530:
            continue
        for a, b in _RANGES:
            if a <= c <= b:
                return True
    return False


def _config(settings):
    cfg = (settings or {}).get("scriptGuard")
    if cfg is True:
        return {"bias": DEFAULT_BIAS, "idsFile": DEFAULT_IDS}
    if isinstance(cfg, dict) and cfg.get("enabled", True):
        return {"bias": float(cfg.get("bias", DEFAULT_BIAS)), "idsFile": cfg.get("idsFile", DEFAULT_IDS)}
    return None


def _ids(name):
    path = name if os.path.isabs(name) else os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if path not in _CACHE:
        try:
            with open(path, encoding="utf-8") as f:
                _CACHE[path] = [int(i) for i in json.load(f)["ids"]]
        except (OSError, ValueError, KeyError, TypeError):
            _CACHE[path] = []
    return _CACHE[path]


def _message_text(m):
    c = m.get("content") if isinstance(m, dict) else None
    parts = []
    if isinstance(c, str):
        parts.append(c)
    elif isinstance(c, list):
        parts += [p.get("text", "") for p in c if isinstance(p, dict) and isinstance(p.get("text"), str)]
    if isinstance(m, dict):
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                parts.append(fn["arguments"])
        for key in ("reasoning_content",):
            if isinstance(m.get(key), str):
                parts.append(m[key])
    return "\n".join(parts)


def conversation_has_foreign_script(obj):
    for m in obj.get("messages") or []:
        if has_foreign_script(_message_text(m)):
            return True
    if isinstance(obj.get("prompt"), str) and has_foreign_script(obj["prompt"]):
        return True
    return False


def apply(body, path, settings):
    """Return the request body with the script bias added when the guard is on and the conversation has no foreign script."""
    cfg = _config(settings)
    if not cfg or not body or not path.endswith("/chat/completions"):
        return body
    try:
        obj = json.loads(body)
    except (ValueError, RecursionError):
        return body
    if not isinstance(obj, dict) or conversation_has_foreign_script(obj):
        return body
    ids = _ids(cfg["idsFile"])
    if not ids:
        return body
    existing = obj.get("logit_bias")
    if existing is None:
        pairs, seen = [], set()
    elif isinstance(existing, list):
        pairs, seen = list(existing), {p[0] for p in existing if isinstance(p, list) and p}
    elif isinstance(existing, dict):
        pairs, seen = [[int(k), v] for k, v in existing.items() if str(k).lstrip("-").isdigit()], {int(k) for k in existing if str(k).lstrip("-").isdigit()}
    else:
        return body
    bias = cfg["bias"]
    pairs += [[i, bias] for i in ids if i not in seen]
    obj["logit_bias"] = pairs
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()
