"""Language guard: the final answer must be in the language the user writes in.

The chat template already tells the model (ornith-ideal4: a line in the user's own language in the system block and a
reminder before every generation). After a long English tool session a model still sometimes answers in English.
This guard is the backstop: for a chat request whose user writes Russian or Kazakh, it watches the streamed reply;
a final answer (no tool calls, finish_reason "stop") whose prose is mostly Latin is held back and the same model is
asked once to rewrite it in the user's language (same prompt prefix -> the KV cache is reused; thinking budget 0).
Replies that start in the right language stream through untouched (only the first ~60 letters are held to decide).

Enabled per model: "langGuard": true in model-launch.json. Pure stdlib; `feed()`/`finish()` work on raw SSE bytes.
"""
import json
import re

CY = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
KZ = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
LA = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
DECIDE_LETTERS = 60     # prose letters needed to decide that a streamed answer starts in the right language
MIN_LETTERS = 40        # shorter answers are never rewritten
LATIN_SHARE = 0.7       # rewrite when the prose is at least this Latin
REWRITE = {
    "Russian": ("Перепиши своё последнее сообщение пользователю целиком на русском языке, не меняя смысла, структуры "
                "и форматирования. Код, команды, пути, имена файлов и идентификаторы оставь как есть; устоявшиеся технические "
                "термины, названия и значения параметров (API, commit, endpoint, high, xhigh) не переводи искусственно. "
                "Выведи только переписанное сообщение, без вступления."),
    "Kazakh": ("Соңғы хабарламаңды пайдаланушыға мағынасын, құрылымын және пішімін өзгертпей толығымен қазақ тілінде "
               "қайта жаз. Код, командалар, жолдар, файл атаулары мен идентификаторлар өзгеріссіз қалсын; қалыптасқан техникалық "
               "терминдерді, атауларды және параметр мәндерін (API, commit, endpoint, high, xhigh) жасанды түрде аударма. "
               "Тек қайта жазылған хабарламаны шығар, кіріспесіз."),
}


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def detect_language(messages):
    """Same rule as the template's language anchor: first prose line of the latest real user message, skipping
    identifier-like words; < 8 letters defers to the previous user message. Returns 'Russian', 'Kazakh', 'mixed' or ''."""
    for m in reversed(messages or []):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        raw = _text(m.get("content")).strip()
        if raw.startswith("<tool_response>") and raw.endswith("</tool_response>"):
            continue
        line = raw.split("```")[0].strip().split("\n")[0][:300]
        words = [w for w in line.split(" ") if not any(c in w for c in "_./(=`") and not (w == w.upper() and w != w.lower())]
        kept = " ".join(words)
        first = ""
        for w in words:
            c, l = any(ch in CY or ch in KZ for ch in w), any(ch in LA for ch in w)
            if c != l:
                first = "c" if c else "l"
            if c or l:
                break
        nk = sum(ch in KZ for ch in kept)
        nc = sum(ch in CY for ch in kept) + nk
        nl = sum(ch in LA for ch in kept)
        if nc + nl < 8:
            continue
        if (first == "c" and nc * 2 >= nc + nl) or nc * 10 >= (nc + nl) * 9:
            return "Kazakh" if nk else "Russian"
        return "mixed" if nc else ""
    return ""


def prose_letters(text):
    """(cyrillic, latin) letter counts outside code spans/blocks."""
    t = re.sub(r"```.*?(```|$)", " ", text, flags=re.S)
    t = re.sub(r"`[^`\n]*`", " ", t)
    t = re.sub(r"https?://\S+|\S+\.(py|js|ts|md|json|html|css|txt|log|sh|yaml|yml|toml)\b|\S*[/_]\S*", " ", t)
    return sum(ch in CY or ch in KZ for ch in t), sum(ch in LA for ch in t)


def wrong_language(text):
    c, l = prose_letters(text)
    return c + l >= MIN_LETTERS and l >= (c + l) * LATIN_SHARE


class StreamGuard:
    """Feed raw SSE bytes from llama-server; get the bytes to send to the client. `rewrite(text) -> str|None`."""

    def __init__(self, lang, rewrite):
        self.lang, self.rewrite = lang, rewrite
        self.buf = b""            # incomplete SSE line data
        self.held = []            # (event_bytes, parsed_or_None) held back while deciding
        self.text = ""
        self.tool_calls = False
        self.finish = None
        self.passing = False
        self.rewritten = False

    def _events(self, data):
        self.buf += data
        while b"\n\n" in self.buf:
            ev, self.buf = self.buf.split(b"\n\n", 1)
            yield ev + b"\n\n"

    def feed(self, data):
        if self.passing:
            return data
        out = []
        for ev in self._events(data):
            if self.passing:
                out.append(ev)
                continue
            obj = None
            s = ev.decode("utf-8", "replace").strip()
            if s.startswith("data:") and s[5:].strip() != "[DONE]":
                try:
                    obj = json.loads(s[5:].strip())
                except ValueError:
                    obj = None
            ch = ((obj or {}).get("choices") or [{}])[0] if obj else {}
            delta = ch.get("delta") or {}
            if delta.get("tool_calls"):
                self.tool_calls = True
            if ch.get("finish_reason"):
                self.finish = ch["finish_reason"]
            if delta.get("content"):
                self.text += delta["content"]
            if not self.held and not delta.get("content") and not self.tool_calls and s != "data: [DONE]":
                out.append(ev)        # thinking / role chunks before the answer: stream immediately
                continue
            self.held.append((ev, obj))
            if self.tool_calls:
                self._release(out)
                continue
            c, l = prose_letters(self.text)
            if c + l >= DECIDE_LETTERS and l < (c + l) * LATIN_SHARE:
                self._release(out)
        return b"".join(out)

    def _release(self, out):
        out.extend(ev for ev, _ in self.held)
        out.append(self.buf)
        self.held, self.buf, self.passing = [], b"", True

    def finish_bytes(self):
        """Call at the end of the upstream stream."""
        if self.passing:
            return self.buf
        events = self.held
        if (not self.tool_calls and self.finish == "stop" and wrong_language(self.text)):
            new = None
            try:
                new = self.rewrite(self.text)
            except Exception:  # noqa: BLE001 - the original answer is still better than no answer
                new = None
            if new and not wrong_language(new):
                self.rewritten = True
                out, done = [], False
                for ev, obj in events:
                    ch = ((obj or {}).get("choices") or [{}])[0] if obj else {}
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        if done:
                            delta["content"] = ""
                            if not ch.get("finish_reason"):
                                continue
                        else:
                            delta["content"], done = new, True
                        ev = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()
                    out.append(ev)
                return b"".join(out) + self.buf
        return b"".join(ev for ev, _ in events) + self.buf


def rewrite_request(req, answer, lang):
    """The same request plus the answer and a rewrite instruction; thinking budget 0 (prompt prefix unchanged -> cached)."""
    r = dict(req)
    r["messages"] = list(req.get("messages") or []) + [{"role": "assistant", "content": answer},
                                                        {"role": "user", "content": REWRITE[lang]}]
    r["stream"] = False
    r.pop("stream_options", None)
    r["tool_choice"] = "none"
    r["reasoning_budget_tokens"] = 0
    r["reasoning_budget_message"] = ""
    r["max_tokens"] = max(1024, min(16384, len(answer)))
    return r


def guard_for(req, path, model_cfg):
    """Return the target language if this request should be guarded, else ''."""
    if not model_cfg.get("langGuard") or not path.endswith("/chat/completions") or not isinstance(req, dict):
        return ""
    lang = detect_language(req.get("messages"))
    return lang if lang in REWRITE else ""
