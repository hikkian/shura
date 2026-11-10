"""Typo guard for Russian answers (backstop for typos the quantised model makes in its own prose, e.g. «Алаты» for «Алматы», «монторинге»).

After the final answer is complete (the guard holds a Russian answer until it ends; tool-call steps are released at once), the words of the
prose (outside code, URLs, paths) are checked with hunspell (ru_RU). The words the dictionary does not know (typos, but also jargon and names)
go to the same model in ONE short extra request (same prompt prefix -> KV cache reused, thinking budget 0, ~60 tokens) which returns a JSON
object {"wrong": "right"} only for real typos. A correction is applied only when it passes the checks in `accept()` (known to the dictionary or
present in the conversation, close in edit distance); the answer is never rewritten as a whole, only those words are replaced.
Enabled per model: "typoGuard": true in model-launch.json (needs the `hunspell` binary and ru_RU in ~/.local/share/hunspell or a system path).
"""
import json
import re
import subprocess

CY = "А-Яа-яЁё"
WORD = re.compile(rf"[{CY}]{{4,}}(?:-[{CY}]+)*")
CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
NOISE = re.compile(r"https?://\S+|[\w./~-]*[/\\_][\w./~-]*|\S+@\S+")
MAX_SUSPECTS = 12
LOG = None   # set by the gateway: log(str)
INSTRUCTION = ("Орфографический словарь не знает эти слова из твоего последнего сообщения: {words}. Ответь одной строкой JSON-объекта "
               '{{"слово": "правильное написание"}} только для слов с настоящей опечаткой (пропущенная, лишняя или переставленная буква). '
               "Названия, имена, термины, жаргон и принятые в разработке написания не трогай и не включай. Если опечаток нет, ответь {{}}.")


def prose_segments(text):
    """[(is_code, chunk)]: fenced and inline code is never touched."""
    out, pos = [], 0
    for m in CODE.finditer(text):
        if m.start() > pos:
            out.append((False, text[pos:m.start()]))
        out.append((True, m.group(0)))
        pos = m.end()
    if pos < len(text):
        out.append((False, text[pos:]))
    return out


def candidate_words(text):
    words = []
    for is_code, chunk in prose_segments(text):
        if is_code:
            continue
        chunk = NOISE.sub(" ", chunk)
        words.extend(WORD.findall(chunk))
    seen, uniq = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def hunspell_unknown(words, dic="ru_RU"):
    if not words:
        return []
    r = subprocess.run(["hunspell", "-d", dic, "-l"], input="\n".join(words), capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:200])
    return [w for w in dict.fromkeys(r.stdout.split()) if w in set(words)]


def edit_distance(a, b):
    a, b = a.lower(), b.lower()
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def conversation_text(messages):
    parts = []
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return "\n".join(parts)


def accept(wrong, right, known_in_context, unknown_after, answer_words=()):
    """A correction is applied only if it is a plausible spelling fix: close to the wrong word and either known to the dictionary, present in
    the conversation, or (jargon the dictionary lacks) a one-letter fix that shares its first four letters with another word of the answer."""
    if not isinstance(right, str) or not right or right == wrong or not re.fullmatch(rf"[{CY}]+(?:-[{CY}]+)*", right):
        return False
    d = edit_distance(wrong, right)
    if d > 3 or abs(len(wrong) - len(right)) > 3:
        return False
    if right in known_in_context or right not in unknown_after:
        return True
    return d == 1 and any(w != wrong and w[:4].lower() == right[:4].lower() for w in answer_words)


def fix_request(req, answer, suspects):
    r = dict(req)
    r["messages"] = list(req.get("messages") or []) + [{"role": "assistant", "content": answer},
                                                        {"role": "user", "content": INSTRUCTION.format(words=", ".join(suspects))}]
    r["stream"] = False
    r.pop("stream_options", None)
    r["tool_choice"] = "none"
    r["reasoning_budget_tokens"] = 0
    r["reasoning_budget_message"] = ""
    r["temperature"] = 0
    r["max_tokens"] = 400
    return r


def parse_mapping(content):
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return {}
    return {k: v for k, v in obj.items() if isinstance(k, str)} if isinstance(obj, dict) else {}


def apply_mapping(text, mapping):
    out = []
    for is_code, chunk in prose_segments(text):
        if not is_code:
            for wrong, right in mapping.items():
                chunk = re.sub(rf"(?<![{CY}A-Za-z]){re.escape(wrong)}(?![{CY}A-Za-z])", right, chunk)
        out.append(chunk)
    return "".join(out)


def make_fixer(req, ask):
    """ask(request_dict) -> content string or None. Returns fix(text) -> text (never raises; on any problem returns the text unchanged)."""
    def fix(text):
        try:
            suspects = hunspell_unknown(candidate_words(text))[:MAX_SUSPECTS]
            if not suspects:
                return text
            content = ask(fix_request(req, text, suspects))
            mapping = parse_mapping(content)
            ctx = conversation_text(req.get("messages"))
            known_ctx = set(WORD.findall(ctx))
            unknown = set(suspects)
            answer_words = set(WORD.findall(text))
            good = {w: r for w, r in mapping.items() if w in unknown and accept(w, r, known_ctx, set(hunspell_unknown([r])), answer_words)}
            if LOG:
                LOG(f"typo guard: suspects={suspects} reply={content!r} applied={good}")
            return apply_mapping(text, good) if good else text
        except Exception:  # noqa: BLE001 - the original answer is better than none
            return text
    return fix
