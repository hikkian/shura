import json, sys
sys.path.insert(0, '/home/hikkian/Desktop/AI/gateway-typo')
import lang_guard as lg, typo_fix as tf
def sse(content=None, finish=None, tool=False):
    d = {"choices": [{"index": 0, "delta": ({"content": content} if content else ({"tool_calls": [{}]} if tool else {})), "finish_reason": finish}]}
    return ("data: " + json.dumps(d, ensure_ascii=False) + "\n\n").encode()
def run(guard, chunks):
    out = b"".join(guard.feed(c) for c in chunks) + guard.finish_bytes()
    text = ""
    for ev in out.split(b"\n\n"):
        s = ev.decode().strip()
        if s.startswith("data:") and "[DONE]" not in s:
            text += (json.loads(s[5:])["choices"][0]["delta"].get("content") or "")
    return text, out
req = {"messages": [{"role": "user", "content": "Опиши адрес Алматы коротко"}]}
fixer = tf.make_fixer(req, lambda r: '{"Алаты": "Алматы"}')
chunks = [sse("Адрес: "), sse("Алаты, "), sse("пр. Абая 150. "), sse(None, "stop"), b"data: [DONE]\n\n"]
g = lg.StreamGuard("Russian", lambda t: None, typo=fixer, check_lang=False)
text, out = run(g, chunks); print("fixed:", text, "| typo_fixed", g.typo_fixed); assert "Алматы" in text and "Алаты" not in text
# no suspects -> passes unchanged
g = lg.StreamGuard("Russian", lambda t: None, typo=tf.make_fixer(req, lambda r: "{}"), check_lang=False)
text, _ = run(g, [sse("Всё хорошо, "), sse("файл создан."), sse(None, "stop"), b"data: [DONE]\n\n"]); print("clean:", text); assert text == "Всё хорошо, файл создан." and not g.typo_fixed
# tool calls release immediately, nothing held
g = lg.StreamGuard("Russian", lambda t: None, typo=fixer, check_lang=False)
o = g.feed(sse("Сейчас проверю Алаты ")) + g.feed(sse(None, None, tool=True)); assert b"\xd0\x90\xd0\xbb\xd0\xb0\xd1\x82\xd1\x8b" in o; print("tool-call step released")
# the model answers garbage -> unchanged
g = lg.StreamGuard("Russian", lambda t: None, typo=tf.make_fixer(req, lambda r: "не знаю"), check_lang=False)
text, _ = run(g, [sse("Алаты"), sse(None, "stop"), b"data: [DONE]\n\n"]); assert text == "Алаты"; print("garbage reply ignored")
# ask raises -> unchanged
def boom(r): raise RuntimeError("x")
g = lg.StreamGuard("Russian", lambda t: None, typo=tf.make_fixer(req, boom), check_lang=False)
text, _ = run(g, [sse("Алаты"), sse(None, "stop"), b"data: [DONE]\n\n"]); assert text == "Алаты"; print("failure ignored")
# language guard still works alone
g = lg.StreamGuard("Russian", lambda t: "Это ответ на русском языке, который был переписан моделью.", check_lang=True)
text, _ = run(g, [sse("This is a long English answer that should be rewritten into Russian by the guard now. "), sse(None, "stop"), b"data: [DONE]\n\n"]); assert text.startswith("Это ответ"); print("language rewrite intact")
print("ALL OK")
