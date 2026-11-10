#!/usr/bin/env python3
"""Vision shootout: same page image, same question, several vision projectors / models, N samples each. Score = exact BIN number + share of key strings found.
usage: shootout.py <model-id> <samples> -> prints one JSON line"""
import base64, json, os, sys, time, urllib.error, urllib.request, re
IMG = "/home/hikkian/Desktop/AI/shuracode-prompt/eval/tasks_heldout/h4_invoice_mockup/attach.png"
KEYS = ["123456789012", "117/2025", "355 500", "319 950", "35 550", "12.03.2025", "150/230", "Алия К.", "45 000", "120 000", "38 500", "7 500"]
model, n = sys.argv[1], int(sys.argv[2])
LAYOUT_Q = ("Опиши структуру этой страницы, не переписывая весь текст: какого цвета шапка, что в шапке слева и справа, "
            "есть ли значок статуса и какого он цвета, из каких колонок состоит таблица, какая строка таблицы выделена и чем, "
            "какие строки идут под таблицей и что в самом низу страницы.")
LAYOUT_FACTS = {"header_blue": ["син"], "badge": ["оплачен частично", "частично"], "highlight_row": ["настройка сети", "выделен", "жёлт", "желт", "оранж"],
                "columns": ["кол-во", "цена", "сумма"], "totals": ["итого", "скидк"], "footer": ["подвал", "внизу", "низу"]}
b64 = base64.b64encode(open(IMG, "rb").read()).decode()
def ask(q=None):
    body = {"model": model, "temperature": 0.3, "max_tokens": 1500, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": q or "Перепиши дословно весь текст с этой страницы, строка за строкой, включая мелкий текст внизу. Цифры и символы переписывай точно."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}]}
    req = urllib.request.Request(os.environ.get("SHOOT_URL", "http://127.0.0.1:8080/v1/chat/completions"), json.dumps(body).encode(), {"Content-Type": "application/json"})
    for attempt in range(45):   # the gateway answers 503 while it restarts or switches models: wait and retry (up to ~15 min)
        try:
            r = json.load(urllib.request.urlopen(req, timeout=900)); break
        except urllib.error.HTTPError as ex:
            if ex.code != 503 or attempt == 44: raise
            time.sleep(20)
    return r["choices"][0]["message"].get("content") or ""
res = []
for i in range(n):
    t = ask(); norm = re.sub(r"[  ]", " ", t)
    found = [k for k in KEYS if k in norm]
    bin_digits = re.findall(r"БИН\s*([0-9 ]+)", norm)
    lay = ask(LAYOUT_Q).lower()
    lay_hits = sum(1 for k, alts in LAYOUT_FACTS.items() if any(a in lay for a in alts))
    res.append({"layout": lay_hits, "layout_of": len(LAYOUT_FACTS), "bin_ok": "123456789012" in norm.replace(" ", ""), "found": len(found), "of": len(KEYS), "bin_read": bin_digits[:1]})
print(json.dumps({"model": model, "n": n, "bin_exact": sum(r["bin_ok"] for r in res), "mean_found": sum(r["found"] for r in res) / (n * len(KEYS)), "mean_layout": sum(r["layout"] for r in res) / (n * len(LAYOUT_FACTS)), "samples": res}, ensure_ascii=False))
