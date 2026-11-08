#!/usr/bin/env python3
"""probe.py <label> [reps=2] [extra-json]
Russian prose prompts through the gateway with token logprobs.
For every hunspell-flagged word: which tokens built it, was each the model's top-1, and its probability.
Answers whether typos come from sampling (fixable by settings) or from the model itself (quant/model).
Env: PORT (default 8080)."""
import json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
LABEL = sys.argv[1]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
EXTRA = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
PORT = int(os.environ.get("PORT", "8080"))
DICT = str(HERE.parent / "dict" / "ru_RU")
ALLOW = set(w.strip().lower() for w in open(HERE / "allow.txt", encoding="utf-8")) if (HERE / "allow.txt").exists() else set()

src = (HERE.parent / "bench.py").read_text()
ns = {}
exec(src[src.index("PARAGRAPH ="):src.index("def main")], ns)
PROMPTS = [(l, m[0]["content"]) for l, m in ns["RU"]] + [
    ("6 тарифы", "Сверстай HTML-разметку блока тарифов облачного хранилища Shura Cloud (без CSS). В шапке ссылки: «Тарифы», «Документация», «Войти». "
                 "Три тарифа: «Старт» 0 ₸/мес (5 ГБ хранилища, 1 проект, поддержка по почте), «Про» 4 900 ₸/мес (100 ГБ хранилища, 10 проектов, "
                 "приоритетная поддержка, автобэкапы), «Бизнес» 14 900 ₸/мес (1 ТБ хранилища, безлимит проектов, поддержка 24/7). Потом кратко по-русски объясни структуру."),
    ("7 инструкция", "Напиши для нового сотрудника подробную инструкцию на русском: как настроить резервное копирование (бэкапы) базы PostgreSQL на сервере, "
                     "проверить восстановление и настроить уведомления о сбоях. Пиши связным текстом с нумерованными шагами."),
]


def chat(content):
    body = {"model": "tiel-coder", "messages": [{"role": "user", "content": content}], "max_tokens": 2500,
            "temperature": 0.4, "top_k": 20, "top_p": 0.95, "logprobs": True, "top_logprobs": 5, "stream": False}
    body.update(EXTRA)
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    ch = d["choices"][0]
    return ch["message"].get("content") or "", (ch.get("logprobs") or {}).get("content") or [], d.get("timings", {})


def hunspell(words):
    out = subprocess.run(["hunspell", "-d", DICT, "-l"], input="\n".join(words), capture_output=True, text=True).stdout.split()
    return set(out)


WORD = re.compile(r"[А-Яа-яЁё]+")


def analyse(text, lps):
    # strip code from the check (typos inside code blocks are not prose)
    prose_spans = []
    pos = 0
    for m in re.finditer(r"```.*?```|`[^`\n]*`", text, flags=re.S):
        prose_spans.append((pos, m.start())); pos = m.end()
    prose_spans.append((pos, len(text)))
    words = [(m.group(), m.start(), m.end()) for a, b in prose_spans for m in WORD.finditer(text, a, b) if len(m.group()) > 1]
    bad = hunspell([w for w, _, _ in words])
    # token offsets: rebuild text from logprob tokens (content only)
    offs, cur = [], 0
    rebuilt = "".join(t.get("token", "") for t in lps)
    shift = rebuilt.find(text[:40]) if text[:40] and rebuilt.find(text[:40]) > 0 else 0
    for t in lps:
        s = cur - shift; cur += len(t.get("token", "")); offs.append((s, cur - shift, t))
    flagged = []
    for w, a, b in words:
        if w not in bad or w.lower() in ALLOW:
            continue
        toks = []
        for s, e, t in offs:
            if e > a and s < b:
                top = t.get("top_logprobs") or []
                top1 = top[0]["token"] if top else None
                p = 2.718281828 ** t.get("logprob", 0)
                alts = [(x["token"], round(2.718281828 ** x["logprob"], 3)) for x in top[:3]]
                toks.append({"tok": t.get("token"), "p": round(p, 3), "top1": t.get("token") == top1, "alts": alts})
        ctx = text[max(0, a - 30):b + 15].replace("\n", " ")
        flagged.append({"word": w, "ctx": ctx, "tokens": toks, "all_top1": all(x["top1"] for x in toks) if toks else None})
    return len(words), flagged


res = []
t0 = time.time()
for rep in range(REPS):
    for label, content in PROMPTS:
        text, lps, tm = chat(content)
        n, fl = analyse(text, lps)
        res.append({"rep": rep, "q": label, "ru_words": n, "flagged": fl, "text": text, "has_logprobs": bool(lps),
                    "decode": round(tm.get("predicted_per_second", 0), 1)})
        print(f"{rep+1}/{REPS} {label}: words {n}, flagged {len(fl)}: " +
              ", ".join(f"{f['word']}{'' if f['all_top1'] else '*'}" for f in fl), flush=True)
words = sum(r["ru_words"] for r in res)
fl = [f for r in res for f in r["flagged"]]
summary = {"label": LABEL, "extra": EXTRA, "answers": len(res), "ru_words": words, "flagged": len(fl),
           "flagged_per_1000": round(1000 * len(fl) / max(1, words), 2),
           "flagged_not_top1": sum(1 for f in fl if f["all_top1"] is False),
           "minutes": round((time.time() - t0) / 60, 1)}
(HERE / f"{LABEL}.json").write_text(json.dumps({"summary": summary, "answers": res}, ensure_ascii=False, indent=1))
print("ИТОГ", json.dumps(summary, ensure_ascii=False))
