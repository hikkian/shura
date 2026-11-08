#!/usr/bin/env python3
"""margin.py <label> : fast deterministic spelling metric (1 token per case).
Cases = real Ornith typos from probe answers (ruspell/*.json): the model's own answer text up to the typo, then the
first token where the correct word and the typo diverge. margin = logp(correct) - logp(typo) at that point
(-20 if the correct token is not in the top-50). Also: share of cases where the correct token is top-1.
Plus code controls (identifiers) so a fix that hurts code shows up.
Env PORT (default 8092): a llama-server (any build), /tokenize + /completion with token-id prompts."""
import json, os, sys, urllib.request
from pathlib import Path

PORT = int(os.environ.get("PORT", "8092"))
R = Path("/home/hikkian/Desktop/AI/bench/ruspell")
FIX = {"уложитьсь": "уложиться", "сдоях": "слоях", "иераргия": "иерархия", "принцимов": "принципов", "бадах": "базах",
       "монторинге": "мониторинге", "убитесь": "убедитесь", "выделённого": "выделенного", "возьнём": "возьмём",
       "переносён": "перенесён", "попавать": "попадать", "предугадить": "предугадать", "ориироваться": "ориентироваться",
       "произольный": "произвольный", "произольную": "произвольную", "благодар": "благодаря", "упомянём": "упомянем",
       "учитайте": "учитывайте", "талицы": "таблицы", "переупорачивать": "переупорядочивать", "бесполен": "бесполезен"}
FILES = ["ornith-gsq-t04.json", "ornith-ad-t04.json", "s8-base-t04.json", "s8-head8-t04.json", "tiel-t04.json",
         "tiel-greedy.json", "tiel-minp05.json"]
# code controls: (prefix, correct, wrong)
CODE = [("import numpy as np\nx = np.arr", "ay", "ey"), ("def __init__(self):\n        super().__in", "it__", "t__"),
        ("for i in range(len(items)):\n    print(ite", "ms[i]", "m[i]"), ("const result = await fetch(url);\nconst data = await result.js", "on()", "o()"),
        ("SELECT name FROM users WHERE id = ? ORDER BY crea", "ted_at", "te_at"), ("git commit -m \"fix\" && git pu", "sh", "ll"),
        ("raise ValueError(f\"invalid durat", "ion", "on"), ("with open(path, encoding=\"ut", "f-8", "f8")]

ns = {}
src = (R.parent / "bench.py").read_text()
exec(src[src.index("PARAGRAPH ="):src.index("def main")], ns)
QTEXT = {l: m[0]["content"] for l, m in ns["RU"]}
psrc = (R / "probe.py").read_text()
g = {"ns": ns}
exec(psrc[psrc.index("PROMPTS ="):psrc.index("def chat")], g)
PR = dict(g["PROMPTS"])


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def tok(text):
    return post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]


def chat_prefix(q, ans):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{ans}"


def case(prefix, good, bad):
    a, b = tok(prefix + good), tok(prefix + bad)
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)
    if i is None:  # one is a prefix of the other (благодар / благодаря): compare at the first extra token
        if len(a) == len(b):
            return None
        i = min(len(a), len(b))
        if i >= len(a) or i >= len(b):  # typo ends where the word continues: score 'continue' vs 'stop' is not possible with one id
            return None
    d = post("/completion", {"prompt": a[:i], "n_predict": 1, "n_probs": 50, "temperature": 0, "cache_prompt": True,
                             "post_sampling_probs": False})
    top = d["completion_probabilities"][0]["top_logprobs"]
    lp = {t["id"]: t["logprob"] for t in top}
    lg, lb = lp.get(a[i], -20.0), lp.get(b[i], -20.0)
    return lg - lb, top[0]["id"] == a[i]


def main():
    cases, seen = [], set()
    for f in FILES:
        if not (R / f).exists():
            continue
        for ans in json.load(open(R / f))["answers"]:
            q = PR.get(ans["q"]) or QTEXT.get(ans["q"])
            if not q:
                continue
            text = ans["text"]
            for w in ans["flagged"]:
                typo = w["word"]
                good = FIX.get(typo.lower())
                if not good:
                    continue
                pos = text.find(typo)
                if pos < 0:
                    continue
                key = (typo.lower(), text[max(0, pos - 40):pos])
                if key in seen:
                    continue
                seen.add(key)
                if typo[0].isupper():
                    good = good[0].upper() + good[1:]
                cases.append((typo, good, chat_prefix(q, text[:pos])))
    res = []
    for typo, good, pre in cases:
        c = case(pre, good, typo)
        if c is None:
            print(f"{typo:>18} -> {good:<18} skipped (prefix case)", flush=True); continue
        m, t1 = c
        res.append({"typo": typo, "good": good, "margin": round(m, 2), "top1": t1})
        print(f"{typo:>18} -> {good:<18} margin {m:+6.2f} {'top1' if t1 else ''}", flush=True)
    code = []
    for pre, good, bad in CODE:
        m, t1 = case(pre, good, bad)
        code.append({"ctx": pre[-25:], "margin": round(m, 2), "top1": t1})
    s = {"label": sys.argv[1], "n": len(res), "mean_margin": round(sum(r["margin"] for r in res) / len(res), 2),
         "top1_share": round(sum(r["top1"] for r in res) / len(res), 3),
         "code_mean_margin": round(sum(c["margin"] for c in code) / len(code), 2), "code_top1": sum(c["top1"] for c in code)}
    Path(__file__).with_name(f"margin-{sys.argv[1]}.json").write_text(json.dumps({"summary": s, "cases": res, "code": code}, ensure_ascii=False, indent=1))
    print("ИТОГ", json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
