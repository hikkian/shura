#!/usr/bin/env python3
"""vision.py <label> [reps=3] : transcribe the t16 mockup; score exact strings. Env PORT (default 8091)."""
import base64, json, os, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
LABEL = sys.argv[1]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
PORT = int(os.environ.get("PORT", "8091"))
IMG = os.environ.get("VIS_IMG") or "/home/hikkian/Desktop/AI/shuracode-prompt/eval/tasks/t16_mockup/attach.png"
TRUTH = ["Shura Cloud", "Тарифы", "Документация", "Войти", "Выберите свой тариф",
         "Платите только за то, что используете. Отмена в любой момент.", "ПОПУЛЯРНЫЙ", "Старт", "Про", "Бизнес",
         "0 ₸", "4 900 ₸", "14 900 ₸", "/мес", "5 ГБ хранилища", "1 проект", "Поддержка по почте", "100 ГБ хранилища",
         "10 проектов", "Приоритетная поддержка", "Автобэкапы", "1 ТБ хранилища", "Безлимит проектов", "Поддержка 24/7",
         "Начать", "© 2024 Shura Cloud"]
b64 = base64.b64encode(open(IMG, "rb").read()).decode()
res = []
for rep in range(REPS):
    body = {"model": os.environ.get("VIS_MODEL") or None, "messages": ([{"role": "system", "content": open(os.environ["VIS_SYSFILE"]).read()}] if os.environ.get("VIS_SYSFILE") else []) + [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        {"type": "text", "text": os.environ.get("VIS_PREFIX","") + "Перепиши дословно весь текст с этой картинки, сверху вниз, каждую надпись с новой строки. Символы валют и значки переписывай точно такими, как на картинке. Ничего не добавляй."}]}],
        **({"reasoning_effort": os.environ["VIS_EFFORT"]} if os.environ.get("VIS_EFFORT") else {}), "max_tokens": int(os.environ.get("VIS_MAXTOK", "2000")), "temperature": 0.4, "top_k": 20, "top_p": 0.95}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    text = d["choices"][0]["message"].get("content") or ""
    norm = text.replace(" ", " ").replace(" ", " ")
    missing = [t for t in TRUTH if t not in norm]
    res.append({"rep": rep, "text": text, "missing": missing, "prompt_ms": d.get("timings", {}).get("prompt_ms")})
    print(f"{rep+1}/{REPS}: ok {len(TRUTH)-len(missing)}/{len(TRUTH)} missing {missing}", flush=True)
s = {"label": LABEL, "ok_avg": round(sum(len(TRUTH) - len(r["missing"]) for r in res) / len(res), 1), "of": len(TRUTH),
     "tenge_ok": sum("₸" in r["text"] for r in res), "backup_ok": sum("Автобэкапы" in r["text"] for r in res), "reps": len(res)}
(HERE / f"vision-{LABEL}.json").write_text(json.dumps({"summary": s, "runs": res}, ensure_ascii=False, indent=1))
print("ИТОГ", json.dumps(s, ensure_ascii=False))
