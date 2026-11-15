#!/usr/bin/env python3
"""EXAM200e: runner for the micro stage (П1). One request per task to an OpenAI-compatible server, then the hidden
tests of the task run in the no-network bwrap sandbox (std_check.run_bwrap).

Per task: system + user message = the statement (statement.ru.md) and the start file; the model must answer with the
whole solution.py in one ```python block. The last python block is taken (the whole reply if there is none).
Score 1 if the hidden script exits 0, else 0. A reply that is cut off (finish_reason "length") is scored as it is.

Result: one JSON line per task in --out (JSON lines): id, score, completion_tokens, seconds, finish_reason, error.
--resume skips ids already in --out. Ports 8080 and 18090 (the live Shura stands) are refused in code.

Usage:
  run_micro.py --tasks DIR --base-url http://127.0.0.1:PORT/v1 --model NAME --out FILE.jsonl
               [--max-tokens 3000] [--limit N] [--resume] [--ids FILE] [--answers DIR]
  --answers DIR: do not call a server; take the answer from DIR/<id>.py (for self-tests: reference or empty).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import std_check  # noqa: E402

REFUSED_PORTS = {8080, 18090}
SYSTEM = ("Ты пишешь код на Python. Ответь одним блоком ```python с полным содержимым файла solution.py. "
          "Никаких пояснений вне блока не нужно.")
CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_code(reply: str) -> str:
    blocks = CODE_RE.findall(reply)
    return (blocks[-1] if blocks else reply).strip() + "\n"


def ask(base_url: str, model: str, prompt: str, max_tokens: int, timeout: int = 900) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "User-Agent": "exam-runner"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return {"text": choice["message"].get("content") or "", "finish_reason": choice.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"), "seconds": round(time.time() - t0, 2)}


def build_prompt(task: Path) -> str:
    statement = (task / "statement.ru.md").read_text(encoding="utf-8")
    start = (task / "start" / "solution.py").read_text(encoding="utf-8")
    return (f"{statement}\n\nИсходный файл solution.py:\n```python\n{start}```\n\n"
            "Верни полный solution.py одним блоком ```python.")


def score(task: Path, code: str) -> int:
    hidden = task / "hidden"
    work = Path(tempfile.mkdtemp(prefix=f"{task.name}-run-", dir=std_check.WORK_ROOT))
    try:
        (work / "solution.py").write_text(code, encoding="utf-8")
        for f in hidden.iterdir():
            shutil.copy(f, work / f.name)
        rc = std_check._run(work)
        return 1 if rc == 0 else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default="local")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ids", type=Path, default=None)
    ap.add_argument("--answers", type=Path, default=None)
    a = ap.parse_args(argv)

    if a.base_url:
        port = urllib.parse.urlparse(a.base_url).port
        if port in REFUSED_PORTS:
            print(f"refused: port {port} belongs to the live stands", file=sys.stderr)
            return 2
    if not a.answers and not a.base_url:
        print("need --base-url or --answers", file=sys.stderr)
        return 2

    done = set()
    if a.resume and a.out.exists():
        for line in a.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    # only tasks with a stdlib hidden script; the EXAM200b tasks (unittest in a repo snapshot) are run by build_micro.py
    ids = [p.name for p in sorted(a.tasks.iterdir())
           if (p / "statement.ru.md").exists() and (p / "hidden" / "test_hidden.py").exists()]
    if a.ids:
        wanted = [x.strip() for x in a.ids.read_text(encoding="utf-8").split() if x.strip()]
        ids = [x for x in ids if x in wanted]
    ids = [x for x in ids if x not in done]
    if a.limit:
        ids = ids[:a.limit]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a", encoding="utf-8") as fh:
        for tid in ids:
            task = a.tasks / tid
            rec = {"id": tid}
            try:
                if a.answers:
                    code = (a.answers / f"{tid}.py").read_text(encoding="utf-8")
                    rec.update({"completion_tokens": None, "seconds": 0.0, "finish_reason": "answers-dir"})
                else:
                    res = ask(a.base_url, a.model, build_prompt(task), a.max_tokens)
                    code = extract_code(res["text"]) if res["text"].strip() else ""
                    rec.update({k: res[k] for k in ("completion_tokens", "seconds", "finish_reason")})
                rec["error"] = None
                rec["score"] = score(task, code) if code.strip() else 0
            except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
                rec.update({"score": 0, "error": repr(exc)[:300], "finish_reason": None,
                            "completion_tokens": None, "seconds": None})
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(tid, rec["score"], rec.get("finish_reason"), rec.get("completion_tokens"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
