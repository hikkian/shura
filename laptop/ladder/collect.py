#!/usr/bin/env python3
"""Сводит результаты бенча в таблицу для отчёта руководству.
Запуск: python3 collect.py  -> печатает pass/total/timeouts по каждой руке (arm)."""
import json, glob, os, collections
R = os.path.expanduser("~/Desktop/AI/shura-report/results")
for f in sorted(glob.glob(R + "/arm_*.jsonl")):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if not rows: continue
    ok = sum(1 for r in rows if r.get("pass"))
    to = sum(1 for r in rows if r.get("secs", 0) >= 1500 and not r.get("pass"))
    ru = [r["ru_ratio"] for r in rows if r.get("ru_ratio") is not None]
    secs = sorted(r.get("secs", 0) for r in rows)
    print(f"{os.path.basename(f)[:-6]:34} {ok:>3}/{len(rows):<3} timeouts={to:<2} median_s={secs[len(secs)//2]:<5} ru_avg={sum(ru)/len(ru) if ru else 0:.2f}")
