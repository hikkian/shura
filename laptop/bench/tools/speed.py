#!/usr/bin/env python3
"""speed.py <model_id> [reps=3] : decode tok/s through the gateway: short prompt and ~30K-token context, thinking off."""
import json, sys, urllib.request
MID = sys.argv[1]; REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
def call(msgs):
    body = {"model": MID, "messages": msgs, "max_tokens": 700, "temperature": 0.4, "reasoning_effort": "none"}
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"}), timeout=1800)
    t = json.load(r).get("timings", {}); return t.get("predicted_per_second", 0), t.get("prompt_n", 0), t.get("prompt_per_second", 0)
task = "Напиши на Python класс LRU-кеша с TTL, с докстрингами и примером использования."
big = open("/home/hikkian/.local/share/shura/shura.cpp/tools/server/server-context.cpp", errors="ignore").read()[:110000]
for name, msgs in (("short", [{"role": "user", "content": task}]),
                   ("ctx30k", [{"role": "user", "content": "Вот код:\n" + big + "\n\n" + task}])):
    res = [call(msgs) for _ in range(REPS)]
    print(MID, name, "tg tok/s", [round(x[0], 1) for x in res], "prompt_n", res[0][1], "pp tok/s", round(res[0][2]), flush=True)
