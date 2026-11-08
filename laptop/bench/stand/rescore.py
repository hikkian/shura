#!/usr/bin/env python3
"""rescore.py <agent> : re-run score2.py on saved work dirs with the current hidden tests -> results/<agent>.final.jsonl"""
import json, os, subprocess, sys
A = sys.argv[1]
H = os.path.dirname(os.path.abspath(__file__))
E = "/home/hikkian/Desktop/AI/shuracode-prompt/eval"
WR = "/home/hikkian/Desktop/AI/shura-report/work-old"
out = []
for l in open(f"{H}/results/{A}.jsonl"):
    r = json.loads(l)
    if r["rep"] < 1:
        continue
    W = f"/home/hikkian/Desktop/AI/shura-report/work/{A}-{r['task']}-{r['rep']}"
    if not os.path.isdir(W): W = f"{WR}/{A}-{r['task']}-{r['rep']}"
    p = subprocess.run([sys.executable, f"{H}/score2.py", E, A, r["task"], str(r["rep"]), W, str(r["secs"]), str(r["rc"])],
                       capture_output=True, text=True, env={**os.environ, "SHOT_DIR": f"{H}/shots/{A}-{r['rep']}"})
    n = json.loads(p.stdout.strip().splitlines()[-1])
    if n["pass"] != r["pass"]:
        print(f"{r['task']} rep{r['rep']}: {r['pass']} -> {n['pass']}")
    out.append(n)
open(f"{H}/results/{A}.final.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
print(A, len(out), "pass", sum(x["pass"] for x in out))
