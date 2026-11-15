"""QHARN1: how many 'rule prevents' denials in old agentic-eval runs were caused by the harness rather than by the model, and which
of them the new harness (rules inside the project, no config path in the prompt, ./.tmp for scratch) removes.

    python3 diagnose_qharn1.py <raw run folder> [<raw run folder> ...]     (each folder: results.jsonl + agent/*.agent.log)

Read-only. Prints one table per run and the totals; --json writes the same numbers as JSON to stdout instead.
Categories of a denied tool call:
  cfg_read     a read of the experiment's config folder (cfg/AGENTS.md, cfg/shuracode.json): reached by the absolute path in the prompt
  tmp          a write or a shell command that used /tmp: outside the project
  outside      a path or shell command in the scratch folder but not in work/: the model left the project folder
  other        anything else
Addressed by QHARN1: cfg_read and tmp (the rules now live in the project and point to ./.tmp). outside is only partly addressed
(the prompt says relative paths); it is counted separately and not claimed as fixed.
"""
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import ADHOC_RE, VERIFY_RE  # the same patterns as the new harness

RULE = "rule which prevents"


def classify(path, command):
    text = path or command or ""
    if "/cfg/" in text:
        return "cfg_read"
    if path.startswith("/tmp") or re.search(r"(^|[\s;&(|])/tmp(/|\s|$)", command or ""):
        return "tmp"
    if "/dev/shm/agentic-eval/" in text:
        return "other" if "/work" in text else "outside"
    if path.startswith("/"):
        return "outside"
    return "other"


def run_stats(folder):
    results = os.path.join(folder, "results.jsonl")
    recs = []
    if os.path.exists(results):
        recs = [json.loads(line) for line in open(results) if line.strip()]
    per_task = {}
    cats = collections.Counter()
    checks = {"runs": 0, "tests_run_project": 0, "adhoc_checks": 0}
    for path in glob.glob(os.path.join(folder, "agent", "*.agent.log")):
        name = os.path.basename(path).split("__")[0]
        blocked = collections.Counter()
        tests = adhoc = 0
        for line in open(path, errors="replace"):
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            p = ev.get("part") or {}
            st = p.get("state") or {}
            inp = st.get("input") or {}
            command = inp.get("command") or ""
            if p.get("tool") == "bash" and command:
                tests += bool(VERIFY_RE.search(command))
                adhoc += len(ADHOC_RE.findall(command))
            if RULE in (st.get("error") or ""):
                cat = classify(inp.get("filePath") or inp.get("path") or "", command)
                blocked[cat] += 1
                cats[cat] += 1
        checks["runs"] += 1
        checks["tests_run_project"] += bool(tests)
        checks["adhoc_checks"] += adhoc
        per_task[name] = {"blocked": dict(blocked), "tests": tests, "adhoc": adhoc}
    solved = {r["task"] + "__" + r["arm"] + "__" + str(r["rep"]): bool(r.get("resolved")) for r in recs}
    tasks_hit = sum(1 for v in per_task.values() if v["blocked"])
    addressed = sum(1 for v in per_task.values() if v["blocked"].get("cfg_read") or v["blocked"].get("tmp"))
    return {"folder": folder, "agent_logs": len(per_task), "results": len(recs), "resolved": sum(bool(r.get("resolved")) for r in recs),
            "blocked_calls": dict(cats), "blocked_total": sum(cats.values()), "tasks_with_a_denial": tasks_hit,
            "tasks_with_an_addressed_denial": addressed, "checks": checks}


def main(argv):
    as_json = "--json" in argv
    folders = [a for a in argv if not a.startswith("--")]
    out = [run_stats(f) for f in folders]
    if as_json:
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return 0
    for r in out:
        print(f"{r['folder']}: logs={r['agent_logs']} results={r['results']} resolved={r['resolved']}")
        print(f"  denied calls: {r['blocked_total']} {r['blocked_calls']}")
        print(f"  tasks with a denial: {r['tasks_with_a_denial']}; with an addressed one (cfg_read/tmp): {r['tasks_with_an_addressed_denial']}")
        print(f"  project test run in {r['checks']['tests_run_project']} of {r['checks']['runs']} runs; ad-hoc checks {r['checks']['adhoc_checks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
