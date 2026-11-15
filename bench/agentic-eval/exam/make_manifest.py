#!/usr/bin/env python3
"""EXAM200: the manifest of exam version v3r1 — which task belongs to which part, with slice labels and units of independence.

Parts (lifelong/MASTER.md §5а):
  search   ideas that already took part in any run (variants are picked here);
  gate-1..3, gate-reserve   ideas that were never run, in waves of fresh tasks for one pre-registered candidate;
  sealed   the untouched test split; at most eight accesses a year.
The unit of independence is the IDEA (family name without the language prefix): js_X and py_X share one specification.
Never-run ideas are ordered by sha256(seed:idea) and dealt into waves, so the assignment is deterministic and never changes.

    python3 exam/make_manifest.py            # run from bench/agentic-eval; writes exam/manifest-v3r1.json
"""
import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = HERE / "tasks_v3r1"
SPLITS = json.loads((HERE.parent / "splits.json").read_text())
REPORTS = Path.home() / "Desktop/AI/reports"
SEED = 20261113
WAVE_IDEAS = 10          # ideas per gate wave
WAVES = 3                # the rest of the never-run search ideas is the reserve


def idea_of(family):
    return family.split("_", 1)[1]


def used_families():
    used = set()
    for path in glob.glob(str(REPORTS / "**" / "results*.jsonl"), recursive=True):
        if "/ll3b/" in path:
            continue
        try:
            with open(path, errors="replace") as fh:
                for line in fh:
                    m = re.search(r'"family": ?"([^"]+)"', line)
                    if m:
                        used.add(m.group(1))
        except OSError:
            pass
    return used


def sentinel_ids(min_runs=4):
    """Tasks the main model solved in every recorded run (at least `min_runs` runs): the watchdog pool that shows forgetting."""
    runs = defaultdict(list)
    for rel in ("qlev1/raw/main/results.jsonl", "sage1/raw/tiel-final/results.jsonl", "qlev1/base/results.jsonl"):
        path = REPORTS / rel
        if path.exists():
            for line in path.read_text(errors="replace").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("score") is not None:
                        runs[r["task"]].append(bool(r["resolved"]))
    return {t for t, v in runs.items() if len(v) >= min_runs and all(v)}


def main():
    sentinels = sentinel_ids()
    log = {row["id"]: row for row in json.loads((HERE / "build-v3r1.log.json").read_text())}
    families = sorted(p.name for p in TASKS.iterdir() if p.is_dir())
    ideas = defaultdict(list)
    for fam in families:
        ideas[idea_of(fam)].append(fam)
    used = used_families()
    split = {i: SPLITS["families"][fams[0]] for i, fams in ideas.items()}
    ran = {i: any(f in used for f in fams) for i, fams in ideas.items()}
    fresh = sorted((i for i in ideas if not ran[i] and split[i] in ("train", "dev")),
                   key=lambda i: hashlib.sha256(f"{SEED}:{i}".encode()).hexdigest())
    part = {}
    for i in ideas:
        if ran[i]:
            part[i] = "search"                      # includes three ideas of the old test split that were run in MODEL3R
        elif split[i] == "test":
            part[i] = "sealed"
    for n, i in enumerate(fresh):
        wave = n // WAVE_IDEAS
        part[i] = f"gate-{wave + 1}" if wave < WAVES else "gate-reserve"
    tasks = []
    for fam in families:
        for tj_path in sorted((TASKS / fam).glob("*/task.json")):
            tj = json.loads(tj_path.read_text())
            row = log[tj["id"]]
            tasks.append({"id": tj["id"], "family": fam, "idea": idea_of(fam), "unit": idea_of(fam), "kind": tj["kind"],
                          "part": part[idea_of(fam)],
                          "slices": {"statement_lang": tj.get("prompt_lang", "en"), "code_lang": tj.get("lang"), "kind": tj["kind"],
                                     "long_horizon": bool(tj.get("long_horizon_steps"))},
                          "fixes": row["fixes"], "hidden_sha256": row["hidden_sha256"], "prompt_sha256": row["prompt_sha256"],
                          "timeout_s": tj.get("timeout_s"),
                          "limits": {"completion_tokens": 30000, "requests": 40}, "sentinel": tj["id"] in sentinels})
    summary = {}
    for p in sorted(set(part.values())):
        rows = [t for t in tasks if t["part"] == p]
        summary[p] = {"ideas": len({t["idea"] for t in rows}), "families": len({t["family"] for t in rows}), "tasks": len(rows),
                      "kinds": dict(Counter(t["kind"] for t in rows)), "sentinels": sum(1 for t in rows if t["sentinel"])}
    manifest = {"version": "v3r1", "date": "2026-11-13", "tasks_root": "exam/tasks_v3r1", "source": "tasks_v3 + exam/build_v3r1.py",
                "unit_of_independence": "idea (family without the language prefix)", "wave_seed": SEED,
                "rules": {"search": "variants are picked only here", "gate": "one pre-registered candidate per wave; a wave is used once per candidate",
                          "sealed": "at most eight accesses a year, each written to the decision journal",
                          "training": "no task of this manifest ever goes into training data"},
                "summary": summary, "tasks": tasks}
    (HERE / "manifest-v3r1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for p, s in summary.items():
        print(f"{p:13} ideas {s['ideas']:3} families {s['families']:3} tasks {s['tasks']:3} {s['kinds']}")


if __name__ == "__main__":
    main()
