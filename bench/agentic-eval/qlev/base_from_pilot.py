"""QLEVP1: the QLEV1 base (the original harness of Tiel on the pre-registered dev tasks) taken from the pilot's runs, in the format
qlev.summary reads: one JSON line per run with the harness's own field names and arm "base". No new runs are made here.

    python3 -B -m qlev.base_from_pilot --fingerprint aafd0d256fcc --out reports/qlev1/base/results.jsonl

Checks (printed and written next to the file as coverage.json):
  - the runs are the ones of one harness fingerprint (the original one), and the same run is not counted twice across the two states;
  - coverage against the 48 pre-registered dev tasks (rsi/dev_subset.json) x 2 repeats: what exists and what is missing;
  - no run against a dead server: a run whose interval (saved time minus its seconds, to the saved time) contains a moment when the
    server was killed (the kill times from reports/rsil1/RESTARTS.md) is dropped and reported; a run with no model request is dropped too.
Standard library only."""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEV_SUBSET = HERE.parent / "rsi" / "dev_subset.json"
STATES = [
    "/home/alisher/Desktop/AI/reports/rsil1/runs/rsil1-tiel/state.json",
    "/home/alisher/Desktop/AI/reports/rsil1/runs/rsil1-tiel-b/state.json",
]
OUT = Path("/home/alisher/Desktop/AI/reports/qlev1/base/results.jsonl")
# the moments the server was killed or stopped, from reports/rsil1/RESTARTS.md (local time, 2026-10-22); a kill at the next restart
# comes right after the last run that finished normally, so each is taken as the moment after that run's save
KILLS = [
    "2026-10-22T02:54:02",   # OOM of the kernel (cgroup 13 GB)
    "2026-10-22T03:20:00",   # safe-run watchdog ("~03:20" in the log)
    "2026-10-22T04:22:41",   # planned restart at a run boundary (step 12 -> 13)
    "2026-10-22T06:37:45",   # watchdog kill, ~18 s after run 40 had started
    "2026-10-22T08:21:17",   # watchdog kill, ~12 s after run 52 had started
    "2026-10-22T09:56:47",   # planned restart by the driver (memory rule), after the run saved at 09:56:46
    "2026-10-22T11:40:31",   # planned restart by the driver (memory rule), after the run saved at 11:40:30
    "2026-10-22T14:04:30",   # planned restart in the continuation run ("~14:04"), before py_flags-refactor t03
]
TIME_FMT = "%Y-%m-%dT%H:%M:%S"


def load_dev_subset_tasks():
    ideas = set(json.loads(DEV_SUBSET.read_text(encoding="utf-8"))["ideas"])
    return ideas


def pilot_runs(state_path, fingerprint):
    """The dev runs of one harness fingerprint, as {(task, rep): record} with the save time and the run's start."""
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    runs = {}
    for key, step in state.get("steps", {}).items():
        if not key.startswith(f"dev.{fingerprint}."):
            continue
        value = step.get("value")
        if not isinstance(value, dict):
            continue
        saved = datetime.strptime(step["at"], TIME_FMT)
        seconds = float(value.get("seconds") or 0.0)
        runs[(str(value["task"]), int(value["rep"]))] = {"record": value, "saved": saved, "started": saved - timedelta(seconds=seconds),
                                                         "key": key, "state": Path(state_path).parent.name}
    return runs


def killed_during(run):
    """The kill moments that fall strictly inside the run (started, saved): the run went on against a server that was gone."""
    kills = [datetime.strptime(k, TIME_FMT) for k in KILLS]
    return [k.strftime(TIME_FMT) for k in kills if run["started"] < k < run["saved"]]


def to_qlev_row(record):
    """The harness's field names, with the arm of the QLEV1 base."""
    row = dict(record)
    row["arm"] = "base"
    row["rep"] = int(record["rep"])
    row["reasoning_tokens"] = record.get("reasoning_tokens")
    return row


def build(fingerprint, states):
    merged, conflicts, dropped, sources = {}, [], [], {}
    for path in states:
        for key, run in pilot_runs(path, fingerprint).items():
            sources.setdefault(run["state"], 0)
            sources[run["state"]] += 1
            if run["record"].get("requests", 0) == 0:
                dropped.append({"task": key[0], "rep": key[1], "state": run["state"], "reason": "no model request"})
                continue
            hit = killed_during(run)
            if hit:
                dropped.append({"task": key[0], "rep": key[1], "state": run["state"], "reason": f"server killed during the run: {hit}"})
                continue
            if key in merged:
                if merged[key]["record"] != run["record"]:
                    conflicts.append({"task": key[0], "rep": key[1]})
                continue
            merged[key] = run
    return merged, conflicts, dropped, sources


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qlev.base_from_pilot")
    ap.add_argument("--fingerprint", default="aafd0d256fcc", help="the original harness's fingerprint in the pilot's state")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--state", action="append", help="a pilot state.json (repeatable); default: the two RSIL1 runs")
    args = ap.parse_args(argv)
    states = args.state or STATES
    merged, conflicts, dropped, sources = build(args.fingerprint, states)
    rows = [to_qlev_row(merged[k]["record"]) for k in sorted(merged, key=lambda k: (k[0], k[1]))]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ideas = load_dev_subset_tasks()
    tasks_all = sorted({k[0] for k in merged})
    have = {(k[0], k[1]) for k in merged}
    tasks_expected = sorted({t for s in states for t in json.loads(Path(s).read_text(encoding="utf-8"))["meta"]["dev"]})
    report = {"fingerprint": args.fingerprint, "states": states, "runs_per_state": sources, "records_written": len(rows),
              "tasks_with_records": len(tasks_all), "pairs_written": len(have), "expected_tasks": len(tasks_expected),
              "ideas": sorted(ideas),
              "missing_pairs": sorted([f"{t} rep {r}" for t in tasks_expected for r in (1, 2) if (t, r) not in have]),
              "conflicts_between_states": conflicts, "dropped": dropped, "out": str(out)}
    (out.parent / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("records_written", "tasks_with_records", "pairs_written", "expected_tasks")}, ensure_ascii=False))
    print("missing:", report["missing_pairs"])
    print("dropped:", len(dropped), "conflicts:", len(conflicts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
