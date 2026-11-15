"""QLEV0 non-test task subsets and selectors that wait for QLEV1 base logs."""
import argparse
import json
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = Path(__file__).resolve().parent / "rsit1_split_snapshot.json"
TASKS_V3 = EVAL_ROOT / "tasks_v3"
D_REQUIRED_FAMILIES = ("js_event_journal", "js_headers", "js_deps_plan")
P_MIN_TASKS = 15


def _split_map(splits_path=SPLITS_PATH):
    data = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    return data.get("families", {})


def _tasks_for_splits(allowed, eval_root=EVAL_ROOT, splits_path=SPLITS_PATH):
    root = Path(eval_root)
    family_splits = _split_map(splits_path)
    rows = []
    # Decide from the family map before opening task.json, so the reserved test partition is never read.
    for path in sorted((root / "tasks_v3").glob("*/*/task.json")):
        family = path.parent.parent.name
        split = family_splits.get(family)
        if split not in allowed:
            continue
        task = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"id": task["id"], "family": task["family"], "kind": task["kind"],
                     "lang": task.get("lang"), "split": split,
                     "long_horizon_steps": task.get("long_horizon_steps", 0)})
    return rows


def dev_code_tasks(eval_root=EVAL_ROOT, splits_path=SPLITS_PATH):
    rows = _tasks_for_splits({"dev"}, eval_root, splits_path)
    if len(rows) != 56:
        raise ValueError(f"expected the 56 RSIT1 dev v3 tasks; found {len(rows)}")
    return rows


def c_tasks(eval_root=EVAL_ROOT, splits_path=SPLITS_PATH):
    return [row for row in dev_code_tasks(eval_root, splits_path) if row["kind"] in ("implement", "refactor")]


def p_train_candidates(eval_root=EVAL_ROOT, splits_path=SPLITS_PATH):
    """Long-horizon v3 train tasks that are safe candidates if the dev base yields fewer than 15 multi-user runs."""
    rows = _tasks_for_splits({"train"}, eval_root, splits_path)
    return [row for row in rows if int(row.get("long_horizon_steps") or 0) >= 2]


def d_required_tasks(eval_root=EVAL_ROOT, splits_path=SPLITS_PATH):
    """QLEV1 explicitly adds these already-seen v2 training tasks to the context-diet subset."""
    rows = []
    root = Path(eval_root) / "tasks_v2"
    for path in sorted(root.glob("*/*/task.json")):
        family = path.parent.parent.name
        if family not in D_REQUIRED_FAMILIES:
            continue
        task = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"id": task["id"], "family": task["family"], "kind": task["kind"],
                     "lang": task.get("lang"), "split": "train-v2", "long_horizon_steps": task.get("long_horizon_steps", 0)})
    return rows


def _records(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _base_user_messages(results, agent_logs=None, base_arm="base"):
    counts = {}
    missing_logs = []
    for row in results:
        if row.get("arm") != base_arm:
            continue
        task_id = row.get("task")
        value = row.get("user_messages")
        if value is None and agent_logs:
            path = Path(agent_logs) / f"{task_id}__{base_arm}__{row.get('rep')}.agent.log"
            if path.is_file():
                value = 0
                for line in path.read_text(errors="replace").splitlines():
                    if not line.startswith("{"):
                        continue
                    try:
                        if json.loads(line).get("type") == "user":
                            value += 1
                    except ValueError:
                        continue
        if value is None:
            missing_logs.append(task_id)
            continue
        counts[task_id] = max(counts.get(task_id, 0), int(value))
    return counts, sorted(set(missing_logs))


def select_p(results_path, agent_logs=None, base_arm="base", eval_root=EVAL_ROOT, splits_path=SPLITS_PATH):
    results = _records(results_path)
    counts, missing = _base_user_messages(results, agent_logs, base_arm)
    dev = dev_code_tasks(eval_root, splits_path)
    train_candidates = p_train_candidates(eval_root, splits_path)
    selected_dev = [dict(row, base_user_messages=counts[row["id"]]) for row in dev if counts.get(row["id"], 0) >= 2]
    selected_train = [dict(row, base_user_messages=counts[row["id"]]) for row in train_candidates
                      if counts.get(row["id"], 0) >= 2]
    selected = list(selected_dev)
    if len(selected) < P_MIN_TASKS:
        for row in selected_train:
            if row["id"] not in {item["id"] for item in selected}:
                selected.append(row)
            if len(selected) >= P_MIN_TASKS:
                break
    unmeasured_train = [row for row in train_candidates if row["id"] not in counts]
    return {"selection": "P", "criterion": "at least two user messages in a base-arm serve log",
            "dev_selected": selected_dev, "train_selected": selected_train,
            "selected": selected, "minimum": P_MIN_TASKS, "underpowered": len(selected) < P_MIN_TASKS,
            "unmeasured_train_candidates": unmeasured_train, "base_records_missing_user_counts": missing}


def select_d(results_path, base_arm="base", threshold=20000, eval_root=EVAL_ROOT, splits_path=SPLITS_PATH):
    results = _records(results_path)
    peaks = {}
    missing = []
    for row in results:
        if row.get("arm") != base_arm:
            continue
        task_id = row.get("task")
        value = row.get("max_prompt_tokens")
        if value is None:
            missing.append(task_id)
            continue
        peaks[task_id] = max(peaks.get(task_id, 0), int(value))
    dev = dev_code_tasks(eval_root, splits_path)
    required = d_required_tasks(eval_root, splits_path)
    selected = []
    for row in dev + required:
        peak = peaks.get(row["id"])
        mandatory = row["family"] in D_REQUIRED_FAMILIES
        if mandatory or (peak is not None and peak > threshold):
            selected.append(dict(row, base_peak_prompt_tokens=peak,
                                 selection_reason="required QLEV1 context family" if mandatory else "base request peak > threshold"))
    return {"selection": "D", "criterion": f"base-arm max prompt_tokens > {threshold}; always include required QLEV1 families",
            "selected": selected, "mandatory_families": list(D_REQUIRED_FAMILIES),
            "base_records_missing_peak_prompt_tokens": sorted(set(missing)),
            "dev_tasks_without_base_records": sorted(row["id"] for row in dev if row["id"] not in peaks)}


def write_static(outdir, eval_root=EVAL_ROOT, splits_path=SPLITS_PATH):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    payloads = {
        "dev-code.json": {"source": "RSIT1 splits.json; tasks_v3; split=dev; test files were not read", "tasks": dev_code_tasks(eval_root, splits_path)},
        "C-implement-refactor.json": {"source": "dev-code subset: kind in implement/refactor", "tasks": c_tasks(eval_root, splits_path)},
        "P-train-candidates.json": {"source": "non-test train v3 tasks with long_horizon_steps >= 2; base logs must confirm >=2 user messages", "tasks": p_train_candidates(eval_root, splits_path)},
        "D-required.json": {"source": "QLEV1 prompt mandatory context families", "tasks": d_required_tasks(eval_root, splits_path)},
    }
    for name, payload in payloads.items():
        (out / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {name: len(payload["tasks"]) for name, payload in payloads.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    static = sub.add_parser("static", help="write dev/C/train-candidate/D-required lists without touching test data")
    static.add_argument("--outdir", required=True)
    p = sub.add_parser("P", help="select multi-user tasks from base serve logs; add only train tasks confirmed by base logs")
    p.add_argument("results")
    p.add_argument("--agent-logs")
    p.add_argument("--base-arm", default="base")
    p.add_argument("--out", required=True)
    d = sub.add_parser("D", help="select context tasks from per-run base max_prompt_tokens")
    d.add_argument("results")
    d.add_argument("--base-arm", default="base")
    d.add_argument("--threshold", type=int, default=20000)
    d.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "static":
        print(json.dumps(write_static(args.outdir), indent=2))
        return 0
    if args.command == "P":
        payload = select_p(args.results, args.agent_logs, args.base_arm)
    else:
        payload = select_d(args.results, args.base_arm, args.threshold)
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selection": payload["selection"], "selected": len(payload["selected"]), "output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
