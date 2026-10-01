"""The whole experiment, unattended: rounds of tasks, every arm on the same tasks in every round, an interim look after each round.

    python3 bench/agentic-eval/run_experiment.py arms.json --out bench/results/agentic-eval-YYYYMMDD [--round-size 12] [--reps 1]
                                                  [--parallel 4] [--max-hours 14] [--seed 20261001]

arms.json: {"server": {"exe": "...", "args": ["-ngl", "99", ...]},
            "arms": [{"name": "tiel", "model": "/path/a.gguf", "extra": {}},
                      {"name": "tiel-no-terse", "model": "/path/a.gguf", "extra": {"chat_template_kwargs": {"terse": false}}}, ...]}

Design points (all fixed in PROTOCOL.md before the first run):
  * tasks are shuffled once with the seed and cut into rounds; in each round EVERY arm does the same tasks, so the data is paired
    and the arms see the same slow or fast hours of the night (a model swap between arms costs about a minute);
  * `parallel` agents run at the same time against one llama-server (slots), each through its own measuring proxy;
  * after every round an interim look: the experiment stops early only if one arm is better than every other with Holm-adjusted
    p < 0.001 (Haybittle-Peto: the final 0.05 test is then not affected), or when the time budget is spent;
  * results are appended to results.jsonl after every single run, so a crash loses nothing and `--resume` continues.
"""
import argparse
import json
import random
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import analyze
import harness
import proxy as proxy_mod

INTERIM_ALPHA = 0.001


def load_done(path):
    done = []
    if Path(path).exists():
        done = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return done


def make_rounds(tasks, size, seed):
    ids = sorted(t["id"] for t in tasks)
    random.Random(seed).shuffle(ids)
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def expendable():
    """preexec_fn: if the machine runs out of memory the kernel kills OUR processes first, not the user's browser or editor."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass


def start_server(cfg, model, port, parallel, log_path):
    """llama-server for one arm. Returns the Popen; the caller stops it."""
    argv = [*cfg.get("prefix", []), cfg["exe"], "-m", model, "--host", "127.0.0.1", "--port", str(port), "-np", str(parallel), "-c",
            str(cfg.get("ctx_per_slot", 32768) * parallel), "--jinja", *cfg.get("args", [])]
    log = open(log_path, "wb")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                            preexec_fn=expendable)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 600:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited while loading {model}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except OSError:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("llama-server did not become healthy in 600 s")


def stop_server(proc):
    import os
    import signal
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def run_arm_round(arm, task_ids, tasks_by_id, reps, parallel, server_cfg, out_dir, base_port, seed, runner=None,
                  server_starter=start_server, server_stopper=stop_server, stop_flag=None):
    """One arm on one round of tasks. Returns the list of result records."""
    server = server_starter(server_cfg, arm["model"], base_port, parallel, Path(out_dir) / f"server-{arm['name']}.log")
    proxies, results, lock = [], [], threading.Lock()
    work = [(tid, rep) for tid in task_ids for rep in range(reps)]
    random.Random(seed).shuffle(work)
    try:
        slots = []
        for i in range(parallel):
            log = Path(out_dir) / f"proxy-{arm['name']}-{i}.jsonl"
            srv = proxy_mod.serve(0, f"http://127.0.0.1:{base_port}", str(log), arm.get("extra") or {})
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            proxies.append(srv)
            slots.append((srv.server_port, log))
        free = list(range(parallel))

        def one(item):
            tid, rep = item
            if stop_flag and stop_flag():                       # the time budget or the watchdog: leave the rest undone
                return None
            with lock:
                i = free.pop()
            try:
                port, log = slots[i]
                kw = {"runner": runner} if runner else {}
                rec = harness.run_one(tasks_by_id[tid], arm["name"], rep, port, log, Path(out_dir) / "agent-logs", **kw)
            finally:
                with lock:
                    free.append(i)
            with lock:
                results.append(rec)
                with open(Path(out_dir) / "results.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
            return rec
        with ThreadPoolExecutor(parallel) as pool:
            list(pool.map(one, work))
    finally:
        for p in proxies:
            p.shutdown()
            p.server_close()
        server_stopper(server)
    return results


def interim(runs, arms):
    """After a round: stop early only for a decisive winner. Returns (stop, text)."""
    if len({r["task"] for r in runs}) < 20:
        return False, "fewer than 20 tasks done: no interim look yet"
    res = analyze.compare(runs, flips=20000, boot=2000)
    ver = analyze.verdict(res)
    if ver["kind"] != "winner":
        return False, "no decisive winner yet"
    worst_p = max(r["p_holm"] for (a, b), r in res["pairs"].items() if ver["arm"] in (a, b))
    return worst_p < INTERIM_ALPHA, f"leader {ver['arm']}: largest Holm-adjusted p against the others = {worst_p:.5f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("arms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--round-size", type=int, default=12)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--max-hours", type=float, default=14.0)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.arms).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = harness.all_tasks()
    by_id = {t["id"]: t for t in tasks}
    done = load_done(out / "results.jsonl") if a.resume else []
    deadline = time.monotonic() + a.max_hours * 3600
    rounds = make_rounds(tasks, a.round_size, a.seed)
    for n, ids in enumerate(rounds, 1):
        for arm in cfg["arms"]:
            todo = [t for t in ids if not any(r["arm"] == arm["name"] and r["task"] == t for r in done)]
            if not todo:
                continue
            if time.monotonic() > deadline:
                print("time budget spent: stopping")
                return finish(out)
            print(f"round {n}/{len(rounds)} arm {arm['name']}: {len(todo)} tasks", flush=True)
            done += run_arm_round(arm, todo, by_id, a.reps, a.parallel, cfg["server"], out, a.port, a.seed + n)
        complete = [r for r in done if sum(1 for q in done if q["task"] == r["task"] and q["rep"] == r["rep"]) == len(cfg["arms"])]
        stop, text = interim(complete, [x["name"] for x in cfg["arms"]]) if complete else (False, "nothing complete")
        print(f"after round {n}: {text}", flush=True)
        if stop:
            print("stopping early: a decisive winner")
            break
    return finish(out)


def finish(out):
    runs = load_done(Path(out) / "results.jsonl")
    if not runs:
        print("no results")
        return 1
    by_cell = {}
    for r in runs:
        by_cell.setdefault((r["task"], r["rep"]), set()).add(r["arm"])
    arms = {r["arm"] for r in runs}
    complete = [r for r in runs if by_cell[(r["task"], r["rep"])] == arms]
    text = analyze.report(complete)[0]
    per_hour = {}
    for a in sorted(arms):
        rs = [r for r in complete if r["arm"] == a]
        hours = sum(r["seconds"] for r in rs) / 3600
        per_hour[a] = sum(r["resolved"] for r in rs) / hours if hours else 0.0
    text += "\n\nSolved tasks per hour of agent time (secondary, measured in the same parallel conditions for every arm):\n"
    text += "\n".join(f"  {a:28} {v:6.1f} per hour" for a, v in per_hour.items())
    (Path(out) / "REPORT.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
