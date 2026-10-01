"""The unattended night: wait for a free GPU, check the plumbing on a few real tasks, run every arm on every task, write the
verdict, delete the models that lost, put the machine back. Everything is logged; every decision is in MORNING.md.

    python3 bench/agentic-eval/overnight.py --out bench/results/agentic-eval-20261002 [--max-hours 9] [--reps 2]

Arms: tiel (as shipped), tiel-no-terse (the same file, terse mode off), base (Qwen3.6-35B-A3B), occamy (Occamy-1.0). The first two
share one server; `base` and `occamy` start as soon as their files have been downloaded (fetch_models.py runs beside this).

Safety: the agent runs inside bubblewrap (it can change nothing outside its scratch folder in /dev/shm); a watchdog stops
everything on heat, low VRAM/RAM or a full disk; the live gateway is switched off for the night and ALWAYS switched back on;
the user's own model file is never deleted.
"""
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import analyze
import fetch_models
import harness
import run_experiment as rx
import safety

REPO = Path(__file__).resolve().parents[2]
HOME = Path.home()
TIEL = HOME / "Desktop/AI/models/Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf"
EVAL_DIR = HOME / "Desktop/AI/models/eval"
SHURA = HOME / ".local/bin/shura"
PILOT_TASKS = ("f01_duration-implement", "f01_duration-bugfix", "f04_expr-implement")
PORT = 8099
RUNNER = None          # tests replace this with a stand-in agent; in the real night it stays None (the real agent)
# server layouts tried in order until one starts and serves the pilot: (parallel agents, expert-cache slots)
LAYOUTS = ((4, 24), (4, 16), (4, 8), (2, 16), (1, 24))


def log(out, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(Path(out) / "overnight.log", "a") as f:
        f.write(line + "\n")


def status(out_dir, **kw):
    p = Path(out_dir) / "status.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    p.write_text(json.dumps(cur, indent=2))


def server_cfg(slots):
    live = json.loads((REPO / "config/model-launch.json").read_text())
    exe = live["exePath"]
    trace = REPO / "config/moe-trace/tiel-coder-agentic.csv"
    args = ["-t", "6", "-tb", "6", "-b", "2048", "-ub", "512", "-ngl", "99", "-ncmoe", "26", "-ctk", "turbo3", "-ctv", "turbo3",
            "-fa", "on", "--load-mode", "none", "--temp", "0.6", "--top-k", "20", "--top-p", "0.95", "--reasoning", "auto",
            "--no-mmproj-auto", "--cache-ram", "1024"]
    if slots:
        args += ["--moe-cache-profile", str(trace), "--moe-cache-slots", str(slots)]
    return {"exe": exe, "args": args, "ctx_per_slot": 32768, "prefix": ["nice", "-n", "10"]}


def gpu_is_free(max_used_mib=2500):
    other = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True).stdout.split()
    try:
        used = safety.Gpu().used_mib()
    except (OSError, RuntimeError):
        used = 0
    return not other and used < max_used_mib


def wait_for_free_gpu(out, max_hours=4.0):
    t0, calm = time.monotonic(), 0
    while time.monotonic() - t0 < max_hours * 3600:
        calm = calm + 1 if gpu_is_free() else 0
        if calm >= 5:                                          # five minutes in a row
            return True
        if calm == 0 and int(time.monotonic() - t0) % 600 < 60:
            log(out, "waiting: the GPU is in use by something else")
        time.sleep(60)
    return False


def shura(cmd):
    try:
        return subprocess.run([str(SHURA), cmd], capture_output=True, text=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class Servers:
    """One llama-server at a time; the arms that share a model file share the server."""

    def __init__(self, cfg, parallel, out):
        self.cfg, self.parallel, self.out = cfg, parallel, Path(out)
        self.proc, self.model = None, None

    def get(self, model):
        if self.proc and self.model == model and self.proc.poll() is None:
            return self.proc
        self.close()
        self.proc = rx.start_server(self.cfg, str(model), PORT, self.parallel, self.out / f"server-{Path(model).stem}.log")
        self.model = model
        return self.proc

    def close(self):
        if self.proc:
            rx.stop_server(self.proc)
        self.proc, self.model = None, None


def plumbing_ok(r):
    """A pilot run proves the plumbing when the model really answered requests through the proxy (the agent need not solve the task)."""
    return r["requests"] > 0 and r["tokens"] > 0 and r["request_errors"] == 0


def pilot(out, tasks_by_id, model, layout, watchdog):
    """Three real tasks through the whole plumbing. True if the plumbing works (the agent need not solve them)."""
    parallel, slots = layout
    servers = Servers(server_cfg(slots), parallel, out)
    pdir = Path(out) / "pilot"
    pdir.mkdir(exist_ok=True)
    try:
        servers.get(model)
        ids = [t for t in PILOT_TASKS if t in tasks_by_id]
        t0 = time.monotonic()
        res = rx.run_arm_round({"name": "pilot", "model": str(model)}, ids, tasks_by_id, 1, min(parallel, len(ids)), {}, pdir, PORT, 1,
                               runner=RUNNER, server_starter=lambda *a: servers.proc, server_stopper=lambda s: None)
        took = time.monotonic() - t0
        ok = bool(res) and all(plumbing_ok(r) for r in res)
        log(out, f"pilot layout np={parallel} slots={slots}: {'OK' if ok else 'FAILED'} in {took:.0f} s; "
                 + "; ".join(f"{r['task']}: resolved={r['resolved']} req={r['requests']} tok={r['tokens']} exit={r['agent_exit']} "
                             f"timeout={r['timed_out']} errors={r['request_errors']}" for r in res))
        return ok, took, res
    except Exception as e:
        log(out, f"pilot layout np={parallel} slots={slots}: exception {e}")
        return False, 0.0, []
    finally:
        servers.close()


def run_all(out, arms, tasks, reps, layout, deadline, watchdog):
    """Every arm on every task (paired), arms that share a model file back to back on one server."""
    parallel, slots = layout
    servers = Servers(server_cfg(slots), parallel, out)
    by_id = {t["id"]: t for t in tasks}
    ids = sorted(by_id)
    done = rx.load_done(Path(out) / "results.jsonl")
    try:
        for arm in arms:
            model = Path(arm["model"])
            if not wait_for_file(model, deadline, out, watchdog):
                log(out, f"arm {arm['name']}: model file not available, skipped")
                continue
            todo = [t for t in ids if sum(1 for r in done if r["arm"] == arm["name"] and r["task"] == t) < reps]
            if not todo:
                continue
            log(out, f"arm {arm['name']}: {len(todo)} tasks x {reps} reps, np={parallel}")
            status(out, current_arm=arm["name"])
            servers.get(model)
            done += rx.run_arm_round(arm, todo, by_id, reps, parallel, server_cfg(slots), out, PORT, 7, runner=RUNNER,
                                     server_starter=lambda *a: servers.proc, server_stopper=lambda s: None,
                                     stop_flag=lambda: watchdog.tripped.is_set() or time.monotonic() > deadline)
            if watchdog.tripped.is_set():
                log(out, f"watchdog tripped: {watchdog.reason}")
                break
            if time.monotonic() > deadline:
                log(out, "time budget spent")
                break
    finally:
        servers.close()
    return done


def wait_for_file(path, deadline, out, watchdog):
    t0 = time.monotonic()
    while not Path(path).exists() or Path(path).with_name(Path(path).name + ".part").exists():
        if time.monotonic() > deadline or watchdog.tripped.is_set():
            return False
        if int(time.monotonic() - t0) % 900 < 30:
            log(out, f"waiting for {Path(path).name} to finish downloading")
        time.sleep(30)
    return True


def decide_and_clean(out, runs, arms):
    complete_arms = sorted({r["arm"] for r in runs})
    cells = {}
    for r in runs:
        cells.setdefault((r["task"], r["rep"]), set()).add(r["arm"])
    complete = [r for r in runs if cells[(r["task"], r["rep"])] == set(complete_arms)]
    if len({r["arm"] for r in complete}) < 2:
        return None, "fewer than two arms completed: no comparison", None
    text, res, ver = analyze.report(complete)
    keep = ver["arm"] if ver["kind"] == "winner" else analyze.tie_break(complete, ver["group"])[0]
    return keep, text, ver


def cleanup_models(out, keep_arm, arms):
    removed, kept = [], []
    for arm in arms:
        p = Path(arm["model"])
        if EVAL_DIR not in p.parents:
            continue                                            # never touch a file that was not downloaded for this test
        if arm["name"] == keep_arm or (keep_arm and any(a["name"] == keep_arm and a["model"] == arm["model"] for a in arms)):
            kept.append(str(p))
            continue
        if p.exists():
            p.unlink()
            removed.append(str(p))
    return removed, kept


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=float, default=9.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--keep-models", action="store_true", help="do not delete the losing candidates")
    ap.add_argument("--min-disk-gb", type=float, default=20.0, help="stop if the disk of --out has less free space than this")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + a.max_hours * 3600
    watchdog = safety.Watchdog(str(out), limits={"disk_free_gb": a.min_disk_gb})
    arms = [{"name": "tiel", "model": str(TIEL), "extra": {}},
            {"name": "tiel-no-terse", "model": str(TIEL), "extra": {"chat_template_kwargs": {"terse": False}}},
            {"name": "base", "model": str(EVAL_DIR / fetch_models.CANDIDATES["base"][1]), "extra": {}},
            {"name": "occamy", "model": str(EVAL_DIR / fetch_models.CANDIDATES["occamy"][1]), "extra": {}}]
    shura_was_on = None
    verdict_text = "the run did not finish"
    try:
        log(out, "start")
        status(out, state="starting", folder=str(out))
        if not Path(harness.ENGINE).exists() or not (Path("/usr/bin/bwrap").exists()):
            raise RuntimeError("the agent engine or bubblewrap is missing: refusing to run an agent without the sandbox")
        tasks = harness.all_tasks()
        tasks_by_id = {t["id"]: t for t in tasks}
        log(out, f"{len(tasks)} tasks in {len({t['family'] for t in tasks})} families")
        if not wait_for_free_gpu(out):
            raise RuntimeError("the GPU never became free")
        shura_was_on = shura("status")
        shura("off")
        watchdog.start()
        status(out, state="pilot")
        layout, took = None, 0.0
        for cand in LAYOUTS:
            ok, took, _ = pilot(out, tasks_by_id, TIEL, cand, watchdog)
            if watchdog.tripped.is_set():
                raise RuntimeError(f"watchdog: {watchdog.reason}")
            if ok:
                layout = cand
                break
        if not layout:
            raise RuntimeError("no server layout passed the pilot: see overnight.log")
        q = max(1, min(layout[0], len(PILOT_TASKS)))
        remaining = deadline - time.monotonic()
        est = {r: len(tasks) * r * len(arms) * took / q for r in (a.reps, 1)}          # seconds, from the pilot's throughput
        reps = a.reps if est[a.reps] <= 0.8 * remaining else 1
        log(out, f"layout {layout}; the pilot's {len(PILOT_TASKS)} tasks took {took:.0f} s; estimate for the night: "
                 f"{est[a.reps] / 3600:.1f} h with {a.reps} reps, {est[1] / 3600:.1f} h with 1; budget left {remaining / 3600:.1f} h; using {reps} rep(s)")
        status(out, state="running", layout=list(layout), reps=reps)
        runs = run_all(out, arms, tasks, reps, layout, deadline, watchdog)
        keep, verdict_text, ver = decide_and_clean(out, runs, arms)
        status(out, state="analysed", keep=keep)
        removed, kept = ([], [])
        if keep and not a.keep_models and not watchdog.tripped.is_set():
            removed, kept = cleanup_models(out, keep, arms)
        (out / "REPORT.txt").write_text(verdict_text + "\n")
        log(out, "verdict:\n" + verdict_text)
        log(out, f"kept: {kept}; removed: {removed}")
        morning(out, verdict_text, keep, kept, removed, layout, watchdog)
        status(out, state="done")
        return 0
    except Exception as e:
        log(out, "FAILED: " + "".join(traceback.format_exception_only(type(e), e)).strip())
        log(out, traceback.format_exc())
        status(out, state="failed", error=str(e))
        morning(out, verdict_text, None, [], [], None, watchdog, error=str(e))
        return 1
    finally:
        watchdog.stop()
        subprocess.run(["pkill", "-f", f"llama-server.*--port {PORT}"], capture_output=True)
        if shura_was_on is not False:
            shura("on")
        log(out, f"gateway switched back on; peak GPU {watchdog.peak['gpu_c']} C, CPU {watchdog.peak['cpu_c']:.0f} C, "
                 f"VRAM {watchdog.peak['vram_used_mib']} MiB")
        subprocess.run(["rm", "-rf", "/dev/shm/agentic-eval"], capture_output=True)


def morning(out, verdict_text, keep, kept, removed, layout, watchdog, error=None):
    lines = ["# Утренний отчёт по ночному тесту моделей", ""]
    if error:
        lines += [f"**Тест не закончился:** {error}", "", "Подробности в `overnight.log`. Живая система возвращена в обычное состояние.", ""]
    if watchdog.tripped.is_set():
        lines += [f"**Сработала защита:** {watchdog.reason}. Тест остановлен, модели не удалялись.", ""]
    lines += ["## Итог", "", "```", verdict_text, "```", ""]
    if keep:
        lines += [f"Оставлена модель: **{keep}**.", f"Файлы оставлены: {kept or 'ничего скачанного'}.", f"Удалены: {removed or 'ничего'}.", ""]
    if layout:
        lines += [f"Режим сервера: {layout[0]} параллельных агента, {layout[1]} слотов кэша экспертов.", ""]
    (Path(out) / "MORNING.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
