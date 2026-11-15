"""RSIL0 dry executor: runs a few TRAIN role tasks through the real harness (serve mode, the real engine) against the fake llama
server of the QA toolkit. No GPU, no real model: the answers are scripted, so the scores mean nothing; what is checked is the plumbing
(workdir, config, engine start, session, logs, hidden-test copy and run). Only this process's own fake server is started and stopped.

    cd bench/agentic-eval && AGENTIC_EVAL_NO_SANDBOX=1 python3 -m rsi.dry_executor
"""
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402
import proxy  # noqa: E402

FAKE = Path("/home/alisher/Projects/shura-qa/fake_llama/fake_llama_server.py")
TASKS = [HERE.parent / "tasks_roles" / "gen_json_extract" / "general-01",
         HERE.parent / "tasks_roles" / "gen_log_cause" / "general-01",
         HERE.parent / "tasks_roles" / "plan_deps_order" / "plan-01"]
OUT = Path("/home/alisher/Desktop/AI/reports/rsil1/dry")
WORK = Path("/dev/shm/RSIL0")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    cfg = WORK / "fake.json"
    cfg.write_text(json.dumps({"text": "ok", "n_tokens": 5, "token_delay_ms": 5, "load_seconds": 0.1}))
    port = free_port()
    fake = subprocess.Popen([sys.executable, str(FAKE), "--host", "127.0.0.1", "--port", str(port), "-m", "fake.gguf"],
                            env={**os.environ, "FAKE_LLAMA_CONFIG": str(cfg)}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    results = []
    srv = None
    try:
        deadline = time.time() + 30
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                if time.time() > deadline:
                    raise RuntimeError("fake server did not start")
                time.sleep(0.3)
        # the measuring proxy sits between the engine and the fake server, as in the real experiment: it writes the request log
        # (requests, tokens) that run_one reads. It runs as a thread of this process, so nothing else is stopped.
        proxy_port = free_port()
        srv = proxy.serve(proxy_port, f"http://127.0.0.1:{port}", str(WORK / "proxy.jsonl"), extra={})
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        for task_dir in TASKS:
            task = harness.load_task(task_dir)
            t0 = time.time()
            rec = harness.run_one(task, "base", 1, proxy_port, WORK / "proxy.jsonl", OUT / "agent", scratch_root=str(WORK / "scratch"),
                                  runner=harness.run_agent_serve)
            rec["wall_s"] = round(time.time() - t0, 1)
            results.append(rec)
            print(json.dumps({k: rec[k] for k in ("task", "resolved", "agent_exit", "timed_out", "events", "seconds", "requests", "tokens")}),
                  flush=True)
    finally:
        if srv is not None:
            srv.shutdown()
            srv.server_close()
        try:
            os.killpg(fake.pid, signal.SIGTERM)
            fake.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
            try:
                os.killpg(fake.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
    with open(OUT / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(results)} tasks run through the engine with the fake server; results in {OUT / 'results.jsonl'}")


if __name__ == "__main__":
    main()
