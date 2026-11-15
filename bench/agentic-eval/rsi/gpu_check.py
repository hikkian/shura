"""RSIL0c: a ten-minute check of the real stand on the GPU (the card's last step). Not part of the cycle.

    python3 -B -m rsi.gpu_check        # holds the GPU (barid), starts Tiel's profile A from the gateway's command line, runs ONE train task
                                       # through the real engine, stops the server by its own PID, checks that VRAM came back, drops the GPU

It writes its numbers to reports/rsil0c/gpu_check.json. The guard inside GpuStand stops the server when less than 1.5 GiB stays free for
the desktop; the amdgpu journal is counted before and after (a rise means a reset or a hang: stop and report).
"""
import json
import shutil
import subprocess
import time
from pathlib import Path

from . import executor, pilot, server_argv

RESULT = Path("/home/alisher/Desktop/AI/reports/rsil0c/gpu_check.json")
WORK = Path("/dev/shm/RSIL0c")
TASK_TIMEOUT_S = 300          # the card: about ten minutes of GPU in all
HEALTH_TIMEOUT_S = 600
GPU_TASK = "RSIL0c"


def journal_count():
    """Lines of the kernel journal that name an amdgpu timeout, a GPU reset or a hang (since this boot). -1 when the journal can't be read."""
    r = subprocess.run(["bash", "-c", "journalctl -k -b --no-pager 2>/dev/null | grep -ciE 'amdgpu.*timeout|gpu reset|gpu hang' || true"],
                       capture_output=True, text=True)
    text = (r.stdout or "").strip()
    return int(text) if text.isdigit() else -1


def free_gib():
    return (int(executor.VRAM_TOTAL.read_text()) - int(executor.VRAM_USED.read_text())) / 2 ** 30


def run():
    result = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "task_timeout_s": TASK_TIMEOUT_S}
    held = pilot.barid("hold", "gpu", "--task", GPU_TASK, "--wait", "3600", "--by", pilot.AGENT)
    result["hold"] = (held.stdout or held.stderr).strip()[:300]
    if held.returncode != 0:
        result["status"] = "GPU not held: nothing was started"
        return _write(result)
    try:
        if WORK.exists():
            shutil.rmtree(WORK)
        WORK.mkdir(parents=True)
        result["journal_before"] = journal_count()
        result["vram_before_gib"] = round(free_gib(), 3)
        argv_file = WORK / "server-argv.json"
        argv_file.write_text(json.dumps(server_argv.build("tiel-coder")), encoding="utf-8")
        result["argv_len"] = len(json.loads(argv_file.read_text()))
        stand = executor.GpuStand(argv_file, WORK / "gpu")
        base_root = WORK / "harness"
        shutil.copytree(pilot.HARNESS_CFG / "tiel", base_root / "tiel")
        task = dict(pilot.select_tasks(1, 0, 0)["train"][0])
        task["timeout_s"] = TASK_TIMEOUT_S
        result["task"] = task["id"]
        with stand:
            result["vram_loaded_free_gib"] = round(free_gib(), 3)
            runner = executor.Runner(stand, WORK / "scratch")
            rec = runner.run(task, base_root, "tiel", "base", 1, WORK / "runs")
            result["run"] = {k: rec.get(k) for k in ("agent_exit", "timed_out", "seconds", "tokens", "prompt_tokens", "requests",
                                                     "request_errors", "resolved", "score", "tests_run_project", "events")}
        result["guard_stop"] = stand.stop_reason
        result["vram_before_start_free_gib"] = round(stand.free_before, 3)
        result["vram_right_after_stop_free_gib"] = round(stand.free_after_stop, 3)
        result["vram_released_free_gib"] = round(stand.free_released, 3)
        result["release_target_gib"] = round(stand.release_target, 3)
        result["release_wait_s"] = stand.release_wait_s
        result["released"] = stand.released
        result["status"] = "ran"
    except Exception as err:   # the run is reported, never hidden
        result["status"] = f"failed: {type(err).__name__}: {err}"[:500]
    finally:
        result["journal_after"] = journal_count()
        drop = pilot.barid("drop", "gpu", "--task", GPU_TASK, "--by", pilot.AGENT)
        result["drop"] = (drop.stdout or drop.stderr).strip()[:300]
        result["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        shutil.rmtree(WORK, ignore_errors=True)
    return _write(result)


def _write(result):
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
