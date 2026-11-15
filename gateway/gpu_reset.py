"""GPURESET1: tell a GPU reset apart from a crash of the model. A full reset of the card (amdgpu: ring timeout, MES does not answer,
`GPU reset begin`, `VRAM is lost`) kills every process that has VRAM on it, so llama-server dies with `unspecified launch failure` (exit -6)
although nothing is wrong with the model. After an unexpected exit the gateway reads the kernel log since the server started and, if it
finds a reset, calls the exit a GPU reset: its own label in last_error and the status, no strike against the model's crash limit (a
separate, higher limit stops a restart loop on a card that keeps resetting), and an optional replay of a request that had not produced a
token yet (`gpuResetRetry`).

Pure parsing and bookkeeping; the command runner is injected, so the tests feed canned kernel logs."""

import os
import re
import subprocess
import time
from collections import deque

MARKERS = ("GPU reset begin", "VRAM is lost", "GPU reset(", "GPU reset succeeded", "GPU reset failed", "reset failed")
RING = re.compile(r"ring (\w+_\d+\.\d+\.\d+) timeout")
PROCESS = re.compile(r"\bProcess (\S+) pid \d+ thread")
LINE = re.compile(r"^(\d+(?:\.\d+)?)\s+\S+\s+kernel:\s*(.*)$")
DEFAULTS = {
    "gpuResetRetry": False,        # replay a request that had no token yet after a GPU reset (needs the recovery path: set by this switch)
    "gpuResetMaxPerWindow": 3,     # more resets than this inside gpuResetWindowSeconds: the card is not healthy, stop restarting (ERROR)
    "gpuResetWindowSeconds": 900,
}


def run_command(cmd, timeout=3.0):
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return cp.returncode, cp.stdout
    except (OSError, subprocess.TimeoutExpired):
        return -1, ""


def parse_kernel_log(text, pid=None):
    """Look for a GPU reset in `journalctl -k -o short-unix` text. None if there is none, else
    {"at": epoch of the reset, "ring": "gfx_0.0.0" or "", "culprit": process named by the ring timeout, "vram_lost": bool,
     "recovered": bool, "ours": the reset touched our server (named in the log), "lines": up to six key lines}."""
    begin = None
    info = {"at": None, "ring": "", "culprit": "", "vram_lost": False, "recovered": False, "ours": False, "lines": []}
    keep = []
    for raw in text.splitlines():
        m = LINE.match(raw.strip())
        if not m:
            continue
        stamp, msg = float(m.group(1)), m.group(2)
        ring = RING.search(msg)
        if ring:
            info["ring"] = ring.group(1)
            info["at"] = info["at"] or stamp
            keep.append(msg)
        elif keep and PROCESS.search(msg):
            info["culprit"] = PROCESS.search(msg).group(1)
            keep.append(msg)
        elif "GPU reset begin" in msg:
            begin = stamp
            info["at"] = begin
            keep.append(msg)
        elif "VRAM is lost" in msg:
            info["vram_lost"] = True
            keep.append(msg)
        elif re.search(r"GPU reset\(\d+\) succeeded|GPU reset succeeded", msg):
            info["recovered"] = True
            keep.append(msg)
        elif pid is not None and re.search(rf"\bpid {pid}\b", msg):
            info["ours"] = True
            keep.append(msg)
    if begin is None and not info["vram_lost"]:
        return None                       # a ring timeout alone (it can be recovered by a ring reset) is not a card reset
    info["lines"] = keep[:6]
    return info


def check_after_exit(run, started_wall, now_wall, pid):
    """The reset found in the kernel log between the server's start and now, or None. Never raises.
    AI_GATEWAY_KERNEL_LOG=off turns the look off (the test suites: a unit test with a mocked server must not read this machine's real log)."""
    if os.environ.get("AI_GATEWAY_KERNEL_LOG") == "off":
        return None
    since = max(started_wall - 5.0, now_wall - 6 * 3600)
    try:
        rc, out = run(["journalctl", "-k", "--no-pager", "-o", "short-unix", "--since", f"@{int(since)}"])
    except Exception:  # noqa: BLE001 - detection is best effort
        return None
    if rc != 0:
        return None
    return parse_kernel_log(out, pid)


def label(returncode, info):
    """The text for last_error."""
    where = f"ring {info['ring']} timeout" if info.get("ring") else "GPU reset"
    who = f" in {info['culprit']}" if info.get("culprit") else ""
    return f"GPU reset ({where}{who}; VRAM lost: {'yes' if info.get('vram_lost') else 'no'}) killed llama-server (exit {returncode}); the model is not at fault"


class ResetLedger:
    """Resets seen by this gateway, in a sliding window; limit_reached() says when to stop restarting."""

    def __init__(self, config=None):
        cfg = {**DEFAULTS, **(config or {})}
        self.max, self.window = cfg["gpuResetMaxPerWindow"], cfg["gpuResetWindowSeconds"]
        self.times = deque()
        self.total = 0
        self.last = None

    def note(self, info, now):
        self.total += 1
        self.last = info
        self.times.append(now)
        while self.times and now - self.times[0] > self.window:
            self.times.popleft()
        return len(self.times) >= self.max

    def snapshot(self):
        return {"total": self.total, "in_window": len(self.times), "last": self.last}


def validate_config(config):
    if type(config.get("gpuResetRetry", False)) is not bool:
        raise ValueError("Invalid guardian config: gpuResetRetry must be true or false")
    for key in ("gpuResetMaxPerWindow", "gpuResetWindowSeconds"):
        value = config.get(key, DEFAULTS[key])
        if type(value) is not int or not 1 <= value <= 86400:
            raise ValueError(f"Invalid guardian config: {key} must be an integer 1..86400")
