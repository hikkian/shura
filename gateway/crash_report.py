"""Small bounded diagnostics for an unexpected llama-server exit."""

import json
import re
import signal
import subprocess
import time
import os
from pathlib import Path

MAX_RECORD_BYTES = 64 * 1024
MAX_RECORDS = 10
SENSITIVE = re.compile(r"(api[-_]?key|authorization|password|secret|token|bearer)", re.I)
LOG_SECRET = re.compile(
    r"(?i)(--?(?:api[-_]?key|authorization|password|secret|token)\s+|"
    r"--?(?:api[-_]?key|authorization|password|secret|token)=|"
    r"\bBearer\s+)([^\s,;]+)"
)


def _safe_args(args):
    result = []
    hide_next = False
    for raw in args or []:
        value = str(raw)
        if hide_next:
            result.append("<redacted>")
            hide_next = False
            continue
        if value.startswith("--") and "=" in value:
            name, _ = value.split("=", 1)
            if SENSITIVE.search(name):
                result.append(name + "=<redacted>")
                continue
        if value.startswith("--") and SENSITIVE.search(value):
            result.append(value)
            hide_next = True
            continue
        result.append(value[:2048])
    return result


def journal_tail(unit=None):
    unit = unit or os.environ.get("AI_GATEWAY_CRASH_JOURNAL_UNIT", "ai-gateway.service")
    if not re.fullmatch(r"[A-Za-z0-9_.@-]{1,128}", unit):
        unit = "ai-gateway.service"
    result = subprocess.run(
        ["journalctl", "--user", "-u", unit, "-n", "150", "--no-pager", "-o", "cat"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=2,
        check=False,
        text=True,
        errors="replace",
    )
    lines = result.stdout.splitlines()[-150:]
    return [LOG_SECRET.sub(r"\1<redacted>", line)[:4096] for line in lines]


def _failure(exit_code):
    if isinstance(exit_code, int) and exit_code < 0:
        try:
            return {"signal": signal.Signals(-exit_code).name, "exit_code": None}
        except ValueError:
            return {"signal": f"SIG{-exit_code}", "exit_code": None}
    return {"signal": None, "exit_code": exit_code if isinstance(exit_code, int) else None}


def _encode(record):
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _prune(directory, keep_path):
    entries = []
    for path in directory.glob("crash-*.json"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            item = json.loads(path.read_text())
            if item.get("service") == "ai-gateway" and item.get("schema") == 1:
                entries.append((item.get("time", 0), path))
        except (OSError, ValueError, AttributeError):
            continue
    entries.sort(key=lambda row: (row[0], row[1].name))
    for _, path in entries[:-MAX_RECORDS]:
        if path != keep_path:
            path.unlink(missing_ok=True)


def write_crash_report(state_dir, args, exit_code, operation, snapshot,
                       read_journal=None, now=None):
    """Write one atomic, secret-filtered record; diagnostics failures never escape."""
    try:
        root = Path(state_dir)
        root.mkdir(parents=True, exist_ok=True)
        directory = root / "crashes"
        directory.mkdir(mode=0o700, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            return None
        stamp = time.time() if now is None else float(now)
        name = time.strftime("%Y%m%dT%H%M%S", time.gmtime(stamp)) + f".{int(stamp % 1 * 1_000_000):06d}Z"
        target = directory / f"crash-{name}.json"
        try:
            lines = list((read_journal or journal_tail)())[-150:]
            journal_error = ""
        except Exception as error:  # noqa: BLE001 - journalctl is optional during failure handling.
            lines = []
            journal_error = type(error).__name__
        lines = [LOG_SECRET.sub(r"\1<redacted>", str(line))[:4096] for line in lines]
        try:
            state = snapshot()
        except Exception as error:  # noqa: BLE001 - partial telemetry must not block recovery.
            state = {"snapshot_error": type(error).__name__}
        record = {
            "schema": 1,
            "service": "ai-gateway",
            "time": stamp,
            **_failure(exit_code),
            "operation": str(operation)[:256],
            "launch_args": _safe_args(args),
            "state": state,
            "server_log_tail": lines,
        }
        if journal_error:
            record["journal_error"] = journal_error
        payload = _encode(record)
        while len(payload) > MAX_RECORD_BYTES and record["server_log_tail"]:
            record["server_log_tail"].pop(0)
            payload = _encode(record)
        if len(payload) > MAX_RECORD_BYTES:
            record["launch_args"] = ["<omitted: record size limit>"]
            record["state"] = {"snapshot_error": "record size limit"}
            record["server_log_tail"] = []
            payload = _encode(record)
        temporary = directory / f".{target.name}.tmp"
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
        temporary.replace(target)
        _prune(directory, target)
        return str(target)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return None
