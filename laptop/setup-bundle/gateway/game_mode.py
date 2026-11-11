"""Opt-in heavy-GPU signals and hysteresis. No GPU initialization on import."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import time

DEFAULTS = {
    "gameMode": False,
    "gameDetectNvml": False,
    "gameDetectGameMode": False,
    "gameDetectSteam": False,
    "gameDetectReaper": False,
    "gameOnSeconds": 5,
    "gameOffSeconds": 60,
    "gameDrainSeconds": 60,
    "gameVramMiB": 2048,
    "gameGpuPercent": 80,
    "gameRetrySeconds": 5,
    "gamePollSeconds": 1,
    "gameDesktopProcesses": [
        "kwin_wayland",
        "plasmashell",
        "xwayland",
        "xorg",
        "gnome-shell",
        "brave",
        "chrome",
        "chromium",
        "firefox",
        "code",
        "telegram",
        "discord",
        "chatgpt",
        "qtwebengineproc",
        "kscreenlocker_greet",
        "steam",
        "steamwebhelper",
    ],
}


class GamePolicy:
    def __init__(self, config):
        self.config = {**DEFAULTS, **config}
        self.enabled = self.config["gameMode"] is True
        self.manual = False
        self.active = False
        self.ready = not self.enabled
        self.reason = ""
        self.errors = []
        self.on_since = None
        self.off_since = None
        self.active_since = None
        self.last_sample = None
        self.last_reasons = []
        self.parked = False
        self.next_save = 0
        self.drain_since = None

    def update(self, now, reasons, errors=()):
        self.ready = True
        self.last_sample = now
        self.last_reasons = reasons
        self.errors = list(errors)
        before = self.active
        if not self.enabled:
            return False
        if self.manual or reasons:
            self.off_since = None
            self.on_since = now if self.on_since is None else self.on_since
            if self.manual or now - self.on_since >= self.config["gameOnSeconds"]:
                self.active = True
                self.active_since = (
                    now if self.active_since is None else self.active_since
                )
                self.reason = "manual pause" if self.manual else "; ".join(reasons)
        elif errors and self.active:
            self.off_since = None
            self.on_since = None
        else:
            self.on_since = None
            self.off_since = now if self.off_since is None else self.off_since
            if self.active and now - self.off_since >= self.config["gameOffSeconds"]:
                self.active = False
                self.active_since = None
                self.reason = ""
                self.drain_since = None
        return before != self.active

    def manual_switch(self, paused, now):
        self.manual = paused
        if paused:
            self.update(now, self.last_reasons, self.errors)
        elif not self.last_reasons and not self.errors:
            self.active = False
            self.active_since = None
            self.reason = ""
            self.drain_since = None
        return self.active

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        return {
            "enabled": self.enabled,
            "active": self.active,
            "manual": self.manual,
            "reason": self.reason,
            "parked": self.parked,
            "errors": self.errors,
            "elapsed_seconds": int(now - self.active_since)
            if self.active_since is not None
            else 0,
            "quiet_seconds": int(now - self.off_since)
            if self.off_since is not None
            else 0,
        }


class GameSignals:
    def __init__(self, config, proc_root=Path("/proc")):
        self.config = {**DEFAULTS, **config}
        self.proc_root = Path(proc_root)
        self.desktop = {x.lower() for x in self.config["gameDesktopProcesses"]}
        self.last_util = 0
        self.util_warning = ""

    def process(self, pid):
        root = self.proc_root / str(pid)
        if root.stat().st_uid != os.getuid():
            return None
        return (root / "comm").read_text().strip()

    def collect(self, model_pid, nvml=None):
        reasons = []
        errors = []
        if self.config["gameDetectSteam"] or self.config["gameDetectReaper"]:
            for root in self.proc_root.iterdir():
                if not root.name.isdigit() or int(root.name) in (
                    model_pid,
                    os.getpid(),
                ):
                    continue
                try:
                    name = self.process(int(root.name))
                    if not name or name.lower() in self.desktop:
                        continue
                    if self.config["gameDetectReaper"] and name == "reaper":
                        reasons.append(f"Steam reaper PID{root.name}")
                    if self.config["gameDetectSteam"]:
                        with (root / "environ").open("rb") as stream:
                            env = stream.read(65536).split(b"\0")
                        if any(
                            re.fullmatch(rb"SteamAppId=[1-9][0-9]*", x) for x in env
                        ):
                            reasons.append(f"SteamAppId PID{root.name}")
                except (OSError, ValueError):
                    continue
        if self.config["gameDetectGameMode"]:
            if not shutil.which("gdbus") or not shutil.which("gamemoded"):
                errors.append("gamemoded unavailable")
            else:
                try:
                    cp = subprocess.run(
                        [
                            "gdbus",
                            "call",
                            "--session",
                            "--dest",
                            "com.feralinteractive.GameMode",
                            "--object-path",
                            "/com/feralinteractive/GameMode",
                            "--method",
                            "org.freedesktop.DBus.Properties.Get",
                            "com.feralinteractive.GameMode",
                            "ClientCount",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if cp.returncode:
                        raise RuntimeError(cp.stderr.strip())
                    match = re.fullmatch(r"\(<(?:int32 )?(\d+)>,\)\s*", cp.stdout)
                    if not match:
                        raise RuntimeError("invalid GameMode ClientCount")
                    if int(match[1]) > 0:
                        reasons.append("gamemoded registered game")
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                    errors.append("gamemoded: " + str(error))
        if self.config["gameDetectNvml"]:
            try:
                devices = nvml.processes()
                try:
                    utilization, self.last_util = nvml.process_utilization(
                        self.last_util
                    )
                except RuntimeError as error:
                    utilization = {}
                    self.util_warning = str(error)
                for pid, mib in devices.items():
                    if pid in (model_pid, os.getpid()):
                        continue
                    try:
                        name = self.process(pid)
                    except OSError:
                        continue
                    if not name or name.lower() in self.desktop:
                        continue
                    if mib is not None and mib > self.config["gameVramMiB"]:
                        reasons.append(f"foreign VRAM PID{pid} {name} {mib}MiB")
                    if utilization.get(pid, 0) > self.config["gameGpuPercent"]:
                        reasons.append(
                            f"foreign GPU PID{pid} {name} {utilization[pid]}%"
                        )
            except (OSError, RuntimeError, AttributeError) as error:
                errors.append("NVML: " + str(error))
        return reasons, errors
