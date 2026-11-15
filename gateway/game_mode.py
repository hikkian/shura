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
    # An active pause is kept through detector errors for at most this long; a detector
    # that never works must not park the model forever.
    "gameErrorHoldSeconds": 300,
    # The last per-process GPU load stays valid for this long when NVML returns no new sample.
    "gameUtilHoldSeconds": 3,
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
        # MEMV2: desk_need's deep sleep parks the model through this machine too (its own detectors may stay off)
        self.desk_deep = (self.config.get("deskNeed") is True and self.config.get("deskDeepSleep", True) is True
                          and self.config.get("vramPolicy") == "governor")
        self.enabled = self.config["gameMode"] is True or self.desk_deep
        self.settled_only = False                # the pause stands on desk_need's own hysteresis alone
        self.manual = False
        self.active = False
        # ready once the own detectors have spoken; with only desk_need's deep sleep there is nothing to wait for (else the first
        # second after a start would refuse every request as "detector initializing")
        self.ready = self.config["gameMode"] is not True
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
        self.error_since = None

    def update(self, now, reasons, errors=(), settled=()):
        """`settled`: reasons that already carry their own hysteresis (desk_need's deep sleep): they pause at once, and a pause
        that stood on them alone ends as soon as they are gone."""
        self.ready = True
        self.last_sample = now
        self.last_reasons = reasons
        self.errors = list(errors)
        before = self.active
        if not self.enabled:
            return False
        if self.manual or reasons:
            self.error_since = None
            self.off_since = None
            self.on_since = now if self.on_since is None else self.on_since
            if not self.active:
                self.settled_only = bool(settled) and not self.manual and all(r in settled for r in reasons)
            elif not all(r in settled for r in reasons):
                self.settled_only = False
            if self.manual or settled or now - self.on_since >= self.config["gameOnSeconds"]:
                self.active = True
                self.active_since = (
                    now if self.active_since is None else self.active_since
                )
                self.reason = "manual pause" if self.manual else "; ".join(reasons)
        elif errors and self.active and (
            self.error_since is None
            or now - self.error_since < self.config["gameErrorHoldSeconds"]
        ):
            # A broken detector is not proof that the game ended, but the hold is bounded.
            self.error_since = now if self.error_since is None else self.error_since
            self.off_since = None
            self.on_since = None
        else:
            if not errors:
                self.error_since = None
            self.on_since = None
            self.off_since = now if self.off_since is None else self.off_since
            if self.active and (self.settled_only or now - self.off_since >= self.config["gameOffSeconds"]):
                self.active = False
                self.active_since = None
                self.reason = ""
                self.drain_since = None
        return before != self.active

    def manual_switch(self, paused, now):
        self.manual = paused
        if paused:
            self.update(now, self.last_reasons, self.errors)
        elif not self.last_reasons:
            # Detector errors cannot veto an explicit resume; a live game signal still can.
            self.active = False
            self.active_since = None
            self.reason = ""
            self.drain_since = None
            self.on_since = self.off_since = self.error_since = None
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
    def __init__(self, config, proc_root=Path("/proc"), clock=time.monotonic):
        self.config = {**DEFAULTS, **config}
        self.proc_root = Path(proc_root)
        self.clock = clock
        # comm is limited to 15 characters, so long names only ever appear truncated.
        self.desktop = set()
        for name in self.config["gameDesktopProcesses"]:
            self.desktop |= {name.lower(), name.lower()[:15]}
        self.last_util = 0
        self.util_warning = ""
        self.util_seen = {}
        self._classified = {}

    def process(self, pid):
        root = self.proc_root / str(pid)
        if root.stat().st_uid != os.getuid():
            return None
        return (root / "comm").read_text().strip()

    def _classify(self, root, name):
        """Expensive per-process checks; cached while the process identity is unchanged."""
        reaper = steam = False
        if self.config["gameDetectReaper"] and name == "reaper":
            # Steam's launcher is "reaper SteamLaunch AppId=..."; the REAPER audio app is not it.
            with (root / "cmdline").open("rb") as stream:
                reaper = b"SteamLaunch" in stream.read(4096).split(b"\0")
        if self.config["gameDetectSteam"]:
            with (root / "environ").open("rb") as stream:
                env = stream.read(65536).split(b"\0")
            steam = any(re.fullmatch(rb"SteamAppId=[1-9][0-9]*", x) for x in env)
        return reaper, steam  # an OSError means "skip this pid, do not cache"

    def collect(self, model_pid, nvml=None, nvml_error=None):
        reasons = []
        errors = []
        if self.config["gameDetectSteam"] or self.config["gameDetectReaper"]:
            uid, own = os.getuid(), os.getpid()
            live = {}
            for root in self.proc_root.iterdir():
                if not root.name.isdigit() or int(root.name) in (model_pid, own):
                    continue
                try:
                    info = root.stat()
                    if info.st_uid != uid:
                        continue
                    name = (root / "comm").read_text().strip()
                    if not name or name.lower() in self.desktop:
                        continue
                    # exec changes comm, a recycled pid gets a new inode: either invalidates the cache.
                    key = (info.st_ino, info.st_mtime_ns, name)
                    cached = self._classified.get(root.name)
                    if cached is None or cached[0] != key:
                        cached = (key, self._classify(root, name))
                    live[root.name] = cached
                    reaper, steam = cached[1]
                    if reaper:
                        reasons.append(f"Steam reaper PID{root.name}")
                    if steam:
                        reasons.append(f"SteamAppId PID{root.name}")
                except (OSError, ValueError):
                    continue
            self._classified = live
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
                    match = re.fullmatch(
                        r"\(<(?:u?int(?:32|64) )?(\d+)>,\)\s*", cp.stdout
                    )
                    if not match:
                        raise RuntimeError("invalid GameMode ClientCount")
                    if int(match[1]) > 0:
                        reasons.append("gamemoded registered game")
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                    errors.append("gamemoded: " + str(error))
        if self.config["gameDetectNvml"] and nvml_error:
            errors.append(nvml_error)
        elif self.config["gameDetectNvml"]:
            try:
                devices = nvml.processes()
                utilization = self._utilization(nvml)
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
                errors.append(getattr(error, "backend", "NVML") + ": " + str(error))
        return reasons, errors

    def _utilization(self, nvml):
        """Per-pid GPU load; an empty NVML poll keeps the last value for a short hold."""
        now = self.clock()
        try:
            fresh, self.last_util = nvml.process_utilization(self.last_util)
            self.util_warning = ""
        except RuntimeError as error:
            fresh = {}
            self.util_warning = str(error)
        for pid, value in fresh.items():
            self.util_seen[pid] = (value, now)
        hold = self.config["gameUtilHoldSeconds"]
        self.util_seen = {
            pid: item for pid, item in self.util_seen.items() if now - item[1] <= hold
        }
        return {pid: item[0] for pid, item in self.util_seen.items()}
