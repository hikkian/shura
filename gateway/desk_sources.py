"""MEMV2: read-only readers of the desk's signals for desk_need.py. Nothing here changes a setting of the desktop, the session or
the GPU: gamemoded's client count over D-Bus, the PipeWire graph (`pw-dump`), the session's idle and lock hints (`loginctl`)
and the amdgpu fdinfo files of the user's own processes. Every reader is injectable (command runner, /proc root, clock), so the
tests feed canned output; a reader that fails names the failure in `errors` and leaves its signal neutral.
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

from desk_need import Signals

BROWSERS = ("brave", "chrome", "chromium", "firefox", "vivaldi", "opera", "msedge")
# Processes of the desktop itself, of the work tools and of Shura: never a "game", whatever they draw.
NOT_GAMES = (
    "kwin_wayland", "kwin_x11", "plasmashell", "xwayland", "xwaylandvideobr", "xorg", "gnome-shell", "krunner", "spectacle",
    "plasma-keyboard", "kscreenlocker_greet", "code", "telegram", "discord", "chatgpt", "qtwebengineproc", "steamwebhelper",
    "llama-server", "llama-bench", "python3", "python", "ghostty", "konsole", "dolphin", "okular", "kate",
) + BROWSERS
VIDEO_ENGINES = ("dec", "enc", "vcn_dec", "vcn_enc", "vcn_unified")   # amdgpu names the video engines of a client like this


def run_command(cmd, timeout=2.0):
    """(returncode, stdout) of a command; (-1, "") if it does not exist or hangs. The default runner of DeskSources."""
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return cp.returncode, cp.stdout
    except (OSError, subprocess.TimeoutExpired):
        return -1, ""


def parse_gamemode(rc, out):
    """gamemoded ClientCount from `gdbus call ... Properties.Get`. Returns the count, or raises ValueError."""
    if rc != 0:
        raise ValueError("gamemoded not reachable")
    match = re.fullmatch(r"\(<(?:u?int(?:32|64) )?(\d+)>,\)\s*", out)
    if not match:
        raise ValueError("invalid GameMode ClientCount")
    return int(match[1])


def parse_pipewire(text):
    """Signals from a `pw-dump` document: (call, audio_out). A call is a running microphone/camera/screen capture stream;
    audio_out is a running playback stream (music, a video's sound)."""
    call = audio_out = False
    for obj in json.loads(text):
        if not isinstance(obj, dict) or obj.get("type") != "PipeWire:Interface:Node":
            continue
        info = obj.get("info") or {}
        props = info.get("props") or {}
        klass = props.get("media.class") or ""
        if info.get("state") != "running":
            continue
        if klass == "Stream/Output/Audio":
            audio_out = True
        elif klass == "Stream/Input/Audio" and not props.get("stream.capture.sink"):
            call = True                        # a recording client (a monitor of the output is not a microphone)
        elif klass in ("Stream/Input/Video", "Video/Source", "Stream/Output/Video"):
            call = True
    return call, audio_out


KWIN_TAG = "shura-desk-signal "
KWIN_ENTRY = "shura-desk-signalEnabled"


def parse_kwin_line(text):
    """The newest line of the KWin script (`shura-desk-signal {json}`) -> dict with active, fullscreen, fullscreen_class; or raises
    ValueError. The text may be several lines (journalctl -n 1 gives one): the last one with the tag wins."""
    for line in reversed(text.splitlines()):
        at = line.find(KWIN_TAG)
        if at < 0:
            continue
        data = json.loads(line[at + len(KWIN_TAG):])
        if not isinstance(data, dict) or data.get("v") != 1 or "error" in data:
            raise ValueError("unusable KWin script line")
        return {"active": str(data.get("active", "")), "fullscreen": bool(data.get("fullscreen", False)),
                "any_fullscreen": bool(data.get("any_fullscreen", False)), "fullscreen_class": str(data.get("fullscreen_class", ""))}
    raise ValueError("no KWin script line")


def kwin_script_enabled(kwinrc_text):
    """Is the script switched on in kwinrc ([Plugins] shura-desk-signalEnabled=true)?"""
    in_plugins = False
    for raw in kwinrc_text.splitlines():
        line = raw.strip()
        if line.startswith("["):
            in_plugins = line == "[Plugins]"
        elif in_plugins and line.replace(" ", "") == KWIN_ENTRY + "=true":
            return True
    return False


def parse_session(text):
    """(idle_s, locked) from `loginctl show-session -p IdleHint -p IdleSinceHintMonotonic -p LockedHint`. KDE does not set
    IdleHint, so idle_s is usually None: only the lock tells that nobody is there."""
    values = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    locked = values.get("LockedHint") == "yes"
    idle_s = None
    if values.get("IdleHint") == "yes":
        try:
            since_us = int(values.get("IdleSinceHintMonotonic", "0"))
            if since_us > 0:
                idle_s = max(0.0, time.monotonic() - since_us / 1e6)
        except ValueError:
            pass
    return idle_s, locked


def parse_fdinfo(text):
    """One amdgpu fdinfo file: {"client": id, "vram_kib": n, "engines": {name: ns}} or None if it is not an amdgpu client."""
    if not re.search(r"^drm-driver:\s*amdgpu\s*$", text, re.M):
        return None
    out = {"client": None, "vram_kib": 0, "engines": {}}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if key == "drm-client-id":
            out["client"] = value
        elif key == "drm-memory-vram":
            out["vram_kib"] = int(value.split()[0])
        elif key.startswith("drm-engine-") and not key.startswith("drm-engine-capacity"):
            try:
                out["engines"][key[len("drm-engine-"):]] = int(value.split()[0])
            except (ValueError, IndexError):
                pass
    return out


class DeskSources:
    def __init__(self, config, run=run_command, proc_root="/proc", clock=time.monotonic, uid=None, session="auto"):
        self.cfg = config
        self.run, self.proc, self.clock = run, Path(proc_root), clock
        self.uid = os.getuid() if uid is None else uid
        self.session = session
        self.previous = None                    # (time, {pid: {"gfx": ns, "video": ns}}) of the last fdinfo scan
        self.kwinrc = Path(os.path.expanduser("~/.config/kwinrc"))
        self.kwin_poll_seconds, self.kwin_retry_seconds = 4.0, 300.0
        self._kwin = (None, -1e9)               # (last parsed line or None, time it was read)
        self._kwin_missing_until = -1e9
        self.fd_rescan_seconds = 30.0
        self._fd_cache, self._fd_scan_at = {}, -1e9

    # --- the readers ----------------------------------------------------------------------------------------------
    def read_game(self, errors):
        rc, out = self.run(["gdbus", "call", "--session", "--dest", "com.feralinteractive.GameMode", "--object-path",
                            "/com/feralinteractive/GameMode", "--method", "org.freedesktop.DBus.Properties.Get",
                            "com.feralinteractive.GameMode", "ClientCount"])
        try:
            return parse_gamemode(rc, out) > 0
        except ValueError as error:
            if "reachable" not in str(error):    # gamemoded not running is not an error: no game
                errors.append(f"gamemoded: {error}")
            return False

    def read_pipewire(self, errors):
        rc, out = self.run(["pw-dump"], timeout=3.0)
        if rc != 0:
            errors.append("pipewire: pw-dump failed")
            return False, False
        try:
            return parse_pipewire(out)
        except (ValueError, TypeError) as error:
            errors.append(f"pipewire: {error}")
            return False, False

    def read_session(self, errors):
        rc, out = self.run(["loginctl", "show-session", self.session, "-p", "IdleHint", "-p", "IdleSinceHintMonotonic", "-p", "LockedHint"])
        if rc != 0:
            errors.append("logind: show-session failed")
            return None, False
        return parse_session(out)

    def read_kwin(self, errors):
        """The KWin script's newest line, or None when the script is not enabled (then the proxy signals are used). The journal is
        searched only when kwinrc says the script is on, and a failed search is not repeated for kwin_retry_seconds."""
        if not self.cfg.get("deskKwinSignal", True):
            return None
        now = self.clock()
        if now - self._kwin[1] < self.kwin_poll_seconds:
            return self._kwin[0]
        try:
            enabled = kwin_script_enabled(self.kwinrc.read_text())
        except OSError:
            enabled = False
        if not enabled or now < self._kwin_missing_until:
            self._kwin = (None, now)
            return None
        rc, out = self.run(["journalctl", "--user", "-b", "-t", "kwin_wayland", "-r", "-n", "1", "--case-sensitive", "-g", KWIN_TAG,
                            "-o", "cat"], timeout=3.0)
        try:
            if rc != 0:
                raise ValueError("journalctl failed")
            data = parse_kwin_line(out)
        except ValueError as error:
            self._kwin_missing_until = now + self.kwin_retry_seconds
            errors.append(f"kwin script: {error}")
            data = None
        self._kwin = (data, now)
        return data

    def read_gpu_clients(self, errors, exclude_pids=()):
        """Per process of the user: VRAM (MiB) and, from the change since the last scan, the load of the graphics and video
        engines in percent. Returns {pid: {"name", "vram_mib", "gfx_pct", "video_pct"}}."""
        now = self.clock()
        current, rows = {}, {}
        try:
            entries = list(self.proc.iterdir())
        except OSError as error:
            errors.append(f"fdinfo: {error}")
            return {}
        fresh_scan = now - self._fd_scan_at >= self.fd_rescan_seconds
        if fresh_scan:
            self._fd_scan_at = now
        live = {}
        for root in entries:
            if not root.name.isdigit() or int(root.name) in exclude_pids:
                continue
            try:
                stat = root.stat()
                if stat.st_uid != self.uid:
                    continue
                key = (stat.st_ino, stat.st_mtime_ns)
                cached = self._fd_cache.get(root.name)
                # the list of a process's graphics fds is looked up when the process is new and then every fd_rescan_seconds;
                # in between only those few files are read (listing every fd of every process costs ~0.2 s of CPU)
                if cached is None or cached[0] != key or fresh_scan:
                    names = []
                    for fd in (root / "fdinfo").iterdir():
                        try:
                            if parse_fdinfo(fd.read_text()) is not None:
                                names.append(fd.name)
                        except (OSError, ValueError):
                            continue
                    cached = (key, names)
                live[root.name] = cached
                clients = {}
                for fd_name in cached[1]:
                    try:
                        info = parse_fdinfo((root / "fdinfo" / fd_name).read_text())
                    except (OSError, ValueError):
                        continue
                    if info is not None and info["client"] is not None:
                        clients[info["client"]] = info          # several fds of one client are one client
                if not clients:
                    continue
                name = (root / "comm").read_text().strip()
            except OSError:
                continue
            vram = sum(c["vram_kib"] for c in clients.values()) // 1024
            gfx = sum(c["engines"].get("gfx", 0) + c["engines"].get("compute", 0) for c in clients.values())
            video = sum(ns for c in clients.values() for eng, ns in c["engines"].items() if eng in VIDEO_ENGINES)
            pid = int(root.name)
            current[pid] = {"gfx": gfx, "video": video}
            rows[pid] = {"name": name, "vram_mib": vram, "gfx_pct": 0.0, "video_pct": 0.0}
        if self.previous is not None:
            before_t, before = self.previous
            dt = now - before_t
            if dt > 0:
                for pid, row in rows.items():
                    prev = before.get(pid)
                    if prev is not None:
                        row["gfx_pct"] = max(0.0, (current[pid]["gfx"] - prev["gfx"]) / (dt * 1e9) * 100.0)
                        row["video_pct"] = max(0.0, (current[pid]["video"] - prev["video"]) / (dt * 1e9) * 100.0)
        self.previous = (now, current)
        self._fd_cache = live
        return rows

    # --- everything together --------------------------------------------------------------------------------------
    def collect(self, model_pids=()):
        errors = []
        cfg = self.cfg
        sig = Signals()
        sig.game = self.read_game(errors)
        call, audio_out = self.read_pipewire(errors)
        sig.call = call
        sig.idle_s, sig.locked = self.read_session(errors)
        kwin = self.read_kwin(errors)
        if kwin is not None:
            sig.fullscreen, sig.active_class, sig.fullscreen_class = kwin["fullscreen"] or kwin["any_fullscreen"], kwin["active"], kwin["fullscreen_class"]
        rows = self.read_gpu_clients(errors, exclude_pids=set(model_pids) | {os.getpid()})
        browser_vram = 0
        video_engine = browser_drawing = False
        for pid, row in rows.items():
            name = row["name"].lower()
            if name in BROWSERS:
                browser_vram += row["vram_mib"]
                browser_drawing = browser_drawing or row["gfx_pct"] >= 10.0   # software decode: the page is drawn, not decoded
            if row["video_pct"] >= 2.0:
                video_engine = True
            if name in NOT_GAMES:
                continue
            if row["gfx_pct"] >= cfg["deskGpuAppBusyPercent"]:
                sig.gpu_app, sig.gpu_app_detail = row["name"], f"gfx {row['gfx_pct']:.0f}%"
            elif row["vram_mib"] >= cfg["deskGpuAppVramMiB"] and not sig.gpu_app:
                sig.gpu_app, sig.gpu_app_detail = row["name"], f"VRAM {row['vram_mib']} MiB"
        sig.browser_vram_mib = browser_vram
        sig.video = audio_out and (video_engine or browser_drawing)
        sig.errors = errors
        return sig
