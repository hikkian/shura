"""MEMV2: how much the person needs the computer right now (human_need), and what the server's VRAM reserve and sleep level
should be for it. Pure logic: the signals are read elsewhere (desk_sources.py), nothing here touches the desktop or the GPU.

human_need is one number in [0, 1] from the signals of the desk (reports/REPORT-MEMVISION1.md, 2.2): 0.0 nobody is there,
0.3 ordinary use, 0.6 video or a call, 1.0 a game or other heavy 3D. The reserve of free VRAM follows it:
    reserve = 0.75 GiB + 1.75 GiB * need     (0.75 at night ... 1.5 ordinary ... 2.5 in a game)
but never below the floor (default 1.5 GiB; lowering the floor is the user's decision, see deskReserveFloorMiB). The need
rises at once and falls only after it has been lower for a while, so a short pause in a video does not give the cache back.
All sizes are MiB, all times are seconds.
"""

import math
import time
from dataclasses import dataclass, field

DEFAULTS = {
    "deskNeed": False,
    "deskReserveFloorMiB": 1536,       # the VRAM sentinel's floor; below it only with the user's word
    "deskNeedDownSeconds": 60,         # the need falls only after the lower value held this long
    "deskIdleSeconds": 600,            # no input for this long (or a locked screen) means nobody is there
    "deskGpuAppBusyPercent": 50,       # a foreign process keeping the graphics engine this busy is a game
    "deskGpuAppVramMiB": 2048,         # a foreign process holding this much VRAM is probably a game
    "deskBrowserVramMiB": 2048,        # a browser holding this much VRAM lifts the need to the video level
    "deskSleepOnSeconds": 5,           # need 1.0 for this long: light sleep (the cache goes to zero, the model stays)
    "deskSleepOffSeconds": 60,         # need below 1.0 for this long: wake
    # MEMV2 deep sleep (L2): the model leaves the card with its context saved in RAM (game mode's parking: drain, save, stop) when
    # the desk has asked for 1.0 for deskDeepSleepSeconds, or at once when the server had to pause during light sleep (the game
    # needs more than the cache could give). It wakes with light sleep; the model then loads on the next request.
    "deskDeepSleep": True,
    "deskDeepSleepSeconds": 120,
    "deskReserveStepMiB": 128,         # the reserve is pushed only when it moved by at least this much
    # VRAMLOGIC3: the governor's working buffer above the floor. 256 on a quiet desk; 512 while the desk is active: need >= 0.6
    # (video, call, full-screen browser, game) or for deskBufferHoldSeconds after the browser's VRAM grew by deskBrowserGrowthMiB
    # between two polls. SAGE1 (43 h): 98 % of the desk's rises > 256 MiB/s are the browser's, and they come in clusters: a 120 s
    # hold after +128 MiB keeps 512 on 27 % of the time and covers 92 % of those rises. Instant signals only, nothing learned.
    "deskBufferQuietMiB": 256,
    "deskBufferActiveMiB": 512,
    "deskBrowserGrowthMiB": 128,
    "deskBufferHoldSeconds": 120,
    "deskPushMinSeconds": 5,
    "deskPollSeconds": 2,
    "deskKwinSignal": True,            # read the KWin script's full-screen line when the script is enabled (kwin/shura-desk-signal)
    "deskSignalsFile": "",             # test hook: read the signals from this JSON file instead of the desktop
}

# application classes (KWin resourceClass, lower case, substring match) that are tools of the person, not games: a full-screen terminal
# or editor does not make the desk busy; a full-screen browser is a video (the level of video), anything else full-screen is a game
WORK_CLASSES = ("konsole", "ghostty", "kitty", "alacritty", "wezterm", "code", "kate", "dolphin", "okular", "telegram", "libreoffice",
                "org.kde", "plasma", "krunner", "spectacle", "gwenview", "obsidian", "zed", "jetbrains", "idea", "pycharm")
BROWSER_CLASSES = ("brave", "chrome", "chromium", "firefox", "vivaldi", "opera", "msedge", "zen")

RESERVE_NIGHT_MIB = 768
RESERVE_SPAN_MIB = 1792                # 0.75 GiB + 1.75 GiB * need: 768 .. 2560
RESERVE_RANGE = (256, 4096)            # the server's range for POST /vram/policy


@dataclass
class Signals:
    """What the desk says. A signal that could not be read is left at its neutral value and named in `errors`: an unknown
    signal never lowers the need below ordinary use."""
    game: bool = False                         # gamemoded has a registered client
    gpu_app: str = ""                          # name of a foreign process that looks like a game (gfx busy / VRAM), "" = none
    gpu_app_detail: str = ""
    call: bool = False                         # camera, microphone or screen sharing is in use
    video: bool = False                        # video is playing (audio stream with the decode engine busy)
    browser_vram_mib: int = 0
    idle_s: float = None                       # seconds without input (None = unknown)
    locked: bool = False
    fullscreen: bool = None                    # the active window is full screen (None = the KWin script is not there)
    active_class: str = ""                     # application class of the active window (KWin script only)
    fullscreen_class: str = ""                 # class of a full-screen window, when there is one
    errors: list = field(default_factory=list)

    @staticmethod
    def from_json(data):
        """Signals from a JSON object (the test hook file); unknown keys are ignored."""
        allowed = {f for f in Signals.__dataclass_fields__}
        return Signals(**{k: v for k, v in data.items() if k in allowed})


def fullscreen_kind(sig):
    """"game" | "browser" | "" for the full-screen window the KWin script reported (nothing when it is not there)."""
    if not sig.fullscreen:
        return ""
    name = (sig.fullscreen_class or sig.active_class or "").lower()
    if not name:
        return ""
    if any(w in name for w in BROWSER_CLASSES):
        return "browser"
    if any(w in name for w in WORK_CLASSES):
        return ""
    return "game"


def target_need(sig, cfg):
    """The need the signals ask for right now, and the short reason. No hysteresis."""
    if sig.game:
        return 1.0, "game: gamemoded"
    if sig.gpu_app:
        return 1.0, f"gpu_app: {sig.gpu_app} {sig.gpu_app_detail}".strip()
    kind = fullscreen_kind(sig)
    if kind == "game":
        return 1.0, f"fullscreen: {sig.fullscreen_class or sig.active_class}"
    if sig.call:
        return 0.6, "call: camera/microphone/screen share"
    if kind == "browser":
        return 0.6, f"fullscreen browser: {sig.fullscreen_class or sig.active_class}"
    if sig.video:
        return 0.6, "video playing"
    if sig.browser_vram_mib >= cfg["deskBrowserVramMiB"]:
        return 0.6, f"browser VRAM {sig.browser_vram_mib} MiB"
    if sig.locked:
        return 0.0, "locked"
    if sig.idle_s is not None and sig.idle_s >= cfg["deskIdleSeconds"]:
        return 0.0, f"idle {int(sig.idle_s)} s"
    return 0.3, "input" if sig.idle_s is not None else "input unknown"


def reserve_for(need, floor_mib):
    """VRAM reserve in MiB for a need, rounded up to 64 and clamped to the floor and to the server's range."""
    raw = RESERVE_NIGHT_MIB + RESERVE_SPAN_MIB * need
    value = max(int(floor_mib), int(math.ceil(raw / 64.0) * 64))
    return max(RESERVE_RANGE[0], min(RESERVE_RANGE[1], value))


class DeskNeed:
    """Need with hysteresis, the sleep decision and what to push. Only injected clock values (`now`, seconds)."""

    def __init__(self, config=None):
        self.cfg = {**DEFAULTS, **(config or {})}
        self.need = 0.3
        self.reason = "starting"
        self.sleep_level = 0                     # 0 awake, 1 light sleep (L1: no expert cache), 2 deep sleep (L2: parked)
        self.deep_reason = ""
        self.pushed = None                       # what the server was last told: {"reserve_mib", "sleep_level", "human_need"}
        self.pushed_at = -math.inf
        self.sleeping_s = 0.0                    # seconds spent in light sleep
        self._lower_since = None
        self._lower_max = 0.0
        self._full_since = None
        self._notfull_since = None
        self._last_tick = None
        self.buffer_mib = self.cfg["deskBufferQuietMiB"]
        self.buffer_reason = "quiet desk"
        self._prev_browser_mib = None
        self._browser_active_until = -math.inf
        self._browser_growth = 0

    # --- need with hysteresis -------------------------------------------------------------------------------------
    def update(self, now, sig):
        cfg = self.cfg
        want, why = target_need(sig, cfg)
        if self._last_tick is not None and self.sleep_level >= 1:
            self.sleeping_s += max(0.0, now - self._last_tick)
        self._last_tick = now
        if want >= self.need:
            self.need, self.reason = want, why
            self._lower_since, self._lower_max = None, 0.0
        else:
            if self._lower_since is None:
                self._lower_since, self._lower_max = now, want
            self._lower_max = max(self._lower_max, want)
            if now - self._lower_since >= cfg["deskNeedDownSeconds"]:
                self.need, self.reason = self._lower_max, why
                self._lower_since, self._lower_max = None, 0.0
            else:
                self.reason = f"{why} (holding {self.need:g} for up to {cfg['deskNeedDownSeconds']} s)"
        # light sleep: starts when the desk itself has asked for 1.0 for a few seconds in a row (not on the held value, so a
        # two-second signal cannot put the model to sleep for the whole hold); wakes when the held need has been lower for a while
        if want >= 1.0:
            self._full_since = now if self._full_since is None else self._full_since
            if self.sleep_level == 0 and now - self._full_since >= cfg["deskSleepOnSeconds"]:
                self.sleep_level = 1
            if (self.sleep_level == 1 and cfg["deskDeepSleep"] and cfg["deskDeepSleepSeconds"] > 0
                    and now - self._full_since >= cfg["deskDeepSleepSeconds"]):
                self.sleep_level, self.deep_reason = 2, f"{why} for {int(now - self._full_since)} s"
        else:
            self._full_since = None
        self._update_buffer(now, sig)
        if self.need >= 1.0:
            self._notfull_since = None
        else:
            self._notfull_since = now if self._notfull_since is None else self._notfull_since
            if self.sleep_level >= 1 and now - self._notfull_since >= cfg["deskSleepOffSeconds"]:
                self.sleep_level, self.deep_reason = 0, ""
        return self.need

    def _update_buffer(self, now, sig):
        """VRAMLOGIC3: the working buffer from instant signals (see DEFAULTS)."""
        cfg = self.cfg
        browser = int(sig.browser_vram_mib or 0)
        if self._prev_browser_mib is not None and browser - self._prev_browser_mib >= cfg["deskBrowserGrowthMiB"]:
            self._browser_active_until = now + cfg["deskBufferHoldSeconds"]
            self._browser_growth = browser - self._prev_browser_mib
        self._prev_browser_mib = browser
        if self.need >= 0.6:
            self.buffer_mib, self.buffer_reason = cfg["deskBufferActiveMiB"], f"active desk: {self.reason}"
        elif now < self._browser_active_until:
            left = int(self._browser_active_until - now)
            self.buffer_mib = cfg["deskBufferActiveMiB"]
            self.buffer_reason = f"browser VRAM grew by {self._browser_growth} MiB (held {left} s more)"
        else:
            self.buffer_mib, self.buffer_reason = cfg["deskBufferQuietMiB"], "quiet desk"

    # --- what to tell the server ----------------------------------------------------------------------------------
    def reserve_mib(self):
        return reserve_for(self.need, self.cfg["deskReserveFloorMiB"])

    def deepen(self, why):
        """MEMV2: the server paused while the model slept lightly: the desk needs more than the cache could give. Deep sleep at
        once (if allowed); True when the level changed."""
        if self.sleep_level == 1 and self.cfg["deskDeepSleep"]:
            self.sleep_level, self.deep_reason = 2, why
            return True
        return False

    def wanted(self):
        # the server knows levels 0 and 1; in deep sleep it is parked, and if it still runs it holds no cache
        return {"reserve_mib": self.reserve_mib(), "sleep_level": min(self.sleep_level, 1), "human_need": round(self.need, 2),
                "buffer_mib": self.buffer_mib}

    def push_due(self, now, force=False):
        """The body for POST /vram/policy if the server should be told something new, else None. The informational
        human_need alone never causes a push: it travels with the next change of the reserve or the sleep level."""
        want = self.wanted()
        if not force and self.pushed is not None:
            moved = abs(want["reserve_mib"] - self.pushed["reserve_mib"]) >= self.cfg["deskReserveStepMiB"]
            slept = want["sleep_level"] != self.pushed["sleep_level"]
            buffered = want["buffer_mib"] != self.pushed.get("buffer_mib", want["buffer_mib"])
            if not (moved or slept or buffered) or now - self.pushed_at < self.cfg["deskPushMinSeconds"]:
                return None
        body = dict(want)
        body["note"] = self.reason[:200]
        return body

    def pushed_ok(self, now, body):
        self.pushed = {k: body[k] for k in ("reserve_mib", "sleep_level", "human_need")}
        self.pushed["buffer_mib"] = body.get("buffer_mib", self.buffer_mib)   # a server that refused the field keeps its own
        self.pushed_at = now

    def server_restarted(self):
        """A new server process starts with the configured policy: tell it everything again at the next chance."""
        self.pushed = None

    def snapshot(self):
        return {"need": round(self.need, 2), "reason": self.reason, "reserve_mib": self.reserve_mib(),
                "buffer_mib": self.buffer_mib, "buffer_reason": self.buffer_reason,
                "floor_mib": self.cfg["deskReserveFloorMiB"], "sleep_level": self.sleep_level, "deep_reason": self.deep_reason,
                "pushed": self.pushed, "sleeping_s": round(self.sleeping_s, 1)}


class Ledger:
    """Daily totals of the concessions, from the server's journal (GET /vram/journal). The server's journal is in memory and
    starts empty with every server process, so the gateway keeps the days. `day` is a local date string, injected."""

    @staticmethod
    def _blank():
        return {"yields": 0, "yielded_mib": 0, "recovered": 0, "recovered_sum_s": 0.0, "recovered_max_s": 0.0,
                "pauses": 0, "sleeps": 0, "sleeping_s": 0.0}

    def __init__(self):
        self.days = {}
        self.last_id = 0            # last journal event id seen from the current server process
        self._counted = set()       # trims whose recovery time is already in the totals

    def day(self, day):
        return self.days.setdefault(day, self._blank())

    def server_restarted(self):
        self.last_id = 0
        self._counted = set()

    def add_events(self, day, events):
        """Count the new events of a journal answer once each (by id; a restarted server starts its ids again)."""
        d = self.day(day)
        for e in events:
            if not isinstance(e, dict) or type(e.get("id")) is not int:
                continue
            if e.get("kind") == "trim" and e.get("recovered_s") is not None and e["id"] not in self._counted:
                # a trim is closed later than it appears: its recovery time is counted when it shows up filled in
                self._counted.add(e["id"])
                d["recovered"] += 1
                d["recovered_sum_s"] += float(e["recovered_s"])
                d["recovered_max_s"] = max(d["recovered_max_s"], float(e["recovered_s"]))
            if e["id"] <= self.last_id:
                continue
            self.last_id = e["id"]
            kind = e.get("kind")
            if kind == "trim":
                d["yields"] += 1
                d["yielded_mib"] += abs(int(e.get("mib") or 0))
            elif kind == "pause":
                d["pauses"] += 1
            elif kind == "sleep":
                d["sleeps"] += 1
        return d

    def to_json(self):
        return {"days": self.days}

    def load(self, data):
        if isinstance(data, dict) and isinstance(data.get("days"), dict):
            for day, values in data["days"].items():
                if isinstance(values, dict):
                    self.days[str(day)] = {**self._blank(), **{k: values[k] for k in self._blank() if k in values}}


def status_line(snapshot, today, language="ru"):
    """One line for the status bar: what Shura gave back today, how fast the card recovered, and the need now."""
    need = snapshot.get("need")
    yields, mib, worst = today.get("yields", 0), today.get("yielded_mib", 0), today.get("recovered_max_s", 0.0)
    level = snapshot.get("sleep_level", 0)
    if language == "en":
        parts = [f"Shura gave way {yields}x today ({mib / 1024:.1f} GiB)" if yields else "Shura has not had to give way today"]
        if yields and today.get("recovered"):
            parts.append(f"card back in <= {worst:.2f} s")
        parts.append("parked: the desk needs the card" if level >= 2 else "asleep: the desk is busy" if level >= 1 else
                     (f"desk need {need:g}" if need is not None else ""))
    else:
        parts = [f"Шура уступила {yields}× за сегодня ({mib / 1024:.1f} ГиБ)" if yields else "Шура сегодня не уступала"]
        if yields and today.get("recovered"):
            parts.append(f"карта освобождалась ≤ {worst:.2f} с")
        parts.append("выгружена: карта нужна столу" if level >= 2 else "спит: стол занят" if level >= 1 else
                     (f"нужда стола {need:g}" if need is not None else ""))
    return " · ".join(p for p in parts if p)


def local_day(timestamp=None):
    return time.strftime("%Y-%m-%d", time.localtime(timestamp if timestamp is not None else time.time()))
