"""MEMV2: the part of the gateway that listens to the desk and tells the server's VRAM governor how polite to be.

Every few seconds: read the desk's signals (desk_sources.py or the test hook file), update the need (desk_need.py), and when the
reserve or the sleep level should change, POST them to /vram/policy together with a short reason (`note`). It also keeps the
daily totals of what the server gave back, read from GET /vram/journal, and a one-line status. It never changes a setting of
the desktop, and it only runs when the VRAM policy is "governor" and deskNeed is on.

The ledger is written to disk at most every `persistSeconds` and only when it changed (about 2 KB each time), so the cost to
the SSD is a fraction of a megabyte a day.
"""

import json
import math
import os
import time

import desk_need
from desk_need import DeskNeed, Ledger, Signals

KEEP_DAYS = 30
JOURNAL_OVERLAP = 32                 # re-read this many events: a trim's recovery time is filled in after the event appears


def validate_desk_config(config):
    """Raise ValueError for a desk* setting that cannot work. Missing keys mean the defaults."""
    d = {**desk_need.DEFAULTS, **{k: v for k, v in config.items() if k in desk_need.DEFAULTS}}
    if type(d["deskNeed"]) is not bool:
        raise ValueError("Invalid guardian config: deskNeed must be true or false")
    lo, hi = desk_need.RESERVE_RANGE
    if type(d["deskReserveFloorMiB"]) is not int or not lo <= d["deskReserveFloorMiB"] <= hi:
        raise ValueError(f"Invalid guardian config: deskReserveFloorMiB must be an integer {lo}..{hi}")
    if type(d["deskDeepSleep"]) is not bool:
        raise ValueError("Invalid guardian config: deskDeepSleep must be true or false")
    for key in ("deskNeedDownSeconds", "deskIdleSeconds", "deskSleepOnSeconds", "deskSleepOffSeconds", "deskPushMinSeconds",
                "deskDeepSleepSeconds"):
        if type(d[key]) not in (int, float) or not math.isfinite(d[key]) or not 0 <= d[key] <= 86400:
            raise ValueError(f"Invalid guardian config: {key} must be a number of seconds 0..86400")
    if type(d["deskPollSeconds"]) not in (int, float) or not 0.5 <= d["deskPollSeconds"] <= 60:
        raise ValueError("Invalid guardian config: deskPollSeconds must be 0.5..60")
    for key in ("deskGpuAppBusyPercent", "deskGpuAppVramMiB", "deskBrowserVramMiB", "deskReserveStepMiB"):
        if type(d[key]) not in (int, float) or not 0 < d[key] <= 1_000_000:
            raise ValueError(f"Invalid guardian config: {key} must be a positive number")
    if type(d["deskSignalsFile"]) is not str:
        raise ValueError("Invalid guardian config: deskSignalsFile must be a path string")
    # VRAMLOGIC3: the adaptive working buffer (the server's buffer_mib range is 0..2048)
    for key in ("deskBufferQuietMiB", "deskBufferActiveMiB"):
        if type(d[key]) is not int or not 0 <= d[key] <= 2048:
            raise ValueError(f"Invalid guardian config: {key} must be an integer 0..2048 MiB")
    if d["deskBufferActiveMiB"] < d["deskBufferQuietMiB"]:
        raise ValueError("Invalid guardian config: deskBufferActiveMiB must not be below deskBufferQuietMiB")
    if type(d["deskBrowserGrowthMiB"]) is not int or not 1 <= d["deskBrowserGrowthMiB"] <= 65536:
        raise ValueError("Invalid guardian config: deskBrowserGrowthMiB must be an integer 1..65536 MiB")
    if type(d["deskBufferHoldSeconds"]) not in (int, float) or not 0 <= d["deskBufferHoldSeconds"] <= 86400:
        raise ValueError("Invalid guardian config: deskBufferHoldSeconds must be a number of seconds 0..86400")


class DeskRuntime:
    def __init__(self, config, http, sources, state_path, event, wall=time.time, persist_seconds=900, language="ru"):
        """http(method, path, body=None) -> (status, json); sources.collect(model_pids) -> Signals;
        event(kind, text) logs a gateway event."""
        self.cfg = {**desk_need.DEFAULTS, **{k: v for k, v in config.items() if k in desk_need.DEFAULTS}}
        self.http, self.sources, self.event, self.wall = http, sources, event, wall
        self.state_path, self.persist_seconds, self.language = state_path, persist_seconds, language
        self.need = DeskNeed(self.cfg)
        self.ledger = Ledger()
        self.signals = Signals()
        self.pid = None
        self.polled_at = -math.inf
        self.journal_at = -math.inf
        self.persisted_at = -math.inf
        self.dirty = False
        self.error = ""
        self.errors = {}                         # last error text per channel ("policy", "journal", ...): a repeat is not logged again
        self.caps = {"extra": True, "journal": True, "buffer": True}   # what the current server process understands (a server before MEMV2 does not know sleep_level, human_need, note, /vram/journal; one before VRAMLOGIC2 not buffer_mib)
        self._sleep_counted = 0.0
        self._load()

    # --- one step of the loop ---------------------------------------------------------------------------------------
    def step(self, now, pid, ready, model_pids=()):
        """Call every second or so with the server's pid (None = not running) and whether it is READY."""
        if pid != self.pid:
            self.pid = pid
            self.need.server_restarted()
            self.ledger.server_restarted()
            self.caps = {"extra": True, "journal": True, "buffer": True}      # a new process may be a newer server: ask again
            self.errors = {}
        if now - self.polled_at >= self.cfg["deskPollSeconds"]:
            self.polled_at = now
            try:
                self.signals = self._collect(model_pids)
            except Exception as error:  # noqa: BLE001 - a broken reader never takes the gateway down
                self.signals = Signals(errors=[f"desk: {type(error).__name__}: {error}"])
            self.need.update(now, self.signals)
            self._count_sleep()
        if pid is None or not ready:
            return
        body = self.need.push_due(now)
        if body is not None:
            self._push(now, body)
        if self.caps["journal"] and now - self.journal_at >= 5.0:
            self.journal_at = now
            self._read_journal()
        self._persist(now)

    def _collect(self, model_pids):
        path = self.cfg["deskSignalsFile"]
        if path:
            try:
                with open(path, "r", encoding="utf-8") as stream:
                    return Signals.from_json(json.load(stream))
            except FileNotFoundError:
                return Signals()
        return self.sources.collect(model_pids)

    def _shape(self, body):
        """The body for a server that does not know the MEMV2 fields: the reserve, and light sleep as the old game mode
        (`max_chunks: 0`, released again with null) when the sleep level changed."""
        shaped = {"reserve_mib": body["reserve_mib"]}
        before = self.need.pushed["sleep_level"] if self.need.pushed else 0
        if body["sleep_level"] != before:
            shaped["max_chunks"] = 0 if body["sleep_level"] >= 1 else None
        return shaped

    def _push(self, now, body):
        if not self.caps["buffer"]:
            body = {k: v for k, v in body.items() if k != "buffer_mib"}
        sent = body if self.caps["extra"] else self._shape(body)
        try:
            code, out = self.http("POST", "/vram/policy", sent)
            if code == 400 and self.caps["buffer"] and "buffer_mib" in sent and "buffer_mib" in json.dumps(out):
                # VRAMLOGIC3: a server before VRAMLOGIC2 refuses the buffer; everything else still goes
                self.caps["buffer"] = False
                self.event("desk_need_compat", "the server does not know buffer_mib: the working buffer stays the server's own until it restarts")
                body = {k: v for k, v in body.items() if k != "buffer_mib"}
                sent = body if self.caps["extra"] else self._shape(body)
                code, out = self.http("POST", "/vram/policy", sent)
            if code == 400 and self.caps["extra"] and "unknown" in json.dumps(out).lower():
                # a server before MEMV2 refuses the whole body (atomically): the reserve must go without the new fields
                self.caps["extra"] = False
                self.event("desk_need_compat", "the server does not know sleep_level/human_need/note: sending the reserve (and sleep as "
                           "max_chunks) only until it restarts")
                code, out = self.http("POST", "/vram/policy", self._shape(body))
        except (OSError, ValueError) as error:
            self._fail("policy", f"POST /vram/policy: {error}")
            return
        if code != 200:
            self._fail("policy", f"POST /vram/policy HTTP {code}: {out}")
            return
        self.errors.pop("policy", None)
        self.error = ""
        previous = self.need.pushed
        self.need.pushed_ok(now, body)
        if (previous is None or previous["reserve_mib"] != body["reserve_mib"] or previous["sleep_level"] != body["sleep_level"]
                or previous.get("buffer_mib") != body.get("buffer_mib", previous.get("buffer_mib"))):
            buffer = f" buffer={body['buffer_mib']} MiB" if "buffer_mib" in body else ""
            self.event("desk_need", f"need={body['human_need']} reserve={body['reserve_mib']} MiB{buffer} sleep={body['sleep_level']}: "
                       f"{body['note']}" + (f"; buffer: {self.need.buffer_reason}" if "buffer_mib" in body else ""))

    def _read_journal(self):
        try:
            code, out = self.http("GET", f"/vram/journal?since={max(0, self.ledger.last_id - JOURNAL_OVERLAP)}&limit=200")
        except (OSError, ValueError) as error:
            self._fail("journal", f"GET /vram/journal: {error}")
            return
        if code == 404:
            self.caps["journal"] = False                 # a server before MEMV2: no journal; stop asking until it restarts
            self.event("desk_need_compat", "the server has no /vram/journal: daily totals of concessions are not collected from it")
            return
        if code != 200 or not isinstance(out, dict) or not isinstance(out.get("events"), list):
            self._fail("journal", f"GET /vram/journal HTTP {code}")
            return
        self.errors.pop("journal", None)
        last = out.get("last_id")
        if type(last) is int and last < self.ledger.last_id:
            self.ledger.server_restarted()      # the server forgot its journal (a restart we did not see)
        before = json.dumps(self.ledger.day(self.today()), sort_keys=True)
        self.ledger.add_events(self.today(), out["events"])
        if json.dumps(self.ledger.day(self.today()), sort_keys=True) != before:
            self.dirty = True

    def _fail(self, channel, text):
        """Log a failure once per channel and text (a failing POST and a failing GET must not take turns logging)."""
        if self.errors.get(channel) != text:
            self.event("desk_need_failed", text)
        self.errors[channel] = text
        self.error = text

    def _count_sleep(self):
        total = self.need.sleeping_s
        if total > self._sleep_counted:
            self.ledger.day(self.today())["sleeping_s"] += total - self._sleep_counted
            self._sleep_counted = total
            self.dirty = True

    # --- ledger on disk ---------------------------------------------------------------------------------------------
    def today(self):
        return desk_need.local_day(self.wall())

    def _load(self):
        try:
            with open(self.state_path, "r", encoding="utf-8") as stream:
                self.ledger.load(json.load(stream))
        except (OSError, ValueError):
            pass

    def _persist(self, now, force=False):
        if not self.dirty or (not force and now - self.persisted_at < self.persist_seconds):
            return
        days = sorted(self.ledger.days)
        for old in days[:-KEEP_DAYS]:
            del self.ledger.days[old]
        tmp = f"{self.state_path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as stream:
                json.dump(self.ledger.to_json(), stream)
            os.replace(tmp, self.state_path)
            self.persisted_at, self.dirty = now, False
        except OSError as error:
            self.persisted_at = now        # do not retry every second
            self._fail("ledger", f"ledger write: {error}")

    def flush(self):
        self._persist(time.monotonic(), force=True)

    # --- what the status shows --------------------------------------------------------------------------------------
    def snapshot(self):
        snap = self.need.snapshot()
        today = dict(self.ledger.day(self.today()))
        sig = self.signals
        return {
            "enabled": True, **snap, "today": today, "status_line": desk_need.status_line(snap, today, self.language),
            "signals": {"game": sig.game, "gpu_app": sig.gpu_app, "call": sig.call, "video": sig.video,
                        "browser_vram_mib": sig.browser_vram_mib, "idle_s": sig.idle_s, "locked": sig.locked, "errors": sig.errors},
            "error": self.error, "server_supports": dict(self.caps),
        }
