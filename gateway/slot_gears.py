"""Rare slot changes: pure per-model policy and a separate transactional mechanism."""
import math

DEFAULTS = {"slotGears": False, "slotGearLevels": [26, 24, 20, 16], "slotGearCorridorMiB": 250,
            "slotGearDownSeconds": 60, "slotGearUpSeconds": 1800, "slotGearMiBPerSlot": 36,
            "slotGearRestartSeconds": 20, "slotGearChangesPerHour": 2, "slotGearChangesPerDay": 6,
            # An up-shift followed by a down-shift this soon means the per-slot cost estimate is too low.
            "slotGearPingPongSeconds": 3600, "slotGearUpBackoffSeconds": 21600,
            # Putting the old gear back after a failed change may take longer than the change itself.
            "slotGearRollbackSeconds": 120}


class Budget:
    def __init__(self):
        self.history = []


class Ratchet:
    def __init__(self, config, initial, budget=None):
        self.config = {**DEFAULTS, **config}
        if type(initial) is not int or initial < 16:
            raise ValueError("slot gears must never go below 16")
        levels = {initial}
        for item in self.config["slotGearLevels"]:
            try:
                level = int(item)
            except (TypeError, ValueError):
                continue  # one bad entry must not disable the whole controller
            if 16 <= level <= initial:
                levels.add(level)
        self.levels = sorted(levels, reverse=True)
        self.current = initial
        self.low_since = self.calm_since = None
        self.budget = budget or Budget()
        self.retry_at = -math.inf
        self.last_up = None
        self.up_blocked_until = -math.inf
        self.reason = "static baseline"

    @property
    def history(self):
        return self.budget.history

    @history.setter
    def history(self, value):
        self.budget.history = value

    def target(self, now, free_mib, busy=False, ready=True):
        c = self.config
        self.history = [t for t in self.history if now - t < 86400]
        if free_mib is None or not math.isfinite(free_mib) or free_mib < 0:
            self.low_since = self.calm_since = None
            self.reason = "fresh VRAM data unavailable"
            return None
        if self.reason == "fresh VRAM data unavailable":
            self.reason = "static baseline"
        if self.current not in self.levels:  # e.g. the level list was edited between runs
            self.levels = sorted(set(self.levels) | {self.current}, reverse=True)
        if not ready:
            # Loading, vision mode or a game: VRAM readings do not describe the normal text server.
            self.low_since = self.calm_since = None
            return None
        i = self.levels.index(self.current)
        low = free_mib < c["slotGearCorridorMiB"]
        high = i > 0 and free_mib >= c["slotGearCorridorMiB"] + (self.levels[i-1] - self.current) * c["slotGearMiBPerSlot"]
        if high and now < self.up_blocked_until:
            high = False
            self.reason = "up-shift paused after a ping-pong"
        self.low_since = (now if self.low_since is None else self.low_since) if low else None
        self.calm_since = (now if self.calm_since is None else self.calm_since) if high else None
        if busy or now < self.retry_at:
            return None
        if sum(now - t < 3600 for t in self.history) >= c["slotGearChangesPerHour"] or len(self.history) >= c["slotGearChangesPerDay"]:
            self.reason = "change budget exhausted"
            return None
        if self.low_since is not None and now - self.low_since >= c["slotGearDownSeconds"] and i + 1 < len(self.levels):
            self.reason = "sustained desktop pressure"
            return self.levels[i + 1]
        if self.calm_since is not None and now - self.calm_since >= c["slotGearUpSeconds"] and i > 0:
            self.reason = "30-minute calm"
            return self.levels[i - 1]
        return None

    def deferred(self, now, reason):
        """The change was refused before anything restarted: no budget is charged, only a short pause."""
        self.retry_at = now + 30
        self.reason = reason

    def completed(self, now, target, ok):
        self.history.append(now)  # Failed attempts also cost pauses; they cannot bypass the budget.
        if ok:
            if target > self.current:
                self.last_up = now
            elif self.last_up is not None and now - self.last_up < self.config["slotGearPingPongSeconds"]:
                self.up_blocked_until = now + self.config["slotGearUpBackoffSeconds"]
            self.current = target
            self.low_since = self.calm_since = None
        else:
            self.retry_at = now + 30
            self.reason = "restart failed; old gear restored"

    def snapshot(self):
        return {"slots": self.current, "reason": self.reason, "changes_today": len(self.history)}


class RollbackFailed(RuntimeError):
    pass


def apply_slots(target, backend):
    """Backend keeps one immutable A2 snapshot until new server restoration succeeds.

    No configuration change or shutdown is permitted before a successful snapshot.
    None means the change was refused before anything was touched, False that the old server and old
    slots were restored after a failed attempt; a rollback failure raises.
    """
    old = backend.current_slots()
    snapshot = backend.save()
    if snapshot is None:
        return None
    release = False
    try:
        if not backend.restart(target, snapshot):
            raise RuntimeError("new gear failed to restore")
        release = True
        return True
    except BaseException as error:
        if not backend.restart(old, snapshot):
            raise RollbackFailed("slot restart and rollback both failed; A2 snapshot retained") from error
        release = True
        if not isinstance(error, Exception):
            raise
        return False
    finally:
        if release:
            backend.release(snapshot)
