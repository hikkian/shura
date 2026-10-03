"""Rare slot changes: pure per-model policy and a separate transactional mechanism."""
import math

DEFAULTS = {"slotGears": False, "slotGearLevels": [26, 24, 20, 16], "slotGearCorridorMiB": 250,
            "slotGearDownSeconds": 60, "slotGearUpSeconds": 1800, "slotGearMiBPerSlot": 36,
            "slotGearRestartSeconds": 20, "slotGearChangesPerHour": 2, "slotGearChangesPerDay": 6}


class Budget:
    def __init__(self):
        self.history = []


class Ratchet:
    def __init__(self, config, initial, budget=None):
        self.config = {**DEFAULTS, **config}
        self.levels = sorted({int(x) for x in self.config["slotGearLevels"] if 16 <= int(x) <= initial} | {initial}, reverse=True)
        if initial < 16:
            raise ValueError("slot gears must never go below 16")
        self.current = initial
        self.low_since = self.calm_since = None
        self.budget = budget or Budget()
        self.retry_at = -math.inf
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
        i = self.levels.index(self.current)
        low = free_mib < c["slotGearCorridorMiB"]
        high = i > 0 and free_mib >= c["slotGearCorridorMiB"] + (self.levels[i-1] - self.current) * c["slotGearMiBPerSlot"]
        self.low_since = (now if self.low_since is None else self.low_since) if low else None
        self.calm_since = (now if self.calm_since is None else self.calm_since) if high else None
        if busy or not ready or now < self.retry_at:
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

    def completed(self, now, target, ok):
        self.history.append(now)  # Failed attempts also cost pauses; they cannot bypass the budget.
        if ok:
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
    False means the old server and old slots were restored; a rollback failure raises.
    """
    old = backend.current_slots()
    snapshot = backend.save()
    if snapshot is None:
        return False
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
