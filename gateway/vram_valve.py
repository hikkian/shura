"""Pure policy for the opt-in live expert-cache VRAM valve."""
import math


DEFAULTS = {
    "vramValve": False,
    "vramValveLevels": [18, 16, 8],
    "vramValveHysteresisMiB": 200,
    "vramValveUpSeconds": 60,
    "vramValveParkSeconds": 2,
    "vramValveResizeRetrySeconds": 2,
}

SLOT_COST_MIN_MIB = 10.0
SLOT_COST_MAX_MIB = 80.0
SLOT_COST_FALLBACK_MIB = 36.0


def valid_slot_cost(value):
    """Return a finite, plausible MiB/slot price, or None for untrusted input."""
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    value = float(value)
    return value if SLOT_COST_MIN_MIB <= value <= SLOT_COST_MAX_MIB else None


class SlotCostEstimator:
    """Keep three accepted resize measurements per model in memory only."""

    def __init__(self):
        self.measured = {}

    def resolve(self, model_id, model_config, default_value):
        error = None
        if "moeCacheSlotMiB" in model_config:
            configured = valid_slot_cost(model_config.get("moeCacheSlotMiB"))
            if configured is not None:
                return {"value_mib": configured, "source": "config", "samples": len(self.measured.get(model_id, ())),
                        "error": None}
            error = "invalid moeCacheSlotMiB; falling back to learned/default price"
        samples = self.measured.get(model_id, ())
        if samples:
            return {"value_mib": sum(samples) / len(samples), "source": "measured",
                    "samples": len(samples), "error": error}
        default = valid_slot_cost(default_value)
        if default is None:
            error = error or "invalid slotGearMiBPerSlot; using safe fallback"
            default = SLOT_COST_FALLBACK_MIB
        return {"value_mib": default, "source": "default", "samples": 0, "error": error}

    @staticmethod
    def _settled_mean(values):
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            return None, "fewer than two VRAM probes"
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in values):
            return None, "invalid VRAM probe"
        values = [float(value) for value in values]
        if max(values) - min(values) > 64:
            return None, "VRAM probe spread exceeded 64 MiB"
        if abs(values[-1] - values[-2]) > 16:
            return None, "VRAM did not settle within 16 MiB"
        return (values[-1] + values[-2]) / 2, None

    def observe_resize(self, model_id, old_slots, new_slots, before_samples, after_samples,
                       foreign_before, foreign_after):
        """Learn absolute MiB/slot only from settled, uncontended, plausible resizes."""
        if type(old_slots) is not int or type(new_slots) is not int or old_slots == new_slots:
            return {"accepted": False, "reason": "resize did not change an integer slot count"}
        before, error = self._settled_mean(before_samples)
        if error:
            return {"accepted": False, "reason": "before: " + error}
        after, error = self._settled_mean(after_samples)
        if error:
            return {"accepted": False, "reason": "after: " + error}
        if not isinstance(foreign_before, dict) or not isinstance(foreign_after, dict):
            return {"accepted": False, "reason": "foreign-process telemetry unavailable"}
        if foreign_before != foreign_after:
            return {"accepted": False, "reason": "foreign GPU load changed during resize"}
        slot_delta = abs(old_slots - new_slots)
        observed = abs(after - before) / slot_delta
        if not math.isfinite(observed) or not SLOT_COST_MIN_MIB <= observed <= SLOT_COST_MAX_MIB:
            return {"accepted": False, "reason": f"measured price {observed:.3f} MiB/slot outside 10-80"}
        samples = self.measured.setdefault(model_id, [])
        samples.append(observed)
        del samples[:-3]
        return {"accepted": True, "value_mib": sum(samples) / len(samples),
                "measurement_mib": observed, "samples": len(samples), "reason": "settled resize"}


class Valve:
    """Pick one supported gear at a time; the caller applies it only between requests."""

    def __init__(self, config, maximum, initial=None):
        self.config = {**DEFAULTS, **config}
        if type(maximum) is not int or maximum < 0:
            raise ValueError("maximum cache slots must be a non-negative integer")
        levels = {maximum}
        # Zero is an exceptional reported state (for example, a failed resize
        # rollback), not a default gear. Keep it only when explicitly configured
        # or when adopting an already-zero server state.
        if initial == 0:
            levels.add(0)
        configured = self.config["vramValveLevels"]
        if not isinstance(configured, list):
            raise ValueError("vramValveLevels must be a list")
        for value in configured:
            if type(value) is int and 0 <= value <= maximum:
                levels.add(value)
        self.levels = sorted(levels, reverse=True)
        self.maximum = maximum
        self.current = maximum if initial is None else initial
        if type(self.current) is not int or self.current not in self.levels:
            raise ValueError("initial cache level is not supported")
        self.low_since = None
        self.calm_since = None
        self.retry_at = -math.inf
        self.last_change = -math.inf
        self.exhausted_since = None
        self.reason = "configured baseline"
        self.slot_cost_mib = None
        self.slot_cost_source = None
        self.slot_cost_samples = 0

    def target(self, now, free_mib, reserve_mib, per_slot_mib=36.0, busy=False, ready=True):
        if (free_mib is None or not isinstance(free_mib, (int, float))
                or not math.isfinite(free_mib) or free_mib < 0):
            self.low_since = self.calm_since = None
            self.reason = "fresh VRAM unavailable"
            return None
        if (type(reserve_mib) is not int or reserve_mib < 1
                or not math.isfinite(per_slot_mib) or per_slot_mib <= 0):
            self.low_since = self.calm_since = None
            self.reason = "invalid reserve or slot cost"
            return None
        if self.current not in self.levels:
            self.levels = sorted(set(self.levels) | {self.current}, reverse=True)
        if not ready:
            self.low_since = self.calm_since = None
            self.reason = "game or model state blocks resizing"
            return None

        idx = self.levels.index(self.current)
        minimum = self.levels[-1]
        if free_mib < reserve_mib and self.current > minimum:
            self.low_since = now if self.low_since is None else self.low_since
            self.calm_since = None
            self.exhausted_since = None
            if busy or now < self.retry_at:
                self.reason = "low VRAM; waiting for request boundary or retry window"
                return None
            # One transition per tick, selecting the highest next gear that can restore the reserve.
            candidates = self.levels[idx + 1:]
            target = next((n for n in candidates
                           if free_mib + (self.current - n) * per_slot_mib >= reserve_mib),
                          candidates[-1] if candidates else None)
            if target is None:
                self.reason = f"minimum valve level {minimum} reached; no lower configured gear"
                return None
            self.reason = f"free {free_mib:.0f} MiB below reserve {reserve_mib}; shrink {self.current}->{target}"
            return target

        self.low_since = None
        if self.current == minimum and free_mib < reserve_mib:
            self.calm_since = None
            self.exhausted_since = now if self.exhausted_since is None else self.exhausted_since
            self.reason = f"reserve still low at minimum {minimum} slots; hand off to vramGuard"
            return None
        self.exhausted_since = None

        if idx > 0:
            higher = self.levels[idx - 1]
            required = reserve_mib + (higher - self.current) * per_slot_mib + self.config["vramValveHysteresisMiB"]
            if free_mib >= required:
                self.calm_since = now if self.calm_since is None else self.calm_since
                if busy:
                    self.reason = f"free {free_mib:.0f} MiB; waiting for request boundary before growing"
                elif now < self.retry_at:
                    self.reason = "waiting for resize retry window"
                elif now - self.calm_since >= self.config["vramValveUpSeconds"]:
                    self.reason = f"free {free_mib:.0f} MiB stable; grow {self.current}->{higher}"
                    return higher
                else:
                    self.reason = f"free {free_mib:.0f} MiB; stable window {now - self.calm_since:.0f}/{self.config['vramValveUpSeconds']} s"
            else:
                self.calm_since = None
                self.reason = f"holding {self.current} slots; need {required:.0f} MiB to grow"
        else:
            self.calm_since = None
            self.reason = "configured maximum reached"
        return None

    def completed(self, now, target, ok):
        if ok:
            self.current = target
            self.last_change = now
            self.low_since = self.calm_since = None
            self.exhausted_since = None
            self.reason = f"cache now has {target} slots"
        else:
            self.retry_at = now + self.config["vramValveResizeRetrySeconds"]
            self.reason = "resize failed; retaining reported server slot count"

    def park_due(self, now):
        return (self.current == self.levels[-1] and self.exhausted_since is not None
                and now - self.exhausted_since >= self.config["vramValveParkSeconds"])

    def snapshot(self):
        return {"slots": self.current, "configured_max": self.maximum, "reason": self.reason,
                "levels": list(self.levels), "slot_cost_mib": self.slot_cost_mib,
                "slot_cost_source": self.slot_cost_source, "slot_cost_samples": self.slot_cost_samples}
