"""Opt-in CPU courtesy. No irreversible scheduling change on an unprivileged host."""

import math
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from cpu_cgroup import Scheduler

DEFAULTS = {
    "politeModel": False,
    "politeMode": "static",
    "politeStaticWeight": 20,
    "politeWeights": [100, 50, 20, 5],
    "politeUnit": "ai-gateway.service",
    "politeCalmSeconds": 30,
    "politeChangeSeconds": 5,
    "politePsiHigh": 10.0,
    "politePsiLow": 2.0,
    "politeLagHighMs": 2.0,
    "politeLagLowMs": 0.5,
    "politeOtherCpuHigh": 2.5,
    "politeOtherCpuLow": 1.0,
}


@dataclass
class Sample:
    psi: float
    lag_ms: float
    other_busy_cpus: float


def proc_ticks(path):
    fields = Path(path).read_text().rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12]), int(fields[19])


class Sensors:
    """Injectable proc tree and clock; own process and model CPU time are excluded."""

    def __init__(self, root="/proc", clock=time.monotonic):
        self.root, self.clock, self.previous = Path(root), clock, None

    def sample(self, model_pid, lag_ms, own_pids=()):
        pressure = self.root.joinpath("pressure/cpu").read_text().splitlines()
        psi = next(
            float(item.split("=")[1])
            for line in pressure
            if line.startswith("some ")
            for item in line.split()
            if item.startswith("avg10=")
        )
        numbers = list(
            map(int, self.root.joinpath("stat").read_text().splitlines()[0].split()[1:])
        )
        # guest is already counted in user/nice; iowait is not CPU execution.
        busy = sum(numbers[i] for i in (0, 1, 2, 5, 6, 7) if i < len(numbers))
        excluded = {}
        for pid in set(own_pids) | {model_pid}:
            if pid is None:
                continue
            try:
                ticks, start = proc_ticks(self.root / str(pid) / "stat")
                excluded[(pid, start)] = ticks
            except (OSError, ValueError, IndexError):
                continue
        now = self.clock()
        other = 0.0
        if self.previous:
            before, prior_busy, prior_excluded = self.previous
            dt = now - before
            # A process that appeared or vanished (model restart, probe restart) leaves a window in
            # which its CPU time cannot be split from the rest: discard it rather than charge a
            # whole new server's start-up to "other".
            if dt > 0 and set(excluded) == set(prior_excluded):
                removed = sum(
                    max(0, ticks - prior_excluded[key])
                    for key, ticks in excluded.items()
                    if key in prior_excluded
                )
                other = max(
                    0.0, (busy - prior_busy - removed) / os.sysconf("SC_CLK_TCK") / dt
                )
        self.previous = now, busy, excluded
        result = Sample(psi, lag_ms, other)
        if not all(
            math.isfinite(value) and value >= 0 for value in asdict(result).values()
        ):
            raise ValueError("invalid CPU sensor sample")
        return result


class Policy:
    def __init__(self, config):
        self.config = {**DEFAULTS, **config}
        self.level, self.changed, self.calm_since = 0, -math.inf, None
        self.sample = None

    def target(self, now, sample):
        self.sample = sample
        c = self.config
        high = (
            sample.psi >= c["politePsiHigh"]
            or sample.lag_ms >= c["politeLagHighMs"]
            or sample.other_busy_cpus >= c["politeOtherCpuHigh"]
        )
        calm = (
            sample.psi <= c["politePsiLow"]
            and sample.lag_ms <= c["politeLagLowMs"]
            and sample.other_busy_cpus <= c["politeOtherCpuLow"]
        )
        self.calm_since = (
            (self.calm_since if self.calm_since is not None else now) if calm else None
        )
        if now - self.changed < c["politeChangeSeconds"]:
            return self.level
        if high:
            return min(3, max(self.level + 1, self.level * 2))
        if (
            self.calm_since is not None
            and now - self.calm_since >= c["politeCalmSeconds"]
        ):
            return max(0, self.level - 1)
        return self.level

    def committed(self, now, level):
        self.level, self.changed, self.calm_since = level, now, None


class Controller:
    RETRY_FIRST, RETRY_MAX, FAILURE_FORGET = 5.0, 300.0, 120.0

    def __init__(self, config, scheduler=None, sensors=None):
        self.config = {**DEFAULTS, **config}
        self.policy = Policy(self.config)
        self.scheduler = scheduler or Scheduler(
            self.config, state_key=self.config.get("politeStateKey", "gateway")
        )
        self.sensors = sensors or Sensors()
        self.lock = threading.RLock()
        self.closed = False
        self.retry_at, self.failures, self.failed_at = None, 0, -math.inf
        self.enabled = self.config["politeModel"] is True
        self.error = ""
        problem = self._config_problem()
        if problem:
            self.available, self.error = False, problem
        else:
            self.available = self.scheduler.reversible() if self.enabled else False
            if self.enabled and not self.available:
                self.error = "cgroup cpu controller or owned user unit unavailable"
        self.identity = None
        self.full_cpus = None

    def _config_problem(self):
        """Settings that can never work are reported once; they do not retry."""
        model = self.config["politeModel"]
        if model is not True and model not in (False, None, 0, ""):
            return "politeModel must be the JSON value true; polite mode stays off"
        if not self.enabled:
            return ""
        if self.config["politeMode"] not in ("static", "dynamic"):
            return "politeMode must be 'static' or 'dynamic'"
        if self.config["politeMode"] == "static":
            weight = self.config["politeStaticWeight"]
            if type(weight) is not int or not 1 <= weight <= 10000:
                return "politeStaticWeight must be an integer from 1 to 10000"
        return ""

    def _static_weight(self):
        return int(self.config["politeStaticWeight"])

    # A transient failure suspends the controller with a growing back-off instead of ending it.
    def retry_due(self, now):
        return (
            self.enabled
            and not self.closed
            and not self.available
            and self.retry_at is not None
            and now >= self.retry_at
        )

    @property
    def recoverable(self):
        return self.enabled and not self.closed and (self.available or self.retry_at is not None)

    def _suspend(self, now):
        self.available = False
        if now - self.failed_at > self.FAILURE_FORGET:
            self.failures = 0
        self.failed_at = now
        self.retry_at = now + min(self.RETRY_MAX, self.RETRY_FIRST * 2**self.failures)
        self.failures += 1

    def _restore(self, pid):
        """Undo every change independently: one failing step must not skip the others."""
        problems = []
        identity = self.identity[1] if self.identity and self.identity[0] == pid else None
        try:
            self.scheduler.restore_affinity(pid, identity, self.full_cpus)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            problems.append("affinity restore failed: " + str(error))
        try:
            self.scheduler.close()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            problems.append("CPU restore failed: " + str(error))
        self.policy.committed(time.monotonic(), 0)
        return "; ".join(problems)

    def _fail(self, now, pid, reason):
        self.error = reason
        problems = self._restore(pid)
        if problems:
            self.error += "; " + problems
        self._suspend(now)

    def attach(self, pid):
        with self.lock:
            if not self.enabled or not self.available or self.closed:
                return
            try:
                identity = proc_ticks(self.scheduler.root / str(pid) / "stat")[1]
                full = self.scheduler.system.sched_getaffinity(pid)
                if self.config["politeMode"] == "static":
                    self.scheduler.weights = [self._static_weight()] * 4
                self.scheduler.apply(pid, identity, self.policy.level, full)
                self.identity, self.full_cpus = (pid, identity), full
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                self._fail(time.monotonic(), pid, str(error))

    def tick(self, now, pid, lag_ms, own_pids=()):
        with self.lock:
            if not self.enabled or self.closed or pid is None:
                return
            if not self.available:
                if not self.retry_due(now):
                    return
                self.available, self.retry_at, self.error = True, None, ""
                self.identity = None
            try:
                identity = proc_ticks(self.scheduler.root / str(pid) / "stat")[1]
                if self.identity != (pid, identity):
                    self.identity = (pid, identity)
                    self.full_cpus = self.scheduler.system.sched_getaffinity(
                        0 if self.policy.level == 3 else pid
                    )
                if self.config["politeMode"] == "static":
                    self.scheduler.weights = [self._static_weight()] * 4
                    self.scheduler.apply(pid, identity, 0, self.full_cpus)
                    self.error = ""
                    return  # No PSI/latency/AIMD dependency for the simple constant-weight mode.
                sample = self.sensors.sample(pid, lag_ms, own_pids)
                level = self.policy.target(now, sample)
                # Also synchronize threads created since the last tick or server restart.
                self.scheduler.apply(pid, identity, level, self.full_cpus)
                if level != self.policy.level:
                    self.policy.committed(now, level)
                self.error = ""
            except (
                OSError,
                ValueError,
                IndexError,
                StopIteration,
                RuntimeError,
                subprocess.SubprocessError,
            ) as e:
                self._fail(now, pid, str(e))

    def stale(self, pid):
        with self.lock:
            if self.available:
                self.error = "Wakeup probe is stale; controller disabled"
                problems = self._restore(pid)
                if problems:
                    self.error += "; " + problems
                self._suspend(time.monotonic())

    def shutdown(self):
        """Final restore at exit; afterwards no tick may re-apply anything."""
        with self.lock:
            self.closed = True
            pid = self.identity[0] if self.identity else None
            return self._restore(pid)

    def snapshot(self):
        return {
            "enabled": self.enabled,
            "available": self.available,
            "level": self.policy.level,
            "mode": self.config["politeMode"],
            "weight": getattr(self.scheduler, "weight", None),
            "change_ms": getattr(self.scheduler, "last_ms", None),
            "other_busy_cpu_threshold": self.config["politeOtherCpuHigh"],
            "error": self.error,
            "sample": asdict(self.policy.sample) if self.policy.sample else None,
        }
