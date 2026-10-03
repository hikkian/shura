"""Opt-in CPU courtesy. No irreversible scheduling change on an unprivileged host."""
import math
import os
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULTS = {"politeModel": False, "politeCalmSeconds": 30, "politeChangeSeconds": 5,
            "politePsiHigh": 10.0, "politePsiLow": 2.0, "politeLagHighMs": 2.0,
            "politeLagLowMs": 0.5, "politeOtherCpuHigh": 2.5, "politeOtherCpuLow": 1.0}


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
        psi = next(float(item.split("=")[1]) for line in pressure if line.startswith("some ")
                   for item in line.split() if item.startswith("avg10="))
        numbers = list(map(int, self.root.joinpath("stat").read_text().splitlines()[0].split()[1:]))
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
            if dt > 0:
                removed = sum(max(0, ticks - prior_excluded[key]) for key, ticks in excluded.items()
                              if key in prior_excluded)
                other = max(0.0, (busy - prior_busy - removed) / os.sysconf("SC_CLK_TCK") / dt)
        self.previous = now, busy, excluded
        result = Sample(psi, lag_ms, other)
        if not all(math.isfinite(value) and value >= 0 for value in asdict(result).values()):
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
        high = (sample.psi >= c["politePsiHigh"] or sample.lag_ms >= c["politeLagHighMs"]
                or sample.other_busy_cpus >= c["politeOtherCpuHigh"])
        calm = (sample.psi <= c["politePsiLow"] and sample.lag_ms <= c["politeLagLowMs"]
                and sample.other_busy_cpus <= c["politeOtherCpuLow"])
        self.calm_since = (self.calm_since if self.calm_since is not None else now) if calm else None
        if now - self.changed < c["politeChangeSeconds"]:
            return self.level
        if high:
            return min(3, max(self.level + 1, self.level * 2))
        if self.calm_since is not None and now - self.calm_since >= c["politeCalmSeconds"]:
            return max(0, self.level - 1)
        return self.level

    def committed(self, now, level):
        self.level, self.changed, self.calm_since = level, now, None


class Scheduler:
    def __init__(self, root="/proc", system=os):
        self.root, self.system = Path(root), system

    def reversible(self):
        # Returning from nice19 / SCHED_IDLE to nice10 needs CAP_SYS_NICE or RLIMIT_NICE >= 10.
        try:
            caps = next(line.split()[1] for line in self.root.joinpath("self/status").read_text().splitlines()
                        if line.startswith("CapEff:"))
            return bool(int(caps, 16) & (1 << 23)) or resource.getrlimit(resource.RLIMIT_NICE)[0] >= 10
        except (OSError, ValueError, StopIteration, AttributeError):
            return False

    def apply(self, pid, identity, level, full_cpus):
        if not self.reversible():
            raise PermissionError("RLIMIT_NICE/CAP_SYS_NICE does not allow return to L0")
        if proc_ticks(self.root / str(pid) / "stat")[1] != identity:
            raise ProcessLookupError("server PID was reused")
        tids = sorted(int(p.name) for p in (self.root / str(pid) / "task").iterdir())
        snapshots = []
        try:
            for tid in tids:
                tid_start = proc_ticks(self.root / str(pid) / "task" / str(tid) / "stat")[1]
                snapshots.append((tid, tid_start, self.system.sched_getscheduler(tid),
                                  self.system.getpriority(os.PRIO_PROCESS, tid), self.system.sched_getaffinity(tid)))
                if proc_ticks(self.root / str(pid) / "task" / str(tid) / "stat")[1] != tid_start:
                    raise ProcessLookupError("server TID was reused")
                self.system.sched_setscheduler(tid, os.SCHED_IDLE if level >= 2 else os.SCHED_OTHER, os.sched_param(0))
                self.system.setpriority(os.PRIO_PROCESS, tid, 19 if level >= 1 else 10)
                cpus = set(sorted(full_cpus)[:max(1, len(full_cpus) // 2)]) if level == 3 else full_cpus
                self.system.sched_setaffinity(tid, cpus)
        except OSError:
            # Restore surviving TIDs; any restoration error is surfaced, never hidden.
            for tid, tid_start, policy, nice, cpus in reversed(snapshots):
                try:
                    if proc_ticks(self.root / str(pid) / "task" / str(tid) / "stat")[1] != tid_start:
                        continue
                    self.system.sched_setscheduler(tid, policy, os.sched_param(0))
                    self.system.setpriority(os.PRIO_PROCESS, tid, nice)
                    self.system.sched_setaffinity(tid, cpus)
                except ProcessLookupError:
                    pass
            raise


class Controller:
    def __init__(self, config, scheduler=None, sensors=None):
        self.config = {**DEFAULTS, **config}
        self.policy = Policy(self.config)
        self.scheduler = scheduler or Scheduler()
        self.sensors = sensors or Sensors()
        self.enabled = self.config["politeModel"] is True
        self.available = self.scheduler.reversible() if self.enabled else False
        self.error = "" if self.available or not self.enabled else "Cannot restore nice10/SCHED_OTHER without RLIMIT_NICE or CAP_SYS_NICE"
        self.identity = None
        self.full_cpus = None

    def tick(self, now, pid, lag_ms, own_pids=()):
        if not self.enabled or not self.available or pid is None:
            return
        try:
            identity = proc_ticks(self.scheduler.root / str(pid) / "stat")[1]
            if self.identity != (pid, identity):
                self.identity = (pid, identity)
                self.full_cpus = self.scheduler.system.sched_getaffinity(0 if self.policy.level == 3 else pid)
            sample = self.sensors.sample(pid, lag_ms, own_pids)
            level = self.policy.target(now, sample)
            # Also synchronize threads created since the last tick or server restart.
            self.scheduler.apply(pid, identity, level, self.full_cpus)
            if level != self.policy.level:
                self.policy.committed(now, level)
            self.error = ""
        except (OSError, ValueError, IndexError, StopIteration) as e:
            self.error = str(e)
            # L0 restoration is attempted while permissions are still known reversible.
            if self.identity and self.full_cpus:
                try:
                    self.scheduler.apply(pid, self.identity[1], 0, self.full_cpus)
                except OSError as rollback:
                    self.error += "; rollback: " + str(rollback)
            self.available = False

    def stale(self, pid):
        self.error = "Wakeup probe is stale; controller disabled"
        if self.available and pid is not None and self.identity and self.full_cpus:
            try:
                self.scheduler.apply(pid, self.identity[1], 0, self.full_cpus)
                self.policy.committed(time.monotonic(), 0)
            except OSError as error:
                self.error += "; L0 restore failed: " + str(error)
        self.available = False

    def snapshot(self):
        return {"enabled": self.enabled, "available": self.available, "level": self.policy.level,
                "error": self.error, "sample": asdict(self.policy.sample) if self.policy.sample else None}
