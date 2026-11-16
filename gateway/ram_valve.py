"""Pure policy for the opt-in RAM valve: what to do when system memory runs short while a model is loaded.

Everything here is off by default (`ramGuard` false, `llamaOomScoreAdj` 0, `ramCacheBudgetPct` 0, `ssdWriteDailyLimitGB` 0 means
"no limit" only when the guard is off). Readers take paths, so the tests run on fake /proc and /sys trees.

Levels, from least to most severe: ok, tight, pressing, critical. A level is entered when a reading crosses its threshold and left
only when the reading recovers past the same threshold plus the hysteresis, so the state does not flap.
"""
import json
import math
import os
import threading
import time
from pathlib import Path


DEFAULTS = {
    "ramGuard": False,
    "ramGuardPollSeconds": 5,
    "ramGuardHysteresisGB": 0.5,
    "ramGuardTightMemAvailableGB": 6.0,
    "ramGuardPressingMemAvailableGB": 3.0,
    "ramGuardCriticalMemAvailableGB": 1.5,
    "ramGuardPsiFullTight": 5.0,
    "ramGuardPsiFullPressing": 15.0,
    "ramGuardPsiFullCritical": 40.0,
    "ramGuardZramPctTight": 60.0,
    "ramGuardZramPctPressing": 80.0,
    "ramGuardZramPctCritical": 90.0,
    "llamaOomScoreAdj": 0,              # 0 = do not write; recommended 800 (the model is killed before the desktop)
    "ramCacheBudgetPct": 0,             # 0 = no budget; recommended 10 (about 3 GB on a 31 GB machine)
    "ssdWriteDailyLimitGB": 8,          # slot saves per day (RAMFIX1: 8 by default as on the live machine); 0 = no limit
    "ramGuardProcRoot": "/proc",        # where the readings come from (tests and the live check point these at fake trees)
    "ramGuardSysRoot": "/sys",
    "ramGuardIdleSeconds": 30,          # pressing acts only after the model has been idle this long
    "ramValveDeferChars": 32000,        # at tight, a request that adds more than this many characters waits (about 8k tokens)
}

LEVELS = ("ok", "tight", "pressing", "critical")
OOM_MIN, OOM_MAX = -1000, 1000
PSI_HYSTERESIS_PCT = 1.0      # PSI full avg10 is a percentage: leave a level only 1 point below its entry threshold
ZRAM_HYSTERESIS_PCT = 2.0     # zram fill is a percentage: leave a level only 2 points below its entry threshold
# ~63 MiB plus ~2 KiB per token for each context checkpoint (REPORT-RAM0b, measured 3k-98k)
CHECKPOINT_BASE_MIB = 63.0
CHECKPOINT_PER_TOKEN_KIB = 2.0


def parse_meminfo_available_gb(text):
    """MemAvailable in GB from /proc/meminfo text, or None when the line is missing or not a plausible number."""
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                return int(fields[1]) / (1024 * 1024)
            return None
    return None


def parse_psi_full_avg10(text):
    """`full avg10` from /proc/pressure/memory text, or None when absent."""
    for line in text.splitlines():
        if line.startswith("full "):
            for part in line.split():
                if part.startswith("avg10="):
                    try:
                        value = float(part.split("=", 1)[1])
                    except ValueError:
                        return None
                    return value if math.isfinite(value) and value >= 0 else None
    return None


def read_readings(proc_root="/proc", sys_root="/sys"):
    """(mem_available_gb, psi_full_avg10, zram_used_percent); each None when unreadable. Never raises."""
    mem = psi = zram = None
    try:
        mem = parse_meminfo_available_gb((Path(proc_root) / "meminfo").read_text())
    except OSError:
        pass
    try:
        psi = parse_psi_full_avg10((Path(proc_root) / "pressure" / "memory").read_text())
    except OSError:
        pass
    try:
        from monitor import _read_zram
        value = _read_zram(Path(sys_root)).get("used_percent")
        zram = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    except Exception:
        zram = None
    return mem, psi, zram


class RamValve:
    """Level state machine. `update` returns the new level and whether it changed."""

    def __init__(self, config=None):
        self.config = {**DEFAULTS, **(config or {})}
        self.level = "ok"
        self.changes = 0

    def _severity(self, mem, psi, zram, c):
        """Level each reading asks for, without hysteresis (ok when a reading is missing)."""
        worst = 0
        if mem is not None:
            if mem < c["ramGuardCriticalMemAvailableGB"]:
                worst = max(worst, 3)
            elif mem < c["ramGuardPressingMemAvailableGB"]:
                worst = max(worst, 2)
            elif mem < c["ramGuardTightMemAvailableGB"]:
                worst = max(worst, 1)
        if psi is not None:
            if psi >= c["ramGuardPsiFullCritical"]:
                worst = max(worst, 3)
            elif psi >= c["ramGuardPsiFullPressing"]:
                worst = max(worst, 2)
            elif psi >= c["ramGuardPsiFullTight"]:
                worst = max(worst, 1)
        if zram is not None:
            if zram >= c["ramGuardZramPctCritical"]:
                worst = max(worst, 3)
            elif zram >= c["ramGuardZramPctPressing"]:
                worst = max(worst, 2)
            elif zram >= c["ramGuardZramPctTight"]:
                worst = max(worst, 1)
        return worst

    def _recovered(self, level_index, mem, psi, zram):
        """True when every reading is past its entry threshold of `level_index` by the hysteresis, so the level may be left."""
        c = self.config
        h = c["ramGuardHysteresisGB"]
        if mem is not None:
            entry = {1: c["ramGuardTightMemAvailableGB"], 2: c["ramGuardPressingMemAvailableGB"],
                     3: c["ramGuardCriticalMemAvailableGB"]}[level_index]
            if mem < entry + h:
                return False
        if psi is not None:
            entry = {1: c["ramGuardPsiFullTight"], 2: c["ramGuardPsiFullPressing"], 3: c["ramGuardPsiFullCritical"]}[level_index]
            if psi >= entry - PSI_HYSTERESIS_PCT:
                return False
        if zram is not None:
            entry = {1: c["ramGuardZramPctTight"], 2: c["ramGuardZramPctPressing"], 3: c["ramGuardZramPctCritical"]}[level_index]
            if zram >= entry - ZRAM_HYSTERESIS_PCT:
                return False
        return True

    def update(self, mem, psi, zram):
        """Feed one set of readings; returns (level, changed). Entering a higher level is immediate, leaving one needs recovery."""
        wanted = self._severity(mem, psi, zram, self.config)
        current = LEVELS.index(self.level)
        new = current
        if wanted > current:
            new = wanted
        elif wanted < current:
            # step down one level at a time, and only when the readings are clear of the current level's entry thresholds
            if self._recovered(current, mem, psi, zram):
                new = current - 1
        if new != current:
            self.level = LEVELS[new]
            self.changes += 1
            return self.level, True
        return self.level, False


def read_mem_total_gb(proc_root="/proc"):
    """MemTotal in GB, or None when it cannot be read."""
    try:
        for line in (Path(proc_root) / "meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                fields = line.split()
                return int(fields[1]) / (1024 * 1024) if len(fields) >= 2 and fields[1].isdigit() else None
    except OSError:
        return None
    return None


def oom_score_adj_write(pid, value, proc_root="/proc"):
    """Write oom_score_adj for a child process we started. Returns (ok, error). Never raises: a failed write is only reported."""
    if not isinstance(value, int) or isinstance(value, bool) or not OOM_MIN <= value <= OOM_MAX:
        return False, "invalid llamaOomScoreAdj"
    if value == 0:
        return False, None                 # off by default: nothing is written
    try:
        (Path(proc_root) / str(int(pid)) / "oom_score_adj").write_text(str(value))
        return True, None
    except OSError as error:
        return False, str(error)


def cache_budget(total_ram_gb, budget_pct, ctx_tokens, ctx_checkpoints, cache_ram_mib, snapshot_mib=0.0):
    """RAM budget for the model's caches (prompt cache, context checkpoints).

    Returns (cache_ram_mib, ctx_checkpoints): the requested values, reduced until the sum fits the budget (the snapshot already
    held in RAM is subtracted first). A budget of 0 means no limit, and the requested values are returned unchanged."""
    if not budget_pct or budget_pct <= 0 or total_ram_gb is None:
        return cache_ram_mib, ctx_checkpoints
    # The RAM session snapshot in /dev/shm (one per model) counts against the same budget.
    budget_mib = max(0.0, total_ram_gb * 1024 * budget_pct / 100.0 - float(snapshot_mib or 0.0))
    per_checkpoint = CHECKPOINT_BASE_MIB + CHECKPOINT_PER_TOKEN_KIB * ctx_tokens / 1024.0
    cache = int(cache_ram_mib)
    checkpoints = int(ctx_checkpoints)
    while checkpoints > 0 and cache + checkpoints * per_checkpoint > budget_mib:
        checkpoints -= 1
    if cache + checkpoints * per_checkpoint > budget_mib:
        cache = max(0, int(budget_mib - checkpoints * per_checkpoint))
    return cache, checkpoints


_BUDGET_LOCK = threading.Lock()


class SsdDailyBudget:
    """Counts bytes written for slot saves per calendar day, in a small state file. Over the limit a save is skipped (logged)."""

    def __init__(self, limit_gb, state_path, clock=time.time):
        self.limit_bytes = int(limit_gb * 1024 ** 3) if limit_gb and limit_gb > 0 else None
        self.path = Path(state_path)
        self.clock = clock

    def _state(self):
        try:
            data = json.loads(self.path.read_text())
            if isinstance(data, dict) and isinstance(data.get("day"), str) and isinstance(data.get("bytes"), int):
                return data
        except (OSError, ValueError):
            pass
        return {"day": "", "bytes": 0}

    def _today(self):
        return time.strftime("%Y-%m-%d", time.localtime(self.clock()))

    def allowed(self, size_bytes):
        if self.limit_bytes is None:
            return True
        state = self._state()
        used = state["bytes"] if state["day"] == self._today() else 0
        return used + size_bytes <= self.limit_bytes

    def reserve(self, size_bytes):
        """allowed() and record() as one step (RAMFIX1): two writers cannot both pass the check and overrun the day's limit. Called
        before the write; a write that fails afterwards keeps its reservation (the budget errs towards fewer writes)."""
        with _BUDGET_LOCK:
            if not self.allowed(size_bytes):
                return False
            self.record(size_bytes)
            return True

    def record(self, size_bytes):
        if self.limit_bytes is None:
            return                         # no limit: nothing is counted and no state file is written
        today = self._today()
        state = self._state()
        used = state["bytes"] if state["day"] == today else 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps({"day": today, "bytes": used + int(size_bytes)}))
            os.replace(tmp, self.path)
        except OSError:
            pass
