"""Desktop RAM protection for the gateway (RAMGUARD1, design reports/REPORT-RAMDESIGN1.md, steps R0-R1). Pure logic.

The gateway owns the RAM budget. "Honest free" memory is not MemAvailable (it counts the whole page cache as free) but
MemFree + half of the inactive file cache, minus a penalty once swap is filling up: evicted pages come back when read.
Admission of a load or a layout switch needs honest free - the server's un-evictable resident part - the shared memory
planned for snapshots, and the desktop's reserve; swap and pressure must be calm. A four-step ladder (ok, tight, pressing,
critical) decides what to shed first. R1: the server runs in its own systemd scope with memory.high, memory.max and no swap.
RAMLOGIC2: the reserve comes from the measured growth of the desktop and its need; when it does not fit, plan_load climbs a
ladder (a smaller prompt cache, a short wait, a guarded load) instead of refusing, and never asks the user to close anything.

All sizes are MiB, times are seconds. Readers take paths so the tests can use fake /proc trees.
"""

import math
import shutil
from pathlib import Path

LEVELS = ("ok", "tight", "pressing", "critical")
# RAMLOGIC2: the reserve is room for the desktop to GROW during and right after a load, not the desktop's size: honest free
# already leaves out what the desktop uses now (the old 6 GiB was the desktop's whole footprint, counted twice). Measured on
# this machine (reports/REPORT-RAMLOGIC2.md, 2 h at 1 Hz, Brave, VS Code, Telegram, agent sessions and builds at work): the
# desktop's anonymous + shared memory grew by p99 1.6 GiB over 60 s. The reserve follows desk_need like the VRAM buffer does.
DESK_RESERVE_IDLE_MIB = 1024         # need 0 (idle, locked): nobody is opening anything
DESK_RESERVE_ORDINARY_MIB = 2048     # need 0.3: the measured p99 minute of growth (1654 MiB), rounded up
DEFAULT_DESK_RESERVE_MIB = DESK_RESERVE_ORDINARY_MIB   # without desk_need
RAM_HARD_MIN_MIB = 1024              # after the short wait a load may go below the reserve, never below this
LIGHT_CACHE_STEP_MIB = 256
SERVER_OVERHEAD_MIB = 1536           # resident_fixed estimate: runtime, buffers, planner (until measured)
DRAFT_ESTIMATE_MIB = 400
SCOPE_SLICE = "shura.slice"
SCOPE_OOM_SCORE = 800                # the server is the first candidate when memory runs out (its scope has no swap)
HYSTERESIS_MIB = 512                 # a level is left only after this much recovery...
HYSTERESIS_PSI = 1.0                 # ...and this much PSI recovery
SWAP_GROWTH_LIMIT_MIB = 1024         # a load may grow swap by this much while it runs; more refuses the next load (zram: swap level is noisy)


def parse_meminfo_mib(text):
    """/proc/meminfo as {key: MiB} for the keys the budget uses; missing keys are absent."""
    out = {}
    wanted = {"MemTotal", "MemFree", "MemAvailable", "Inactive(file)", "Active(file)", "Dirty", "SwapTotal", "SwapFree", "Shmem"}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in wanted:
            parts = rest.split()
            if parts and parts[0].isdigit():
                out[key] = int(parts[0]) / 1024.0
    return out


def parse_pressure(text):
    """/proc/pressure/memory: {"some": {"avg10", "avg60"}, "full": {"avg10"}} (percent)."""
    out = {}
    for line in text.splitlines():
        kind, _, rest = line.partition(" ")
        if kind not in ("some", "full"):
            continue
        values = {}
        for item in rest.split():
            name, _, value = item.partition("=")
            if name in ("avg10", "avg60"):
                try:
                    values[name] = float(value)
                except ValueError:
                    pass
        out[kind] = values
    return out


def read_meminfo(proc_root="/proc"):
    try:
        return parse_meminfo_mib((Path(proc_root) / "meminfo").read_text())
    except OSError:
        return None


def read_pressure(proc_root="/proc"):
    try:
        return parse_pressure((Path(proc_root) / "pressure" / "memory").read_text())
    except OSError:
        return None


def swap_used_mib(mem):
    return max(0.0, mem.get("SwapTotal", 0.0) - mem.get("SwapFree", 0.0))


def swap_used_fraction(mem):
    total = mem.get("SwapTotal", 0.0)
    return swap_used_mib(mem) / total if total > 0 else 0.0


def honest_free_mib(mem):
    """MemFree + 0.5 x Inactive(file) - penalty. On this machine swap is zram (compressed inactive pages kept in RAM): 9-10 of 16 GiB
    used is normal and says nothing about pressure, so the penalty only starts when swap is more than 60 % full (the pressure
    signal is PSI)."""
    free = mem.get("MemFree", 0.0)
    inactive = mem.get("Inactive(file)", 0.0)
    total = mem.get("SwapTotal", 0.0)
    penalty = max(0.0, swap_used_mib(mem) - 0.60 * total)
    return free + 0.5 * inactive - penalty


def swap_growth_too_big(before_mem, after_mem, limit_mib=SWAP_GROWTH_LIMIT_MIB):
    """True when swap grew by more than limit_mib between two readings (a load that pushes memory out to swap)."""
    return swap_used_mib(after_mem) - swap_used_mib(before_mem) > limit_mib


def level_of(honest_free, psi, desk_reserve_mib=DEFAULT_DESK_RESERVE_MIB, swap_fraction=0.0):
    """The level each reading asks for, without hysteresis (the worst of the three readings)."""
    some10 = (psi or {}).get("some", {}).get("avg10", 0.0)
    full10 = (psi or {}).get("full", {}).get("avg10", 0.0)
    # swap is only an emergency limit (zram makes its level a poor signal); the decisive readings are honest free and PSI
    if honest_free < desk_reserve_mib / 2 or full10 > 40 or swap_fraction > 0.90:
        return "critical"
    if honest_free < desk_reserve_mib or some10 > 15:
        return "pressing"
    if honest_free < desk_reserve_mib + 2048 or some10 > 5:
        return "tight"
    return "ok"


class RamLadder:
    """The level with hysteresis: entering a level is immediate; leaving it needs recovery by HYSTERESIS_MIB of honest free
    and HYSTERESIS_PSI of pressure, so the actions do not flap."""

    def __init__(self, desk_reserve_mib=DEFAULT_DESK_RESERVE_MIB):
        self.desk_reserve_mib = desk_reserve_mib
        self.level = "ok"
        self.changes = 0

    def update(self, mem, psi):
        honest = honest_free_mib(mem)
        wanted = level_of(honest, psi, self.desk_reserve_mib, swap_used_fraction(mem))
        if LEVELS.index(wanted) > LEVELS.index(self.level):
            self._set(wanted)
        elif LEVELS.index(wanted) < LEVELS.index(self.level):
            # recover one step at a time, and only when the reading is clearly better than the level's own threshold
            some10 = (psi or {}).get("some", {}).get("avg10", 0.0)
            clear = honest >= self._threshold(self.level) + HYSTERESIS_MIB and some10 <= self._psi_threshold(self.level) - HYSTERESIS_PSI
            if clear:
                self._set(LEVELS[LEVELS.index(self.level) - 1])
        return self.level

    def _threshold(self, level):
        return {"ok": 0, "tight": self.desk_reserve_mib + 2048, "pressing": self.desk_reserve_mib,
                "critical": self.desk_reserve_mib / 2}[level]

    def _psi_threshold(self, level):
        return {"ok": 5.0, "tight": 5.0, "pressing": 15.0, "critical": 40.0}[level]

    def _set(self, level):
        if level != self.level:
            self.level = level
            self.changes += 1


def resident_fixed_estimate_mib(ncmoe, cache_ram_mib, draft=True):
    """Until a layout is measured: 0.3 GiB per CPU expert layer (pinned copies) + runtime + cache-ram + draft."""
    return 300.0 * max(0, int(ncmoe)) + SERVER_OVERHEAD_MIB + max(0.0, float(cache_ram_mib)) + (DRAFT_ESTIMATE_MIB if draft else 0)


def desk_reserve_for(need):
    """RAM reserve (MiB) for a desk need (0 idle .. 1 game): the line through idle (0 -> 1 GiB) and ordinary (0.3 -> 2 GiB),
    rounded up to 256 MiB: a video or a call (0.6) gets 3 GiB, a game (1.0) 4.5 GiB."""
    need = min(1.0, max(0.0, float(need)))
    raw = DESK_RESERVE_IDLE_MIB + (DESK_RESERVE_ORDINARY_MIB - DESK_RESERVE_IDLE_MIB) * need / 0.3
    return int(math.ceil(raw / 256.0) * 256)


def _gib(mib):
    return f"{mib / 1024:.1f} GiB"


def plan_load(mem, psi, resident_mib, cache_ram_mib, shm_planned_mib=0.0, reserve_mib=DEFAULT_DESK_RESERVE_MIB,
              waited_out=False, hard_min_mib=RAM_HARD_MIN_MIB, credit_mib=0.0):
    """RAMLOGIC2: the ladder that replaces a refusal. resident_mib includes the prompt cache in RAM (cache_ram_mib).

    Returns {"step", "cache_ram_mib", "free_mib", "need_mib", "reason"}; free is honest free minus planned shm plus credit_mib
    (what a running server frees when a switch stops it), need is what the chosen start asks for, reserve included. Steps:
      load     the layout as configured leaves the reserve;
      light    the same layout with a smaller prompt cache in RAM (cache_ram_mib, a multiple of 256, maybe 0) leaves it;
      wait     not even that: wait a little (the caller bounds it) and say for what;
      guarded  the wait is over: load with no prompt cache if that leaves at least hard_min_mib (the swap-growth check
               watches the load and the reserve is for growth that may not come);
      refuse   below the hard minimum after the wait, or memory pressure after the wait, or no reading.
    Pressure (PSI some avg60 >= 5 %, swap >= 90 %) waits like a shortage and is never loaded through."""
    out = {"step": "refuse", "cache_ram_mib": 0, "free_mib": 0, "need_mib": 0, "reason": ""}
    if mem is None or psi is None:
        out["reason"] = "no fresh memory or pressure reading"
        return out
    free = honest_free_mib(mem) - shm_planned_mib + max(0.0, float(credit_mib))
    cache = max(0.0, float(cache_ram_mib))
    fixed = max(0.0, float(resident_mib) - cache)
    out.update(free_mib=int(free), need_mib=int(fixed + cache + reserve_mib), cache_ram_mib=int(cache))
    some60 = (psi or {}).get("some", {}).get("avg60", 0.0)
    if swap_used_fraction(mem) >= 0.90 or some60 >= 5.0:
        why = (f"swap is {swap_used_fraction(mem) * 100:.0f} % full (emergency limit 90 %)" if swap_used_fraction(mem) >= 0.90
               else f"memory pressure some avg60 {some60:.1f} % (limit 5 %)")
        out.update(step="refuse" if waited_out else "wait", reason=why)
        return out
    if free - fixed - cache >= reserve_mib:
        out.update(step="load")
        return out
    fit = math.floor((free - fixed - reserve_mib) / LIGHT_CACHE_STEP_MIB) * LIGHT_CACHE_STEP_MIB
    if fit >= 0:
        out.update(step="light", cache_ram_mib=int(fit), need_mib=int(fixed + fit + reserve_mib),
                   reason=f"prompt cache in RAM {cache:.0f} -> {fit} MiB to keep {_gib(reserve_mib)} for the desktop")
        return out
    detail = (f"{_gib(free)} free, {_gib(fixed + reserve_mib)} needed "
              f"(the model {_gib(fixed)} + room for the desktop {_gib(reserve_mib)})")
    if not waited_out:
        out.update(step="wait", cache_ram_mib=0, need_mib=int(fixed + reserve_mib), reason=f"waiting for memory: {detail}")
        return out
    if free - fixed >= hard_min_mib:
        out.update(step="guarded", cache_ram_mib=0, need_mib=int(fixed + hard_min_mib),
                   reason=f"loading with less room for the desktop than planned: {detail}")
        return out
    out.update(step="refuse", cache_ram_mib=0, need_mib=int(fixed + hard_min_mib),
               reason=f"not enough memory for the model yet: {_gib(free)} free, the model needs {_gib(fixed)} "
                      f"and {_gib(hard_min_mib)} must stay with the desktop")
    return out


# RAMFIX1: the gateway's own files in tmpfs (RAM) that a load may take back. A session snapshot written at an idle unload sat in
# /dev/shm (2 GiB) and honest free counted it as used: the model was refused because of Shura's own file. Kinds, cheapest first:
#   cache       a copy in the session catalogue: evicted (a cache entry; the prompt is re-read if it is ever needed);
#   slot        the saved session of a model: moved to the disk spill folder and restored from there;
#   checkpoint  the RAM checkpoint of a pressure unload: moved to the disk spill folder as that model's saved session.
RELEASE_RANK = {"cache": 0, "slot": 1, "checkpoint": 2}


def release_order(entries, deficit_mib):
    """Which own tmpfs files to release for deficit_mib more honest free. entries: dicts with "kind", "mib", "inode", "mtime".
    Hard links (same inode) are one group: freeing it needs every link gone, and the group is handled as its most valuable
    member (a catalogue copy linked to the saved session is moved with it, not dropped). Cheapest kind first, oldest first;
    stops once the deficit is covered. Returns a list of groups (lists of entries); all groups when the deficit is not covered."""
    groups = {}
    for entry in entries:
        groups.setdefault(entry.get("inode") or id(entry), []).append(entry)
    ordered = sorted(groups.values(), key=lambda g: (max(RELEASE_RANK.get(e["kind"], 9) for e in g),
                                                     min(float(e.get("mtime", 0)) for e in g)))
    out, freed = [], 0.0
    for group in ordered:
        if freed >= deficit_mib:
            break
        out.append(group)
        freed += max(float(e.get("mib", 0)) for e in group)
    return out


def releasable_mib(entries):
    """Total MiB the own tmpfs files hold, hard links counted once."""
    seen = {}
    for entry in entries:
        key = entry.get("inode") or id(entry)
        seen[key] = max(seen.get(key, 0.0), float(entry.get("mib", 0)))
    return sum(seen.values())


def admit_load(mem, psi, resident_fixed_mib, shm_planned_mib=0.0, desk_reserve_mib=DEFAULT_DESK_RESERVE_MIB):
    """May a load or a layout switch start? Returns (ok, reason). Missing readings refuse (no fresh picture, no load)."""
    if mem is None or psi is None:
        return False, "no fresh memory or pressure reading"
    honest = honest_free_mib(mem)
    after = honest - resident_fixed_mib - shm_planned_mib
    if after < desk_reserve_mib:
        return False, f"honest free {honest:.0f} MiB - resident {resident_fixed_mib:.0f} - shm {shm_planned_mib:.0f} leaves " \
                      f"{after:.0f} MiB, below the desktop reserve {desk_reserve_mib} MiB"
    if swap_used_fraction(mem) >= 0.90:
        return False, f"swap is {swap_used_fraction(mem) * 100:.0f} % full (emergency limit 90 %)"
    some60 = (psi or {}).get("some", {}).get("avg60", 0.0)
    if some60 >= 5.0:
        return False, f"memory pressure some avg60 {some60:.1f} % (limit 5 %)"
    return True, ""


def scope_available(which=shutil.which):
    return which("systemd-run") is not None


def build_scope_command(argv, memory_high_mib, memory_max_mib, unit="shura-llama", available=True):
    """R1: the server's command inside its own user scope (systemd-run --user --scope --slice=shura.slice), with
    MemoryHigh, MemoryMax, MemorySwapMax=0 and OOMScoreAdjust. Without systemd-run the argv is returned unchanged."""
    if not available:
        return list(argv)
    return ["systemd-run", "--user", "--scope", f"--unit={unit}", f"--slice={SCOPE_SLICE}",
            "-p", f"MemoryHigh={int(memory_high_mib)}M", "-p", f"MemoryMax={int(memory_max_mib)}M",
            "-p", "MemorySwapMax=0", "-p", f"OOMScoreAdjust={SCOPE_OOM_SCORE}", "--"] + list(argv)


def scope_limits(resident_fixed_mib, pinned_margin_mib=1536, max_extra_mib=2048):
    """memory.high = measured resident + margin; memory.max = high + extra (the server dies before the desktop does)."""
    high = float(resident_fixed_mib) + pinned_margin_mib
    return high, high + max_extra_mib


# Actions per level (RAMGUARD1, Opus's rule): under memory pressure the cache is RESET, not written to the SSD. A snapshot on disk
# is the rare last resort at critical only. No periodic writes anywhere.
LADDER_ACTIONS = {
    "ok": {"writes_disk": False, "actions": ()},
    "tight": {"writes_disk": False, "actions": ("drop_tmpfs_snapshots_older_than_600s", "no_cache_ram_growth")},
    "pressing": {"writes_disk": False, "actions": ("reset_prompt_cache", "reset_checkpoints", "drop_tmpfs_snapshots",
                                                   "no_new_loads_except_smaller_layout")},
    "critical": {"writes_disk": True, "actions": ("snapshot_to_nvme_once", "stop_server_after_snapshot")},
}
SNAPSHOT_KB_PER_TOKEN = 14.0          # measured in SESS2b: a 32k session snapshot was 460 MB (about 14 KB per token)


def snapshot_gb(ctx_tokens):
    """Size of one slot snapshot (decimal GB) from the measured per-token size."""
    return float(ctx_tokens) * SNAPSHOT_KB_PER_TOKEN / 1e6


def may_write_snapshot(level, written_today_gb, daily_limit_gb, ctx_tokens):
    """A disk snapshot only at critical, and only while the daily write budget has room for it. Returns (allowed, reason)."""
    if level != "critical":
        return False, f"no disk write at level {level}: the cache is reset instead"
    size = snapshot_gb(ctx_tokens)
    if written_today_gb + size > daily_limit_gb:
        return False, f"daily SSD write budget would be exceeded ({written_today_gb + size:.2f} > {daily_limit_gb} GB)"
    return True, ""
