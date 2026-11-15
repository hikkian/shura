"""Session cache on NVMe (DESIGN-SESSION-CACHE.md, level 2): catalogue, quotas, LRU, TTL, disk reserve, daily write budget,
atomic writes, compatibility key, the write-behind policy and the restore-or-recompute decision.

Pure and testable: every path, clock and disk reading is injected, so the tests run on a fake filesystem. Off by default
(`sessionCacheEnabled` false); with the default config nothing here is called.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path


DEFAULTS = {
    "sessionCacheEnabled": False,
    "sessionCacheDir": "",               # empty = ~/.local/state/ai-gateway/sessions
    "sessionCacheQuotaGB": 30.0,         # the smaller of this and sessionCacheQuotaPct of the partition
    "sessionCacheQuotaPct": 8.0,
    "sessionCacheReserveGB": 100.0,      # a write never leaves less than this free on the partition
    "sessionCacheTTLDays": 14,
    "sessionCacheIdleSeconds": 120,      # write-behind only after this much idle time
    "sessionCacheMinGrowthTokens": 4000,
    "sessionCacheMinSlotTokens": 8000,
    "sessionCacheIoPressureMax": 10.0,   # PSI io avg10 must be below this for a write
    "sessionCacheRestoreRatio": 0.25,    # restore only if it costs less than this fraction of recomputing the prefill
    "sessionCacheReadGBps": 3.0,         # NVMe read speed used for the decision (measure on the machine; RAM0 gives the numbers)
    "sessionCacheLoadGBps": 3.0,         # speed of putting the snapshot into video memory
    "sessionCachePrefillTps": 500.0,     # recompute speed used for the decision (tokens per second)
    # Second level on the SSD (SESS3): only long sessions, only at a gateway stop or an idle unload, only the newest snapshot.
    "sessionPersistDir": "",             # empty = off (nothing is written to the SSD)
    "sessionPersistMinTokens": 30000,    # shorter sessions are cheap to re-read: not written
    "sessionPersistMaxGB": 4.0,          # quota of the SSD level
    "sessionPersistKeep": 1,             # how many of the newest snapshots stay on the SSD
    "sessionPersistOnIdle": False,       # also copy at an idle unload; off: the RAM copy survives an unload, only a reboot loses it
}

SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
META_SUFFIX = ".json"


def idle_copy_command(src, dst):
    """The command that moves a snapshot to disk in the idle I/O class (ionice -c3), so it never competes with the desktop."""
    return ["ionice", "-c3", "cp", "--sparse=never", str(src), str(dst)]


def new_session_id():
    """A random per-session id (uuid4 hex form). It carries no path and no personal data."""
    import uuid
    return str(uuid.uuid4())


def valid_session_id(value):
    return isinstance(value, str) and bool(SESSION_ID_RE.match(value))


def compatibility_key(model_sha256, vision, chat_template_sha256, kv_types):
    """A snapshot restores only into the same model file, mode, chat template and KV types (design: key of compatibility)."""
    raw = json.dumps({"model": model_sha256, "vision": bool(vision), "template": chat_template_sha256, "kv": list(kv_types)},
                     sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def prefix_hash(text, chars=4096):
    """Hash of the first `chars` characters of the conversation: the fallback key when no session header is sent."""
    return hashlib.sha256(text[:chars].encode("utf-8", "replace")).hexdigest()


def quota_bytes(partition_total_bytes, quota_gb, quota_pct):
    """min(quota_gb, quota_pct of the partition), in bytes. A missing value counts as no limit on that term."""
    limits = []
    if quota_gb and quota_gb > 0:
        limits.append(quota_gb * 1024 ** 3)
    if quota_pct and quota_pct > 0 and partition_total_bytes:
        limits.append(partition_total_bytes * quota_pct / 100.0)
    return int(min(limits)) if limits else 0


def should_write(idle_s, growth_tokens, slot_tokens, io_pressure, game_or_video, ram_level, cfg=None, budget_ok=True, disk_ok=True):
    """Write-behind decision. Returns (write, reason). Never during work (idle_s counts from the last request)."""
    c = {**DEFAULTS, **(cfg or {})}
    if game_or_video:
        return False, "game or fullscreen video is running"
    if ram_level in ("pressing", "critical"):
        return False, f"RAM valve is {ram_level}"
    if io_pressure is not None and io_pressure >= c["sessionCacheIoPressureMax"]:
        return False, f"disk I/O pressure {io_pressure:.1f} >= {c['sessionCacheIoPressureMax']}"
    if idle_s < c["sessionCacheIdleSeconds"]:
        return False, "not idle long enough"
    if growth_tokens < c["sessionCacheMinGrowthTokens"]:
        return False, "too little new context since the last snapshot"
    if slot_tokens < c["sessionCacheMinSlotTokens"]:
        return False, "slot too small: recomputing is cheaper than writing"
    if not budget_ok:
        return False, "daily write budget used"
    if not disk_ok:
        return False, "disk reserve would be breached"
    return True, "idle, grown, quiet"


def should_restore(size_bytes, read_bps, load_bps, prefill_tps, tokens, ratio=0.25):
    """Restore if the snapshot costs less than `ratio` of recomputing the same tokens. Returns (restore, t_restore, t_recompute)."""
    if not all(isinstance(v, (int, float)) and v > 0 for v in (read_bps, load_bps, prefill_tps)) or tokens <= 0:
        return False, None, None
    t_restore = size_bytes / read_bps + size_bytes / load_bps
    t_recompute = tokens / prefill_tps
    return t_restore < ratio * t_recompute, t_restore, t_recompute


def persist_should_write(tokens, reason, cfg=None, budget_ok=True, disk_ok=True):
    """SSD level decision (SESS3). Returns (write, why). Only a gateway stop or an idle unload may write; never a manual
    unload, a memory-pressure unload or a model switch; and only a session long enough that re-reading it is expensive."""
    c = {**DEFAULTS, **(cfg or {})}
    allowed = ("gateway shutdown", "idle for") if c["sessionPersistOnIdle"] else ("gateway shutdown",)
    if not str(reason).startswith(allowed):
        return False, f"not a stop{' or an idle unload' if c['sessionPersistOnIdle'] else ''} ({reason})"
    if tokens < c["sessionPersistMinTokens"]:
        return False, f"short session ({tokens} tokens < {c['sessionPersistMinTokens']}): re-reading it is cheap"
    if not budget_ok:
        return False, "daily write budget used"
    if not disk_ok:
        return False, "SSD quota or reserve would be breached"
    return True, "stop with a long session"


class SessionCatalog:
    """One snapshot per session in `root`: `<id>.bin` plus `<id>.json` (key, tokens, times, prefix hash, size).

    Writes are atomic (temp file, fsync, rename, then the previous snapshot of the same session is removed). The catalogue
    refuses any write that would break the quota or leave less than the reserve free on the partition."""

    def __init__(self, root, quota_bytes_value, reserve_bytes, ttl_days, clock=time.time, disk_stats=os.statvfs):
        self.root = Path(root)
        self.quota = int(quota_bytes_value)
        self.reserve = int(reserve_bytes)
        self.ttl = float(ttl_days) * 86400.0
        self.clock = clock
        self.disk_stats = disk_stats

    # --- disk and index helpers
    def _free_bytes(self):
        st = self.disk_stats(self.root if self.root.exists() else self.root.parent)
        return st.f_bavail * st.f_frsize

    def _entries(self):
        out = []
        if not self.root.is_dir():
            return out
        for meta in self.root.glob("*" + META_SUFFIX):
            try:
                data = json.loads(meta.read_text())
                if valid_session_id(data.get("session_id")) and isinstance(data.get("size"), int):
                    out.append(data)
            except (OSError, ValueError):
                continue
        return out

    def used_bytes(self):
        return sum(e["size"] for e in self._entries())

    def write_allowed(self, size_bytes):
        """False when the snapshot would break the quota or leave less than the reserve free on the partition."""
        if self.quota and self.used_bytes() + size_bytes > self.quota:
            return False
        # A full partition never takes a write, whatever the reserve says: at least the reserve, and never below zero.
        if self._free_bytes() - size_bytes < max(self.reserve, 0):
            return False
        return True

    def fits_after_eviction(self, size_bytes):
        """Write admission when the least-recently-used snapshots may be evicted first (make_room). The quota term is a single
        snapshot never larger than the quota; the reserve term is the same as write_allowed (evictions only free more)."""
        if self.quota and size_bytes > self.quota:
            return False
        return self._free_bytes() - size_bytes >= max(self.reserve, 0)

    def make_room(self, size_bytes, keep=None):
        """Evict least-recently-used snapshots until `size_bytes` more fit under the quota. `keep` is the session being
        replaced (its old snapshot leaves anyway). Called only right before a write that will really happen: a skipped write
        must not evict. Returns the removed session ids."""
        if not self.quota:
            return []
        entries = [e for e in self._entries() if e["session_id"] != keep]
        entries.sort(key=lambda e: float(e.get("last_used", 0)))
        used = sum(e["size"] for e in entries)
        removed = []
        while entries and used + size_bytes > self.quota:
            victim = entries.pop(0)
            used -= victim["size"]
            self.remove(victim["session_id"])
            removed.append(victim["session_id"])
        return removed

    # --- writes
    def save(self, session_id, model_key, tokens, prefix, payload_path, now=None):
        """Atomically store the snapshot in `payload_path` (already written by the server) as the session's one snapshot.
        Returns (ok, reason)."""
        if not valid_session_id(session_id):
            return False, "invalid session id"
        src = Path(payload_path)
        if not src.is_file() or src.is_symlink():
            return False, "payload missing"
        size = src.stat().st_size
        if not self.write_allowed(size):
            return False, "quota or disk reserve would be breached"
        self.root.mkdir(parents=True, exist_ok=True)
        now = self.clock() if now is None else now
        target = self.root / f"{session_id}.bin"
        tmp = self.root / f".{session_id}.tmp"
        try:
            with open(src, "rb") as read, open(tmp, "wb") as write:
                while True:
                    chunk = read.read(1 << 20)
                    if not chunk:
                        break
                    write.write(chunk)
                write.flush()
                os.fsync(write.fileno())
            os.replace(tmp, target)
        except OSError as error:
            tmp.unlink(missing_ok=True)
            return False, f"write failed: {error}"
        meta = {"session_id": session_id, "model_key": model_key, "tokens": int(tokens), "prefix_hash": prefix,
                "size": size, "created": now, "last_used": now, "file": target.name}
        meta_tmp = self.root / f".{session_id}.json.tmp"
        meta_tmp.write_text(json.dumps(meta))
        os.replace(meta_tmp, self.root / f"{session_id}{META_SUFFIX}")
        return True, "saved"

    def adopt_copied(self, session_id, model_key, tokens, prefix, saved, tmp_path, now=None):
        """Take a snapshot that was already copied into the catalogue as `tmp_path` (by the idle I/O copy). Makes it the session's
        one snapshot: fsync, rename, then the metadata (which keeps the gateway's own slot record in `saved`)."""
        if not valid_session_id(session_id):
            return False, "invalid session id"
        tmp = Path(tmp_path)
        if not tmp.is_file() or tmp.is_symlink() or tmp.parent != self.root:
            return False, "temporary payload missing"
        size = tmp.stat().st_size
        now = self.clock() if now is None else now
        try:
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            target = self.root / f"{session_id}.bin"
            os.replace(tmp, target)
        except OSError as error:
            tmp.unlink(missing_ok=True)
            return False, f"adopt failed: {error}"
        meta = {"session_id": session_id, "model_key": model_key, "tokens": int(tokens), "prefix_hash": prefix,
                "size": size, "created": now, "last_used": now, "file": target.name, "saved": saved}
        meta_tmp = self.root / f".{session_id}.json.tmp"
        meta_tmp.write_text(json.dumps(meta))
        os.replace(meta_tmp, self.root / f"{session_id}{META_SUFFIX}")
        return True, "adopted"

    def touch(self, session_id, now=None):
        meta_path = self.root / f"{session_id}{META_SUFFIX}"
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            return False
        meta["last_used"] = self.clock() if now is None else now
        meta_path.write_text(json.dumps(meta))
        return True

    def find(self, session_id, model_key):
        """The snapshot of this session if its key matches; a mismatched one is removed (design: it cannot be used)."""
        if not valid_session_id(session_id):
            return None
        meta_path = self.root / f"{session_id}{META_SUFFIX}"
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            return None
        if meta.get("model_key") != model_key:
            self.remove(session_id)
            return None
        payload = self.root / meta.get("file", "")
        if not payload.is_file():
            self.remove(session_id)
            return None
        return {**meta, "path": str(payload)}

    def remove(self, session_id):
        for name in (f"{session_id}.bin", f"{session_id}{META_SUFFIX}"):
            (self.root / name).unlink(missing_ok=True)

    # --- housekeeping
    def prune(self, now=None):
        """TTL first, then least-recently-used until the quota holds. Temporary files are removed. Returns removed session ids."""
        now = self.clock() if now is None else now
        removed = []
        for tmp in list(self.root.glob(".*.tmp")) if self.root.is_dir() else []:
            tmp.unlink(missing_ok=True)
        entries = self._entries()
        for e in entries:
            if now - float(e.get("last_used", 0)) > self.ttl:
                self.remove(e["session_id"])
                removed.append(e["session_id"])
        entries = [e for e in self._entries()]
        entries.sort(key=lambda e: float(e.get("last_used", 0)))
        while self.quota and sum(e["size"] for e in entries) > self.quota and entries:
            victim = entries.pop(0)
            self.remove(victim["session_id"])
            removed.append(victim["session_id"])
        return removed

    def keep_newest(self, count, keep_id=None):
        """Remove all but the `count` most recently used snapshots (`keep_id` always stays). Used by the SSD level: one long
        session is the whole point, and older ones only cost wear. Returns the removed session ids."""
        removed = []
        entries = sorted(self._entries(), key=lambda e: float(e.get("last_used", 0)), reverse=True)
        kept = 0
        for e in entries:
            if e["session_id"] == keep_id:
                kept += 1
                continue
            if kept < max(int(count), 1):
                kept += 1
                continue
            self.remove(e["session_id"])
            removed.append(e["session_id"])
        return removed

    def start_check(self, model_key_of):
        """At start: remove snapshots whose key no longer matches (model_key_of(entry) gives the current key for that entry)."""
        dropped = []
        for e in self._entries():
            if model_key_of(e) != e.get("model_key"):
                self.remove(e["session_id"])
                dropped.append(e["session_id"])
        return dropped


class DailyWriteBudget:
    """Counts snapshot bytes written per calendar day in a small state file (shared with RAM1's ssdWriteDailyLimitGB)."""

    def __init__(self, limit_gb, state_path, clock=time.time):
        self.limit = int(limit_gb * 1024 ** 3) if limit_gb and limit_gb > 0 else None
        self.path = Path(state_path)
        self.clock = clock

    def _today(self):
        return time.strftime("%Y-%m-%d", time.localtime(self.clock()))

    def _used(self):
        try:
            data = json.loads(self.path.read_text())
            if data.get("day") == self._today():
                return int(data.get("bytes", 0))
        except (OSError, ValueError):
            pass
        return 0

    def allowed(self, size_bytes):
        return self.limit is None or self._used() + size_bytes <= self.limit

    def record(self, size_bytes):
        if self.limit is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"day": self._today(), "bytes": self._used() + int(size_bytes)}))
        os.replace(tmp, self.path)
