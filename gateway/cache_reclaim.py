"""Give clean page cache back to the kernel on demand, without root and without touching swap.

The user's systemd cgroup (user@UID.service) owns a writable memory.reclaim file. Writing "<N>M swappiness=0" there reclaims
about N MiB, file pages only. The gateway's own cgroup is walked upwards to the nearest ancestor whose memory.reclaim is writable
(the user@UID.service level is preferred). Nothing here raises; a failure comes back as an error string.

An older kernel that does not know the swappiness key answers EINVAL: that is returned as an error and never retried as a plain
reclaim (a plain reclaim may push anonymous pages to swap).
"""
import errno
import os
import re
import threading
import time
from pathlib import Path

REPORT_MARGIN_MIB = 512         # reclaim the shortfall plus this much
MAX_WRITE_SECONDS = 15.0        # the write is synchronous; a stuck one is abandoned by the caller's thread budget, not retried


def meminfo_mib(text):
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key] = int(parts[0]) / 1024.0
    return out


def reclaimable_mib(mem):
    """Clean file cache that can go: Active(file) + Inactive(file) - Dirty (never below 0). mem: {key: MiB}.

    Shmem (tmpfs, /dev/shm) is not subtracted: it is swap-backed and lives on the anon LRU lists, so it is not part of
    Active(file)/Inactive(file) at all (it is only part of Cached)."""
    if not mem:
        return 0.0
    total = mem.get("Active(file)", 0.0) + mem.get("Inactive(file)", 0.0)
    return max(0.0, total - mem.get("Dirty", 0.0))


def _own_cgroup(proc_root):
    try:
        for line in (Path(proc_root) / "self" / "cgroup").read_text().splitlines():
            if line.startswith("0::"):
                return line[3:].strip()
    except OSError:
        pass
    return None


def find_reclaim_file(proc_root="/proc", cgroup_root="/sys/fs/cgroup", uid=None):
    """The memory.reclaim to write: the user@UID.service level when it is on the way up, else the nearest writable ancestor."""
    rel = _own_cgroup(proc_root)
    if not rel:
        return None
    parts = [p for p in rel.split("/") if p]
    candidates = [Path(cgroup_root).joinpath(*parts[:n], "memory.reclaim") for n in range(len(parts), 0, -1)]
    writable = [c for c in candidates if c.exists() and os.access(c, os.W_OK)]
    for c in writable:
        if re.fullmatch(r"user@\d+\.service", c.parent.name):
            return c
    return writable[0] if writable else None


def reclaim_page_cache(want_mib, proc_root="/proc", cgroup_root="/sys/fs/cgroup", clock=time.monotonic):
    """Reclaim up to want_mib of clean file cache. Returns (freed_mib, error); error is "" on success. Never raises.

    want_mib is capped to the reclaimable clean cache. freed is measured as the rise of MemFree."""
    try:
        want = float(want_mib)
    except (TypeError, ValueError):
        return 0, "bad size"
    try:
        before = meminfo_mib((Path(proc_root) / "meminfo").read_text())
    except OSError as error:
        return 0, f"no meminfo: {error}"
    want = min(want, reclaimable_mib(before))
    if want < 64:
        return 0, "no clean page cache to reclaim"
    target = find_reclaim_file(proc_root, cgroup_root)
    if target is None:
        return 0, "no writable memory.reclaim in this cgroup or above"
    result = {"error": None}

    def write():
        try:
            fd = os.open(target, os.O_WRONLY)
            try:
                os.write(fd, f"{int(want)}M swappiness=0".encode())
            finally:
                os.close(fd)
            result["error"] = ""
        except OSError as e:
            if e.errno == errno.EAGAIN:
                result["error"] = ""                # partial reclaim: whatever was freed counts
            elif e.errno == errno.EINVAL:
                result["error"] = "kernel does not accept swappiness= for memory.reclaim (not falling back: a plain reclaim may swap)"
            else:
                result["error"] = f"memory.reclaim write failed: {e}"
        except Exception as e:                      # noqa: BLE001 - never raises
            result["error"] = f"memory.reclaim write failed: {e}"

    # the write is synchronous in the kernel: run it in its own thread and stop waiting after MAX_WRITE_SECONDS
    started = clock()
    worker = threading.Thread(target=write, daemon=True, name="memory-reclaim")
    worker.start()
    worker.join(MAX_WRITE_SECONDS)
    error = result["error"]
    if error is None:
        error = f"memory.reclaim did not return within {MAX_WRITE_SECONDS:.0f} s (left running; not retried)"
    elif not error and clock() - started > MAX_WRITE_SECONDS:
        error = f"memory.reclaim took {clock() - started:.0f} s"
    if error:
        return 0, error
    try:
        after = meminfo_mib((Path(proc_root) / "meminfo").read_text())
        freed = max(0.0, after.get("MemFree", 0.0) - before.get("MemFree", 0.0))
    except OSError:
        freed = 0.0
    return int(freed), ""
