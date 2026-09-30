"""RAM bandwidth measurement. Runs the small C helper `tools/membw.c` (a prebuilt one next to it, else compiled with the
system C compiler into a temporary directory). If neither works the result is None and the planner uses a conservative
estimate: an unknown number is reported as unknown, never guessed. Takes a few seconds and uses at most a quarter of
the free RAM."""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "tools" / "membw.c"
MAX_MIB_PER_THREAD = 256


def parse(output):
    m = re.search(r"membw_gbs\s+([0-9.]+)", output or "")
    return float(m.group(1)) if m else None


def plan_threads(physical_cores, available_bytes):
    """Threads and MiB per thread that stay within a quarter of the available RAM (and a sane ceiling)."""
    threads = max(1, min(physical_cores, 256))
    per = min(MAX_MIB_PER_THREAD, int(available_bytes * 0.25 / threads / (1 << 20)))
    return threads, per


def physical_cpu_ids(read=None):
    """One CPU id per physical core (the lowest id of each SMT sibling group), from Linux sysfs; [] elsewhere.
    Pinning measurement threads to these avoids two threads sharing one core, whatever the numbering scheme is."""
    read = read or _read
    ids, seen = [], set()
    n = 0
    while True:
        text = read(f"/sys/devices/system/cpu/cpu{n}/topology/thread_siblings_list")
        if text is None:
            break
        siblings = text.strip()
        if siblings not in seen:
            seen.add(siblings)
            ids.append(n)
        n += 1
    return ids


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def build(dest_dir):
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if not cc or not SOURCE.exists():
        return None
    out = Path(dest_dir) / "membw"
    r = subprocess.run([cc, "-O2", "-pthread", str(SOURCE), "-o", str(out)], capture_output=True, timeout=60)
    return out if r.returncode == 0 else None


def measure(physical_cores, available_bytes, passes=4, timeout=60):
    """Measured read bandwidth in GB/s across all cores, or None if it could not be measured."""
    threads, per = plan_threads(physical_cores, available_bytes)
    if per < 4:
        return None                                   # too little free RAM to measure without pressure
    prebuilt = HERE / "tools" / "membw"
    with tempfile.TemporaryDirectory(prefix="shura-membw-") as tmp:
        exe = prebuilt if prebuilt.exists() and os.access(prebuilt, os.X_OK) else build(tmp)
        if not exe:
            return None
        try:
            args = [str(exe), str(threads), str(per), str(passes)]
            ids = physical_cpu_ids()
            if len(ids) >= threads:
                args.append(",".join(str(i) for i in ids[:threads]))
            r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            return None
    return parse(r.stdout) if r.returncode == 0 else None
