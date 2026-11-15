"""AMD GPU telemetry without NVML: the same calls as gpu_nvml.NVML for the parts the gateway uses.

Everything is read from files, read only, with no tool and no root:
  - the card: /sys/class/drm/cardN/device (amdgpu): mem_info_vram_total, mem_info_vram_used, gpu_busy_percent,
    current_link_speed, current_link_width; temperature and power come from its hwmon folder;
  - each process's VRAM and GPU load: the fdinfo of its DRM file descriptors (drm-memory-vram, drm-engine-*).
A reading that is not available raises AMDError (a RuntimeError, as NVML does); monitor_info turns it into None.
The roots can be moved for tests: SHURA_SYSFS_ROOT (default /sys) and SHURA_PROC_ROOT (default /proc).
"""

import os
import re
import time
from pathlib import Path

MIB = 1 << 20
AMD_VENDOR = "0x1002"
# A card with less VRAM than this is integrated graphics (a share of system memory): it is never chosen.
MIN_DISCRETE_VRAM = 2 << 30
# Reading /proc for DRM file descriptors costs about 50 ms on a busy desktop; one scan is reused for this long.
SCAN_TTL_SECONDS = 0.5
# GPU load of a process = busy share of these engines (graphics and compute) over the wall time between two scans.
ENGINES = ("drm-engine-gfx", "drm-engine-compute")
UNITS = {"KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}
PCI_GENERATION = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}
PCI_ADDRESS = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
CARD = re.compile(r"card\d+")


class AMDError(RuntimeError):
    """A reading that is not available. `backend` names the source in the status text."""

    backend = "AMD"


def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _int(path):
    try:
        return int(_read(path))
    except (TypeError, ValueError):
        return None


def parse_fdinfo(text):
    """The DRM facts of one fdinfo file. Lines that are not what amdgpu writes are skipped, never fatal."""
    fact = {"driver": None, "client": None, "pdev": None, "vram_kib": None, "engine_ns": None}
    for line in text.splitlines():
        key, colon, value = line.partition(":")
        if not colon:
            continue
        key, parts = key.strip(), value.split()
        if key == "drm-driver" and parts:
            fact["driver"] = " ".join(parts)
        elif key == "drm-client-id" and len(parts) == 1 and parts[0].isdigit():
            fact["client"] = int(parts[0])
        elif key == "drm-pdev" and len(parts) == 1:
            fact["pdev"] = parts[0]
        elif key == "drm-memory-vram" and len(parts) == 2 and parts[0].isdigit() and parts[1] in UNITS:
            fact["vram_kib"] = int(parts[0]) * UNITS[parts[1]] // 1024
        elif key in ENGINES and len(parts) == 2 and parts[0].isdigit() and parts[1] == "ns":
            fact["engine_ns"] = (fact["engine_ns"] or 0) + int(parts[0])
    return fact


def find_card(sys_root):
    """(device folder, VRAM bytes) of the AMD card with the most VRAM, or None. Integrated graphics is not chosen."""
    best = None
    for card in sorted(Path(sys_root, "class", "drm").glob("card*")):
        if not CARD.fullmatch(card.name):
            continue  # card1-DP-1 and similar are connectors, not cards
        device = card / "device"
        if _read(device / "vendor") != AMD_VENDOR:
            continue
        total = _int(device / "mem_info_vram_total")
        if total is None or total < MIN_DISCRETE_VRAM:
            continue
        if best is None or total > best[1]:
            best = (device, total)
    return best


class AMDDevice:
    """The NVML interface for an AMD card. Methods are called by the gateway as they are called on gpu_nvml.NVML."""

    def __init__(self, sys_root=None, proc_root=None, clock=time.monotonic):
        self.sys_root = sys_root or os.environ.get("SHURA_SYSFS_ROOT") or "/sys"
        self.proc_root = Path(proc_root or os.environ.get("SHURA_PROC_ROOT") or "/proc")
        self.clock = clock
        found = find_card(self.sys_root)
        if found is None:
            raise AMDError(f"no AMD graphics card with at least {MIN_DISCRETE_VRAM >> 30} GiB of VRAM under {self.sys_root}")
        self.device = found[0]
        resolved = self.device.resolve().name  # the PCI address, which fdinfo names in drm-pdev
        self.pci = resolved if PCI_ADDRESS.fullmatch(resolved) else None
        self.needs_reinit = False  # sysfs has no handle that can die
        self.unreadable = 0        # DRM file descriptors of processes that could not be read, at the last scan
        self._scan_cache = None    # (time, {pid: {client: fact}})
        self._engine_seen = {}     # (pid, client) -> (engine ns, scan time)
        self._load = (None, {})    # (scan time the last result is for, result)

    def close(self):
        return None

    # --- the card (sysfs) ---

    def _vram(self):
        total = _int(self.device / "mem_info_vram_total")
        used = _int(self.device / "mem_info_vram_used")
        if total is None or used is None or total <= 0:
            raise AMDError("AMD VRAM counters are not readable")
        return total, used

    def memory_info(self):
        total, used = self._vram()
        return {"total_mib": total // MIB, "free_mib": max(0, total - used) // MIB, "used_mib": used // MIB}

    def card_free_mib(self):
        return self.memory_info()["free_mib"]

    def utilization(self):
        value = _int(self.device / "gpu_busy_percent")
        if value is None:
            raise AMDError("AMD gpu_busy_percent is not readable")
        return {"gpu_percent": value}

    def _hwmon(self, name):
        for folder in sorted(self.device.glob("hwmon/hwmon*")):
            value = _int(folder / name)
            if value is not None:
                return value
        return None

    def temperature_c(self):
        value = self._hwmon("temp1_input")  # millidegrees Celsius: the GPU edge sensor
        if value is None:
            raise AMDError("AMD temperature is not readable")
        return value // 1000

    def power_usage_w(self):
        for name in ("power1_average", "power1_input"):  # microwatts
            value = self._hwmon(name)
            if value is not None:
                return value / 1_000_000
        raise AMDError("AMD power is not readable")

    def pcie_link(self):
        width = _int(self.device / "current_link_width")
        match = re.match(r"(\d+(?:\.\d+)?) GT/s", _read(self.device / "current_link_speed") or "")
        generation = PCI_GENERATION.get(float(match.group(1))) if match else None
        if width is None or generation is None:
            raise AMDError("AMD PCIe link is not readable")
        return {"generation": generation, "width": width}

    # --- the processes (fdinfo) ---

    def _scan(self):
        """{pid: {client: fact}} for the DRM clients of this card. A scan is reused for SCAN_TTL_SECONDS."""
        now = self.clock()
        if self._scan_cache is not None and now - self._scan_cache[0] < SCAN_TTL_SECONDS:
            return self._scan_cache[1]
        try:
            entries = list(os.scandir(self.proc_root))
        except OSError as error:
            raise AMDError(f"the process list is not readable at {self.proc_root}: {error.strerror or error}") from error
        found, unreadable = {}, 0
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            fd_dir = self.proc_root / entry.name / "fd"
            try:
                names = os.listdir(fd_dir)
            except OSError:
                continue  # the process ended, or it is not ours to read
            for name in names:
                try:
                    target = os.readlink(fd_dir / name)
                except OSError:
                    continue
                if not target.startswith("/dev/dri/"):
                    continue
                try:
                    text = (self.proc_root / entry.name / "fdinfo" / name).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    unreadable += 1
                    continue
                fact = parse_fdinfo(text)
                if fact["driver"] != "amdgpu":
                    continue
                if self.pci and fact["pdev"] and fact["pdev"] != self.pci:
                    continue  # a client of another AMD card
                # one client may hold several file descriptors: it is counted once
                key = fact["client"] if fact["client"] is not None else ("fd", name)
                found.setdefault(pid, {}).setdefault(key, fact)
        self.unreadable = unreadable
        self._scan_cache = (now, found)
        return found

    @staticmethod
    def _mib(clients):
        known = [fact["vram_kib"] for fact in clients.values() if fact["vram_kib"] is not None]
        return sum(known) // 1024 if known else None

    def processes(self):
        """{pid: MiB of VRAM held by the process's AMD clients}; None where no client shows a VRAM figure."""
        return {pid: self._mib(clients) for pid, clients in self._scan().items()}

    def proc_mib(self, pid):
        clients = self._scan().get(pid)
        return self._mib(clients) if clients else None

    def process_utilization(self, last_seen=0):
        """({pid: percent}, time). The percent is the busy share of the graphics and compute engines since the previous scan.
        The first call has nothing to compare with and returns {}. The time is the time of the scan, in ns.
        last_seen is kept for the NVML interface."""
        self._scan()
        stamp, clients = self._scan_cache
        if stamp == self._load[0]:
            return dict(self._load[1]), int(stamp * 1e9)
        totals, current = {}, {}
        for pid, per_client in clients.items():
            for key, fact in per_client.items():
                if fact["engine_ns"] is None:
                    continue
                ns = fact["engine_ns"]
                current[(pid, key)] = (ns, stamp)
                before = self._engine_seen.get((pid, key))
                if before is not None and stamp > before[1] and ns >= before[0]:
                    totals[pid] = totals.get(pid, 0.0) + (ns - before[0]) / ((stamp - before[1]) * 1e9) * 100
        self._engine_seen = current
        result = {pid: min(100, int(round(value))) for pid, value in totals.items()}
        self._load = (stamp, result)
        return dict(result), int(stamp * 1e9)

    # --- one sample for the monitor: every field degrades to None on its own ---

    def monitor_info(self, pid=None):
        result = {"source": "gpu_amd"}
        try:
            result.update(self.memory_info())
        except (OSError, RuntimeError, ValueError):
            result.update({"total_mib": None, "free_mib": None, "used_mib": None})
        try:
            result["utilization_percent"] = self.utilization()["gpu_percent"]
        except (OSError, RuntimeError, ValueError):
            result["utilization_percent"] = None
        for key, method in (("temperature_c", self.temperature_c), ("power_w", self.power_usage_w)):
            try:
                result[key] = method()
            except (OSError, RuntimeError, ValueError):
                result[key] = None
        try:
            result.update({f"pcie_{key}": value for key, value in self.pcie_link().items()})
        except (OSError, RuntimeError, ValueError):
            result.update({"pcie_generation": None, "pcie_width": None})
        try:
            result["model_process_mib"] = self.processes().get(pid, 0 if pid is None else None)
        except (OSError, RuntimeError, ValueError):
            result["model_process_mib"] = 0 if pid is None else None
        return result
