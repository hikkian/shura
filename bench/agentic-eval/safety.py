"""Hard limits for an unattended run. A watchdog thread reads the machine every few seconds and trips an event when a limit is
broken for long enough; the experiment then stops its server and exits cleanly. Nobody is awake to look at it, so the limits
are the ones used for every GPU test on this machine: GPU above 83 C or CPU above 90 C for 30 s, free VRAM below 150 MiB twice
in a row, less than 1.5 GiB of RAM available twice in a row, less than 20 GB free on the disk."""
import ctypes
import shutil
import threading
import time
from pathlib import Path

LIMITS = {"gpu_c": 83, "cpu_c": 90, "hot_seconds": 30, "vram_free_mib": 150, "ram_avail_gib": 1.5, "disk_free_gb": 20}


class _Mem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class Gpu:
    """Just enough of NVML through ctypes (no nvidia-smi: it can hang)."""

    def __init__(self, index=0):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise RuntimeError("nvmlInit failed")
        self.h = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.h)) != 0:
            raise RuntimeError("no GPU")

    def free_mib(self):
        m = _Mem()
        self.lib.nvmlDeviceGetMemoryInfo(self.h, ctypes.byref(m))
        return m.free >> 20

    def used_mib(self):
        m = _Mem()
        self.lib.nvmlDeviceGetMemoryInfo(self.h, ctypes.byref(m))
        return m.used >> 20

    def temp_c(self):
        t = ctypes.c_uint()
        self.lib.nvmlDeviceGetTemperature(self.h, 0, ctypes.byref(t))
        return t.value


def cpu_temp_c():
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (hw / "name").read_text().strip() in ("k10temp", "coretemp"):
                return int((hw / "temp1_input").read_text()) / 1000
        except (OSError, ValueError):
            continue
    return None


def ram_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576
    return None


class Watchdog(threading.Thread):
    def __init__(self, path_for_disk, limits=None, period=5.0, gpu=None, reader=None):
        super().__init__(daemon=True)
        self.limits, self.period, self.disk = {**LIMITS, **(limits or {})}, period, path_for_disk
        self.tripped = threading.Event()
        self.reason = ""
        self.stop_event = threading.Event()
        self.peak = {"gpu_c": 0, "cpu_c": 0, "vram_used_mib": 0}
        self.read = reader or self._read
        try:
            self.gpu = gpu if gpu is not None else Gpu()
        except (OSError, RuntimeError):
            self.gpu = None

    def _read(self):
        g = self.gpu
        return {"gpu_c": g.temp_c() if g else None, "cpu_c": cpu_temp_c(), "vram_free": g.free_mib() if g else None,
                "vram_used": g.used_mib() if g else None, "ram": ram_available_gib(),
                "disk": shutil.disk_usage(self.disk).free / 1e9}

    def run(self):
        hot_since, low_vram, low_ram = None, 0, 0
        L = self.limits
        while not self.stop_event.wait(self.period):
            try:
                r = self.read()
            except Exception:
                continue                                    # a failed read is not a reason to stop the night
            for k, v in (("gpu_c", r.get("gpu_c")), ("cpu_c", r.get("cpu_c")), ("vram_used_mib", r.get("vram_used"))):
                if v is not None:
                    self.peak[k] = max(self.peak[k], v)
            hot = (r.get("gpu_c") or 0) > L["gpu_c"] or (r.get("cpu_c") or 0) > L["cpu_c"]
            hot_since = (hot_since or time.monotonic()) if hot else None
            low_vram = low_vram + 1 if (r.get("vram_free") is not None and r["vram_free"] < L["vram_free_mib"]) else 0
            low_ram = low_ram + 1 if (r.get("ram") is not None and r["ram"] < L["ram_avail_gib"]) else 0
            if hot_since and time.monotonic() - hot_since >= L["hot_seconds"]:
                self._trip(f"too hot for {L['hot_seconds']} s (GPU {r.get('gpu_c')} C, CPU {r.get('cpu_c')} C)")
            elif low_vram >= 2:
                self._trip(f"free VRAM {r['vram_free']} MiB")
            elif low_ram >= 2:
                self._trip(f"available RAM {r['ram']:.1f} GiB")
            elif r.get("disk") is not None and r["disk"] < L["disk_free_gb"]:
                self._trip(f"disk free {r['disk']:.0f} GB")

    def _trip(self, why):
        self.reason = why
        self.tripped.set()
        self.stop_event.set()

    def stop(self):
        self.stop_event.set()
