"""Small, read-only telemetry source adapters used by Shura Monitor.

Each adapter has an availability probe and a sample method. Missing files,
permissions, libraries, or sensors degrade only that source to empty data.
"""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import time


GPU_EMPTY = {
    "total_mib": None, "used_mib": None, "free_mib": None,
    "model_process_mib": None, "utilization_percent": None,
    "temperature_c": None, "power_w": None,
    "pcie_generation": None, "pcie_width": None,
}


def _read_number(path, scale=1.0):
    try:
        value = float(Path(path).read_text().strip()) / scale
        return value if math.isfinite(value) else None
    except (OSError, ValueError, OverflowError):
        return None


def drm_cards(sys_root="/sys"):
    root = Path(sys_root)
    for device in sorted(root.glob("class/drm/card[0-9]*/device")):
        try:
            vendor = device.joinpath("vendor").read_text().strip().lower()
        except OSError:
            continue
        yield vendor, device


class TelemetrySource:
    name = "none"

    def available(self):
        return False

    def sample(self):
        return {}


class NoGPU(TelemetrySource):
    """Explicit no-data GPU adapter; it never guesses a device."""


class NvidiaGPU(TelemetrySource):
    name = "gpu_nvidia"

    def __init__(self, provider=None):
        self.provider = provider

    def available(self):
        return self.provider is not None

    def sample(self):
        if not self.available():
            return {}
        try:
            data = self.provider()
            if isinstance(data, dict) and any(data.get(k) is not None for k in (
                "total_mib", "free_mib", "used_mib", "utilization_percent", "temperature_c", "power_w")):
                return dict(data)
        except Exception:
            pass
        return {}


class SysfsGPU(TelemetrySource):
    vendor_id = ""
    name = "gpu"

    def __init__(self, sys_root="/sys"):
        self.sys_root = Path(sys_root)

    def devices(self):
        return [path for vendor, path in drm_cards(self.sys_root) if vendor == self.vendor_id]

    @staticmethod
    def _hwmon(device):
        for path in device.glob("hwmon/hwmon*"):
            yield path

    def _sensor_values(self, device):
        temperatures, powers = [], []
        for hwmon in self._hwmon(device):
            for path in hwmon.glob("temp*_input"):
                value = _read_number(path, 1000)
                if value is not None:
                    temperatures.append(value)
            for name in ("power1_average", "power1_input"):
                value = _read_number(hwmon / name, 1_000_000)
                if value is not None:
                    powers.append(value)
        return max(temperatures) if temperatures else None, max(powers) if powers else None

    def _device_sample(self, device):
        used = _read_number(device / "mem_info_vram_used", 1048576)
        total = _read_number(device / "mem_info_vram_total", 1048576)
        busy = _read_number(device / "gpu_busy_percent")
        temp, power = self._sensor_values(device)
        if all(value is None for value in (used, total, busy, temp, power)):
            return {}
        return {"total_mib": total, "used_mib": used,
                "free_mib": max(0.0, total-used) if total is not None and used is not None else None,
                "model_process_mib": None, "utilization_percent": busy,
                "temperature_c": temp, "power_w": power,
                "pcie_generation": None, "pcie_width": None}

    def available(self):
        return any(bool(self._device_sample(device)) for device in self.devices())

    def sample(self):
        samples = [self._device_sample(device) for device in self.devices()]
        samples = [sample for sample in samples if sample]
        if not samples:
            return {}
        # Report aggregate memory for multi-card machines and mean busy load.
        keys_sum = ("total_mib", "used_mib", "free_mib")
        result = dict(samples[0])
        for key in keys_sum:
            values = [sample.get(key) for sample in samples]
            result[key] = sum(values) if values and all(value is not None for value in values) else None
        busy = [sample["utilization_percent"] for sample in samples if sample.get("utilization_percent") is not None]
        result["utilization_percent"] = sum(busy)/len(busy) if busy else None
        result["model_process_mib"] = None
        return result


class AMDGPU(SysfsGPU):
    name = "gpu_amd"
    vendor_id = "0x1002"


class IntelGPU(SysfsGPU):
    name = "gpu_intel"
    vendor_id = "0x8086"


class LinuxSystemSource(TelemetrySource):
    name = "system_linux"

    def __init__(self, proc_root="/proc", sys_root="/sys", home="/", wall_clock=time.time):
        self.proc_root, self.sys_root, self.home = Path(proc_root), Path(sys_root), Path(home)
        self.wall_clock = wall_clock
        self.cpu_previous = None
        self.nvme_base = None
        self.nvme_delta = None

    def available(self):
        return self.proc_root.joinpath("meminfo").is_file() or self.proc_root.joinpath("stat").is_file()

    def capabilities(self):
        return {
            "system_linux": self.available(),
            "zram": any(self.sys_root.glob("block/zram*/mm_stat")),
            "psi": self.proc_root.joinpath("pressure/memory").is_file(),
            "cpu_temp": cpu_temperature_available(self.sys_root),
            "nvme": any(self.sys_root.glob("block/nvme*/stat")),
        }

    def sample(self, readers):
        def read(key, *args, default=None):
            try:
                return readers[key](*args)
            except Exception:
                return default
        total_kib, avail_kib = read("meminfo", self.proc_root, default=(None, None)) or (None, None)
        zram = read("zram", self.sys_root, default={}) or {}
        psi = read("psi", self.proc_root, default={}) or {}
        current, cores, frequency = read("cpu", self.proc_root, self.sys_root, self.cpu_previous,
                                         default=(self.cpu_previous or {}, [], None))
        self.cpu_previous = current
        nvme = read("nvme", self.sys_root)
        if self.nvme_base is None:
            self.nvme_base = nvme
        if nvme is not None and self.nvme_base is not None:
            self.nvme_delta = max(0, nvme-self.nvme_base)
        try:
            disk = shutil.disk_usage(self.home).free / 1e9
        except OSError:
            disk = None
        return {
            "memory": {"ram_total_mib": total_kib/1024 if total_kib is not None else None,
                       "ram_available_mib": avail_kib/1024 if avail_kib is not None else None,
                       "ram_used_mib": (total_kib-avail_kib)/1024 if total_kib is not None and avail_kib is not None else None,
                       "zram_original_mib": zram.get("original_bytes")/1048576 if zram.get("original_bytes") is not None else None,
                       "zram_compressed_mib": zram.get("compressed_bytes")/1048576 if zram.get("compressed_bytes") is not None else None,
                       "zram_used_percent": zram.get("used_percent"),
                       "zram_disk_mib": zram.get("disk_bytes")/1048576 if zram.get("disk_bytes") is not None else None,
                       "psi_some_avg10": psi.get("some"), "psi_full_avg10": psi.get("full")},
            "cpu": {"temperature_c": read("cpu_temp", self.sys_root),
                    "frequency_mhz": frequency, "cores": cores},
            "ssd_free_gb": disk, "nvme_written_bytes": self.nvme_delta,
            "sampled_at": self.wall_clock(),
        }


class WindowsSystemSource(TelemetrySource):
    """Standard-library-only Windows fallback; counters are intentionally unavailable for now."""
    name = "system_windows"

    def available(self):
        return False

    def sample(self):
        return {}


def cpu_temperature_available(sys_root="/sys"):
    recognized = {"k10temp", "coretemp", "zenpower", "cpu_thermal", "acpitz", "peci_cputemp"}
    for hwmon in Path(sys_root).glob("class/hwmon/hwmon*"):
        try:
            if hwmon.joinpath("name").read_text().strip() in recognized and any(hwmon.glob("temp*_input")):
                return True
        except OSError:
            continue
    return False


def gpu_sources(sys_root="/sys", nvidia_provider=None):
    nvidia = NvidiaGPU(nvidia_provider)
    amd, intel = AMDGPU(sys_root), IntelGPU(sys_root)
    return {"gpu_nvidia": nvidia, "gpu_amd": amd, "gpu_intel": intel,
            "gpu_none": NoGPU()}


def sample_gpu(sys_root="/sys", nvidia_provider=None):
    sources = gpu_sources(sys_root, nvidia_provider)
    for capability in ("gpu_nvidia", "gpu_amd", "gpu_intel"):
        source = sources[capability]
        if source.available():
            data = source.sample()
            if data:
                data.setdefault("source", capability)
                return data
    return {}
