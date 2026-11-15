#!/usr/bin/env python3
"""Create a minimal, privacy-filtered Shura monitor hardware snapshot."""
from __future__ import annotations

import argparse
from pathlib import Path
import platform
import re
import sys

GATEWAY = Path(__file__).resolve().parents[2] / "gateway"
sys.path.insert(0, str(GATEWAY))
from monitor_sources import LinuxSystemSource, drm_cards, gpu_sources  # noqa: E402


def _safe(value):
    text = str(value or "unavailable").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"(?i)\b(serial|uuid|hostname|host|api[_-]?key|token)\b\s*[:=]\s*\S+", "[redacted]", text)
    text = re.sub(r"(?:[A-Za-z]:)?/(?:[^\s]+)", "[path]", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[address]", text)
    text = re.sub(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b", "[address]", text)
    return text[:120]


def hardware_dump(*, sys_root="/sys", proc_root="/proc", nvml_factory=None):
    """Read only whitelisted hardware attributes; never list paths or process identity."""
    lines = ["Shura hardware snapshot", f"Platform: {_safe(platform.system())}",
             f"Kernel: {_safe(platform.release())}"]
    nvml = None
    if nvml_factory is not None:
        try:
            nvml = nvml_factory()
        except Exception:
            nvml = None
    cards = list(drm_cards(sys_root))
    if nvml is not None:
        try:
            lines += [f"GPU model: {_safe(nvml.device_name())}", f"GPU driver: {_safe(nvml.driver_version())}"]
        except Exception:
            lines.append("GPU model/driver: unavailable from NVIDIA telemetry")
    for index, (vendor, device) in enumerate(cards, 1):
        vendor_name = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel"}.get(vendor, "Other")
        product = ""
        for key in ("product_name", "product_number"):
            try:
                value = device.joinpath(key).read_text().strip()
                if value:
                    product = _safe(value)
                    break
            except OSError:
                continue
        try:
            driver = device.joinpath("driver").resolve(strict=True).name
        except OSError:
            driver = "unavailable"
        driver_version = "unavailable"
        try:
            driver_version = _safe(device.joinpath("driver/module/version").read_text().strip())
        except OSError:
            pass
        device_id = "unavailable"
        try:
            device_id = _safe(device.joinpath("device").read_text().strip())
        except OSError:
            pass
        lines.append(f"GPU {index}: {vendor_name}" + (f" {product}" if product else " model unavailable") +
                     f"; PCI device {device_id}; driver {_safe(driver)} {_safe(driver_version)}")
        for key in ("mem_info_vram_total", "gpu_busy_percent"):
            try:
                value = device.joinpath(key).read_text().strip()
            except OSError:
                value = "unavailable"
            lines.append(f"  {key}: {_safe(value)}")
        for hwmon in sorted(device.glob("hwmon/hwmon*")):
            for key in ("temp1_input", "power1_average"):
                try:
                    value = hwmon.joinpath(key).read_text().strip()
                except OSError:
                    continue
                lines.append(f"  {key}: {_safe(value)}")
    try:
        nvidia_source = gpu_sources(sys_root, (lambda: {}) if nvml is not None else None)["gpu_nvidia"].available()
    except Exception:
        nvidia_source = False
    try:
        system_source = LinuxSystemSource(proc_root=proc_root, sys_root=sys_root)
        syscaps = system_source.capabilities()
        linux_source = system_source.available()
    except Exception:
        syscaps, linux_source = {}, False
    vendors = {vendor for vendor, _ in cards}
    available = [
        f"gpu_nvidia={'yes' if nvidia_source else 'no'}",
        f"gpu_amd={'yes' if '0x1002' in vendors else 'no'}",
        f"gpu_intel={'yes' if '0x8086' in vendors else 'no'}",
        f"system_linux={'yes' if linux_source else 'no'}",
    ] + [f"{key}={'yes' if syscaps.get(key) else 'no'}" for key in ("zram", "psi", "cpu_temp", "nvme")]
    available += ["slots=runtime", "timings=runtime", "expert_cache=runtime"]
    lines.append("Monitor sources: " + ", ".join(available))
    return "\n".join(lines) + "\n"


def main(argv=None, *, nvml_factory=None, sys_root="/sys", proc_root="/proc"):
    parser = argparse.ArgumentParser(prog="shura doctor --hw-dump")
    parser.add_argument("--hw-dump", action="store_true", required=True)
    parser.add_argument("--output", default="shura-hw-dump.txt")
    args = parser.parse_args(argv)
    if nvml_factory is None:
        try:
            from gpu_nvml import NVML
            nvml_factory = NVML
        except Exception:
            pass
    text = hardware_dump(nvml_factory=nvml_factory, sys_root=sys_root, proc_root=proc_root)
    Path(args.output).write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    print("Saved filtered report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
