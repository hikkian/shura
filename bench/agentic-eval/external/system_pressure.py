"""Supplement S6 with the pilot's zram-capacity and PSI memory-full limits."""
from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Callable


def read_system_pressure() -> dict[str, float | None]:
    zram_pct = None
    try:
        for line in Path("/proc/swaps").read_text().splitlines()[1:]:
            fields = line.split()
            if fields and fields[0].startswith("/dev/zram") and len(fields) >= 4:
                size_kib, used_kib = int(fields[2]), int(fields[3])
                zram_pct = 100.0 * used_kib / size_kib if size_kib else 0.0
                break
    except (OSError, ValueError):
        pass
    psi_full = None
    try:
        for line in Path("/proc/pressure/memory").read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == "full":
                psi_full = float(next(item.split("=", 1)[1] for item in fields if item.startswith("avg10=")))
                break
    except (OSError, ValueError, StopIteration):
        pass
    return {"zram_pct": zram_pct, "psi_memory_full_avg10": psi_full}


class SystemPressureGuard(threading.Thread):
    def __init__(self, trip: Callable[[str], None], *, period: float = 5.0,
                 zram_limit_pct: float = 92.0, psi_limit_pct: float = 10.0,
                 reader: Callable[[], dict] = read_system_pressure):
        super().__init__(daemon=True)
        self.trip_callback = trip
        self.period = period
        self.zram_limit_pct = zram_limit_pct
        self.psi_limit_pct = psi_limit_pct
        self.reader = reader
        self.stop_event = threading.Event()
        self.baseline = self._read()
        self.peak_zram_pct = self.baseline.get("zram_pct")
        self.peak_psi_pct = self.baseline.get("psi_memory_full_avg10")
        self.samples = 0
        self.psi_high_streak = 0

    def _read(self):
        try:
            return self.reader()
        except Exception:
            return {"zram_pct": None, "psi_memory_full_avg10": None}

    def sample_once(self):
        sample = self._read()
        self.samples += 1
        zram = sample.get("zram_pct")
        psi = sample.get("psi_memory_full_avg10")
        if zram is not None:
            self.peak_zram_pct = max(self.peak_zram_pct or zram, zram)
            if zram > self.zram_limit_pct:
                self.trip_callback(f"zram occupancy {zram:.1f}% > {self.zram_limit_pct:.1f}%")
        if psi is not None:
            self.peak_psi_pct = max(self.peak_psi_pct or psi, psi)
            self.psi_high_streak = self.psi_high_streak + 1 if psi > self.psi_limit_pct else 0
            if self.psi_high_streak >= 2:
                self.trip_callback(f"PSI memory full avg10 {psi:.2f}% > {self.psi_limit_pct:.2f}% twice")
        return sample

    def run(self):
        while not self.stop_event.wait(self.period):
            self.sample_once()

    def stop(self):
        self.stop_event.set()
