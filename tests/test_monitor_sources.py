import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gateway"))
import monitor_sources as sources


def put(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(value), encoding="utf-8")
    return path


class GPUFixtureSources(unittest.TestCase):
    def _device(self, root, vendor, device_id="0x1234"):
        base = root / "class/drm/card0/device"
        put(root, "class/drm/card0/device/vendor", vendor)
        put(root, "class/drm/card0/device/device", device_id)
        return base

    def test_amd_sysfs_vram_utilization_temperature_and_power(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            root = Path(tmp)
            dev = self._device(root, "0x1002")
            put(root, "class/drm/card0/device/mem_info_vram_used", 2 * 1024**3)
            put(root, "class/drm/card0/device/mem_info_vram_total", 8 * 1024**3)
            put(root, "class/drm/card0/device/gpu_busy_percent", 67)
            put(root, "class/drm/card0/device/hwmon/hwmon0/temp1_input", 71000)
            put(root, "class/drm/card0/device/hwmon/hwmon0/power1_average", 183000000)
            gpu = sources.AMDGPU(root)
            self.assertTrue(gpu.available())
            sample = gpu.sample()
            self.assertEqual(sample["used_mib"], 2048)
            self.assertEqual(sample["free_mib"], 6144)
            self.assertEqual(sample["utilization_percent"], 67)
            self.assertEqual(sample["temperature_c"], 71)
            self.assertEqual(sample["power_w"], 183)

    def test_intel_sysfs_source_uses_only_readable_attributes(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            root = Path(tmp)
            self._device(root, "0x8086")
            put(root, "class/drm/card0/device/mem_info_vram_used", 1024**3)
            put(root, "class/drm/card0/device/mem_info_vram_total", 4 * 1024**3)
            put(root, "class/drm/card0/device/gpu_busy_percent", 22)
            gpu = sources.IntelGPU(root)
            self.assertTrue(gpu.available())
            self.assertEqual(gpu.sample()["free_mib"], 3072)
            self.assertEqual(gpu.sample()["utilization_percent"], 22)

    def test_no_data_and_failed_sensor_leave_other_source_independent(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            root = Path(tmp)
            empty = sources.AMDGPU(root)
            self.assertFalse(empty.available())
            self.assertEqual(empty.sample(), {})
            self.assertEqual(sources.NoGPU().sample(), {})
            put(root, "class/drm/card1/device/vendor", "0x1002")
            put(root, "class/drm/card1/device/gpu_busy_percent", "not-a-number")
            put(root, "class/drm/card1/device/mem_info_vram_used", "bad")
            self.assertFalse(sources.AMDGPU(root).available())

    def test_failed_nvidia_probe_falls_back_to_amd_sysfs_source(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            root = Path(tmp)
            self._device(root, "0x1002")
            put(root, "class/drm/card0/device/mem_info_vram_used", 1024**3)
            put(root, "class/drm/card0/device/mem_info_vram_total", 8 * 1024**3)
            def broken():
                raise OSError("NVML unavailable")
            result = sources.sample_gpu(root, broken)
            self.assertEqual(result["source"], "gpu_amd")
            self.assertEqual(result["free_mib"], 7168)

    def test_nvidia_adapter_degrades_without_nvml_and_accepts_injected_data(self):
        absent = sources.NvidiaGPU()
        self.assertFalse(absent.available())
        self.assertEqual(absent.sample(), {})
        sample = sources.NvidiaGPU(lambda: {"total_mib": 12000, "free_mib": 5000}).sample()
        self.assertEqual(sample["free_mib"], 5000)


class LinuxSystemFixtures(unittest.TestCase):
    def test_capabilities_and_sample_are_from_fixture_roots(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            root = Path(tmp)
            proc, sysroot = root / "proc", root / "sys"
            put(proc, "meminfo", "MemTotal: 8192000 kB\nMemAvailable: 4096000 kB\n")
            put(proc, "stat", "cpu 1 2 3 4 5 6 7 8\ncpu0 10 0 5 80 5 0 0 0 0 0\n")
            put(proc, "pressure/memory", "some avg10=1.0 avg60=1 total=1\nfull avg10=0.1 avg60=0 total=1\n")
            put(sysroot, "block/zram0/mm_stat", "2097152 1048576 524288 0 0 0 0\n")
            put(sysroot, "block/zram0/disksize", str(1024 * 1024 * 1024))
            put(sysroot, "devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 3500000)
            put(sysroot, "class/hwmon/hwmon0/name", "coretemp")
            put(sysroot, "class/hwmon/hwmon0/temp1_input", 61000)
            put(sysroot, "block/nvme0n1/stat", "0 0 0 0 0 0 1024 0\n")
            source = sources.LinuxSystemSource(proc, sysroot, root, wall_clock=lambda: 123.0)
            caps = source.capabilities()
            self.assertTrue(caps["system_linux"])
            self.assertTrue(caps["zram"])
            self.assertTrue(caps["psi"])
            self.assertTrue(caps["cpu_temp"])
            self.assertTrue(caps["nvme"])
            sample = source.sample({
                "meminfo": lambda p: (8192000, 4096000), "zram": lambda _: {"original_bytes": 2097152, "compressed_bytes": 1048576, "used_bytes": 524288, "disk_bytes": 1024**3, "used_percent": .05},
                "psi": lambda _: {"some": 1.0, "full": .1}, "cpu": lambda _p, _s, _prev: ({0: [1]}, [{"id": 0, "load_percent": 20}], 3500),
                "cpu_temp": lambda _: 61, "nvme": lambda _: 4096,
            })
            self.assertEqual(sample["memory"]["ram_available_mib"], 4000)
            self.assertEqual(sample["cpu"]["temperature_c"], 61)
            self.assertEqual(sample["nvme_written_bytes"], 0)
            self.assertEqual(sample["sampled_at"], 123.0)

    def test_absent_linux_files_return_false_capabilities_without_raising(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            source = sources.LinuxSystemSource(Path(tmp) / "proc", Path(tmp) / "sys", tmp)
            self.assertFalse(source.available())
            self.assertEqual(source.capabilities(), {"system_linux": False, "zram": False, "psi": False, "cpu_temp": False, "nvme": False})

    def test_one_broken_system_reader_does_not_discard_other_system_values(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            root = Path(tmp)
            source = sources.LinuxSystemSource(root / "proc", root / "sys", root)
            sample = source.sample({
                "meminfo": lambda _p: (_ for _ in ()).throw(OSError("meminfo unavailable")),
                "zram": lambda _s: {"original_bytes": None, "compressed_bytes": None, "used_bytes": None, "disk_bytes": None, "used_percent": None},
                "psi": lambda _p: {"full": 1.25},
                "cpu": lambda *_args: ({}, [{"id": 0, "load_percent": 5}], 2000),
                "cpu_temp": lambda _s: 55,
                "nvme": lambda _s: None,
            })
            self.assertIsNone(sample["memory"]["ram_available_mib"])
            self.assertEqual(sample["memory"]["psi_full_avg10"], 1.25)
            self.assertEqual(sample["cpu"]["temperature_c"], 55)

    def test_windows_fallback_is_explicitly_unavailable_on_non_windows(self):
        source = sources.WindowsSystemSource()
        if sys.platform != "win32":
            self.assertFalse(source.available())
        self.assertEqual(source.sample(), {})


if __name__ == "__main__":
    unittest.main()
