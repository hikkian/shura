import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "installer" / "universal"))
import hw_dump


def write(root, rel, value):
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


class PrivateHardwareDump(unittest.TestCase):
    def test_dump_whitelists_hardware_data_and_redacts_sensitive_strings(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            root = Path(tmp)
            write(root, "class/drm/card0/device/vendor", "0x1002")
            write(root, "class/drm/card0/device/device", "0x744c")
            write(root, "class/drm/card0/device/product_name", "AMD GPU serial=SERIAL123 /home/exampleuser 192.168.1.7")
            write(root, "class/drm/card0/device/gpu_busy_percent", "35")
            write(root, "class/drm/card0/device/mem_info_vram_total", "25769803776")
            write(root, "class/drm/card0/device/hwmon/hwmon0/temp1_input", "72000")
            write(root, "class/hwmon/hwmon1/name", "k10temp")
            write(root, "class/hwmon/hwmon1/temp1_input", "64000")
            write(root, "block/zram0/mm_stat", "10 5 5")
            write(root, "block/zram0/disksize", "100")
            proc = root / "proc"
            proc.mkdir()
            (proc / "meminfo").write_text("MemTotal: 10 kB\n", encoding="utf-8")
            text = hw_dump.hardware_dump(sys_root=root, proc_root=proc)
            self.assertIn("Platform:", text)
            self.assertIn("Kernel:", text)
            self.assertIn("GPU 1: AMD", text)
            self.assertIn("mem_info_vram_total: 25769803776", text)
            self.assertIn("gpu_busy_percent: 35", text)
            self.assertIn("gpu_amd=yes", text)
            self.assertIn("cpu_temp=yes", text)
            for secret in ("SERIAL123", "/home/exampleuser", "192.168.1.7", "hostname", str(Path.home())):
                self.assertNotIn(secret, text)

    def test_nvidia_name_and_driver_are_scrubbed(self):
        class FakeNVML:
            def device_name(self): return "RTX 4070 SUPER serial=xyz"
            def driver_version(self): return "580.1"
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            text = hw_dump.hardware_dump(sys_root=tmp, proc_root=Path(tmp) / "proc", nvml_factory=FakeNVML)
            self.assertIn("GPU model: RTX 4070 SUPER [redacted]", text)
            self.assertIn("GPU driver: 580.1", text)
            self.assertNotIn("xyz", text)
            self.assertIn("gpu_nvidia=yes", text)

    def test_cli_writes_the_same_filtered_text_it_prints(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            out = Path(tmp) / "dump.txt"
            capture = io.StringIO()
            with patch.object(hw_dump, "hardware_dump", return_value="safe dump\n"), contextlib.redirect_stdout(capture):
                self.assertEqual(hw_dump.main(["--hw-dump", "--output", str(out)], nvml_factory=None), 0)
            self.assertEqual(out.read_text(encoding="utf-8"), "safe dump\n")
            self.assertIn("safe dump", capture.getvalue())
            self.assertIn("Saved filtered report.", capture.getvalue())

    def test_everyday_shura_script_dispatches_doctor_dump(self):
        script = (Path(__file__).resolve().parents[1] / "scripts" / "shura").read_text(encoding="utf-8")
        self.assertIn('doctor) if [ "${2:-}" = "--hw-dump" ]', script)
        self.assertIn('installer/universal/hw_dump.py', script)


if __name__ == "__main__":
    unittest.main()
