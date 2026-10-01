"""Hardware detection tests. Parsers are fed recorded output: real captures from the reference machine
(tests/fixtures/detect/) and small synthetic samples for machines we do not own. Nothing touches the real system."""
import json
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
from universal import detect as D, schema  # noqa: E402

FX = ROOT / "tests/fixtures/detect"
MiB = 1024 * 1024


def fx(name):
    return (FX / name).read_text()


class FakeEnv:
    """run() returns canned text by command name, read() serves files, exists()/listdir() a fake filesystem."""

    def __init__(self, commands=None, files=None):
        self.commands, self.files = commands or {}, files or {}

    def run(self, cmd, timeout=15):
        return self.commands.get(cmd[0])

    def read(self, path):
        return self.files.get(path)

    def exists(self, path):
        return path in self.files or any(f.startswith(path + "/") for f in self.files)

    def listdir(self, path):
        return sorted({f[len(path) + 1:].split("/")[0] for f in self.files if f.startswith(path + "/")})


class PowerShellEnv(FakeEnv):
    """Answers each PowerShell script with its own canned JSON."""

    def __init__(self, cpu=None, video=None, vram=None, smi=None):
        super().__init__()
        self.by = {"Win32_Processor": cpu, "Win32_VideoController": video, "HardwareInformation": vram}
        self.smi = smi

    def run(self, cmd, timeout=15):
        if cmd[0] == "nvidia-smi":
            return self.smi
        if cmd[0] in ("powershell", "pwsh"):
            return next((v for k, v in self.by.items() if k in cmd[-1]), None)
        return None


class WindowsAmd(unittest.TestCase):
    """The machine that made this necessary: Ryzen 7 5700X3D, 48 GB, Radeon RX 9070 16 GB, Windows 11."""
    CPU = json.dumps({"Name": "AMD Ryzen 7 5700X3D 8-Core Processor", "NumberOfCores": 8, "NumberOfLogicalProcessors": 16})
    VIDEO = json.dumps({"Name": "AMD Radeon RX 9070", "DriverVersion": "32.0.21001.1009"})
    VRAM = json.dumps({"DriverDesc": "AMD Radeon RX 9070", "HardwareInformation.qwMemorySize": 17163091968})

    def detect(self, **kw):
        env = PowerShellEnv(**{"cpu": self.CPU, "video": self.VIDEO, "vram": self.VRAM, **kw})
        with unittest.mock.patch.object(D, "_windows_memory", return_value=(48 * 1024 ** 3, 40 * 1024 ** 3)):
            return D.detect_windows(env)

    def test_the_radeon_is_found_with_its_real_vram_and_cores(self):
        cpu, mem, gpus, notes = self.detect()
        self.assertEqual((cpu["physical_cores"], cpu["logical_cores"]), (8, 16))
        self.assertEqual(len(gpus), 1)
        g = gpus[0]
        self.assertEqual((g["vendor"], g["vram_total"]), ("amd", 17163091968))
        self.assertEqual(g["backends"], ["rocm", "vulkan"])
        self.assertTrue(g["display"])
        self.assertEqual(D.gpu_bandwidth(g["name"]), 640)

    def test_vram_stored_as_raw_bytes_is_read_too(self):
        raw = list((17163091968).to_bytes(8, "little"))
        self.assertEqual(D.parse_windows_vram(json.dumps({"DriverDesc": "X", "HardwareInformation.qwMemorySize": raw})),
                         {"X": 17163091968})

    def test_unknown_vram_is_a_flagged_placeholder_not_a_missing_gpu(self):
        _, _, gpus, notes = self.detect(vram=None)
        self.assertEqual(len(gpus), 1)
        self.assertTrue(any("unknown" in n for n in notes))

    def test_no_powershell_falls_back_to_estimates_and_says_so(self):
        cpu, _, gpus, notes = self.detect(cpu=None, video=None, vram=None)
        self.assertEqual(gpus, [])
        self.assertTrue(any("estimated" in n for n in notes))

    def test_nvidia_comes_from_nvidia_smi_and_is_not_listed_twice(self):
        smi = "NVIDIA GeForce RTX 3060, 12288, 900, 555.85\n"
        video = json.dumps({"Name": "NVIDIA GeForce RTX 3060", "DriverVersion": "32.0.15.5585"})
        _, _, gpus, _ = self.detect(smi=smi, video=video, vram=None)
        self.assertEqual([g["vendor"] for g in gpus], ["nvidia"])
        self.assertEqual(gpus[0]["vram_total"], 12288 * 1024 ** 2)

    def test_two_sockets_are_summed(self):
        two = json.dumps([{"Name": "Xeon", "NumberOfCores": 12, "NumberOfLogicalProcessors": 24}] * 2)
        self.assertEqual(D.parse_windows_cpu(two)[1:], (24, 48, 2))


class Parsers(unittest.TestCase):
    def test_lscpu_real_capture(self):
        cpu = D.parse_lscpu(fx("lscpu_ryzen5600.txt"))
        self.assertEqual((cpu["physical_cores"], cpu["logical_cores"], cpu["numa_nodes"]), (6, 12, 1))
        self.assertIn("avx2", cpu["flags"])
        self.assertIn("Ryzen 5 5600", cpu["model"])

    def test_lscpu_dual_socket_numa(self):
        text = "CPU(s): 128\nModel name: AMD EPYC 7551\nSocket(s): 2\nCore(s) per socket: 32\nNUMA node(s): 8\n"
        cpu = D.parse_lscpu(text)
        self.assertEqual((cpu["physical_cores"], cpu["numa_nodes"]), (64, 8))

    def test_meminfo_real_capture(self):
        mem = D.parse_meminfo(fx("meminfo_32g.txt"))
        self.assertEqual(mem["total"], 32774700 * 1024)
        self.assertGreater(mem["available"], 0)

    def test_nvidia_smi_real_capture(self):
        (g,) = D.parse_nvidia_smi(fx("nvidia_smi_4070s.txt"))
        self.assertEqual((g["vendor"], g["vram_total"]), ("nvidia", 12282 * MiB))
        self.assertIn("4070 SUPER", g["name"])
        self.assertTrue(g["driver"].startswith("615"))

    def test_lspci_real_capture_and_amd_intel_samples(self):
        (g,) = D.parse_lspci(fx("lspci_nvidia.txt"))
        self.assertEqual(g["vendor"], "nvidia")
        amd = '0a:00.0 "VGA compatible controller" "Advanced Micro Devices, Inc. [AMD/ATI]" "Navi 31 [Radeon RX 7900 XTX] [1002:744c]"'
        self.assertEqual(D.parse_lspci(amd)[0]["vendor"], "amd")
        intel = '03:00.0 "VGA compatible controller" "Intel Corporation" "DG2 [Arc A770] [8086:56a0]"'
        self.assertEqual(D.parse_lspci(intel)[0]["vendor"], "intel")
        self.assertEqual(D.parse_lspci('00:1f.3 "Audio device" "Intel Corporation" "x"'), [])

    def test_macos_sysctl(self):
        s = D.parse_sysctl("hw.memsize: 68719476736\nhw.physicalcpu: 12\nhw.logicalcpu: 12\n"
                           "machdep.cpu.brand_string: Apple M2 Max\n")
        self.assertEqual((s["memsize"], s["physical"], s["brand"]), (68719476736, 12, "Apple M2 Max"))

    def test_windows_video_json_does_not_trust_adapter_ram(self):
        out = D.parse_windows_video(json.dumps([{"Name": "NVIDIA GeForce RTX 3060", "AdapterRAM": 4293918720,
                                                 "DriverVersion": "31.0.15.3623"}]))
        self.assertEqual(out[0]["vendor"], "nvidia")
        self.assertNotIn("vram_total", out[0])

    def test_bandwidth_table_uses_the_longest_match(self):
        self.assertEqual(D.gpu_bandwidth("NVIDIA GeForce RTX 4070 Ti SUPER"), 672)
        self.assertEqual(D.gpu_bandwidth("NVIDIA GeForce RTX 4070 SUPER"), 504)
        self.assertIsNone(D.gpu_bandwidth("Some Unknown GPU"))


class Flow(unittest.TestCase):
    def test_reference_machine_profile_is_valid_and_complete(self):
        env = FakeEnv({"lscpu": fx("lscpu_ryzen5600.txt"), "nvidia-smi": fx("nvidia_smi_4070s.txt"),
                       "lspci": fx("lspci_nvidia.txt"), "vulkaninfo": "Devices:\n"},
                      {"/proc/meminfo": fx("meminfo_32g.txt")})
        cpu, mem, gpus, notes = D.detect_linux(env)
        hw = D.build_profile("linux", cpu, mem, gpus, notes, ram_bandwidth_gbs=35.0, disk_free=10**11)
        self.assertEqual(schema.validate_hardware(hw), [])
        g = hw["gpus"][0]
        self.assertEqual((g["bandwidth_gbs"], g["backends"], g["display"]), (504, ["cuda", "vulkan"], True))
        self.assertEqual(hw["memory"]["bandwidth_gbs"], 35.0)
        self.assertEqual(notes, [])

    def test_amd_gpu_reads_vram_from_sysfs_and_offers_rocm_only_with_kfd(self):
        lspci = '0a:00.0 "VGA compatible controller" "Advanced Micro Devices, Inc. [AMD/ATI]" "Navi 31 [Radeon RX 7900 XTX] [1002:744c]"'
        files = {"/proc/meminfo": "MemTotal: 67108864 kB\nMemAvailable: 50000000 kB\n",
                 "/sys/class/drm/card0/device/vendor": "0x1002\n",
                 "/sys/class/drm/card0/device/mem_info_vram_total": str(24 * 1024 * MiB),
                 "/sys/class/drm/card0/device/mem_info_vram_used": str(800 * MiB)}
        with_kfd = D.detect_linux(FakeEnv({"lscpu": fx("lscpu_ryzen5600.txt"), "lspci": lspci}, {**files, "/dev/kfd": ""}))
        without = D.detect_linux(FakeEnv({"lscpu": fx("lscpu_ryzen5600.txt"), "lspci": lspci}, files))
        self.assertEqual(with_kfd[2][0]["backends"], ["rocm", "vulkan"])
        self.assertEqual(without[2][0]["backends"], ["vulkan"])
        self.assertEqual(with_kfd[2][0]["vram_total"], 24 * 1024 * MiB)

    def test_intel_gpu_is_skipped_with_a_note_instead_of_guessing_vram(self):
        lspci = '03:00.0 "VGA compatible controller" "Intel Corporation" "DG2 [Arc A770] [8086:56a0]"'
        cpu, mem, gpus, notes = D.detect_linux(FakeEnv({"lscpu": fx("lscpu_ryzen5600.txt"), "lspci": lspci},
                                                       {"/proc/meminfo": fx("meminfo_32g.txt")}))
        self.assertEqual(gpus, [])
        self.assertTrue(any("VRAM" in n for n in notes))

    def test_cpu_only_linux_machine(self):
        cpu, mem, gpus, notes = D.detect_linux(FakeEnv({"lscpu": fx("lscpu_ryzen5600.txt")},
                                                       {"/proc/meminfo": fx("meminfo_32g.txt")}))
        hw = D.build_profile("linux", cpu, mem, gpus, notes)
        self.assertEqual(hw["gpus"], [])
        self.assertEqual(schema.validate_hardware(hw), [])

    def test_missing_lscpu_is_noted_not_fatal(self):
        cpu, mem, gpus, notes = D.detect_linux(FakeEnv({}, {"/proc/meminfo": fx("meminfo_32g.txt")}))
        self.assertGreaterEqual(cpu["physical_cores"], 1)
        self.assertTrue(any("lscpu" in n for n in notes))

    def test_macos_apple_silicon_is_unified_memory(self):
        env = FakeEnv({"sysctl": "hw.memsize: 68719476736\nhw.physicalcpu: 12\nhw.logicalcpu: 12\n"
                                 "machdep.cpu.brand_string: Apple M2 Max\n"})
        cpu, mem, gpus, notes = D.detect_macos(env)
        hw = D.build_profile("macos", cpu, mem, gpus, notes, arch="arm64")
        self.assertEqual(schema.validate_hardware(hw), [])
        self.assertTrue(hw["gpus"][0]["unified"])
        self.assertEqual(hw["gpus"][0]["bandwidth_gbs"], 400)


if __name__ == "__main__":
    unittest.main()
