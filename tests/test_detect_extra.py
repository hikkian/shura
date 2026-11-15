"""INSTALL1 step 1: extended detection (all GPUs, display GPU, driver versions, swap/zram, cgroup v2, power, disk speed).
Everything runs on recorded output in tests/fixtures/detect/extra_*; no command is run and no real file is read."""
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
from universal import detect as D, schema  # noqa: E402

FX = ROOT / "tests/fixtures/detect"
MiB = 1024 * 1024


class FakeEnv:
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


def machine(name, **cmds):
    """FakeEnv from a fixture set: sysfs.json plus <command>.txt files named in cmds (command -> file)."""
    d = FX / name
    files = json.loads((d / "sysfs.json").read_text())
    return FakeEnv({c: (d / f).read_text() for c, f in cmds.items()}, files)


def laptop():
    return machine("extra_laptop_intel_nvidia", lspci="lspci_nn.txt", nvidia_smi="nvidia_smi.txt",
                   vulkaninfo="vulkaninfo_summary.txt", **{"sycl-ls": "sycl_ls.txt", "nvidia-smi": "nvidia_smi.txt"})


def amd():
    return machine("extra_amd_gfx1101", lspci="lspci_nn.txt", vulkaninfo="vulkaninfo_summary.txt")


def cpu_only():
    return machine("extra_cpu_only", lspci="lspci_nn.txt")


class Parsers(unittest.TestCase):
    def test_lspci_nn_laptop(self):
        g = D.parse_lspci_nn(FX / "extra_laptop_intel_nvidia/lspci_nn.txt")
        self.assertEqual([(x["slot"], x["vendor"], x["kind"], x["integrated"]) for x in g],
                         [("00:02.0", "intel", "vga", True), ("01:00.0", "nvidia", "3d", None)])
        self.assertIn("Iris Xe", g[0]["name"])
        self.assertEqual(g[1]["device_id"], "25a2")

    def test_lspci_nn_text_and_none(self):
        self.assertEqual(D.parse_lspci_nn((FX / "extra_cpu_only/lspci_nn.txt").read_text()), [])
        self.assertEqual(D.parse_lspci_nn(""), [])
        a = D.parse_lspci_nn(FX / "extra_amd_gfx1101/lspci_nn.txt")
        self.assertEqual(len(a), 1)                      # the audio function is not a video card
        self.assertEqual(a[0]["vendor"], "amd")

    def test_intel_dgpu_not_integrated(self):
        g = D.parse_lspci_nn("03:00.0 VGA compatible controller [0300]: Intel Corporation DG2 [Arc A770] [8086:56a0] (rev 08)\n")
        self.assertIs(g[0]["integrated"], False)

    def test_drm_connector_wins_over_boot_vga(self):
        e = [{"card": "card0", "slot": "0000:00:02.0", "boot_vga": "0\n", "connected": 1},
             {"card": "card1", "slot": "0000:01:00.0", "boot_vga": "1\n", "connected": 0}]
        r = D.parse_drm_cards(e)
        self.assertEqual((r["display_slot"], r["method"], r["boot_vga_slot"]), ("00:02.0", "connector", "01:00.0"))

    def test_drm_boot_vga_fallback_and_unknown(self):
        e = [{"card": "card0", "slot": "00:02.0", "boot_vga": "1", "connected": 0},
             {"card": "card1", "slot": "01:00.0", "boot_vga": "0", "connected": 0}]
        self.assertEqual(D.parse_drm_cards(e)["method"], "boot_vga")
        e[1]["connected"] = e[0]["connected"] = 1                    # two screens on two cards, two boot flags unknown
        e[0]["boot_vga"] = e[1]["boot_vga"] = None
        self.assertIsNone(D.parse_drm_cards(e)["display_slot"])

    def test_nvidia_header(self):
        self.assertEqual(D.parse_nvidia_smi_header(FX / "extra_laptop_intel_nvidia/nvidia_smi.txt"),
                         {"driver": "560.35.03", "cuda": "12.6"})
        self.assertEqual(D.parse_nvidia_smi_header("NVIDIA-SMI 470.1  Driver Version: 470.1  CUDA Version: N/A"),
                         {"driver": "470.1", "cuda": None})
        self.assertEqual(D.parse_nvidia_smi_header(""), {"driver": None, "cuda": None})

    def test_rocm_and_gfx(self):
        self.assertEqual(D.parse_rocm_version(FX / "extra_amd_gfx1101/rocm_version.txt"), "7.2.4-45")
        self.assertIsNone(D.parse_rocm_version("garbage"))
        self.assertEqual(D.parse_kfd_gfx(FX / "extra_amd_gfx1101/kfd_properties_node1.txt"), "gfx1101")
        self.assertIsNone(D.parse_kfd_gfx(FX / "extra_amd_gfx1101/kfd_properties_node0.txt"))
        self.assertEqual(D.parse_kfd_gfx("gfx_target_version 90010\n"), "gfx90a")

    def test_vulkaninfo(self):
        v = D.parse_vulkaninfo_summary(FX / "extra_laptop_intel_nvidia/vulkaninfo_summary.txt")
        self.assertEqual(v["instance"], "1.3.280")
        self.assertEqual([(d["type"], d["driver"]) for d in v["devices"]],
                         [("integrated_gpu", "Mesa 24.0.5"), ("discrete_gpu", "560.35.03")])
        self.assertEqual(D.parse_vulkaninfo_summary("")["devices"], [])

    def test_vulkan_icd_names(self):
        self.assertEqual(D.parse_vulkan_icd_names(["nvidia_icd.json", "README", "intel_icd.x86_64.json"]),
                         ["intel_icd.x86_64.json", "nvidia_icd.json"])

    def test_sycl_ls(self):
        s = D.parse_sycl_ls(FX / "extra_laptop_intel_nvidia/sycl_ls.txt")
        self.assertEqual([(x["backend"], x["type"]) for x in s], [("opencl", "cpu"), ("opencl", "gpu"), ("level_zero", "gpu")])
        self.assertIn("Iris", s[2]["name"])

    def test_swaps(self):
        sw = json.loads((FX / "extra_laptop_intel_nvidia/sysfs.json").read_text())["/proc/swaps"]
        self.assertEqual(D.parse_proc_swaps(sw), [{"kind": "zram", "size": 8388604 * 1024, "used": 1048576 * 1024, "priority": 100}])
        f = json.loads((FX / "extra_amd_gfx1101/sysfs.json").read_text())["/proc/swaps"]
        r = D.parse_proc_swaps(f)
        self.assertEqual((r[0]["kind"], r[0]["priority"]), ("file", -2))
        self.assertNotIn("swapfile", json.dumps(r))          # file names are never kept
        self.assertEqual(D.parse_proc_swaps("Filename Type Size Used Priority\n"), [])

    def test_zram(self):
        z = D.parse_zram_device("8589934592\n", "lzo lzo-rle [zstd] 842\n", "2147483648 536870912 570425344 0 1 2 0 0\n")
        self.assertEqual(z, {"disksize": 8589934592, "algorithm": "zstd", "orig_bytes": 2147483648, "compressed_bytes": 536870912})
        self.assertEqual(D.parse_zram_device("", "", ""), {"disksize": None, "algorithm": None, "orig_bytes": None, "compressed_bytes": None})

    def test_cgroup_controllers(self):
        self.assertEqual(D.parse_cgroup_controllers("memory cpu io\n"), ["cpu", "io", "memory"])
        self.assertEqual(D.parse_cgroup_controllers(""), [])

    def test_power(self):
        bat = {"type": "Battery\n", "status": "Discharging\n"}
        self.assertEqual(D.parse_power_supply([bat, {"type": "Mains\n", "online": "0\n"}]),
                         {"has_battery": True, "on_battery": True, "mains_online": False})
        self.assertFalse(D.parse_power_supply([bat, {"type": "Mains\n", "online": "1\n"}])["on_battery"])
        self.assertEqual(D.parse_power_supply([{"type": "Mains", "online": "1"}]),
                         {"has_battery": False, "on_battery": False, "mains_online": True})
        self.assertEqual(D.parse_power_supply([]), {"has_battery": False, "on_battery": False, "mains_online": None})
        self.assertTrue(D.parse_power_supply([bat])["on_battery"])                 # no mains entry: the battery status decides
        self.assertIsNone(D.parse_power_supply([{"type": "Battery", "status": "Unknown"}])["on_battery"])


class DiskRead(unittest.TestCase):
    def test_measure_with_fake_file_and_clock(self):
        data = io.BytesIO(b"\0" * (64 * MiB))
        ticks = iter([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5])
        speed, note = D.measure_disk_read("x", opener=lambda p, m: data, clock=lambda: next(ticks))
        self.assertIsNone(note)
        self.assertGreater(speed, 0)
        self.assertAlmostEqual(speed, 64 / 8.5, places=1)      # 16 chunks, 8.5 s between first and last tick read

    def test_too_small_file_is_not_a_number(self):
        speed, note = D.measure_disk_read("x", opener=lambda p, m: io.BytesIO(b"\0" * MiB))
        self.assertIsNone(speed)
        self.assertIn("16 MiB", note)

    def test_missing_file_and_timeout(self):
        def boom(p, m):
            raise FileNotFoundError(p)
        speed, note = D.measure_disk_read("x", opener=boom)
        self.assertIsNone(speed)
        self.assertIn("cannot be opened", note)
        t = iter(range(0, 1000, 20))                               # every call is 20 s later: stops after the first chunk
        speed, note = D.measure_disk_read("x", opener=lambda p, m: io.BytesIO(b"\0" * (64 * MiB)), clock=lambda: float(next(t)))
        self.assertIsNone(speed)                                   # only 4 MiB read: below the 16 MiB floor


class Collect(unittest.TestCase):
    def test_laptop_igpu_and_nvidia(self):
        gpus = [{"vendor": "nvidia", "name": "NVIDIA GeForce RTX 3050 Laptop GPU", "vram_total": 4096 * MiB, "vram_used": 600 * MiB}]
        s, disk, notes = D.collect_extra(laptop(), gpus)
        self.assertEqual(len(s["gpus_all"]), 2)
        self.assertEqual((s["display"]["display_slot"], s["display"]["method"], s["display"]["boot_vga_slot"]), ("00:02.0", "connector", "00:02.0"))
        self.assertEqual([g["display"] for g in s["gpus_all"]], [True, False])
        self.assertIs(gpus[0]["display"], False)                   # the NVIDIA card does not draw the screen here
        self.assertEqual((s["drivers"]["nvidia_driver"], s["drivers"]["cuda"]), ("560.35.03", "12.6"))
        self.assertEqual(len(s["drivers"]["vulkan"]["devices"]), 2)
        self.assertEqual(len(s["drivers"]["oneapi"]), 3)
        self.assertEqual(s["swap"][0]["kind"], "zram")
        self.assertEqual(s["zram"][0]["algorithm"], "zstd")
        self.assertTrue(s["cgroup_v2"])
        self.assertEqual(s["power"], {"has_battery": True, "on_battery": True, "mains_online": False})
        self.assertIn("disk read speed: not measured (no sample file given)", notes)
        self.assertIsNone(disk["read_mibs"])
        self.assertFalse([n for n in notes if "nvidia-smi" in n])

    def test_cpu_only(self):
        s, disk, notes = D.collect_extra(cpu_only(), [])
        self.assertEqual(s["gpus_all"], [])
        self.assertIsNone(s["drivers"]["vulkan"])                   # no GPU: nothing to ask Vulkan about
        self.assertEqual(s["drivers"]["nvidia_driver"], None)
        self.assertEqual(s["swap"], [])
        self.assertEqual(s["power"], {"has_battery": False, "on_battery": False, "mains_online": True})
        self.assertIsNone(s["display"]["display_slot"])
        self.assertTrue(any("draws the screen" in n for n in notes) or any("/sys/class/drm" in n for n in notes))
        self.assertFalse(any("nvidia-smi" in n for n in notes))     # expected absence on a CPU-only box is not noise

    def test_amd_gfx1101_rocm(self):
        gpus = [{"vendor": "amd", "name": "Radeon RX 7800 XT", "bus": "2d:00.0", "vram_total": 16 << 30, "vram_used": 700 * MiB}]
        s, _, notes = D.collect_extra(amd(), gpus)
        self.assertEqual((s["drivers"]["rocm"], s["drivers"]["rocm_gfx"]), ("7.2.4-45", ["gfx1101"]))
        self.assertEqual(s["display"]["display_slot"], "2d:00.0")
        self.assertIs(gpus[0]["display"], True)
        self.assertEqual(s["swap"][0]["kind"], "file")
        self.assertEqual(s["drivers"]["vulkan"]["devices"][0]["type"], "discrete_gpu")
        self.assertFalse([n for n in notes if "ROCm" in n or "gfx" in n])

    def test_missing_commands_go_to_notes(self):
        env = FakeEnv({}, {})                                       # nothing at all: no commands, no sysfs
        s, disk, notes = D.collect_extra(env, [])
        text = "\n".join(notes)
        for part in ("lspci -nn is not available", "/sys/class/drm is not readable", "/proc/swaps is not readable",
                     "cgroup v2 not found", "disk read speed: not measured"):
            self.assertIn(part, text)
        self.assertFalse(s["cgroup_v2"])
        self.assertEqual(s["gpus_all"], [])

    def test_missing_nvidia_smi_and_rocm_noted(self):
        env = machine("extra_laptop_intel_nvidia", lspci="lspci_nn.txt")             # NVIDIA card listed, no nvidia-smi, no vulkaninfo
        s, _, notes = D.collect_extra(env, [])
        text = "\n".join(notes)
        self.assertIn("nvidia-smi is not available", text)
        self.assertIn("sycl-ls is not available", text)
        self.assertIn("vulkaninfo is not available: 2 Vulkan ICD file(s)", text)
        self.assertIsNone(s["drivers"]["cuda"])
        env2 = machine("extra_amd_gfx1101", lspci="lspci_nn.txt")
        env2.files.pop("/opt/rocm/.info/version")
        _, _, notes2 = D.collect_extra(env2, [])
        self.assertTrue(any("ROCm version not found" in n for n in notes2))

    def test_cgroup_v1_machine(self):
        env = cpu_only()
        del env.files["/sys/fs/cgroup/cgroup.controllers"]
        s, _, notes = D.collect_extra(env, [])
        self.assertFalse(s["cgroup_v2"])
        self.assertTrue(any("cgroup v2 not found" in n for n in notes))
        s2, _, _ = D.collect_extra(cpu_only(), [])
        self.assertTrue(s2["cgroup_v2"])
        self.assertIn("memory", s2["cgroup_controllers"])

    def test_disk_benchmark_uses_measure(self):
        with mock.patch.object(D, "measure_disk_read", return_value=(1234.5, None)) as m:
            _, disk, notes = D.collect_extra(cpu_only(), [], disk_sample="sample.bin", disk_benchmark=True)
        m.assert_called_once()
        self.assertEqual(disk["read_mibs"], 1234.5)
        self.assertFalse(any("disk read speed" in n for n in notes))
        with mock.patch.object(D, "measure_disk_read", return_value=(None, "disk read speed: read failed (OSError)")):
            _, disk, notes = D.collect_extra(cpu_only(), [], disk_sample="s", disk_benchmark=True)
        self.assertIsNone(disk["read_mibs"])
        self.assertIn("disk read speed: read failed (OSError)", notes)

    def test_benchmark_disabled_is_not_measured(self):
        with mock.patch.object(D, "measure_disk_read") as m:
            _, disk, notes = D.collect_extra(cpu_only(), [], disk_sample="sample.bin", disk_benchmark=False)
        m.assert_not_called()
        self.assertIsNone(disk["read_mibs"])
        self.assertIn("disk read speed: not measured", notes)

    def test_fixtures_have_no_private_data(self):
        for f in FX.glob("extra_*/*"):
            t = f.read_text().lower()
            for bad in ("/home/", "serial", "alisher", "hostname"):
                self.assertNotIn(bad, t, f)


class DetectEndToEnd(unittest.TestCase):
    def test_detect_attaches_system_block_and_stays_valid(self):
        env = laptop()
        env.commands.update({"lscpu": (FX / "lscpu_ryzen5600.txt").read_text()})
        env.files["/proc/meminfo"] = (FX / "meminfo_32g.txt").read_text()
        with mock.patch.object(D.platform, "system", return_value="Linux"), \
                mock.patch.object(D.shutil, "disk_usage", return_value=mock.Mock(free=100 << 30)):
            hw = D.detect(env)
        self.assertEqual(schema.validate_hardware(hw), [])
        self.assertIn("system", hw)
        self.assertIsNone(hw["disk"]["read_mibs"])
        self.assertTrue(all(isinstance(n, str) for n in hw["detection_notes"]))
        json.dumps(hw)


class TextNeverReadsStrings(unittest.TestCase):
    def test_a_string_that_spells_a_real_path_is_text_not_a_file(self):
        import builtins
        real_open = builtins.open
        opened = []

        def spy(file, *a, **k):
            opened.append(str(file))
            return real_open(file, *a, **k)
        with mock.patch("builtins.open", spy):
            self.assertEqual(D._text("/etc/hostname"), "/etc/hostname")
            self.assertEqual(D.parse_lspci_nn("/etc/hostname"), [])
        self.assertEqual(opened, [])

    def test_a_path_object_is_read(self):
        self.assertEqual(D._text(FX / "extra_laptop_intel_nvidia/lspci_nn.txt"),
                         (FX / "extra_laptop_intel_nvidia/lspci_nn.txt").read_text())


if __name__ == "__main__":
    unittest.main()
