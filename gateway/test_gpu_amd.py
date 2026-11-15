"""The AMD backend on a fake sysfs and /proc tree: a normal card, no card, two cards (AMD and integrated), no fdinfo, garbage.

Run from the gateway folder:  python3 -m unittest discover -s . -p 'test_gpu_amd.py'
No GPU, no driver, no model: the roots are temporary folders (SHURA_SYSFS_ROOT, SHURA_PROC_ROOT are set per test)."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import gpu_backend
from game_mode import GameSignals
from gpu_amd import AMDDevice, AMDError, find_card, parse_fdinfo
from gpu_backend import GPUError

PCI = "0000:2d:00.0"
GIB = 1 << 30
RENDER = "/dev/dri/renderD128"


def put(path, text=None, link=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if link is not None:
        os.symlink(link, path)
    else:
        path.write_text(text, encoding="utf-8")


def make_card(sys_root, card="card1", pci=PCI, vendor="0x1002", total=16 * GIB, used=1 * GIB, busy="40",
              temp="45500", power="123400000", speed="16.0 GT/s PCIe", width="16"):
    """The layout of a real card: the PCI device folder, and class/drm/cardN/device as a link to it."""
    device = Path(sys_root, "devices", pci)
    values = {"vendor": vendor, "mem_info_vram_total": None if total is None else str(total),
              "mem_info_vram_used": None if used is None else str(used), "gpu_busy_percent": busy,
              "current_link_speed": speed, "current_link_width": width}
    for name, text in values.items():
        if text is not None:
            put(device / name, text + "\n")
    if temp is not None:
        put(device / "hwmon" / "hwmon4" / "temp1_input", temp + "\n")
    if power is not None:
        put(device / "hwmon" / "hwmon4" / "power1_average", power + "\n")
    drm = Path(sys_root, "class", "drm", card)
    put(drm / "device", link=os.path.relpath(device, drm))
    return device


def fdinfo(client=None, vram_kib=None, pdev=PCI, driver="amdgpu", gfx=0, compute=0):
    lines = ["pos:\t0", "flags:\t02100002"]
    if driver is not None:
        lines.append(f"drm-driver:\t{driver}")
    if client is not None:
        lines.append(f"drm-client-id:\t{client}")
    if pdev is not None:
        lines.append(f"drm-pdev:\t{pdev}")
    if vram_kib is not None:
        lines.append(f"drm-memory-vram:\t{vram_kib} KiB")
    lines += [f"drm-engine-gfx:\t{gfx} ns", f"drm-engine-compute:\t{compute} ns"]
    return "\n".join(lines) + "\n"


def make_process(proc, pid, fds, comm=None):
    """fds: {name: (link target, fdinfo text or None)}. A DRM file is what the kernel shows as /dev/dri/renderD128."""
    for name, (target, info) in fds.items():
        put(proc / str(pid) / "fd" / name, link=target)
        if info is not None:
            put(proc / str(pid) / "fdinfo" / name, info)
    if comm is not None:
        put(proc / str(pid) / "comm", comm + "\n")


class Fixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.sys = self.root / "sys"
        self.proc = self.root / "proc"
        self.proc.mkdir()
        self.device_folder = make_card(self.sys)
        env = patch.dict(os.environ, {"SHURA_SYSFS_ROOT": str(self.sys), "SHURA_PROC_ROOT": str(self.proc)})
        env.start()
        self.addCleanup(env.stop)

    def clear_cards(self):
        shutil.rmtree(self.sys / "class" / "drm")
        (self.sys / "class" / "drm").mkdir(parents=True)

    def processes_fixture(self):
        """The model (pid 100) with two fds of one client and a second client; a game (200); other drivers, another card, and a non-DRM fd."""
        make_process(self.proc, 100, {
            "3": (RENDER, fdinfo(client=7, vram_kib=4096)),
            "4": (RENDER, fdinfo(client=7, vram_kib=4096)),   # the same client through a second fd: counted once
            "5": (RENDER, fdinfo(client=8, vram_kib=512)),
        }, comm="llama-server")
        make_process(self.proc, 200, {"3": (RENDER, fdinfo(client=9, vram_kib=2048))}, comm="game")
        make_process(self.proc, 300, {"3": (RENDER, fdinfo(client=10, vram_kib=99999, driver="i915"))}, comm="other")
        make_process(self.proc, 400, {"3": ("/dev/null", "drm-driver:\tamdgpu\n")}, comm="cat")
        make_process(self.proc, 500, {"3": (RENDER, fdinfo(client=11, vram_kib=9999, pdev="0000:03:00.0"))}, comm="other-card")
        make_process(self.proc, 600, {"3": (RENDER, fdinfo(client=12, vram_kib=None))}, comm="novram")


class ParsingTests(unittest.TestCase):
    def test_the_facts_of_one_client_are_read(self):
        fact = parse_fdinfo(fdinfo(client=45, vram_kib=76176, gfx=52961998, compute=309139))
        self.assertEqual((fact["driver"], fact["client"], fact["pdev"]), ("amdgpu", 45, PCI))
        self.assertEqual(fact["vram_kib"], 76176)
        self.assertEqual(fact["engine_ns"], 52961998 + 309139)

    def test_garbage_lines_and_values_are_skipped_not_fatal(self):
        text = ("drm-memory-vram:\tabc KiB\n"
                "drm-memory-vram:\t12 furlongs\n"
                "drm-client-id:\tx\n"
                "drm-engine-gfx:\t-5 ns\n"
                "no colon here\n"
                "drm-memory-vram:\t-4 KiB\n")
        fact = parse_fdinfo(text)
        self.assertIsNone(fact["vram_kib"])
        self.assertIsNone(fact["client"])
        self.assertIsNone(fact["engine_ns"])

    def test_binary_bytes_are_read_as_no_facts(self):
        fact = parse_fdinfo("\x00\xff\xfe\x01 drm-driver")
        self.assertIsNone(fact["driver"])
        self.assertIsNone(fact["client"])


class CardChoiceTests(Fixture):
    def test_the_card_with_the_most_vram_is_chosen_and_its_pci_address_is_known(self):
        make_card(self.sys, card="card0", pci="0000:01:00.0", total=8 * GIB)
        device = AMDDevice()
        self.assertEqual(device.device.resolve(), self.device_folder.resolve())
        self.assertEqual(device.pci, PCI)

    def test_integrated_graphics_with_a_small_share_of_memory_is_not_chosen(self):
        make_card(self.sys, card="card0", pci="0000:01:00.0", total=512 << 20)
        self.assertEqual(AMDDevice().pci, PCI)              # the 16 GiB card wins over the 512 MiB one
        self.clear_cards()
        make_card(self.sys, card="card0", pci="0000:01:00.0", total=512 << 20)
        self.assertIsNone(find_card(self.sys))              # alone, integrated graphics is not a GPU to read
        with self.assertRaises(AMDError):
            AMDDevice()

    def test_a_card_of_another_vendor_is_not_chosen(self):
        make_card(self.sys, card="card0", pci="0000:01:00.0", vendor="0x10de", total=24 * GIB)
        self.assertEqual(AMDDevice().pci, PCI)

    def test_a_connector_entry_is_not_a_card(self):
        Path(self.sys, "class", "drm", "card1-DP-1").mkdir(parents=True)
        self.assertEqual(AMDDevice().pci, PCI)

    def test_no_card_at_all_is_an_error_not_a_guess(self):
        self.clear_cards()
        with self.assertRaises(AMDError):
            AMDDevice()


class ReadingTests(Fixture):
    def test_the_card_readings_match_the_sysfs_values(self):
        device = AMDDevice()
        self.assertEqual(device.memory_info(), {"total_mib": 16384, "free_mib": 15360, "used_mib": 1024})
        self.assertEqual(device.card_free_mib(), 15360)
        self.assertEqual(device.utilization(), {"gpu_percent": 40})
        self.assertEqual(device.temperature_c(), 45)
        self.assertAlmostEqual(device.power_usage_w(), 123.4)
        self.assertEqual(device.pcie_link(), {"generation": 4, "width": 16})
        self.assertFalse(device.needs_reinit)
        device.close()

    def test_each_process_holds_the_vram_of_its_clients_counted_once(self):
        self.processes_fixture()
        device = AMDDevice()
        processes = device.processes()
        self.assertEqual(processes[100], 4)       # 4096 + 512 KiB: the duplicate fd of client 7 is not added again
        self.assertEqual(processes[200], 2)
        self.assertIsNone(processes[600])         # a client with no VRAM figure
        for pid in (300, 400, 500):               # another driver, a non-DRM fd, another AMD card
            self.assertNotIn(pid, processes)
        self.assertEqual(device.proc_mib(100), 4)
        self.assertIsNone(device.proc_mib(300))
        self.assertIsNone(device.proc_mib(999))

    def test_the_monitor_sample_has_the_same_fields_as_the_nvidia_one(self):
        self.processes_fixture()
        sample = AMDDevice().monitor_info(100)
        self.assertEqual(sample, {"source": "gpu_amd", "total_mib": 16384, "free_mib": 15360, "used_mib": 1024,
                                  "utilization_percent": 40, "temperature_c": 45, "power_w": 123.4,
                                  "pcie_generation": 4, "pcie_width": 16, "model_process_mib": 4})

    def test_the_monitor_sample_without_a_model_process(self):
        self.assertEqual(AMDDevice().monitor_info(None)["model_process_mib"], 0)
        self.assertIsNone(AMDDevice().monitor_info(4242)["model_process_mib"])


class DegradeTests(Fixture):
    def test_missing_vram_counters_raise_in_the_method_and_are_none_in_the_monitor(self):
        (self.device_folder / "mem_info_vram_used").unlink()
        device = AMDDevice()
        with self.assertRaises(AMDError):
            device.memory_info()
        sample = device.monitor_info(None)
        self.assertIsNone(sample["total_mib"])
        self.assertIsNone(sample["free_mib"])
        self.assertEqual(sample["utilization_percent"], 40)       # the other fields still come

    def test_garbage_in_the_sysfs_files_is_none_in_the_monitor(self):
        for name, text in (("gpu_busy_percent", "abc\n"), ("current_link_width", "w\n"), ("current_link_speed", "fast\n"),
                           ("hwmon/hwmon4/temp1_input", "x\n"), ("hwmon/hwmon4/power1_average", "??\n")):
            put(self.device_folder / name, text)
        device = AMDDevice()
        for method in (device.utilization, device.temperature_c, device.power_usage_w, device.pcie_link):
            with self.assertRaises(AMDError):
                method()
        sample = device.monitor_info(None)
        self.assertIsNone(sample["utilization_percent"])
        self.assertIsNone(sample["temperature_c"])
        self.assertIsNone(sample["power_w"])
        self.assertIsNone(sample["pcie_generation"])

    def test_no_process_list_is_an_error_for_processes_and_none_in_the_monitor(self):
        device = AMDDevice(proc_root=self.root / "no-such-proc")
        with self.assertRaises(AMDError):
            device.processes()
        self.assertIsNone(device.monitor_info(100)["model_process_mib"])
        self.assertEqual(device.monitor_info(None)["model_process_mib"], 0)

    def test_a_process_whose_fdinfo_cannot_be_read_is_counted_and_skipped(self):
        make_process(self.proc, 700, {"3": (RENDER, None)})      # the fd exists, its fdinfo does not
        device = AMDDevice()
        self.assertEqual(device.processes(), {})
        self.assertEqual(device.unreadable, 1)

    def test_garbage_fdinfo_files_do_not_raise(self):
        make_process(self.proc, 800, {"3": (RENDER, None), "4": (RENDER, None)})
        (self.proc / "800" / "fdinfo").mkdir(parents=True, exist_ok=True)
        (self.proc / "800" / "fdinfo" / "3").write_bytes(b"\x00\xff\xfe\x01")
        (self.proc / "800" / "fdinfo" / "4").write_text("drm-driver:\tamdgpu\ndrm-client-id:\tzz\ndrm-memory-vram:\t-4 KiB\n")
        # the binary file has no driver line and is skipped; the text file has a client that is not a number and no usable VRAM
        self.assertEqual(AMDDevice().processes(), {800: None})


class LoadTests(Fixture):
    def setUp(self):
        super().setUp()
        self.now = [100.0]
        make_process(self.proc, 200, {"3": (RENDER, fdinfo(client=9, vram_kib=2048, gfx=0))}, comm="game")
        self.device = AMDDevice(clock=lambda: self.now[0])

    def set_gfx(self, ns):
        put(self.proc / "200" / "fdinfo" / "3", fdinfo(client=9, vram_kib=2048, gfx=ns))

    def test_the_first_call_has_nothing_to_compare_then_the_busy_share_is_reported(self):
        first, _ = self.device.process_utilization()
        self.assertEqual(first, {})
        self.set_gfx(500_000_000)                 # half a second of GFX time
        self.now[0] += 1.0
        second, _ = self.device.process_utilization()
        self.assertEqual(second, {200: 50})

    def test_the_share_is_capped_at_100_percent(self):
        self.device.process_utilization()
        self.set_gfx(3_000_000_000)
        self.now[0] += 1.0
        self.assertEqual(self.device.process_utilization()[0], {200: 100})

    def test_a_counter_that_went_backwards_gives_no_share(self):
        self.set_gfx(900_000_000)
        self.device.process_utilization()
        self.set_gfx(100_000_000)
        self.now[0] += 1.0
        self.assertEqual(self.device.process_utilization()[0], {})

    def test_within_the_scan_time_the_same_result_is_reused(self):
        self.device.process_utilization()
        self.set_gfx(500_000_000)
        self.now[0] += 1.0
        first = self.device.process_utilization()[0]
        self.assertEqual(self.device.process_utilization()[0], first)


class GameSignalsOnTheAmdCard(Fixture):
    def test_a_foreign_process_on_the_card_is_a_game_reason(self):
        now = [100.0]
        make_process(self.proc, 100, {"3": (RENDER, fdinfo(client=7, vram_kib=4096, gfx=0))}, comm="llama-server")
        make_process(self.proc, 200, {"3": (RENDER, fdinfo(client=9, vram_kib=2048, gfx=0))}, comm="game")
        device = AMDDevice(clock=lambda: now[0])
        config = {"gameDetectNvml": True, "gameVramMiB": 1, "gameGpuPercent": 80}
        signals = GameSignals(config, proc_root=self.proc, clock=lambda: now[0])
        reasons, errors = signals.collect(100, device)
        self.assertIn("foreign VRAM PID200 game 2MiB", reasons)
        self.assertEqual(errors, [])
        put(self.proc / "200" / "fdinfo" / "3", fdinfo(client=9, vram_kib=2048, gfx=900_000_000))
        now[0] += 1.0
        reasons, errors = signals.collect(100, device)
        self.assertIn("foreign GPU PID200 game 90%", reasons)
        self.assertEqual(errors, [])


class BackendChoiceTests(Fixture):
    def test_without_the_nvidia_library_the_amd_card_is_used(self):
        device = gpu_backend.open_device(nvml=Mock(side_effect=OSError("libnvidia-ml.so.1: cannot open shared object file")))
        self.assertIsInstance(device, AMDDevice)

    def test_nvidia_is_used_when_its_library_answers(self):
        sentinel = object()
        self.assertIs(gpu_backend.open_device(nvml=Mock(return_value=sentinel)), sentinel)

    def test_without_any_gpu_the_error_names_both_backends(self):
        self.clear_cards()
        with self.assertRaises(GPUError) as cm:
            gpu_backend.open_device(nvml=Mock(side_effect=OSError("libnvidia-ml.so.1: cannot open")))
        self.assertIn("NVIDIA", str(cm.exception))
        self.assertIn("AMD", str(cm.exception))
        self.assertEqual(cm.exception.backend, "GPU")

    def test_the_gateway_calls_exist_on_the_amd_device(self):
        for name in ("card_free_mib", "memory_info", "processes", "process_utilization", "proc_mib", "monitor_info", "close"):
            self.assertTrue(callable(getattr(AMDDevice, name)), name)
        self.assertFalse(AMDDevice().needs_reinit)


if __name__ == "__main__":
    unittest.main()
