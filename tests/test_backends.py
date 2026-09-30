"""Backend build selection tests against the real asset list of llama.cpp release b11301 (names only)."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
from universal import backends as B  # noqa: E402

ASSETS = json.loads((ROOT / "tests/fixtures/release_assets_b11301.json").read_text())["assets"]
HW = {p.stem: json.loads(p.read_text()) for p in (ROOT / "tests/fixtures/hardware").glob("*.json")}


def pick(os_family, arch, backend, driver=None):
    a = B.pick_asset(ASSETS, os_family, arch, backend, driver)
    return a["name"] if a else None


class Parse(unittest.TestCase):
    def test_parses_backend_version_and_variant(self):
        p = B.parse_asset("llama-b11301-bin-ubuntu-cuda-13.4-x64.tar.gz")
        self.assertEqual((p["os"], p["arch"], p["backend"], p["version"]), ("linux", "x86_64", "cuda", "13.4"))
        self.assertEqual(B.parse_asset("llama-b11301-bin-ubuntu-sycl-fp16-x64.tar.gz")["variant"], "fp16")
        self.assertEqual(B.parse_asset("llama-b11301-bin-macos-arm64.tar.gz")["backend"], "metal")
        self.assertEqual(B.parse_asset("llama-b11301-bin-win-cpu-x64.zip")["backend"], "cpu")

    def test_ignores_what_is_not_a_desktop_build(self):
        for n in ("llama-b11301-ui.tar.gz", "cudart-llama-bin-ubuntu-cuda-12.8-x64.tar.gz",
                  "llama-b11301-bin-linux-arm64-snapdragon.tar.gz", "llama-b11301-bin-win-opencl-adreno-arm64.zip",
                  "llama-b11301-bin-android-arm64.tar.gz", "llama-b11301-xcframework.zip"):
            self.assertIsNone(B.parse_asset(n), n)


class Pick(unittest.TestCase):
    def test_linux_builds_per_backend(self):
        self.assertEqual(pick("linux", "x86_64", "cpu"), "llama-b11301-bin-ubuntu-x64.tar.gz")
        self.assertEqual(pick("linux", "x86_64", "vulkan"), "llama-b11301-bin-ubuntu-vulkan-x64.tar.gz")
        self.assertEqual(pick("linux", "x86_64", "rocm"), "llama-b11301-bin-ubuntu-rocm-10.0-x64.tar.gz")
        self.assertEqual(pick("linux", "x86_64", "sycl"), "llama-b11301-bin-ubuntu-sycl-fp16-x64.tar.gz")
        self.assertEqual(pick("linux", "x86_64", "openvino"), "llama-b11301-bin-ubuntu-openvino-2026.4-x64.tar.gz")

    def test_cuda_follows_what_the_driver_can_run(self):
        self.assertEqual(pick("linux", "x86_64", "cuda", "615.71.09"), "llama-b11301-bin-ubuntu-cuda-13.4-x64.tar.gz")
        self.assertEqual(pick("linux", "x86_64", "cuda", "550.120"), "llama-b11301-bin-ubuntu-cuda-12.8-x64.tar.gz")
        self.assertEqual(pick("windows", "x86_64", "cuda", "551.0"), "llama-b11301-bin-win-cuda-12.4-x64.zip")
        self.assertIsNone(pick("linux", "x86_64", "cuda", "470.1"))

    def test_apple_windows_and_arm(self):
        self.assertEqual(pick("macos", "arm64", "metal"), "llama-b11301-bin-macos-arm64.tar.gz")
        self.assertEqual(pick("windows", "x86_64", "vulkan"), "llama-b11301-bin-win-vulkan-x64.zip")
        self.assertEqual(pick("windows", "x86_64", "rocm"), "llama-b11301-bin-win-rocm-10.0-x64.zip")
        self.assertEqual(pick("linux", "arm64", "vulkan"), "llama-b11301-bin-ubuntu-vulkan-arm64.tar.gz")

    def test_a_missing_build_is_none_not_a_wrong_one(self):
        self.assertIsNone(pick("macos", "arm64", "cuda"))
        self.assertIsNone(pick("windows", "x86_64", "metal"))


class BuildsForMachines(unittest.TestCase):
    def test_reference_machine_gets_cuda_then_vulkan_then_cpu(self):
        hw = HW["ref_rtx4070s_12g_32g"]
        hw["gpus"][0]["driver"] = "615.71.09"
        got = [b for b, _ in B.builds_for(hw, ["cuda", "vulkan", "cpu"], ASSETS)]
        self.assertEqual(got, ["cuda", "vulkan", "cpu"])

    def test_amd_offers_rocm_and_vulkan_builds(self):
        got = [b for b, _ in B.builds_for(HW["rx7900xtx_24g_64g"], ["rocm", "vulkan", "cpu"], ASSETS)]
        self.assertEqual(got, ["rocm", "vulkan", "cpu"])

    def test_unavailable_backends_are_dropped(self):
        got = [b for b, _ in B.builds_for(HW["apple_m2max_64g"], ["metal", "cpu"], ASSETS)]
        self.assertEqual(got, ["metal"])            # there is no separate CPU build listed for macOS arm64 here


class Verify(unittest.TestCase):
    def test_sha256_check(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"hello")
            f.flush()
            good = hashlib.sha256(b"hello").hexdigest()
            self.assertTrue(B.verify_sha256(f.name, "sha256:" + good))
            self.assertTrue(B.verify_sha256(f.name, good.upper()))
            self.assertFalse(B.verify_sha256(f.name, "sha256:" + "0" * 64))


if __name__ == "__main__":
    unittest.main()
