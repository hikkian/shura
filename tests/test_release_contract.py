"""The release contract (docs/RELEASE-CONTRACT.md): one archive name, one manifest, build -> installer. Fake network, no builds."""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
sys.path.insert(0, str(ROOT))
from tests.test_release_manifest import (ARCHIVE, GOOD, HW, NAME, URL, Fake, ManifestCase, manifest, release)  # noqa: E402
from universal import backends, engine, release_manifest as rm  # noqa: E402


class Chosen(ManifestCase):
    def test_a_fitting_build_is_chosen_checked_and_unpacked(self):
        with Fake(self.tmp) as f:
            (backend, server, res), out = f.choose()
        self.assertEqual((backend, res[0]["engine_path"]), ("rocm", "manifest"))
        self.assertIn("SHA-256 matched", res[0]["engine_why"])
        self.assertEqual(f.opened, [URL])
        self.assertTrue(server.is_file())
        self.assertEqual(f.source_calls, [])

    def test_a_misfit_names_the_failed_condition_and_goes_to_source(self):
        cases = {
            "gfx": (dict(hw={**HW, "gpus": [{**HW["gpus"][0], "gfx": "gfx1100"}]}), "gfx1201"),
            "runtime too old": (dict(rocm="6.0"), "rocm min 6.2"),
            "runtime too new": (dict(rocm="8.0"), "rocm max 7.9"),
            "glibc": (dict(glibc="2.28"), "glibc 2.31"),
        }
        for name, (kw, needle) in cases.items():
            with self.subTest(name), Fake(self.tmp, **kw) as f:
                (_, _, res), out = f.choose()
                self.assertEqual(res[0]["engine_path"], "source")
                self.assertEqual(f.opened, [])
                self.assertIn(needle, out.text)
                self.assertIn("building shura.cpp from source", out.text)

    def test_a_family_the_machine_cannot_use_is_named(self):
        m = {"os": "linux", "arch": "x86_64", "backends": ["vulkan"], "gpu_targets": ["gfx1201"], "rocm": "6.4", "glibc": "2.39"}
        rel, why = rm.best_release(manifest(), m)
        self.assertIsNone(rel)
        self.assertIn("HIP is not a backend this machine can use", why)

    def test_wrong_sum_refuses_this_build_and_tries_no_second_one(self):
        other = release(id="other", version="0.9", assets=[{"name": "shura-cpp-0.9-hip-linux-x86_64.tar.gz",
                                                             "url": "https://example.invalid/o.tar.gz", "sha256": GOOD}])
        bad = release(assets=[{"name": NAME, "url": URL, "sha256": "f" * 64}])
        with Fake(self.tmp, mtext=json.dumps(manifest(bad, other))) as f:
            (_, _, res), out = f.choose()
        self.assertEqual(f.opened, [URL])
        self.assertEqual(len(f.source_calls), 1)
        self.assertEqual(res[0]["engine_path"], "source")
        self.assertIn("сумма не сошлась", out.text)
        self.assertFalse(list((self.tmp / "home" / "engines").glob("*shura*")))

    def test_a_path_archive_name_is_refused_everywhere(self):
        for bad in ("../x", "../../x.tar.gz", "sub/shura-cpp-1.0-hip-linux-x86_64.tar.gz"):
            with self.subTest(bad):
                m = manifest(release(assets=[{"name": bad, "url": URL, "sha256": GOOD}]))
                with self.assertRaises(rm.ManifestError):
                    rm.validate(m)
                with Fake(self.tmp, mtext=json.dumps(m)) as f:
                    (_, _, res), out = f.choose()
                self.assertEqual((f.opened, res[0]["engine_path"]), ([], "source"))
                self.assertIn("manifest is unusable", out.text)
        # even if validation were bypassed, the installer refuses before any download
        rel = release(assets=[{"name": "../x", "url": URL, "sha256": GOOD}])
        with Fake(self.tmp) as f, self.assertRaises(engine.EngineError):
            engine.install_manifest_release(rel, self.tmp / "e")
        self.assertEqual(f.opened, [])

    def test_name_must_agree_with_the_release(self):
        for kw in (dict(version="2.0"), dict(family="Vulkan", gpu_targets=[], runtime={"vulkan_min": "1.2"}), dict(arch="arm64")):
            with self.subTest(kw), self.assertRaises(rm.ManifestError):
                rm.validate(manifest(release(**kw)))

    def test_fields_of_the_contract_are_required(self):
        for k in ("parts", "built_with"):
            r = release()
            del r[k]
            with self.subTest(k), self.assertRaises(rm.ManifestError):
                rm.validate(manifest(r))
        with self.assertRaises(rm.ManifestError):
            rm.validate(manifest(release(built_with={"cmake_flags": []})))


class Names(unittest.TestCase):
    def test_every_family_parses_and_round_trips(self):
        for fam, backend in (("HIP", "rocm"), ("CUDA", "cuda"), ("Vulkan", "vulkan"), ("SYCL", "sycl"), ("CPU", "cpu")):
            for arch in ("x86_64", "arm64"):
                n = backends.release_asset_name("1.2.3", fam, "linux", arch)
                p = backends.parse_release_asset(n)
                self.assertEqual((p["family"], p["backend"], p["version"], p["os"], p["arch"]), (fam, backend, "1.2.3", "linux", arch))
                self.assertEqual(n, f"shura-cpp-1.2.3-{fam.lower()}-linux-{arch}.tar.gz")

    def test_wrong_names_give_none(self):
        for n in ("", None, 5, "llama-b11301-bin-ubuntu-vulkan-x64.tar.gz", "shura-cpp-1.0-rocm-linux-x86_64.tar.gz",
                  "shura-cpp-1.0-hip-windows-x86_64.tar.gz", "shura-cpp-1.0-hip-linux-x64.tar.gz", "shura-cpp-hip.tar.gz",
                  "shura-cpp-1.0-hip-linux-x86_64.zip", "../shura-cpp-1.0-hip-linux-x86_64.tar.gz", "shura-cpp-x-hip-linux-x86_64.tar.gz",
                  "shura-cpp-1.0-hip-linux-x86_64.tar.gz\n"):
            self.assertIsNone(backends.parse_release_asset(n), n)

    def test_upstream_names_keep_their_own_parser(self):
        self.assertIsNone(backends.parse_release_asset("llama-b1-bin-ubuntu-vulkan-x64.tar.gz"))
        self.assertEqual(backends.parse_asset("llama-b1-bin-ubuntu-vulkan-x64.tar.gz")["backend"], "vulkan")


class CpuIsa(unittest.TestCase):
    M = {"os": "linux", "arch": "x86_64", "backends": ["cpu"], "gpu_targets": [], "glibc": "2.39"}

    def cpu(self, **kw):
        return release(id="cpu", family="CPU", gpu_targets=[], runtime={}, parts=["mtp"],
                       assets=[{"name": "shura-cpp-1.0-cpu-linux-x86_64.tar.gz", "url": URL, "sha256": GOOD}], **kw)

    def test_cpu_isa_is_a_field_and_changes_no_choice(self):
        plain, isa = self.cpu(), self.cpu(cpu_isa=["avx2", "avx512f", "avx512_vnni"])
        rm.validate(manifest(isa))
        self.assertEqual(rm.best_release(manifest(isa), self.M)[0]["id"], "cpu")
        self.assertIsNone(rm.why_not(isa, self.M))
        self.assertEqual(rm.best_release(manifest(plain), self.M)[0]["id"], rm.best_release(manifest(isa), self.M)[0]["id"])
        self.assertEqual(rm.best_release(manifest(isa), {**self.M, "isa": ["sse4"]})[0]["id"], "cpu")
        with self.assertRaises(rm.ManifestError):
            rm.validate(manifest(self.cpu(cpu_isa="avx2")))

    def test_vulkan_needs_no_gfx_and_must_not_list_any(self):
        vk = release(id="vk", family="Vulkan", gpu_targets=[], runtime={"vulkan_min": "1.2"}, version="1.0",
                     assets=[{"name": "shura-cpp-1.0-vulkan-linux-x86_64.tar.gz", "url": URL, "sha256": GOOD}])
        m = {**self.M, "backends": ["vulkan"], "vulkan": "1.3"}
        self.assertIsNone(rm.why_not(vk, m))
        with self.assertRaises(rm.ManifestError):
            rm.validate(manifest({**copy.deepcopy(vk), "gpu_targets": ["gfx1201"]}))


if __name__ == "__main__":
    unittest.main()
