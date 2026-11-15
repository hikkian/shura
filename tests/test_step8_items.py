"""INSTALL1c step 8: --disk-benchmark key, release_manifest_url setting, source build commands. Nothing runs or downloads."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
from universal import build_cmds, cli, detect, engine  # noqa: E402

BASE = Path(os.environ.get("TMPDIR") or "/dev/shm")


class DiskBenchmark(unittest.TestCase):
    def run_load(self, argv, sample):
        args = mock.Mock(profile=None, ram_gbs=None, no_measure=True, catalog=None, disk_benchmark="--disk-benchmark" in argv)
        hw = {"detection_notes": []}
        with mock.patch.object(detect, "detect", return_value=hw) as d, mock.patch.object(cli, "disk_sample_path", return_value=sample):
            cli.load(args)
        return d, hw

    def test_off_by_default(self):
        d, _ = self.run_load([], None)
        d.assert_called_once_with()
        self.assertTrue(cli.argparse.ArgumentParser)  # key exists in the parser below
        with mock.patch.object(cli, "load", side_effect=RuntimeError("stop")):
            with self.assertRaises(RuntimeError):
                cli.main(["check"])

    def test_key_without_sample_says_not_measured(self):
        d, hw = self.run_load(["--disk-benchmark"], None)
        d.assert_called_once_with(disk_sample=None, disk_benchmark=True)
        self.assertTrue(any("не измерено" in n for n in hw["detection_notes"]))

    def test_key_with_sample_passes_it(self):
        d, _ = self.run_load(["--disk-benchmark"], "/x/m.gguf")
        d.assert_called_once_with(disk_sample="/x/m.gguf", disk_benchmark=True)

    def test_parser_accepts_key_and_sets_flag(self):
        seen = {}
        def fake_load(a):
            seen["v"] = a.disk_benchmark
            raise RuntimeError
        for argv, want in ((["check"], False), (["check", "--disk-benchmark"], True)):
            with mock.patch.object(cli, "load", side_effect=fake_load):
                with self.assertRaises(RuntimeError):
                    cli.main(argv)
            self.assertEqual(seen["v"], want)

    def test_sample_path_none_without_models(self):
        with tempfile.TemporaryDirectory(dir=BASE) as t, mock.patch.dict(os.environ, {"SHURA_HOME": t}):
            self.assertIsNone(cli.disk_sample_path())


class ManifestUrlSetting(unittest.TestCase):
    def cfg(self, text):
        d = tempfile.TemporaryDirectory(dir=BASE)
        self.addCleanup(d.cleanup)
        p = Path(d.name) / "installer.json"
        if text is not None:
            p.write_text(text)
        return mock.patch.object(engine, "INSTALLER_CONFIG", p)

    def test_shipped_config_is_empty(self):
        data = json.loads((ROOT / "catalog" / "installer.json").read_text())
        self.assertEqual(data["release_manifest_url"], "")
        self.assertIn("_note", data)

    def test_empty_or_missing_means_none(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(engine.MANIFEST_ENV, None)
            for text in ('{"release_manifest_url": ""}', None, "not json", '{"release_manifest_url": 5}'):
                with self.cfg(text):
                    self.assertIsNone(engine.manifest_source())

    def test_config_value_used_and_fetched(self):
        url = "https://example.invalid/release-manifest.json"
        with mock.patch.dict(os.environ), self.cfg(json.dumps({"release_manifest_url": url})):
            os.environ.pop(engine.MANIFEST_ENV, None)
            self.assertEqual(engine.manifest_source(), url)
            with mock.patch.object(engine, "_open", side_effect=engine.EngineError("offline")) as op:
                with self.assertRaises(engine.release_manifest.ManifestError):
                    engine.fetch_manifest(engine.manifest_source())
            self.assertEqual(op.call_args[0][0], url)

    def test_env_wins(self):
        with mock.patch.dict(os.environ, {engine.MANIFEST_ENV: "/tmp/m.json"}), self.cfg('{"release_manifest_url": "https://a.invalid/x"}'):
            self.assertEqual(engine.manifest_source(), "/tmp/m.json")


class BuildCmds(unittest.TestCase):
    def test_flags_per_backend(self):
        for backend, flag in (("cuda", "-DGGML_CUDA=ON"), ("vulkan", "-DGGML_VULKAN=ON")):
            cfg, build = build_cmds.build_commands(backend, "/src/shura.cpp-v1", jobs=4)
            self.assertIn(flag, cfg)
            self.assertEqual(cfg[:3], ["cmake", "-S", "/src/shura.cpp-v1"])
            self.assertIn("Ninja", cfg)
            self.assertEqual(build[:2], ["cmake", "--build"])
            self.assertEqual(build[-2:], ["-j", "4"])
        cfg, _ = build_cmds.build_commands("cpu", "/src/x")
        self.assertFalse([a for a in cfg if a.startswith("-DGGML_") ])

    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            build_cmds.build_commands("hip", "/src/x")

    def test_all_present(self):
        self.assertEqual(build_cmds.check_tools("cuda", which=lambda t: "/bin/" + t, exists=lambda p: True, family="fedora"), [])

    def test_missing_tools_text(self):
        which = lambda t: None if t in ("nvcc", "ninja") else "/bin/" + t
        lines = build_cmds.check_tools("cuda", which=which, exists=lambda p: True, family="debian")
        text = "\n".join(lines)
        self.assertIn("nvcc", text)
        self.assertIn("apt-get update && apt-get install -y ninja-build nvidia-cuda-toolkit", text)
        self.assertNotIn("sudo", text)
        v = "\n".join(build_cmds.check_tools("vulkan", which=lambda t: "/bin/" + t if t != "glslc" else None, exists=lambda p: False, family="fedora"))
        self.assertIn("Vulkan headers", v)
        self.assertIn("dnf install -y vulkan-headers vulkan-loader-devel glslc", v)
        self.assertNotIn("nvcc", "\n".join(build_cmds.check_tools("cpu", which=lambda t: None, exists=lambda p: False, family="arch")))

    def test_nothing_is_run(self):
        with mock.patch.object(subprocess, "run") as r, mock.patch.object(subprocess, "Popen") as po, \
                mock.patch.object(subprocess, "check_call") as cc, mock.patch.object(subprocess, "check_output") as co, \
                mock.patch.object(os, "system") as sy:
            for b in ("cuda", "vulkan", "cpu"):
                build_cmds.build_commands(b, "~/Projects/shura.cpp-v1")
                build_cmds.check_tools(b, which=lambda t: None, exists=lambda p: False, family="fedora")
        for m in (r, po, cc, co, sy):
            self.assertEqual(m.call_count, 0)


if __name__ == "__main__":
    unittest.main()
