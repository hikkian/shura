"""build-llama.sh: the measured HIP patch stack, its order, idempotent reruns, and the commit check. No network."""
import json
import os
import re
import sys
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build-llama.sh"
FORK = Path.home() / "Projects" / "llama.cpp-ring"
MEASURED_COMMIT = "95d26d19e332eea80caf9b84e30366aba4a21659"
HIP_STACK = ["upstream-picks", "forktune1", "gqavec2", "op-profile", "fuse1", "fuse3",
             "path-a-prefillprof4", "fattn2-upstream-28102",
             "upstream-fattn-fixes", "gemma-fa512", "cachepf1-prefill-off",
             "cachex1c-pinned-source", "cachex1c-stage3-two-channel",
             "cachex1d-gemma-architecture", "cachex1e-live-resize"]


class BuildStackScript(unittest.TestCase):
    def test_default_commit_is_the_measured_fork_commit(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(f'COMMIT="{MEASURED_COMMIT}"', text)
        self.assertNotIn("27c54b4bbcefadedcec6397477cc2e866c1db716", text)
        self.assertIn("SHURA_LLAMA_REPO", text)
        self.assertIn("FORKPUB1", text)

    def test_hip_stack_order_in_script_matches_the_list(self):
        text = SCRIPT.read_text(encoding="utf-8")
        m = re.search(r'HIP_STACK="([^"]+)"', text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1).split(), HIP_STACK)

    def test_every_stack_patch_exists_with_a_header_and_a_diff(self):
        for name in HIP_STACK:
            path = ROOT / "patches" / f"{name}.patch"
            self.assertTrue(path.exists(), name)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertTrue(lines[0].startswith("# "), f"{name}: header missing")
            self.assertTrue(any(l.startswith(("diff --git", "--- ")) for l in lines), f"{name}: no diff")

    def test_hip_configure_matches_the_measured_build_flags(self):
        # the measured builds (BUILD-INFO of picks/fattn2/final) were static, without curl or the prebuilt UI
        text = SCRIPT.read_text(encoding="utf-8")
        for flag in ("-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF", "-DGPU_TARGETS="):
            self.assertIn(flag, text, flag)

    @unittest.skipUnless(FORK.joinpath(".git").exists(), "local fork checkout not available")
    def test_auto_target_and_selected_targets_are_checked_before_build(self):
        with tempfile.TemporaryDirectory(prefix="shura-hip-target-") as name:
            tmp = Path(name)
            tools = tmp / "tools"
            tools.mkdir()
            (tools / "rocminfo").write_text("#!" + sys.executable + "\nprint(\"Name: gfx1101\")\n")
            cmake = r"""import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if "--build" in args:
    Path(os.environ["BUILD_RECORD"]).write_text(json.dumps(args))
else:
    dest = Path(args[args.index("-B") + 1])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "CMakeCache.txt").write_text("GPU_TARGETS:UNINITIALIZED=" + os.environ["MOCK_ARCH"] + "\n")
    Path(os.environ["CONFIG_RECORD"]).write_text(json.dumps(args))
"""
            (tools / "cmake").write_text("#!" + sys.executable + "\n" + cmake)
            for tool in tools.iterdir():
                tool.chmod(0o755)
            for arch in ("gfx1101", "gfx1201"):
                with self.subTest(configured_arch=arch):
                    record = tmp / ("build-" + arch + ".json")
                    configure = tmp / ("configure-" + arch + ".json")
                    env = dict(os.environ, SHURA_LLAMA_REPO=str(FORK),
                               PATH=str(tools) + os.pathsep + os.environ["PATH"],
                               SHURA_BUILD_TARGETS="llama-server llama-perplexity test-backend-ops",
                               SHURA_BUILD_EXAMPLES="OFF", MOCK_ARCH=arch,
                               BUILD_RECORD=str(record), CONFIG_RECORD=str(configure))
                    env.pop("SHURA_STOP_AFTER_PATCH", None)
                    r = subprocess.run([str(SCRIPT), "--hip", str(tmp / arch)],
                                       env=env, capture_output=True, text=True, timeout=120)
                    if arch == "gfx1101":
                        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                        self.assertIn("Configured HIP GPU target: gfx1101", r.stdout)
                        args = json.loads(record.read_text())
                        self.assertEqual(args[args.index("--target") + 1:],
                                         ["llama-server", "llama-perplexity", "test-backend-ops"])
                        self.assertIn("-DLLAMA_BUILD_EXAMPLES=OFF", json.loads(configure.read_text()))
                    else:
                        self.assertNotEqual(r.returncode, 0)
                        self.assertIn("HIP target mismatch", r.stdout + r.stderr)
                        self.assertFalse(record.exists())

    def test_bash_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_missing_commit_in_local_repo_gives_the_clear_error(self):
        tmp = Path(tempfile.mkdtemp(prefix="shura-stack-test-"))
        try:
            src = tmp / "other"
            src.mkdir()
            subprocess.run(["git", "init", "-q", str(src)], check=True)
            (src / "f.txt").write_text("x\n")
            subprocess.run(["git", "-C", str(src), "add", "f.txt"], check=True)
            subprocess.run(["git", "-C", str(src), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"], check=True)
            env = dict(os.environ, SHURA_LLAMA_REPO=str(src), SHURA_STOP_AFTER_PATCH="1")
            r = subprocess.run([str(SCRIPT), "--hip", "gfx1101", str(tmp / "dest")], env=env, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("not found", r.stderr)
            self.assertIn("FORKPUB1", r.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipUnless(FORK.joinpath(".git").exists(), "local fork checkout not available")
    def test_stack_applies_twice_from_the_local_fork(self):
        tmp = Path(tempfile.mkdtemp(prefix="shura-stack-test-"))
        try:
            env = dict(os.environ, SHURA_LLAMA_REPO=str(FORK), SHURA_STOP_AFTER_PATCH="1")
            dest = tmp / "dest"
            for run in (1, 2):
                r = subprocess.run([str(SCRIPT), "--hip", "gfx1101", str(dest)], env=env, capture_output=True, text=True, timeout=600)
                self.assertEqual(r.returncode, 0, f"run {run}: {r.stderr[-400:]}")
            self.assertIn("HIP patch stack already applied", r.stdout)
            stamp = (dest / ".shura-hip-stack").read_text().split()
            self.assertEqual(len(stamp), len(HIP_STACK))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
