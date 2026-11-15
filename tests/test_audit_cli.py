"""Regression tests for the CLI / install-side audit (reports/audit-2026-10-06/agent-cli.md): uninstall safety, pid reuse,
engine extraction, `shura` command-line behaviour against a fake gateway, install.sh/uninstall.sh in a throw-away HOME with
stub systemctl/docker, the trace/rollback/graft tools, and the setup helpers. Everything runs in /dev/shm (or the system temp
folder), offline, with no GPU and no real service."""
import importlib.util
import io
import json
import os
import shutil
import socket
import socketserver
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
from universal import cli, engine, install, launch, report  # noqa: E402
import shura_setup as S  # noqa: E402

FAKE = ROOT / "tests/fixtures/fake_llama_server.py"
POSIX = sys.platform != "win32"
TMP_BASE = "/dev/shm" if os.path.isdir("/dev/shm") else None


def tmpdir(case):
    d = tempfile.mkdtemp(dir=TMP_BASE, prefix="shura-audit-")
    case.addCleanup(shutil.rmtree, d, ignore_errors=True)
    return Path(d)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_exe(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


# ------------------------------------------------------------------------------------------------ H1, uninstall
class Uninstall(unittest.TestCase):
    def setUp(self):
        self.root = tmpdir(self)
        self.home = self.root / "home"
        self.home.mkdir()
        self.shura = self.home / ".local/share/shura"

    def run_cli(self, *args, **kw):
        """`python cli.py ...` with HOME pointing into the throw-away tree, never the real one."""
        env = {k: v for k, v in os.environ.items() if k not in ("SHURA_HOME", "XDG_DATA_HOME")}
        env["HOME"] = str(self.home)
        return subprocess.run([sys.executable, str(ROOT / "installer/universal/cli.py"), *args], env=env,
                              capture_output=True, text=True, timeout=60, **kw)

    def make_shared_home(self):
        """What setup.sh + `shura install` leave in the same folder."""
        for rel in ("app/config/guardian.json", "app/gateway/ai_gateway.py", "llama.cpp-perf/build/bin/llama-server",
                    "models/Tiel-Coder-gateway.gguf", "models/mine.gguf", "models/stories15M-q4_0.gguf",
                    "engines/b1-cpu/llama-server", "logs/server.log", "shuracode/x"):
            (self.shura / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.shura / rel).write_text("data")
        (self.shura / "state.json").write_text(json.dumps({"model_path": str(self.shura / "models/mine.gguf")}))

    def test_shared_gateway_home_keeps_the_gateway_install(self):
        self.make_shared_home()
        r = self.run_cli("uninstall", "--yes")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        for kept in ("app/config/guardian.json", "app/gateway/ai_gateway.py", "llama.cpp-perf/build/bin/llama-server",
                     "models/Tiel-Coder-gateway.gguf", "shuracode/x"):
            self.assertTrue((self.shura / kept).exists(), f"{kept} was deleted")
        for gone in ("engines", "logs", "state.json", "models/mine.gguf", "models/stories15M-q4_0.gguf"):
            self.assertFalse((self.shura / gone).exists(), f"{gone} should have been removed")
        self.assertIn("keeps everything else", r.stdout)

    def test_a_universal_only_home_is_removed_completely(self):
        for rel in ("engines/b1-cpu/llama-server", "models/a.gguf", "logs/x.log", "state.json", launch.HOME_MARKER):
            (self.shura / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.shura / rel).write_text("x")
        r = self.run_cli("uninstall", "--yes")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(self.shura.exists())

    def test_the_users_home_folder_is_never_deleted(self):
        (self.home / "Documents").mkdir()
        (self.home / "Documents/thesis.txt").write_text("precious")
        (self.home / "state.json").write_text("{}")             # even a folder that looks marked
        for target in (str(self.home), str(self.home / ".")):
            r = self.run_cli("uninstall", "--yes", "--force", "--dir", target)
            self.assertEqual(r.returncode, 1, target)
            self.assertIn("refusing", r.stderr)
        # the file-system root is only ever checked through the planner: a regression must never be able to delete it
        env_home = mock.patch.object(Path, "home", return_value=self.home)
        with env_home:
            for target in (Path("/"), self.home.parent, self.home):
                self.assertIsNotNone(cli.uninstall_plan(target, None, force=True)[2], target)
        self.assertTrue((self.home / "Documents/thesis.txt").exists())
        self.assertTrue((self.home / "state.json").exists())

    def test_a_git_checkout_is_refused(self):
        repo = self.root / "repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "logs").mkdir()
        (repo / "state.json").write_text("{}")
        r = self.run_cli("uninstall", "--yes", "--dir", str(repo))
        self.assertEqual(r.returncode, 1)
        self.assertTrue((repo / "logs").exists())

    def test_a_folder_that_is_not_a_shura_home_needs_force_and_loses_only_its_own_entries(self):
        other = self.root / "stuff"
        (other / "logs").mkdir(parents=True)
        (other / "notes.txt").write_text("keep me")
        r = self.run_cli("uninstall", "--yes", "--dir", str(other))
        self.assertEqual(r.returncode, 1)
        self.assertIn("--force", r.stderr)
        self.assertTrue((other / "logs").exists())
        r = self.run_cli("uninstall", "--yes", "--force", "--dir", str(other))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((other / "notes.txt").exists())
        self.assertFalse((other / "logs").exists())

    def test_declining_the_prompt_deletes_nothing(self):
        self.make_shared_home()
        r = self.run_cli("uninstall", input="n\n")
        self.assertIn("Nothing was deleted", r.stdout)
        self.assertTrue((self.shura / "engines").exists())


# ------------------------------------------------------------------------------------------------ L1, stale pid
@unittest.skipUnless(POSIX, "signals")
class StalePid(unittest.TestCase):
    def test_a_recycled_pid_of_another_program_is_not_signalled(self):
        home = tmpdir(self)
        bystander = subprocess.Popen(["sleep", "300"], start_new_session=True)       # a group leader, like a shell or terminal
        self.addCleanup(lambda: (bystander.kill(), bystander.wait()))
        (home / "server.pid").write_text(str(bystander.pid))
        self.assertFalse(launch.alive(bystander.pid))
        self.assertFalse(launch.stop(home))
        time.sleep(0.3)
        self.assertIsNone(bystander.poll(), "stop() killed a process that is not a llama-server")
        self.assertFalse((home / "server.pid").exists())

    def test_a_real_server_is_still_recognised_and_stopped(self):
        home = tmpdir(self)
        port = free_port()
        launch.start(home, [sys.executable, str(FAKE), "--port", str(port)], port, wait_s=30)
        pid = launch.read_pid(home)
        proc = launch._PROCS.pop(pid)                 # as a new `shura stop` process would see it: only the pid file
        self.assertTrue(launch.alive(pid))
        self.assertTrue(launch.stop(home))
        proc.wait(timeout=10)
        self.assertFalse(launch.health(port))


class PortInUse(unittest.TestCase):
    def test_start_says_which_port_is_taken_instead_of_failing_later(self):
        home = tmpdir(self)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen()
            port = s.getsockname()[1]
            with self.assertRaises(launch.LaunchError) as cm:
                launch.start(home, [sys.executable, str(FAKE), "--port", str(port)], port, wait_s=5)
        self.assertIn("already in use", str(cm.exception))
        self.assertIn("--port", str(cm.exception))


# ------------------------------------------------------------------------------------------------ M9
@unittest.skipUnless(POSIX, "shell-script stand-in")
class InstallWithRunningServer(unittest.TestCase):
    def test_a_server_that_is_already_running_is_stopped_openly_and_does_not_degrade_the_result(self):
        sys.path.insert(0, str(ROOT / "tests"))
        import test_install as T
        d, bin_dir = tmpdir(self), tmpdir(self)
        server = T.fake_server(bin_dir)
        with T.Hub({"Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf": os.urandom(200_000)}) as hub:
            port = free_port()
            launch.start(d, [str(server), "--port", str(port)], port, wait_s=30)       # `shura start` from an earlier install
            old_pid = launch.read_pid(d)
            old_proc = launch._PROCS.pop(old_pid)
            self.addCleanup(lambda: old_proc.poll() is None and old_proc.kill())
            cat, out = T.tiny_catalog(0), T.Out()

            def fake_choose(hw, plan, root, tag, out_, release=None, only_backend=None):
                return only_backend or plan["backend_candidates"][0], server, []
            with mock.patch.object(install, "choose_engine", side_effect=fake_choose):
                code = install.run(T.args(d, hub, port=port), T.HW["rx9070xt_16g_32g"], cat, out)
            self.assertEqual(code, 0, out.text)
            self.assertIn("started earlier is running", out.text)
            argv = launch.load_state(d)["argv"]
            self.assertNotIn("draft-mtp", argv, "the default Vulkan path keeps MTP disabled")
            self.assertNotIn("may not run", out.text)
            old_proc.wait(timeout=10)
            self.assertFalse(launch.alive(old_pid))


    def test_a_busy_port_stops_the_install_with_the_real_reason_instead_of_blaming_the_build(self):
        sys.path.insert(0, str(ROOT / "tests"))
        import test_install as T
        d, bin_dir = tmpdir(self), tmpdir(self)
        server = T.fake_server(bin_dir)
        with T.Hub({"Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf": os.urandom(200_000)}) as hub, socket.socket() as gateway:
            gateway.bind(("127.0.0.1", 0))                     # "the Shura gateway on 8080"
            gateway.listen()
            cat, out = T.tiny_catalog(0), T.Out()

            def fake_choose(hw, plan, root, tag, out_, release=None, only_backend=None):
                return only_backend or plan["backend_candidates"][0], server, []
            with mock.patch.object(install, "choose_engine", side_effect=fake_choose):
                code = install.run(T.args(d, hub, port=gateway.getsockname()[1]), T.HW["rx9070xt_16g_32g"], cat, out)
        self.assertEqual(code, 1, out.text)
        self.assertIn("already in use", out.text)
        self.assertNotIn("Trying without the MTP", out.text)
        self.assertNotIn("may not run", out.text)


# ------------------------------------------------------------------------------------------------ M10, L2
def make_archive(path, members):
    with tarfile.open(path, "w:gz") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            t.addfile(info, io.BytesIO(data))
    return path


class Extraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tmpdir(self)
        self.archive = make_archive(self.tmp / "b1.tar.gz", {
            "build/llama-server": b"#!/bin/sh\nexit 0\n", "build/libggml.so": b"lib"})
        self.release = {"tag": "b1"}
        self.asset = {"name": "llama-b1-bin-ubuntu-x64.tar.gz", "url": "https://x/y", "digest": "sha256:" + "0" * 64}
        self.dest = self.tmp / "engines"

    def fake_download(self, *a, **kw):
        out = Path(a[1])
        shutil.copy(self.archive, out)
        return out

    def test_an_interrupted_extract_is_not_mistaken_for_an_installed_build(self):
        real_extract = engine.extract

        def dying_extract(archive, dest):
            Path(dest).mkdir(parents=True, exist_ok=True)
            (Path(dest) / "build").mkdir()
            (Path(dest) / "build/llama-server").write_text("#!/bin/sh\nexit 0\n")     # the binary arrived, the libraries did not
            raise OSError("No space left on device")
        with mock.patch.object(engine, "download", side_effect=self.fake_download):
            with mock.patch.object(engine, "extract", dying_extract), self.assertRaises(OSError):
                engine.install_build(self.release, self.asset, self.dest, label="cpu")
            with mock.patch.object(engine, "extract", real_extract):
                server = engine.install_build(self.release, self.asset, self.dest, label="cpu")
        self.assertTrue((server.parent / "libggml.so").exists(), "the second run reused the half-extracted folder")

    def test_a_complete_build_is_reused_without_downloading(self):
        with mock.patch.object(engine, "download", side_effect=self.fake_download) as dl:
            engine.install_build(self.release, self.asset, self.dest, label="cpu")
            engine.install_build(self.release, self.asset, self.dest, label="cpu")
        self.assertEqual(dl.call_count, 1)

    def test_a_folder_from_before_the_marker_is_kept_only_if_the_server_really_starts(self):
        root = self.dest / "b1-cpu"
        write_exe(root / "llama-server", "#!/bin/sh\nexit 0\n")                  # legacy, works
        with mock.patch.object(engine, "download", side_effect=AssertionError("must not download")):
            self.assertEqual(engine.install_build(self.release, self.asset, self.dest, label="cpu"), root / "llama-server")
        shutil.rmtree(root)
        write_exe(root / "llama-server", "#!/bin/sh\necho 'error while loading shared libraries: libggml.so' >&2\nexit 127\n")
        with mock.patch.object(engine, "download", side_effect=self.fake_download) as dl:
            server = engine.install_build(self.release, self.asset, self.dest, label="cpu")
        self.assertEqual(dl.call_count, 1)
        self.assertTrue((server.parent / "libggml.so").exists())

    def test_an_archive_with_a_hard_link_out_of_the_folder_is_refused(self):
        evil = self.tmp / "evil.tar.gz"
        with tarfile.open(evil, "w:gz") as t:
            link = tarfile.TarInfo("d/link")
            link.type, link.linkname = tarfile.LNKTYPE, "../outside"       # hard link targets are relative to the archive root
            t.addfile(link)
        with self.assertRaises(engine.EngineError):
            engine.extract(evil, self.tmp / "out")


class UnverifiedDownloads(unittest.TestCase):
    def test_a_download_without_a_checksum_is_refused_before_any_request(self):
        d = tmpdir(self)
        with mock.patch.object(engine, "_open", side_effect=AssertionError("no request may be made")):
            with self.assertRaises(engine.EngineError) as cm:
                engine.download("https://x/y", d / "f", None)
        self.assertIn(engine.ALLOW_UNVERIFIED_ENV, str(cm.exception))

    def test_install_build_without_any_digest_is_refused(self):
        d = tmpdir(self)
        asset = {"name": "llama-b1-bin-ubuntu-x64.tar.gz", "url": "https://x/y", "digest": None}
        with mock.patch.object(engine, "_open", side_effect=AssertionError("no request may be made")):
            with self.assertRaises(engine.EngineError):
                engine.install_build({"tag": "b1"}, asset, d, label="cpu")

    def test_the_override_is_explicit(self):
        d = tmpdir(self)

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        with mock.patch.object(engine, "_open", return_value=Resp(b"data")), \
                mock.patch.dict(os.environ, {engine.ALLOW_UNVERIFIED_ENV: "1"}):
            self.assertEqual(engine.download("https://x/y", d / "f", None).read_bytes(), b"data")


# ------------------------------------------------------------------------------------------------ I3
class ReportScrubbing(unittest.TestCase):
    def test_a_shura_home_outside_the_home_folder_does_not_leak_its_path(self):
        text = "cannot write /mnt/data/exampleuser-private/shura/engines/x"
        out = report.scrub(text, home="/home/exampleuser", user="exampleuser", host="box", shura_home="/mnt/data/exampleuser-private/shura")
        self.assertNotIn("exampleuser", out)
        self.assertIn("<shura-home>", out)


# ------------------------------------------------------------------------------------------------ M8, rollback
class RollbackRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rb = load_module("rollback_audit", ROOT / "scripts/rollback_gateway.py")

    def setUp(self):
        self.root = tmpdir(self)
        self.file = self.root / "guardian.json"
        self.original = json.dumps({"vramGuard": True, "gatewayHost": "127.0.0.1", "gatewayPort": 8081}).encode()
        self.file.write_bytes(self.original)

    def test_a_recovery_that_fails_still_switches_the_ai_back_on_and_names_the_first_error(self):
        calls = []

        def fake_call(url, path, body=None):
            calls.append(path)
            return {"previous_override": "AUTO"} if path == "/guardian/prepare-rollback" else {}
        with mock.patch.object(self.rb, "call", side_effect=fake_call), \
                mock.patch.object(self.rb, "restart", side_effect=[RuntimeError("first failure"), RuntimeError("second failure")]), \
                mock.patch.object(self.rb, "wait_gateway", return_value={}):
            with self.assertRaises(RuntimeError) as cm:
                self.rb.rollback(self.root)
        self.assertIn("/guardian/ai-on", calls, "the AI was left switched OFF")
        self.assertIn("first failure", str(cm.exception))
        self.assertIn("shura on", str(cm.exception))
        self.assertEqual(self.file.read_bytes(), self.original)

    def test_a_gateway_that_is_slow_to_return_is_given_more_than_twenty_seconds(self):
        import inspect
        self.assertGreaterEqual(inspect.signature(self.rb.wait_gateway).parameters["timeout"].default, 60)

    def test_a_missing_config_is_a_clear_error_not_a_traceback(self):
        with self.assertRaisesRegex(RuntimeError, "Config not found"):
            self.rb.rollback(self.root / "nope")
        err = io.StringIO()
        with mock.patch.object(sys, "argv", ["rollback_gateway.py", "--config-dir", str(self.root / "nope")]), \
                redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            self.rb.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("rollback_gateway:", err.getvalue())

    def test_a_config_without_the_gateway_address_is_a_clear_error(self):
        self.file.write_text('{"vramGuard": true}')
        with self.assertRaisesRegex(RuntimeError, "gatewayHost or gatewayPort"):
            self.rb.rollback(self.root)

    def test_switching_back_on_failing_after_a_good_rollback_is_reported(self):
        def fake_call(url, path, body=None):
            if path == "/guardian/ai-on":
                raise urllib.error.URLError("refused")
            return {"previous_override": "AUTO"}
        with mock.patch.object(self.rb, "call", side_effect=fake_call), mock.patch.object(self.rb, "restart"), \
                mock.patch.object(self.rb, "wait_gateway", return_value={}), self.assertRaisesRegex(RuntimeError, "shura on"):
            self.rb.rollback(self.root)


# ------------------------------------------------------------------------------------------------ L3, graft
class GraftGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = load_module("graft_audit", ROOT / "scripts/graft_mtp_head.py")

    def test_output_equal_to_an_input_is_refused_before_anything_is_opened(self):
        d = tmpdir(self)
        body = d / "body.gguf"
        body.write_bytes(b"GGUF" + b"0" * 1000)
        head = d / "head.gguf"
        head.write_bytes(b"GGUF")
        (d / "alias.gguf").symlink_to(body)
        for out in (body, d / "." / "body.gguf", d / "alias.gguf", head):
            with self.assertRaises(SystemExit) as cm:
                self.g.check_output(out, [body, head])
            self.assertIn("same file", str(cm.exception))
        self.assertEqual(body.read_bytes(), b"GGUF" + b"0" * 1000)

    def test_not_enough_free_space_is_reported_up_front(self):
        d = tmpdir(self)
        body = d / "body.gguf"
        body.write_bytes(b"0" * 4096)
        with mock.patch.object(self.g.shutil, "disk_usage", return_value=SimpleNamespace(total=10, used=9, free=1024)):
            with self.assertRaises(SystemExit) as cm:
                self.g.check_output(d / "out.gguf", [body])
        self.assertIn("free space", str(cm.exception))

    def test_a_killed_run_leaves_no_truncated_output(self):
        d = tmpdir(self)
        out = d / "out.gguf"
        out.write_bytes(b"old complete model")
        with self.assertRaises(KeyboardInterrupt):
            with self.g.atomic_output(out) as tmp:
                tmp.write_bytes(b"half")
                raise KeyboardInterrupt
        self.assertEqual(out.read_bytes(), b"old complete model")
        self.assertEqual([f.name for f in d.iterdir()], ["out.gguf"])
        with self.g.atomic_output(out) as tmp:
            tmp.write_bytes(b"new complete model")
        self.assertEqual(out.read_bytes(), b"new complete model")
        self.assertEqual([f.name for f in d.iterdir()], ["out.gguf"])

    def test_a_complete_head_is_twenty_tensors(self):
        names = self.g.expected_head_names(40)
        self.assertEqual(len(names), 20)
        self.assertIn("blk.40.nextn.eh_proj.weight", names)
        self.assertIn("blk.40.ffn_up_exps.weight", names)

    def test_gguf_py_is_searched_in_the_places_the_docs_name_and_the_variable_wins(self):
        c = self.g.gguf_py_candidates({"GGUF_PY": "/x/gguf-py", "SHURA_HOME": "/s"})
        self.assertEqual(c[0], Path("/x/gguf-py"))
        self.assertIn(Path("/s/llama.cpp-perf/gguf-py"), c)
        self.assertIn(self.g.REPO.parent / "llama.cpp-perf" / "gguf-py", c)


# ------------------------------------------------------------------------------------------------ L4, L5, L7, L9: shura_setup
class SetupHelpers(unittest.TestCase):
    def fake_place(self, *a, **kw):
        return {"cache_ram_mb": 2048, "vram_llama": 9 * S.GB, "ram_llama": 14 * S.GB, "vision_ncmoe": 30, "vision_slots": 8}

    def test_rerunning_the_setup_keeps_the_users_settings(self):
        repo = tmpdir(self)
        cfg = repo / "config"
        cfg.mkdir()
        (cfg / "guardian.example.json").write_text(json.dumps({"vramGuard": False, "idleUnloadSeconds": 900, "newKey": 1,
                                                                "ramFreeMinGBToLoad": 4.0}))
        (cfg / "guardian.json").write_text(json.dumps({"vramGuard": True, "idleUnloadSeconds": 120, "gameMode": {"enabled": True},
                                                       "ramFreeMinGBToLoad": 99}))
        (cfg / "model-launch.json").write_text(json.dumps({
            "exePath": "/old/llama-server", "defaultModel": "occamy",
            "models": {"tiel-coder": {"modelPath": "/old.gguf", "ctxSize": 65536, "temperature": 0.2, "nCpuMoe": 5},
                       "occamy": {"modelPath": "/occamy.gguf", "ctxSize": 100000}}}))
        args = SimpleNamespace(llama="/new/llama-server", model="/new.gguf", mmproj="/new-mmproj.gguf")
        best = {"ncmoe": 22, "slots": 40}
        with mock.patch.object(S, "REPO_DIR", repo), mock.patch.object(S, "place", self.fake_place), redirect_stdout(io.StringIO()) as out:
            S.write_config(args, {}, {"gpus": [{}], "ram_total": 32 * S.GB}, best, "turbo3", 6, 6 * S.GB)
        guardian = json.loads((cfg / "guardian.json").read_text())
        self.assertTrue(guardian["vramGuard"])
        self.assertEqual(guardian["idleUnloadSeconds"], 120)
        self.assertEqual(guardian["gameMode"], {"enabled": True})
        self.assertEqual(guardian["newKey"], 1)                       # keys added to the example later still arrive
        self.assertEqual(guardian["ramFreeMinGBToLoad"], 15.0)        # measured by this run
        model = json.loads((cfg / "model-launch.json").read_text())
        self.assertEqual(model["defaultModel"], "occamy")
        self.assertIn("occamy", model["models"])                       # a second model is not thrown away
        tiel = model["models"]["tiel-coder"]
        self.assertEqual((tiel["ctxSize"], tiel["temperature"]), (65536, 0.2))
        self.assertEqual((tiel["modelPath"], tiel["nCpuMoe"], tiel["moeCacheSlots"]), ("/new.gguf", 22, 40))
        self.assertEqual(model["exePath"], "/new/llama-server")
        self.assertEqual(len(list(cfg.glob("*.json.bak-*"))), 2)
        self.assertIn("your settings are kept", out.getvalue())

    def test_the_autotune_kv_slot_is_deleted_even_when_the_tuning_fails(self):
        cache = tmpdir(self)
        work = cache / "autotune"
        work.mkdir()
        (work / "tune.bin").write_bytes(b"0" * 1000)
        (work / "cand0.log").write_text("log")
        with mock.patch.object(S, "CACHE_DIR", cache), mock.patch.object(S, "_autotune", side_effect=SystemExit("no configuration")):
            with self.assertRaises(SystemExit):
                S.autotune(SimpleNamespace())
        self.assertFalse((work / "tune.bin").exists())
        self.assertTrue((work / "cand0.log").exists())

    def test_a_gpu_name_with_a_comma_is_read_whole(self):
        out = "NVIDIA GeForce RTX 4070, SUPER Edition, 12282, 700, 8.9, 575.51\n"
        with mock.patch.object(S.subprocess, "run", return_value=SimpleNamespace(stdout=out)):
            hw = S.detect()
        self.assertEqual(hw["gpus"][0]["name"], "NVIDIA GeForce RTX 4070, SUPER Edition")
        self.assertEqual(hw["gpus"][0]["cuda_arch"], "89")

    def test_a_garbled_nvidia_smi_line_is_skipped_not_fatal(self):
        with mock.patch.object(S.subprocess, "run", return_value=SimpleNamespace(stdout="garbage\n")), \
                redirect_stderr(io.StringIO()) as err:
            hw = S.detect()
        self.assertEqual(hw["gpus"], [])
        self.assertIn("could not read", err.getvalue())

    def test_several_gpus_are_announced(self):
        out = "A, 12000, 1, 8.9, 1\nB, 24000, 1, 8.9, 1\n"
        with mock.patch.object(S.subprocess, "run", return_value=SimpleNamespace(stdout=out)), redirect_stderr(io.StringIO()) as err:
            hw = S.detect()
        self.assertEqual(len(hw["gpus"]), 2)
        self.assertIn("first one (A)", err.getvalue())

    def test_a_server_that_ignores_range_does_not_fill_the_ram_and_the_cache_is_keyed_by_size(self):
        cache = tmpdir(self)
        want = 16 * S.MB
        big = b"G" * (want + 8 * S.MB)

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)                      # ignores Range, sends everything
                self.send_header("Content-Length", str(len(big)))
                self.end_headers()
                try:
                    self.wfile.write(big)
                except OSError:
                    pass

        class Quick(socketserver.ThreadingMixIn, HTTPServer):
            daemon_threads = True

            def server_bind(self):
                socketserver.TCPServer.server_bind(self)
                self.server_name, self.server_port = "127.0.0.1", self.server_address[1]
        srv = Quick(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        files = {S.QUANT_FILE.format("Q4"): (111, "")}
        with mock.patch.object(S, "CACHE_DIR", cache), mock.patch.object(S, "HF_BASE", f"http://127.0.0.1:{srv.server_port}"), \
                mock.patch.object(S, "parse_gguf_header", return_value={"ok": 1}):
            S.remote_layout("Q4", files)
            cached = list(cache.glob("*.header"))
            self.assertEqual(len(cached), 1)
            self.assertEqual(cached[0].stat().st_size, want)
            self.assertIn(".111.", cached[0].name)
            files[S.QUANT_FILE.format("Q4")] = (222, "")             # re-uploaded: the old header must not be reused
            S.remote_layout("Q4", files)
            names = [f.name for f in cache.glob("*.header")]
            self.assertEqual(len(names), 1)
            self.assertIn(".222.", names[0])


# ------------------------------------------------------------------------------------------------ L9: monitor window
class MonitorWindowHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "installer/universal"))
        sys.path.insert(0, str(ROOT / "tests"))
        import monitor_window
        import test_monitor_window
        cls.app = monitor_window
        cls.kwin_run = staticmethod(lambda calls: test_monitor_window.MonitorIntegration._kwin_run(None, calls))

    def env(self, temp):
        return mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": temp, "XDG_CURRENT_DESKTOP": "KDE", "XDG_SESSION_TYPE": "x11"}, clear=False)

    def which(self):
        return mock.patch.object(self.app.shutil, "which",
                                 side_effect=lambda n: "/tmp/" + n if n in {"kwriteconfig6", "busctl"} else None)

    def test_backups_do_not_pile_up_forever(self):
        temp = tmpdir(self)
        calls = []
        with self.env(str(temp)), self.which(), mock.patch.object(self.app.subprocess, "run", side_effect=self.kwin_run(calls)), \
                mock.patch.object(self.app, "_screen_position", return_value="0,0"), redirect_stdout(io.StringIO()):
            for _ in range(9):
                self.assertEqual(self.app.install_rule(), 0)
        self.assertLessEqual(len(list(temp.glob("kwinrulesrc.bak-*"))), self.app.KEEP_BACKUPS)

    def test_a_failing_kwriteconfig_restores_the_file_and_reports_it(self):
        temp = tmpdir(self)
        config = temp / "kwinrulesrc"
        config.write_text("[General]\ncount=1\nrules=pip\n\n[pip]\nDescription=Picture in Picture\n", encoding="utf-8")
        before = config.read_text()
        calls = []
        good = self.kwin_run(calls)

        def run(command, **kw):
            if Path(command[0]).name == "kwriteconfig6" and sum(1 for c in calls if Path(c[0]).name == "kwriteconfig6") >= 3:
                raise subprocess.CalledProcessError(1, command, stderr="kwriteconfig6: disk full")
            return good(command, **kw)
        err = io.StringIO()
        with self.env(str(temp)), self.which(), mock.patch.object(self.app.subprocess, "run", side_effect=run), \
                mock.patch.object(self.app, "_screen_position", return_value="0,0"), redirect_stderr(err):
            self.assertEqual(self.app.install_rule(), 1)
        self.assertEqual(config.read_text(), before)
        self.assertIn("disk full", err.getvalue())

    def test_the_autostart_entry_does_not_depend_on_the_session_path(self):
        temp = tmpdir(self)
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(temp)}, clear=False), \
                mock.patch.object(self.app.shutil, "which", return_value=None), redirect_stdout(io.StringIO()):
            self.app.autostart("on")
        line = next(l for l in (temp / "autostart/shura-monitor.desktop").read_text().splitlines() if l.startswith("Exec="))
        self.assertTrue(line.startswith("Exec=/") or line.startswith('Exec="/'), line)
        self.assertTrue(line.endswith("scripts/shura monitor") or line.endswith('scripts/shura" monitor'), line)


if __name__ == "__main__":
    unittest.main()
