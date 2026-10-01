"""Install flow tests: launch flags, model download (resume, hash, disk), start/stop, and the whole `shura install`
against a local fake Hugging Face and a fake llama-server. Offline, tiny files, no GPU."""
import copy
import hashlib
import io
import json
import os
import socket
import socketserver
import stat
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
from universal import cli, install, launch, modelstore, planner, report  # noqa: E402

FAKE = ROOT / "tests/fixtures/fake_llama_server.py"
HW = {p.stem: json.loads(p.read_text()) for p in (ROOT / "tests/fixtures/hardware").glob("*.json")}
REAL = json.loads((ROOT / "catalog/models.json").read_text())
POSIX = sys.platform != "win32"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class QuickServer(HTTPServer):
    def server_bind(self):                      # skip the reverse lookup that stalls HTTPServer on some runners (macOS CI)
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


class FakeHub(BaseHTTPRequestHandler):
    """Serves a listing and one file with Range support, like Hugging Face."""
    files, ignore_range, drop_after = {}, False, None

    def log_message(self, *a):
        pass

    def do_GET(self):
        if "/api/models/" in self.path:
            data = json.dumps([{"path": n, "size": len(b), "lfs": {"oid": hashlib.sha256(b).hexdigest(), "size": len(b)}}
                               for n, b in self.files.items()]).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        body = self.files.get(self.path.rsplit("/", 1)[-1])
        if body is None:
            self.send_error(404)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and not self.ignore_range:
            start = int(rng.split("=")[1].split("-")[0])
        chunk = body[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        cut = self.drop_after
        self.wfile.write(chunk if cut is None else chunk[:cut])


class Hub:
    def __init__(self, files, **attrs):
        handler = type("H", (FakeHub,), {"files": files, **attrs})
        self.srv = QuickServer(("127.0.0.1", 0), handler)
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()


def sha(b):
    return hashlib.sha256(b).hexdigest()


class ServerArgs(unittest.TestCase):
    def plan(self, name, **kw):
        return planner.plan(HW[name], REAL, **kw)

    def test_cpu_server_plan_becomes_cpu_flags(self):
        p = self.plan("epyc7551x2_256g_cpu")
        a = launch.server_args(p, "/x/llama-server", "/m/model.gguf", port=9000)
        self.assertEqual(a[a.index("-ngl") + 1], "0")
        self.assertEqual(a[a.index("--numa") + 1], "distribute")
        self.assertEqual(a[a.index("-c") + 1], str(p["settings"]["context"]))
        self.assertNotIn("--n-cpu-moe", a)
        self.assertEqual(a[a.index("--port") + 1], "9000")
        self.assertEqual(a[a.index("--host") + 1], "127.0.0.1")          # never exposed to the network by default

    def test_mtp_flags_follow_the_plan(self):
        p = self.plan("rx9070xt_16g_32g")
        self.assertTrue(p["settings"]["mtp"])
        self.assertIn("draft-mtp", launch.server_args(p, "s", "m"))
        off = planner.plan(HW["rx9070xt_16g_32g"], REAL, config={"mtp_standard": False})
        self.assertNotIn("--spec-type", launch.server_args(off, "s", "m"))
        cpu = self.plan("epyc7551x2_256g_cpu")                      # no GPU: no draft model pass
        self.assertNotIn("--spec-type", launch.server_args(cpu, "s", "m"))

    def test_the_mtp_draft_is_paid_for_in_vram(self):
        with_mtp = self.plan("rx9070xt_16g_32g")
        without = planner.plan(HW["rx9070xt_16g_32g"], REAL, config={"mtp_standard": False})
        self.assertLessEqual(with_mtp["settings"]["context"], without["settings"]["context"])
        self.assertLessEqual(with_mtp["memory"]["vram_need"], with_mtp["memory"]["vram_budget"])

    def test_hybrid_moe_gets_n_cpu_moe_and_kv_type(self):
        p = self.plan("rx9070xt_16g_32g")
        a = launch.server_args(p, "s", "m")
        self.assertEqual(a[a.index("--n-cpu-moe") + 1], str(p["settings"]["n_cpu_moe"]))
        self.assertEqual(a[a.index("-ctk") + 1], p["settings"]["kv_type"])
        self.assertEqual(a[a.index("-ngl") + 1], "99")

    def test_the_fork_tier_is_not_launched_here(self):
        with self.assertRaises(launch.LaunchError):
            launch.server_args(self.plan("ref_rtx4070s_12g_32g"), "s", "m")

    def test_no_plan_is_refused(self):
        with self.assertRaises(launch.LaunchError):
            launch.server_args({"ok": False}, "s", "m")

    def test_home_dir_per_platform(self):
        self.assertEqual(launch.home_dir({"SHURA_HOME": "/custom"}, "linux"), Path("/custom"))
        self.assertEqual(launch.home_dir({"XDG_DATA_HOME": "/xdg"}, "linux"), Path("/xdg/shura"))
        self.assertEqual(launch.home_dir({"LOCALAPPDATA": "C:/L"}, "win32"), Path("C:/L/shura"))
        self.assertIn("Application Support", str(launch.home_dir({}, "darwin")))


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.body = os.urandom(3 * 1024 * 1024 + 123)

    def tearDown(self):
        self.tmp.cleanup()

    def test_download_verifies_and_is_skipped_when_intact(self):
        with Hub({"m.gguf": self.body}) as hub:
            dest = modelstore.fetch(f"{hub.base}/x/resolve/main/m.gguf", self.dir / "m.gguf", len(self.body), sha(self.body),
                                    allow_local=True)
            self.assertEqual(dest.read_bytes(), self.body)
            self.assertFalse((self.dir / "m.gguf.part").exists())
        # the server is gone: an intact file must not need it
        self.assertEqual(modelstore.fetch("https://invalid.invalid/m", dest, len(self.body), sha(self.body)), dest)

    def test_an_interrupted_download_resumes(self):
        with Hub({"m.gguf": self.body}, drop_after=1_000_000) as hub:
            url = f"{hub.base}/x/resolve/main/m.gguf"
            with self.assertRaises(modelstore.StoreError):
                modelstore.fetch(url, self.dir / "m.gguf", len(self.body), sha(self.body), allow_local=True)
            part = self.dir / "m.gguf.part"
            self.assertEqual(part.stat().st_size, 1_000_000)
            hub.srv.RequestHandlerClass.drop_after = None
            modelstore.fetch(url, self.dir / "m.gguf", len(self.body), sha(self.body), allow_local=True)
        self.assertEqual((self.dir / "m.gguf").read_bytes(), self.body)

    def test_a_server_that_ignores_range_restarts_cleanly(self):
        (self.dir / "m.gguf.part").write_bytes(b"garbage" * 10)
        with Hub({"m.gguf": self.body}, ignore_range=True) as hub:
            modelstore.fetch(f"{hub.base}/x/resolve/main/m.gguf", self.dir / "m.gguf", len(self.body), sha(self.body),
                             allow_local=True)
        self.assertEqual((self.dir / "m.gguf").read_bytes(), self.body)

    def test_a_corrupt_file_is_moved_aside_and_never_used(self):
        with Hub({"m.gguf": self.body}) as hub:
            with self.assertRaises(modelstore.StoreError) as cm:
                modelstore.fetch(f"{hub.base}/x/resolve/main/m.gguf", self.dir / "m.gguf", len(self.body), sha(b"other"),
                                 allow_local=True)
        self.assertIn("SHA-256", str(cm.exception))
        self.assertFalse((self.dir / "m.gguf").exists())
        self.assertTrue((self.dir / "m.gguf.corrupt").exists())

    def test_not_enough_disk_is_reported_before_downloading(self):
        with mock.patch("shutil.disk_usage", return_value=SimpleNamespace(total=10, used=5, free=1024 ** 3)):
            with self.assertRaises(modelstore.StoreError) as cm:
                modelstore.fetch("https://h/x", self.dir / "big.gguf", 20 * 1024 ** 3, None)
        self.assertIn("disk space", str(cm.exception))
        self.assertFalse((self.dir / "big.gguf.part").exists())

    def test_only_https_is_accepted(self):
        with self.assertRaises(modelstore.StoreError):
            modelstore.fetch("http://example.com/m.gguf", self.dir / "m.gguf", 10, None)

    def test_listing_reads_lfs_hashes(self):
        with Hub({"a.gguf": b"abc"}) as hub:
            self.assertEqual(modelstore.listing("x/y", allow_local=True, base=hub.base), {"a.gguf": (3, sha(b"abc"))})


def fake_server(tmp, *, oom_first=0, no_mtp=False, no_turbo=False, no_q5=False):
    """An executable that behaves like llama-server (POSIX). The first `oom_first` runs die with an out-of-memory message."""
    counter = Path(tmp) / "runs"
    script = Path(tmp) / "llama-server"
    script.write_text(f"""#!/bin/sh
case "$*" in *--list-devices*) exit 0;; esac            # not a server start: does not count as a run
{'case "$*" in *draft-mtp*) echo "error: unknown spec type draft-mtp" >&2; exit 2;; esac' if no_mtp else ''}
{'case "$*" in *turbo3*) echo "error: unsupported cache type turbo3" >&2; exit 2;; esac' if no_turbo else ''}
{'case "$*" in *q5_0*) echo "error: flash attention does not support q5_0 K/V" >&2; exit 2;; esac' if no_q5 else ''}
n=$(cat "{counter}" 2>/dev/null || echo 0); n=$((n+1)); echo $n > "{counter}"
if [ "$n" -le {oom_first} ]; then echo "ggml_cuda: cudaMalloc failed: out of memory" >&2; exit 3; fi
exec "{sys.executable}" "{FAKE}" "$@"
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


@unittest.skipUnless(POSIX, "uses a shell-script stand-in for llama-server")
class Launch(unittest.TestCase):
    def test_start_status_stop(self):
        with tempfile.TemporaryDirectory() as d:
            port = free_port()
            argv = [str(fake_server(d)), "--port", str(port)]
            pid = launch.start(d, argv, port, wait_s=30)
            self.assertTrue(launch.alive(pid) and launch.health(port))
            with self.assertRaises(launch.LaunchError):                 # a second start must not spawn a second server
                launch.start(d, argv, port, wait_s=5)
            self.assertTrue(launch.stop(d))
            self.assertFalse(launch.health(port))
            self.assertFalse(launch.stop(d))                            # stopping twice is harmless

    def test_a_crash_shows_the_log_tail(self):
        with tempfile.TemporaryDirectory() as d:
            port = free_port()
            with self.assertRaises(launch.LaunchError) as cm:
                launch.start(d, [str(fake_server(d, oom_first=9)), "--port", str(port)], port, wait_s=10)
            self.assertIn("out of memory", str(cm.exception))


class Out(install.Out):
    def __init__(self):
        self.lines = []

    def say(self, text=""):
        self.lines.append(text)

    def ask(self, question, default=True):
        return True

    @property
    def text(self):
        return "\n".join(self.lines)


def args(d, hub=None, **kw):
    base = dict(dir=str(d), yes=True, dry_run=False, quant=None, model_id=None, context=None, port=free_port(), wait=30,
                latest=False, tag="b0", hf_base=hub.base if hub else None)
    base.update(kw)
    return SimpleNamespace(**base)


def tiny_catalog(size):
    cat = copy.deepcopy(REAL)
    m = cat["models"][0]
    m["source"]["hf_repo"] = "x/y"
    return cat


@unittest.skipUnless(POSIX, "uses a shell-script stand-in for llama-server")
class EndToEnd(unittest.TestCase):
    HW = "rx9070xt_16g_32g"        # upstream tier: the fork tier is installed by setup.sh

    def run_install(self, d, hub, server, **kw):
        cat, out = tiny_catalog(0), Out()
        def fake_choose(hw, plan, root, tag, out_, release=None, only_backend=None):      # the backend asked for, like the real one
            return only_backend or plan["backend_candidates"][0], server, []
        with mock.patch.object(install, "choose_engine", side_effect=fake_choose):
            code = install.run(args(d, hub, **kw), HW[self.HW], cat, out)
        return code, out

    def model_files(self):
        name = "Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf"
        return {name: os.urandom(200_000)}

    def test_dry_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            out = Out()
            code = install.run(args(d, dry_run=True), HW[self.HW], REAL, out)
            self.assertEqual(code, 0)
            self.assertIn("Dry run", out.text)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_full_install_then_start_status_stop(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bin_dir, Hub(self.model_files()) as hub:
            code, out = self.run_install(d, hub, fake_server(bin_dir))
            self.assertEqual(code, 0, out.text)
            self.assertIn("Ready", out.text)
            state = launch.load_state(d)
            self.assertTrue(state["measured"]["verified"])
            self.assertTrue(Path(state["model_path"]).exists())
            self.assertTrue((Path(d) / "shura-hardware-report.md").exists())
            report_text = (Path(d) / "shura-hardware-report.md").read_text()
            self.assertNotIn(str(Path.home()), report_text)
            # `shura report` after an install carries the speed that was measured
            with redirect_stdout(io.StringIO()):
                cli.main(["report", "--dir", d, "--profile", str(ROOT / "tests/fixtures/hardware" / f"{self.HW}.json"),
                          "-o", str(Path(d) / "again.md")])
            self.assertIn("Measured on this machine", (Path(d) / "again.md").read_text())
            # the same files through the command line
            with redirect_stdout(io.StringIO()) as buf:
                self.assertEqual(cli.main(["start", "--dir", d, "--wait", "30"]), 0)
                self.assertEqual(cli.main(["status", "--dir", d]), 0)
                self.assertEqual(cli.main(["stop", "--dir", d]), 0)
                self.assertEqual(cli.main(["status", "--dir", d]), 1)
            self.assertIn("running", buf.getvalue())

    def test_out_of_memory_retries_with_more_room_and_ends_working(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bin_dir, Hub(self.model_files()) as hub:
            code, out = self.run_install(d, hub, fake_server(bin_dir, oom_first=1))
            self.assertEqual(code, 0, out.text)
            self.assertIn("out of memory", out.text)
            self.assertIn("Trying again", out.text)
            self.assertTrue(launch.load_state(d)["measured"]["verified"])

    def test_a_measured_speed_far_below_the_prediction_replans_without_downloading_again(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bin_dir, Hub(self.model_files()) as hub, \
                mock.patch.dict(os.environ, {"FAKE_TOK_S": "12"}):             # the plan expects ~50
            code, out = self.run_install(d, hub, fake_server(bin_dir))
            self.assertEqual(code, 0, out.text)
            self.assertIn("Planning again with the measured speed", out.text)
            state = launch.load_state(d)
            cal = state["measured"]["calibration"]
            self.assertTrue(cal["replanned"] and cal["adopted"])
            self.assertLess(cal["ratio"], 0.7)
            self.assertEqual(len(list((Path(d) / "models").glob("*.gguf"))), 1)         # the same file, no second download

    def test_every_build_is_measured_on_the_real_model_and_the_vendors_own_stack_is_favoured(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bin_dir, Hub(self.model_files()) as hub:
            code, out = self.run_install(d, hub, fake_server(bin_dir))              # all builds "measure" the same speed
            self.assertEqual(code, 0, out.text)
            m = launch.load_state(d)["measured"]
            self.assertEqual(set(m["engine_comparison"]), {"TurboQuant+", "llama.cpp (rocm)", "llama.cpp (vulkan)"})
            self.assertEqual(m["chosen"], "llama.cpp (rocm)")                       # AMD: ROCm wins a tie, Vulkan needs +10%
            self.assertIn("rocm is this GPU's own stack", out.text)

    def test_a_faster_non_native_build_wins_only_by_enough(self):
        speeds = {"turbo3": "70", "rocm": "50"}                                      # +40%: clearly worth leaving ROCm

        def run(tq_speed):
            with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bin_dir, Hub(self.model_files()) as hub:
                server = Path(bin_dir) / "llama-server"
                server.write_text(f"""#!/bin/sh
case "$*" in *--list-devices*) exit 0;; esac
case "$*" in *turbo3*) export FAKE_TOK_S={tq_speed};; *) export FAKE_TOK_S=50;; esac
exec "{sys.executable}" "{FAKE}" "$@"
""")
                server.chmod(server.stat().st_mode | stat.S_IEXEC)
                self.run_install(d, hub, server)
                return launch.load_state(d)["measured"]["chosen"]
        self.assertEqual(run(54), "llama.cpp (rocm)")                                # +8%: not enough to replace ROCm
        self.assertEqual(run(70), "TurboQuant+")                                     # +40%: enough
        self.assertEqual(speeds["turbo3"], "70")

    def test_a_turboquant_build_that_does_not_start_falls_back_to_standard_llama_cpp(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bin_dir, Hub(self.model_files()) as hub:
            code, out = self.run_install(d, hub, fake_server(bin_dir, no_turbo=True))
            self.assertEqual(code, 0, out.text)
            self.assertIn("did not work here", out.text)
            state = launch.load_state(d)
            self.assertEqual(state["measured"]["engine"], "llama.cpp")
            self.assertIn(state["argv"][state["argv"].index("-ctk") + 1], ("q8_0", "q4_0"))

    def test_a_build_that_cannot_run_the_q5_0_kv_cache_falls_back_to_q8_0_or_q4_0(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bin_dir, Hub(self.model_files()) as hub:
            cat, out = tiny_catalog(0), Out()

            def fake_choose(hw, plan, root, tag, out_, release=None, only_backend=None):
                return only_backend or plan["backend_candidates"][0], fake_server(bin_dir, no_q5=True, no_turbo=True), []
            with mock.patch.object(install, "choose_engine", side_effect=fake_choose):
                code = install.run(args(d, hub, no_turbo=True), HW[self.HW], cat, out)
            self.assertEqual(code, 0, out.text)
            argv = launch.load_state(d)["argv"]
            self.assertIn(argv[argv.index("-ctk") + 1], ("q8_0", "q4_0"))

    def test_no_turbo_never_touches_the_third_party_build(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bin_dir, Hub(self.model_files()) as hub:
            cat, out = tiny_catalog(0), Out()
            seen = []

            def fake_choose(hw, plan, root, tag, out_, release=None, only_backend=None):
                seen.append(plan.get("engine"))
                return "vulkan", fake_server(bin_dir), []
            with mock.patch.object(install, "choose_engine", side_effect=fake_choose):
                code = install.run(args(d, hub, no_turbo=True), HW[self.HW], cat, out)
            self.assertEqual(code, 0, out.text)
            self.assertEqual(set(seen), {"llama.cpp"})
            self.assertNotIn("TurboQuant+ build (third-party", out.text)

    def test_a_build_without_mtp_is_retried_without_it(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bin_dir, Hub(self.model_files()) as hub:
            code, out = self.run_install(d, hub, fake_server(bin_dir, no_mtp=True))
            self.assertEqual(code, 0, out.text)
            self.assertIn("Trying without the MTP draft", out.text)
            state = launch.load_state(d)
            self.assertNotIn("--spec-type", state["argv"])
            self.assertFalse(state["plan"]["settings"].get("mtp"))

    def test_mtp_is_used_when_the_build_runs_it(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bin_dir, Hub(self.model_files()) as hub:
            code, out = self.run_install(d, hub, fake_server(bin_dir))
            self.assertEqual(code, 0, out.text)
            argv = launch.load_state(d)["argv"]
            self.assertEqual(argv[argv.index("--spec-type") + 1], "draft-mtp")

    def test_a_prediction_close_to_the_measurement_keeps_the_first_plan(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bin_dir, Hub(self.model_files()) as hub:
            code, out = self.run_install(d, hub, fake_server(bin_dir))              # the fake reports 50, the plan ~51
            self.assertEqual(code, 0, out.text)
            self.assertFalse(launch.load_state(d)["measured"]["calibration"]["replanned"])
            self.assertNotIn("Planning again", out.text)

    def test_always_failing_leaves_a_report_and_the_downloads(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bin_dir, Hub(self.model_files()) as hub:
            code, out = self.run_install(d, hub, fake_server(bin_dir, oom_first=99))
            self.assertEqual(code, 1)
            self.assertIn("No configuration passed", out.text)
            self.assertTrue((Path(d) / "shura-hardware-report.md").exists())
            self.assertTrue(any((Path(d) / "models").glob("*.gguf")))          # kept: a rerun does not download again
            self.assertIsNone(launch.load_state(d))

    def test_a_download_problem_stops_with_advice_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bin_dir, Hub({}) as hub:
            code, out = self.run_install(d, hub, fake_server(bin_dir))
            self.assertEqual(code, 1)
            self.assertIn("not in the", out.text)
            self.assertIn("run the same command again", out.text)

    def test_the_fork_tier_points_to_setup_sh(self):
        with tempfile.TemporaryDirectory() as d:
            out = Out()
            code = install.run(args(d, dry_run=True), HW["ref_rtx4070s_12g_32g"], REAL, out)
            self.assertEqual(code, 0)
            self.assertIn("setup.sh", out.text)
            self.assertIn("IQ4_XS", out.text)


class ChooseEngine(unittest.TestCase):
    RELEASE = {"tag": "b1", "assets": [{"name": "llama-b1-bin-ubuntu-vulkan-x64.tar.gz", "url": "https://x/v", "digest": None},
                                       {"name": "llama-b1-bin-ubuntu-x64.tar.gz", "url": "https://x/c", "digest": None}]}

    def run_choose(self, results):
        hw, out = copy.deepcopy(HW["rx9070xt_16g_32g"]), Out()
        hw["gpus"][0]["backends"] = ["vulkan"]
        plan = planner.plan(hw, REAL)
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(install.engine, "smoke_model", return_value=Path("m.gguf")), \
                mock.patch.object(install.engine, "install_build", side_effect=lambda rel, a, root: Path(a["name"])), \
                mock.patch.object(install.engine, "selftest", side_effect=lambda server, model, backend: results[backend]):
            return install.choose_engine(hw, plan, Path(d), None, out, self.RELEASE), out

    def test_the_working_gpu_build_is_used(self):
        (backend, server, tested), _ = self.run_choose({"vulkan": {"backend": "vulkan", "ok": True, "tok_s": 100},
                                                        "cpu": {"backend": "cpu", "ok": True, "tok_s": 40}})
        self.assertEqual(backend, "vulkan")

    def test_a_broken_gpu_build_falls_back_to_cpu(self):
        (backend, _, _), out = self.run_choose({"vulkan": {"backend": "vulkan", "ok": False, "error": "no device"},
                                                "cpu": {"backend": "cpu", "ok": True, "tok_s": 40}})
        self.assertEqual(backend, "cpu")
        self.assertIn("does not work here", out.text)

    def test_nothing_working_is_an_error_with_a_reason(self):
        with self.assertRaises(install.engine.EngineError):
            self.run_choose({"vulkan": {"backend": "vulkan", "ok": False}, "cpu": {"backend": "cpu", "ok": False}})


class Reconcile(unittest.TestCase):
    def hw(self, total_gib=4, used_gib=0.5):
        h = copy.deepcopy(HW["rx9070xt_16g_32g"])
        h["gpus"][0].update(vram_total=int(total_gib * 1024 ** 3), vram_used=int(used_gib * 1024 ** 3))
        return h

    def devices(self, total_mib, free_mib):
        return [{"id": "Vulkan0", "name": "AMD Radeon RX 9070", "total": total_mib * 1024 ** 2, "free": free_mib * 1024 ** 2}]

    def test_a_wrong_placeholder_is_replaced_by_what_the_build_reports(self):
        out = Out()
        with mock.patch.object(install.engine, "list_devices", return_value=self.devices(16304, 15000)):
            fixed = install.reconcile(self.hw(4), "server", "vulkan", out)
        self.assertEqual(fixed["gpus"][0]["vram_total"], 16304 * 1024 ** 2)
        self.assertIn("planning with that", out.text)

    def test_a_matching_description_is_left_alone(self):
        hw = self.hw(16, 0.9)
        with mock.patch.object(install.engine, "list_devices", return_value=self.devices(16384, 15400)):
            self.assertIs(install.reconcile(hw, "server", "vulkan", Out()), hw)

    def test_cpu_backend_or_no_report_changes_nothing(self):
        hw = self.hw(4)
        self.assertIs(install.reconcile(hw, "server", "cpu", Out()), hw)
        with mock.patch.object(install.engine, "list_devices", return_value=[]):
            self.assertIs(install.reconcile(hw, "server", "vulkan", Out()), hw)


class TurboBuild(unittest.TestCase):
    RELEASE = {"tag": "tqp-v0.4.0", "assets": [
        {"name": "turboquant-plus-tqp-v0.4.0-linux-x64-vulkan.tar.gz", "url": "https://x/v", "digest": "sha256:" + "0" * 64},
        {"name": "turboquant-plus-tqp-v0.4.0-linux-x64-cpu.tar.gz", "url": "https://x/c", "digest": None}]}

    def plan(self):
        return planner.plan(HW["rx9070xt_16g_32g"], REAL, config={"kv_unavailable": ("q8_0", "q5_0", "q4_0")}, menu=False)

    def test_the_plan_names_the_engine_and_the_only_backend_it_has(self):
        p = self.plan()
        self.assertEqual((p["tier"], p["engine"], p["backend_candidates"], p["needs_probe"]),
                         ("tqp", "turboquant-plus", ["vulkan"], False))
        self.assertEqual(p["settings"]["kv_type"], "turbo3")
        self.assertGreater(p["settings"]["moe_cache_mib"], 0)

    def test_it_can_be_switched_off(self):
        p = planner.plan(HW["rx9070xt_16g_32g"], REAL, config={"tqp_enabled": False}, menu=False)
        self.assertNotEqual(p["tier"], "tqp")

    def test_the_archive_must_match_the_pinned_hash_not_the_release_page(self):
        seen = {}

        def fake_install(release, asset, root, label=None, sha256=None):
            seen["sha"], seen["label"] = sha256, label
            return Path("llama-server")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(install.engine, "fetch_release", return_value=self.RELEASE), \
                mock.patch.object(install.engine, "install_build", side_effect=fake_install), \
                mock.patch.object(install.engine, "smoke_model", return_value=Path("m")), \
                mock.patch.object(install.engine, "selftest", return_value={"ok": True, "tok_s": 9}):
            backend, _, _ = install.choose_engine(HW["rx9070xt_16g_32g"], self.plan(), Path(d), None, Out())
        self.assertEqual(backend, "vulkan")
        self.assertEqual(seen["sha"], install.engine.TQP_SHA256["turboquant-plus-tqp-v0.4.0-linux-x64-vulkan.tar.gz"])

    def test_an_unpinned_build_is_refused(self):
        release = {"tag": "tqp-v9", "assets": [{"name": "turboquant-plus-tqp-v9-linux-x64-vulkan.tar.gz", "url": "https://x", "digest": None}]}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(install.engine, "fetch_release", return_value=release):
            with self.assertRaises(install.engine.EngineError) as cm:
                install.choose_engine(HW["rx9070xt_16g_32g"], self.plan(), Path(d), None, Out())
        self.assertIn("pinned", str(cm.exception))

    def test_a_failing_selftest_raises_so_the_installer_can_fall_back(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(install.engine, "fetch_release", return_value=self.RELEASE), \
                mock.patch.object(install.engine, "install_build", return_value=Path("llama-server")), \
                mock.patch.object(install.engine, "smoke_model", return_value=Path("m")), \
                mock.patch.object(install.engine, "selftest", return_value={"ok": False, "error": "no device"}):
            with self.assertRaises(install.engine.EngineError):
                install.choose_engine(HW["rx9070xt_16g_32g"], self.plan(), Path(d), None, Out())

    def test_launch_flags_give_the_cache_exactly_the_planned_budget(self):
        p = self.plan()
        argv = launch.server_args(p, "s", "m")
        self.assertEqual(argv[argv.index("--moe-cache") + 1], str(p["settings"]["moe_cache_mib"]))
        self.assertEqual(argv[argv.index("-ctk") + 1], "turbo3")


class Relax(unittest.TestCase):
    def test_each_attempt_leaves_more_free_and_the_second_shrinks_the_window(self):
        hw = HW["rx9070xt_16g_32g"]
        first, second, third = (planner.plan(hw, REAL, config=install.relaxed_config({}, n, hw)) for n in range(3))
        self.assertGreaterEqual(second["settings"]["n_cpu_moe"], first["settings"]["n_cpu_moe"])
        self.assertLessEqual(third["settings"]["context"], 131072)

    def test_issue_link_is_a_github_url_and_bounded(self):
        url = report.issue_url("x" * 50000)
        self.assertTrue(url.startswith("https://github.com/hikkian/shura/issues/new?"))
        self.assertLess(len(url), 20000)


if __name__ == "__main__":
    unittest.main()
