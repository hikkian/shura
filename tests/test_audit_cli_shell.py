"""Shell-side regression tests for the CLI audit: scripts/shura against a fake gateway, scripts/install.sh and uninstall.sh in a
throw-away HOME with stub systemctl/docker/git, capture-moe-trace.sh with a fake tracer, build-llama.sh with stub git/cmake,
setup.sh's gateway restore. Nothing touches the real HOME, services, GPU or network; temp files live in /dev/shm."""
import json
import os
import re
import shutil
import socketserver
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_audit_cli import free_port, tmpdir, write_exe  # noqa: E402

BASH = shutil.which("bash")


class Quick(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True

    def server_bind(self):                      # skip the reverse lookup that stalls HTTPServer on some runners
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


class FakeGateway:
    """routes: {(METHOD, path): (status, body, delay_seconds)}; anything else is a 404."""

    def __init__(self, routes):
        gateway = self
        self.routes, self.requests = routes, []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _serve(self):
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(n)
                gateway.requests.append((self.command, self.path))
                status, body, delay = gateway.routes.get((self.command, self.path), (404, b"not found", 0))
                time.sleep(delay)
                data = body if isinstance(body, bytes) else body.encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass
            do_GET = do_POST = _serve

        self.srv = Quick(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()


class Sandbox:
    """A throw-away HOME plus a folder of stub commands that comes first in PATH and logs every call."""

    def __init__(self, case):
        self.root = tmpdir(case)
        self.home = self.root / "home"
        self.home.mkdir()
        self.bin = self.root / "stubs"
        self.log = self.root / "calls.log"
        self.log.write_text("")
        for name in ("notify-send", "update-desktop-database", "gtk-update-icon-cache"):
            write_exe(self.bin / name, "#!/bin/sh\nexit 0\n")
        write_exe(self.bin / "systemctl", '''#!/bin/sh
echo "systemctl $*" >> "$STUB_LOG"
case "$*" in
  *is-active*) exit "${STUB_ACTIVE:-1}" ;;
  *" cat "*) exit "${STUB_UNIT_EXISTS:-0}" ;;
esac
exit 0
''')

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if k not in ("SHURA_HOME", "XDG_DATA_HOME", "SHURA_GATEWAY", "SHURA_DOCKER",
                                                                 "SHURACODE_DIR", "SHURACODE_REF", "SHURA_SEARXNG_IMAGE")}
        env.update(HOME=str(self.home), PATH=f"{self.bin}:{os.environ['PATH']}", STUB_LOG=str(self.log))
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def calls(self):
        return self.log.read_text().splitlines()

    def run(self, cmd, timeout=60, **extra):
        return subprocess.run(cmd, env=self.env(**extra), capture_output=True, text=True, timeout=timeout)


# ------------------------------------------------------------------------------------------------ M1-M3, L6, L7: scripts/shura
class ShuraCommand(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox(self)

    def shura(self, gateway_url, *args, timeout=40, **extra):
        return self.sb.run([BASH, str(ROOT / "scripts/shura"), *args], timeout=timeout, SHURA_GATEWAY=gateway_url, **extra)

    def dead_url(self):
        return f"http://127.0.0.1:{free_port()}"

    # M1
    def test_off_succeeds_only_when_the_gateway_says_so(self):
        with FakeGateway({("POST", "/guardian/ai-off"): (200, '{"ok": true}', 0)}) as gw:
            r = self.shura(gw.url, "off")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("AI disabled", r.stdout)

    def test_off_does_not_claim_success_on_an_http_error(self):
        with FakeGateway({}) as gw:                       # something else on the port: 404
            r = self.shura(gw.url, "off")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("AI disabled", r.stdout)
        self.assertIn("NOT disabled", r.stderr)
        self.assertIn("404", r.stderr)

    def test_off_and_on_and_unload_fail_loudly_when_the_gateway_is_down(self):
        for command, wording in (("off", "NOT disabled"), ("on", "NOT enabled"), ("unload", "not unloaded")):
            r = self.shura(self.dead_url(), command)
            self.assertNotEqual(r.returncode, 0, command)
            self.assertIn("not running", r.stderr, command)
            self.assertIn(wording, r.stderr, command)

    def test_a_gateway_that_never_answers_cannot_hang_the_command(self):
        with FakeGateway({("POST", "/guardian/ai-off"): (200, "{}", 30)}) as gw:
            t0 = time.monotonic()
            r = self.shura(gw.url, "off", timeout=25, SHURA_HTTP_TIMEOUT=1)
        self.assertLess(time.monotonic() - t0, 15)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no answer", r.stderr)

    def test_on_and_unload_report_a_5xx(self):
        with FakeGateway({("POST", "/guardian/ai-on"): (500, "boom", 0), ("POST", "/guardian/unload"): (503, "busy", 0)}) as gw:
            on = self.shura(gw.url, "on")
            unload = self.shura(gw.url, "unload")
        self.assertNotEqual(on.returncode, 0)
        self.assertNotEqual(unload.returncode, 0)
        self.assertNotIn("AI enabled", on.stdout)
        self.assertNotIn("Model unloaded", unload.stdout)

    def test_off_warns_when_the_model_is_still_loaded_afterwards(self):
        routes = {("POST", "/guardian/ai-off"): (200, '{"ok": true}', 0), ("GET", "/guardian/status"): (200, '{"status": "READY"}', 0)}
        with FakeGateway(routes) as gw:
            r = self.shura(gw.url, "off", timeout=60, SHURA_OFF_WAIT=1)
        self.assertEqual(r.returncode, 0)
        self.assertIn("still reports READY", r.stderr)

    # M2
    def test_warm_does_not_report_a_load_when_the_gateway_refuses(self):
        routes = {("GET", "/guardian/status"): (200, "{}", 0),
                  ("POST", "/v1/chat/completions"): (503, '{"error": "AI is manually disabled (ai-off)"}', 0)}
        with FakeGateway(routes) as gw:
            r = self.shura(gw.url, "warm")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("Model loaded", r.stdout)
        self.assertIn("manually disabled", r.stderr)

    def test_warm_reports_the_load_when_it_worked(self):
        routes = {("GET", "/guardian/status"): (200, "{}", 0), ("POST", "/v1/chat/completions"): (200, "{}", 0)}
        with FakeGateway(routes) as gw:
            r = self.shura(gw.url, "warm")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Model loaded", r.stdout)

    # M3
    def test_a_busy_gateway_is_not_started_again_or_declared_dead(self):
        routes = {("GET", "/guardian/status"): (200, "{}", 4), ("POST", "/v1/chat/completions"): (200, "{}", 0)}
        with FakeGateway(routes) as gw:
            r = self.shura(gw.url, "warm", SHURA_PROBE_TIMEOUT=1, SHURA_BUSY_WAIT=2)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse([c for c in self.sb.calls() if " start " in c or c.endswith(" start")], self.sb.calls())
        self.assertIn("busy", r.stderr)

    def test_status_of_a_busy_gateway_does_not_say_to_restart_it(self):
        with FakeGateway({("GET", "/guardian/status"): (200, "{}", 4)}) as gw:
            r = self.shura(gw.url, "status", SHURA_PROBE_TIMEOUT=1)
        self.assertIn("busy", r.stdout)
        self.assertNotIn("not running", r.stdout)
        self.assertNotEqual(r.returncode, 0)

    def test_status_of_a_dead_gateway_says_so(self):
        r = self.shura(self.dead_url(), "status")
        self.assertIn("not running", r.stdout)

    def test_a_dead_gateway_is_started_through_systemd(self):
        r = self.shura(self.dead_url(), "warm", SHURA_START_WAIT=1)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("did not start", r.stderr)
        self.assertIn("systemctl --user start ai-gateway.service", self.sb.calls())

    # L7
    def test_without_the_gateway_service_the_error_explains_the_universal_path(self):
        r = self.shura(self.dead_url(), "warm", STUB_UNIT_EXISTS=1)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("shura start", r.stderr)
        self.assertFalse([c for c in self.sb.calls() if " start " in c])

    def test_status_honours_xdg_data_home_like_the_python_side(self):
        data = self.sb.root / "xdg"
        (data / "shura").mkdir(parents=True)
        (data / "shura/state.json").write_text(json.dumps({"port": free_port(), "backend": "cpu",
                                                           "plan": {"settings": {"context": 4096}, "quant": "Q4"}}))
        r = self.shura(self.dead_url(), "status", XDG_DATA_HOME=data, STUB_UNIT_EXISTS=1)
        self.assertIn("stopped", r.stdout, r.stdout + r.stderr)

    # L6
    def test_help_shows_every_command(self):
        r = self.shura(self.dead_url(), "help")
        for word in ("shura start|stop", "shura uninstall", "shura install", "shura off | on"):
            self.assertIn(word, r.stdout)


# ------------------------------------------------------------------------------------------------ install.sh / uninstall.sh
def make_repo(sb, name="repo"):
    """A copy of just what install.sh reads, so that its config/*.json copies land in the throw-away tree."""
    repo = sb.root / name
    for rel in ("scripts", "systemd", "desktop", "searxng", "assets/icons"):
        shutil.copytree(ROOT / rel, repo / rel, ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "config").mkdir()
    for f in (ROOT / "config").glob("*.example.json"):
        shutil.copy(f, repo / "config" / f.name)
    return repo


class InstallSh(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox(self)
        self.repo = make_repo(self.sb)

    def install(self, *args, repo=None, **extra):
        repo = repo or self.repo
        return self.sb.run([BASH, str(repo / "scripts/install.sh"), "--no-shuracode", *args], **extra)

    def unit(self):
        return self.sb.home / ".config/systemd/user/ai-gateway.service"

    def add_docker_stub(self, name="docker"):
        write_exe(self.sb.bin / name, f'''#!/bin/sh
echo "{name} $*" >> "$STUB_LOG"
case "$1" in
  ps) [ "${{STUB_DOCKER_PS_RC:-0}}" = 0 ] || {{ echo "Cannot connect to the Docker daemon" >&2; exit "$STUB_DOCKER_PS_RC"; }}
      printf '%s' "${{STUB_DOCKER_NAMES:-}}"; exit 0 ;;
  run) exit "${{STUB_DOCKER_RUN_RC:-0}}" ;;
esac
exit 0
''')

    # M4
    def test_an_edited_unit_is_backed_up_before_it_is_replaced(self):
        self.unit().parent.mkdir(parents=True)
        self.unit().write_text("[Service]\n# my local edit\nEnvironment=FOO=bar\n")
        r = self.install("--no-searxng", "--no-desktop")
        self.assertEqual(r.returncode, 0, r.stderr)
        backups = list(self.unit().parent.glob("ai-gateway.service.bak-*"))
        self.assertEqual(len(backups), 1)
        self.assertIn("my local edit", backups[0].read_text())
        self.assertIn(str(self.repo), self.unit().read_text())

    def test_a_rerun_leaves_a_running_gateway_alone_unless_asked(self):
        self.install("--no-searxng", "--no-desktop", STUB_ACTIVE=1)                     # first install: starts it
        self.assertIn("systemctl --user start ai-gateway.service", self.sb.calls())
        self.sb.log.write_text("")
        r = self.install("--no-searxng", "--no-desktop", STUB_ACTIVE=0)                 # it is running now
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse([c for c in self.sb.calls() if "restart" in c or " start " in c], self.sb.calls())
        self.assertIn("left alone", r.stdout)
        self.sb.log.write_text("")
        self.install("--no-searxng", "--no-desktop", "--restart", STUB_ACTIVE=0)
        self.assertIn("systemctl --user restart ai-gateway.service", self.sb.calls())

    def test_a_regular_file_in_place_of_the_command_is_backed_up(self):
        link = self.sb.home / ".local/bin/shura"
        link.parent.mkdir(parents=True)
        link.write_text("#!/bin/sh\necho my own shura\n")
        r = self.install("--no-searxng", "--no-desktop", "--no-service")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(link.is_symlink())
        backups = list(link.parent.glob("shura.bak-*"))
        self.assertEqual(len(backups), 1)
        self.assertIn("my own shura", backups[0].read_text())

    # M5
    def test_the_searxng_secret_is_never_empty_even_when_openssl_fails(self):
        self.add_docker_stub()
        write_exe(self.sb.bin / "openssl", "#!/bin/sh\nexit 1\n")
        r = self.install("--no-service", "--no-desktop")
        self.assertEqual(r.returncode, 0, r.stderr)
        text = (self.sb.home / ".config/searxng/settings.yml").read_text()
        self.assertRegex(text, r'secret_key: "[0-9a-f]{64}"')
        self.assertNotIn("@SEARXNG_SECRET@", text)

    def test_a_settings_file_left_with_an_empty_secret_is_repaired(self):
        self.add_docker_stub()
        conf = self.sb.home / ".config/searxng/settings.yml"
        conf.parent.mkdir(parents=True)
        conf.write_text('server:\n  secret_key: ""\n')
        r = self.install("--no-service", "--no-desktop")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertRegex(conf.read_text(), r'secret_key: "[0-9a-f]{64}"')
        self.assertEqual(len(list(conf.parent.glob("settings.yml.bak-*"))), 1)

    # M6
    def test_an_unusable_docker_does_not_abort_the_rest_of_the_install(self):
        self.add_docker_stub()
        r = self.install("--no-service", STUB_DOCKER_PS_RC=1)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("WARNING", r.stderr)
        self.assertTrue((self.sb.home / ".local/bin/shura").is_symlink())
        self.assertTrue((self.sb.home / ".local/share/applications/io.github.hikkian.Shura.desktop").exists())
        self.assertFalse((self.sb.home / ".config/searxng/settings.yml").exists())
        self.assertIn("problems in optional parts", r.stdout)

    def test_a_failing_container_start_is_a_warning_not_a_crash(self):
        self.add_docker_stub()
        r = self.install("--no-service", STUB_DOCKER_RUN_RC=125)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("container failed", r.stderr)
        self.assertTrue((self.sb.home / ".local/bin/shura").is_symlink())

    def test_an_existing_container_is_left_alone(self):
        self.add_docker_stub()
        r = self.install("--no-service", "--no-desktop", STUB_DOCKER_NAMES="other\nsearxng\n")
        self.assertIn("already exists", r.stdout)
        self.assertFalse([c for c in self.sb.calls() if c.startswith("docker run")])

    def test_shura_docker_can_name_podman(self):
        self.add_docker_stub("podman")
        r = self.install("--no-service", "--no-desktop", SHURA_DOCKER="podman")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue([c for c in self.sb.calls() if c.startswith("podman run")], self.sb.calls())

    def test_the_searxng_image_can_be_pinned(self):
        self.add_docker_stub()
        self.install("--no-service", "--no-desktop", SHURA_SEARXNG_IMAGE="searxng/searxng@sha256:abc")
        run = [c for c in self.sb.calls() if c.startswith("docker run")][0]
        self.assertTrue(run.endswith("searxng/searxng@sha256:abc"), run)

    # L7
    def test_a_repository_path_with_spaces_and_sed_metacharacters_gives_a_valid_unit(self):
        repo = make_repo(self.sb, "my repo|a&b")
        r = self.install("--no-searxng", "--no-desktop", repo=repo)
        self.assertEqual(r.returncode, 0, r.stderr)
        text = self.unit().read_text()
        self.assertIn(f'ExecStart=/usr/bin/python3 "{repo}/gateway/ai_gateway.py"', text)
        self.assertIn(f'Environment="AI_GATEWAY_CONFIG={repo}/config"', text)

    def test_a_path_systemd_cannot_hold_is_refused_with_a_message(self):
        repo = make_repo(self.sb, "per%cent")
        r = self.install("--no-searxng", "--no-desktop", repo=repo)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cannot hold", r.stderr)

    def test_the_unit_finds_cuda_on_arch_too(self):
        template = (ROOT / "systemd/ai-gateway.service.in").read_text()
        self.assertIn("/opt/cuda/bin", template)
        self.assertIn("/opt/cuda/lib64", template)

    # L8
    def add_git_stub(self, clone_rc=0):
        write_exe(self.sb.bin / "git", f'''#!/bin/sh
echo "git $*" >> "$STUB_LOG"
if [ "$1" = clone ]; then
  [ {clone_rc} = 0 ] || exit {clone_rc}
  for last; do :; done
  mkdir -p "$last/.git"
  printf '#!/bin/sh\\necho shuracode installed\\n' > "$last/install.sh"; chmod +x "$last/install.sh"
fi
exit 0
''')

    def test_shuracode_can_be_pinned_to_a_ref_and_the_commit_is_shown(self):
        self.add_git_stub()
        r = self.sb.run([BASH, str(self.repo / "scripts/install.sh"), "--no-searxng", "--no-desktop", "--no-service"],
                        SHURACODE_REF="v1.2.3")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        calls = self.sb.calls()
        self.assertTrue([c for c in calls if c.startswith("git clone")])
        self.assertTrue([c for c in calls if "fetch -q --depth 1 origin v1.2.3" in c], calls)
        self.assertTrue([c for c in calls if "checkout -q FETCH_HEAD" in c], calls)
        self.assertIn("pinned: v1.2.3", r.stdout)

    def test_a_failed_shuracode_download_is_reported_and_the_rest_installed(self):
        self.add_git_stub(clone_rc=128)
        r = self.sb.run([BASH, str(self.repo / "scripts/install.sh"), "--no-searxng", "--no-service"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ShuraCode was not installed", r.stderr)
        self.assertTrue((self.sb.home / ".local/bin/shura").is_symlink())

    def test_the_header_states_what_is_downloaded_and_run(self):
        head = (ROOT / "scripts/install.sh").read_text()[:2500]
        self.assertIn("SHURACODE_REF", head)
        self.assertIn("SHURA_SEARXNG_IMAGE", head)


class UninstallSh(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox(self)
        self.repo = make_repo(self.sb)
        r = self.sb.run([BASH, str(self.repo / "scripts/install.sh"), "--no-shuracode", "--no-searxng"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.sb.log.write_text("")

    def uninstall(self, *args):
        return self.sb.run([BASH, str(self.repo / "scripts/uninstall.sh"), *args])

    def test_install_leaves_things_to_remove(self):
        self.assertTrue((self.sb.home / ".local/bin/shura").is_symlink())
        self.assertTrue((self.sb.home / ".config/systemd/user/ai-gateway.service").exists())

    def test_dry_run_removes_nothing(self):
        r = self.uninstall("--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("remove the command", r.stdout)
        self.assertTrue((self.sb.home / ".local/bin/shura").is_symlink())
        self.assertEqual(self.sb.calls(), [])

    def test_it_removes_what_install_made_and_nothing_else(self):
        models = self.sb.home / ".local/share/shura/models/big.gguf"
        models.parent.mkdir(parents=True)
        models.write_text("20 GB, really")
        r = self.uninstall("--yes")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse((self.sb.home / ".local/bin/shura").exists())
        self.assertFalse((self.sb.home / ".config/systemd/user/ai-gateway.service").exists())
        self.assertFalse((self.sb.home / ".local/share/applications/io.github.hikkian.Shura.desktop").exists())
        self.assertFalse((self.sb.home / ".local/share/icons/hicolor/48x48/apps/shura.png").exists())
        self.assertTrue(models.exists())
        self.assertTrue((self.repo / "config/guardian.json").exists())
        self.assertIn("systemctl --user disable --now ai-gateway.service", self.sb.calls())

    def test_a_command_that_points_elsewhere_is_left_alone(self):
        link = self.sb.home / ".local/bin/shura"
        link.unlink()
        link.symlink_to("/usr/bin/true")
        r = self.uninstall("--yes")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(link.is_symlink())

    def test_without_yes_and_without_a_terminal_it_refuses(self):
        r = self.uninstall()
        self.assertNotEqual(r.returncode, 0)
        self.assertTrue((self.sb.home / ".local/bin/shura").is_symlink())


# ------------------------------------------------------------------------------------------------ M7
class CaptureTrace(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox(self)
        self.llama = self.sb.root / "llama"
        write_exe(self.llama / "build/bin/llama-moe-trace", '''#!/bin/sh
while [ $# -gt 0 ]; do [ "$1" = -f ] && f="$2"; shift; done
case "$(basename "$f")" in "$TRACER_FAIL_ON") echo "CUDA error: out of memory" >&2; exit 9 ;; esac
printf -- '-1,0,1\\n0,3,4\\n1,5,6\\n' > "$MOE_TRACE_OUT"
''')
        self.model = self.sb.root / "m.gguf"
        self.model.write_text("x")
        self.prompts = self.sb.root / "prompts"
        self.prompts.mkdir()
        for n in ("a", "b"):
            (self.prompts / f"{n}.txt").write_text(n)
        self.out = self.sb.root / "trace.csv"
        self.out.write_text("PRECIOUS,ROWS\n")

    def capture(self, **extra):
        return self.sb.run([BASH, str(ROOT / "scripts/capture-moe-trace.sh"), str(self.model), str(self.prompts), str(self.out),
                            str(self.llama)], **extra)

    def test_a_missing_tracer_does_not_destroy_the_output_file(self):
        (self.llama / "build/bin/llama-moe-trace").unlink()
        r = self.capture()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("tracer not found", r.stderr)
        self.assertEqual(self.out.read_text(), "PRECIOUS,ROWS\n")

    def test_a_failing_prompt_leaves_the_output_alone_and_shows_why(self):
        r = self.capture(TRACER_FAIL_ON="b.txt")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("b.txt", r.stderr)
        self.assertIn("out of memory", r.stderr)
        self.assertIn("was not changed", r.stderr)
        self.assertEqual(self.out.read_text(), "PRECIOUS,ROWS\n")

    def test_success_writes_only_generated_token_rows(self):
        r = self.capture()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.out.read_text().splitlines(), ["0,3,4", "1,5,6", "0,3,4", "1,5,6"])
        self.assertIn("Wrote", r.stdout)

    def test_no_prompt_files_is_an_error_not_a_run_on_a_literal_glob(self):
        for f in self.prompts.glob("*.txt"):
            f.unlink()
        r = self.capture()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no *.txt", r.stderr)
        self.assertEqual(self.out.read_text(), "PRECIOUS,ROWS\n")

    def test_a_missing_output_folder_is_reported_at_once(self):
        self.out = self.sb.root / "nodir/trace.csv"
        r = self.capture()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not exist", r.stderr)


# ------------------------------------------------------------------------------------------------ L9: build-llama.sh
class BuildLlama(unittest.TestCase):
    def setUp(self):
        from test_attention_build import STUB
        self.sb = Sandbox(self)
        self.dest = self.sb.root / "fork"
        (self.dest / ".git").mkdir(parents=True)
        (self.dest / "build").mkdir()
        stub = self.sb.root / "stub.py"
        stub.write_text(STUB)
        boot = self.sb.root / "bash-env.sh"
        boot.write_text("\n".join(f'{n}() {{ command python3 "$SHURA_TEST_STUB" {n} "$@"; }}' for n in ("git", "cmake", "nvcc")) + "\n")
        self.extra = {"BASH_ENV": boot, "SHURA_TEST_STUB": stub, "SHURA_TEST_LOG": self.sb.root / "calls.jsonl",
                      "SHURA_TEST_STATE": self.sb.root}

    def build(self, *args, **extra):
        return self.sb.run([BASH, str(ROOT / "scripts/build-llama.sh"), *args, str(self.dest)], **{**self.extra, **extra})

    def calls(self):
        return [json.loads(l) for l in (self.sb.root / "calls.jsonl").read_text().splitlines()]

    def test_a_failed_first_configure_keeps_the_objects_already_built(self):
        (self.dest / "build/obj.o").write_text("hours of work")
        (self.dest / "build/CMakeCache.txt").write_text("stale")
        r = self.build(SHURA_TEST_FAIL_ONCE=1)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.dest / "build/obj.o").exists())
        self.assertFalse((self.dest / "build/CMakeCache.txt").exists())

    def test_an_attention_patch_left_by_an_earlier_run_is_removed_when_not_asked_for(self):
        (self.sb.root / "applied").touch()
        r = self.build()
        self.assertEqual(r.returncode, 0, r.stderr)
        reverses = [c for c in self.calls() if c[0] == "git" and "--reverse" in c and c[-1].endswith("attention-decode.patch")
                    and "--check" not in c]
        self.assertEqual(len(reverses), 1)
        self.assertIn("Removed attention-decode.patch", r.stdout)

    def test_nvcc_is_found_in_a_cuda_folder_outside_path(self):
        cuda = self.sb.root / "opt-cuda/bin"
        write_exe(cuda / "nvcc", "#!/bin/sh\necho nvcc\n")
        only = self.sb.root / "minimal-path"
        only.mkdir()
        for tool in ("dirname", "basename", "rm", "cat", "mkdir", "sed", "grep", "env"):
            real = shutil.which(tool)
            if real:
                (only / tool).symlink_to(real)
        boot = self.sb.root / "bash-env2.sh"
        boot.write_text("\n".join(f'{n}() {{ command python3 "$SHURA_TEST_STUB" {n} "$@"; }}' for n in ("git", "cmake")) + "\n")
        python = shutil.which("python3")
        (only / "python3").symlink_to(python)
        r = subprocess.run([BASH, str(ROOT / "scripts/build-llama.sh"), str(self.dest)], capture_output=True, text=True, timeout=60,
                           env={**self.sb.env(**{**self.extra, "BASH_ENV": boot, "SHURA_CUDA_BIN": cuda}), "PATH": str(only)})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_a_missing_nvcc_names_the_places_it_looked(self):
        only = self.sb.root / "minimal-path"
        only.mkdir()
        for tool in ("dirname", "basename"):
            (only / tool).symlink_to(shutil.which(tool))
        nowhere = self.sb.root / "no-cuda-here"
        r = subprocess.run([BASH, str(ROOT / "scripts/build-llama.sh"), str(self.dest)], capture_output=True, text=True, timeout=60,
                           env={"HOME": str(self.sb.home), "PATH": str(only), "SHURA_CUDA_BIN": str(nowhere)})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(str(nowhere), r.stdout)
        self.assertIn("SHURA_CUDA_BIN", r.stdout)


# ------------------------------------------------------------------------------------------------ L5: setup.sh
class SetupShGateway(unittest.TestCase):
    def preflight(self):
        sb = Sandbox(self)
        os_release = sb.root / "os-release"
        os_release.write_text('ID=fedora\nVERSION_ID=44\nPRETTY_NAME="Test Fedora 44"\n')
        meminfo = sb.root / "meminfo"
        meminfo.write_text("MemTotal:       33554432 kB\n")
        write_exe(sb.bin / "uname", '#!/bin/sh\ncase "$1" in -s) echo Linux ;; -m) echo x86_64 ;; *) exit 1 ;; esac\n')
        write_exe(sb.bin / "df", '#!/bin/sh\necho Available\necho "${STUB_DISK_GB:-40}G"\n')
        write_exe(sb.bin / "nvidia-smi", '#!/bin/sh\ncase "$*" in *query-gpu=name,memory.total*) echo "NVIDIA test GPU, 24576 MiB" ;; *) echo "Driver Version: 555.42 CUDA Version: 12.4" ;; esac\n')
        write_exe(sb.bin / "curl", '#!/bin/sh\necho "curl $*" >> "$STUB_LOG"\nif [ -n "${PREFLIGHT_FAIL_HOST:-}" ]; then case "$*" in *"$PREFLIGHT_FAIL_HOST"*) exit 7 ;; esac; fi\nexit 0\n')
        write_exe(sb.bin / "sudo", '#!/bin/sh\necho "sudo $*" >> "$STUB_LOG"\nexit 99\n')
        script = f'set -euo pipefail\nSHURA_SOURCE_ONLY=1 source "{ROOT / "setup.sh"}"\ndetect_family\npreflight_checks\n'
        return sb, os_release, meminfo, script

    def test_an_error_after_stopping_the_gateway_starts_it_again(self):
        sb = Sandbox(self)
        script = f'''set -euo pipefail
SHURA_SOURCE_ONLY=1 source "{ROOT}/setup.sh"
GATEWAY_STOPPED_BY_SETUP=1
die "auto-tune failed"
'''
        r = sb.run([BASH, "-c", script])
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("systemctl --user start ai-gateway", sb.calls())

    def test_nothing_is_started_when_setup_did_not_stop_it(self):
        sb = Sandbox(self)
        script = f'''set -euo pipefail
SHURA_SOURCE_ONLY=1 source "{ROOT}/setup.sh"
die "no GPU"
'''
        sb.run([BASH, "-c", script])
        self.assertEqual(sb.calls(), [])

    def test_read_only_preflight_checks_github_and_huggingface_before_sudo(self):
        sb, os_release, meminfo, script = self.preflight()
        r = sb.run([BASH, "-c", script], SHURA_OS_RELEASE=os_release, SHURA_MEMINFO=meminfo,
                   SHURA_HOME=sb.home / ".local/share/shura")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Network: GitHub and Hugging Face reachable", r.stdout)
        self.assertTrue(any("https://github.com/" in call for call in sb.calls()))
        self.assertTrue(any("https://huggingface.co/" in call for call in sb.calls()))
        self.assertFalse(any(call.startswith("sudo ") for call in sb.calls()))

    def test_unreachable_huggingface_refuses_before_any_sudo(self):
        sb, os_release, meminfo, script = self.preflight()
        r = sb.run([BASH, "-c", script], SHURA_OS_RELEASE=os_release, SHURA_MEMINFO=meminfo,
                   SHURA_HOME=sb.home / ".local/share/shura", PREFLIGHT_FAIL_HOST="huggingface.co")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cannot reach Hugging Face", r.stderr)
        self.assertFalse(any(call.startswith("sudo ") for call in sb.calls()))


if __name__ == "__main__":
    unittest.main()
