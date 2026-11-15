"""INSTALL1 steps 9 and 10 (the part without a GPU): `install --dry-run` leaves no trace, nothing is downloaded without consent,
and a matrix of recorded machines ends in a clear outcome (a choice, a clear message, never a traceback).

No network (every way to open a connection is replaced by a counter that raises), no real server, no real model, no real home
folder: HOME, TMPDIR and SHURA_HOME point into a throw-away folder under /dev/shm/INSTALL1 (or TMPDIR if that does not exist).
The machines are recorded output of lscpu / lspci / nvidia-smi / vulkaninfo / sysfs from tests/fixtures/detect.

EXPECTATIONS (written before the first run; EXPECT below is the same list in code, the tests check it):
  a  RX 7800 XT 16 GiB      -> a plan; the ROCm (HIP) backend is among the candidates; the mode is chosen; no traceback
  b  laptop Intel iGPU + NVIDIA -> no traceback; a plan; the Intel GPU is recorded in the profile as the one that draws the screen
  c  CPU only                -> the CPU path, or a clear "no suitable video card" message; no traceback
  d  no free disk space      -> a clear "not enough disk space" error BEFORE any model download; no traceback
  e  no cmake / ninja / git  -> a clear line naming the missing tools and the commands to install them, without sudo, no traceback
  f  RTX 8 / 12 / 16 GiB     -> a plan with the CUDA or Vulkan backend, no traceback
  g  no cgroup v2            -> no traceback; the profile says cgroup_v2 is False; the dry run still ends with exit code 0
Run: python3 -B -m unittest tests.test_install_dryrun_matrix
Set SHURA_MATRIX_MD=/path/matrix.md to also write the outcome table.
"""
import contextlib
import copy
import hashlib
import http.client
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
sys.path.insert(0, str(ROOT / "tests"))
from universal import cli, detect as D, install, modelstore, planner  # noqa: E402
sys.path.insert(0, str(ROOT))
from tests._installer_env import FAKE_FREE  # noqa: E402
from test_detect_matrix import MatrixEnv, MATRIX  # noqa: E402

FX = ROOT / "tests/fixtures/detect"
GiB = 1024 ** 3
SCRATCH = Path("/dev/shm/INSTALL1") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
REAL_DETECT = D.detect

# machine -> (matrix folder for CPU/RAM/GPU memory, extra folder for the extended detection or None)
MACHINES = {
    "rx7800xt":      ("rx7800xt_16g_32g", "extra_amd_gfx1101"),
    "laptop_intel_nvidia": ("rtx4060_8g_32g", "extra_laptop_intel_nvidia"),
    "cpu_only":      ("cpu_i5_12400_32g", "extra_cpu_only"),
    "rtx_8g":        ("rtx4060_8g_32g", None),
    "rtx_12g":       ("rtx3060_12g_16g", None),
    "rtx_16g":       ("rtx4070tis_16g_64g", None),
}

# what was expected, written before the first run: machine row -> expectation text (the tests check each one)
EXPECT = {
    "a RX 7800 XT 16 GiB": "plan ok; hip/rocm among the backend candidates; mode chosen; no traceback",
    "b laptop Intel iGPU + NVIDIA": "no traceback; plan ok; Intel GPU recorded as the display GPU in the profile",
    "c CPU only": "CPU path or a clear 'no suitable video card' message; no traceback",
    "d no free disk": "clear 'not enough disk space' error before any model download; no traceback",
    "e no compiler": "clear line naming missing tools + install commands, no sudo, no traceback",
    "f RTX 8 GiB": "plan ok with cuda/vulkan; no traceback",
    "f RTX 12 GiB": "plan ok with cuda/vulkan; no traceback",
    "f RTX 16 GiB": "plan ok with cuda/vulkan; no traceback",
    "g no cgroup v2": "no traceback; profile cgroup_v2 False; dry run exit code 0",
}
OUTCOMES = {}


class CompositeEnv:
    """A matrix machine (CPU, RAM, GPU memory) with the extended-detection fixtures of an extra_* folder laid over it.
    `lspci -nn`, bare `nvidia-smi` and `sycl-ls` come from the extra folder; every other command from the matrix machine
    (or, for vulkaninfo, the extra folder when it has one)."""

    def __init__(self, matrix, extra=None, *, drop_files=()):
        self.base = MatrixEnv(MATRIX / matrix)
        self.extra_cmds, self.extra_files = {}, {}
        if extra:
            d = FX / extra
            self.extra_files = json.loads((d / "sysfs.json").read_text())
            if (d / "lspci_nn.txt").exists():
                self.extra_cmds[("lspci", "-nn")] = (d / "lspci_nn.txt").read_text()
            if (d / "nvidia_smi.txt").exists():
                self.extra_cmds[("nvidia-smi",)] = (d / "nvidia_smi.txt").read_text()
            if (d / "sycl_ls.txt").exists():
                self.extra_cmds[("sycl-ls",)] = (d / "sycl_ls.txt").read_text()
            if (d / "vulkaninfo_summary.txt").exists():
                self.extra_cmds[("vulkaninfo", "--summary")] = (d / "vulkaninfo_summary.txt").read_text()
            for name, path in (("rocm_version.txt", "/opt/rocm/.info/version"),):
                if (d / name).exists():
                    self.extra_files[path] = (d / name).read_text()
        self.extra = bool(extra)
        self.files = {**self.base.files, **self.extra_files}
        for p in drop_files:
            self.files.pop(p, None)

    def run(self, cmd, timeout=15):
        key = tuple(cmd)
        if key in self.extra_cmds:
            return self.extra_cmds[key]
        if self.extra and key[0] == "lspci" and key[1:] == ("-nn",):
            return None
        if self.extra and key[0] == "nvidia-smi" and len(key) == 1:
            return None
        return self.base.run(cmd, timeout)

    def read(self, path):
        return self.files.get(path)

    def exists(self, path):
        return path in self.files or any(f.startswith(path + "/") for f in self.files)

    def listdir(self, path):
        return sorted({f[len(path) + 1:].split("/")[0] for f in self.files if f.startswith(path + "/")})


def make_env(name, **kw):
    matrix, extra = MACHINES[name]
    return CompositeEnv(matrix, extra, **kw)


class Network:
    """Every way to reach the network, replaced. Counts requests by kind; nothing is ever sent."""

    def __init__(self):
        self.calls = []

    def kind(self, url):
        url = str(url)
        if "/resolve/" in url:
            return "model_download"
        if "/api/models" in url:
            return "metadata"
        if "api.github.com" in url or "/releases" in url:
            return "engine_release"
        if "/download/" in url or url.endswith((".zip", ".tar.gz", ".tgz")):
            return "engine_download"
        return "other"

    def urlopen(self, req, *a, **k):
        url = getattr(req, "full_url", req)
        self.calls.append((self.kind(url), str(url)))
        raise OSError("fake network: no connection")

    def low_level(self, *a, **k):
        self.calls.append(("socket", repr(a[:1])))
        raise OSError("fake network: no connection")

    def count(self, *kinds):
        return sum(1 for k, _ in self.calls if k in kinds)

    DOWNLOADS = ("model_download", "engine_download")

    @contextlib.contextmanager
    def patched(self):
        from installer.universal import netsafe
        with mock.patch.object(urllib.request, "urlopen", self.urlopen), \
             mock.patch.object(netsafe, "urlopen", self.urlopen), \
             mock.patch.object(socket, "create_connection", self.low_level), \
             mock.patch.object(socket.socket, "connect", self.low_level), \
             mock.patch.object(http.client.HTTPConnection, "connect", self.low_level):
            yield self


def tree(root):
    """{relative path: sha256 or 'dir' or 'link'} of everything under root, including empty directories."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        rel = str(p.relative_to(root))
        if p.is_symlink():
            out[rel] = "link:" + os.readlink(p)
        elif p.is_dir():
            out[rel] = "dir"
        else:
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class QuietOut(install.Out):
    """The real Out (so ask() is the real code) that keeps the text instead of printing it."""

    def __init__(self):
        self.lines = []

    def say(self, text=""):
        self.lines.append(text)

    @property
    def text(self):
        return "\n".join(self.lines)


class Sandbox:
    """HOME, TMPDIR, SHURA_HOME in a fresh folder; the machine; the network; the tools on PATH; the disk."""

    def __init__(self, test, machine, *, free=None, tools=True, drop_files=(), env=None):
        self.test = test
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="run-", dir=SCRATCH))
        test.addCleanup(shutil.rmtree, self.root, True)
        self.home = self.root / "home"
        (self.home / ".config").mkdir(parents=True)
        (self.home / ".config" / "marker.txt").write_text("user data\n")      # something that must stay exactly as it is
        (self.root / "tmp").mkdir()
        self.shura_home = self.home / ".local/share/shura"
        self.net = Network()
        self.env = env or make_env(machine, drop_files=drop_files)
        self.free, self.tools = free, tools

    def detect(self, *a, **k):
        return REAL_DETECT(self.env, ram_bandwidth_gbs=40)

    @contextlib.contextmanager
    def active(self):
        real_usage = shutil.disk_usage

        def usage(path):
            if self.free is None:
                if os.environ.get("SHURA_TEST_REAL_DISK") == "1":
                    return real_usage(path)
                return shutil._ntuple_diskusage(1000 * GiB, 1000 * GiB - FAKE_FREE, FAKE_FREE)
            return shutil._ntuple_diskusage(100 * GiB, 100 * GiB - self.free, self.free)

        def which(name, *a, **k):
            return f"/usr/bin/{name}" if self.tools is True or name in self.tools else None

        environ = {"HOME": str(self.home), "TMPDIR": str(self.root / "tmp"), "SHURA_HOME": str(self.shura_home),
                   "XDG_DATA_HOME": str(self.home / ".local/share")}
        with mock.patch.dict(os.environ, environ), self.net.patched(), \
             mock.patch.object(D, "detect", self.detect), \
             mock.patch.object(shutil, "disk_usage", usage), \
             mock.patch.object(shutil, "which", which):
            yield

    def cli(self, *argv, stdin=None):
        """cli.main(argv) -> (exit code, stdout+stderr text, exception or None)."""
        buf = io.StringIO()
        exc = None
        code = None
        with self.active(), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                code = cli.main(["install", "--ram-gbs", "40", *argv])
            except SystemExit as e:
                code = e.code
            except Exception as e:                                           # noqa: BLE001 - a traceback is what we look for
                exc = e
        return code, buf.getvalue(), exc


def no_traceback(test, exc, text, label):
    test.assertIsNone(exc, f"{label}: raised {exc!r}")
    test.assertNotIn("Traceback", text, f"{label}: a traceback in the output")


def plan_of(machine_name, tools=True, free=None):
    """The plan the installer's own first planning step gives (same call as cli.main), no network, no files."""
    with mock.patch.object(shutil, "which", lambda n, *a, **k: f"/usr/bin/{n}" if tools is True or n in tools else None):
        hw = REAL_DETECT(make_env(machine_name), ram_bandwidth_gbs=40)
    catalog = json.loads((ROOT / "catalog/models.json").read_text())
    return hw, catalog, planner.plan(hw, catalog, profile="balanced")


class DryRunIsClean(unittest.TestCase):
    """1. `install --dry-run` creates and changes nothing; it opens no connection at all."""

    def check(self, name):
        box = Sandbox(self, name)
        before = tree(box.root)
        code, text, exc = box.cli("--dry-run")
        after = tree(box.root)
        no_traceback(self, exc, text, name)
        self.assertEqual(code, 0, text)
        self.assertIn("Dry run: nothing was downloaded or changed.", text)
        self.assertEqual(before, after, f"{name}: the dry run changed files: "
                         f"{sorted(set(after) ^ set(before) | {k for k in before if k in after and before[k] != after[k]})}")
        self.assertFalse(box.shura_home.exists(), "the Shura home must not even be created by a dry run")
        self.assertEqual(box.net.calls, [], f"{name}: the dry run touched the network: {box.net.calls}")
        OUTCOMES[f"dryrun {name}"] = (len(before), len(after), len(box.net.calls))

    def test_amd_gfx1101_machine(self):
        self.check("rx7800xt")

    def test_laptop_intel_nvidia_machine(self):
        self.check("laptop_intel_nvidia")

    def test_cpu_only_machine(self):
        self.check("cpu_only")

    def test_dry_run_with_a_requested_hip_backend_is_clean_too(self):
        box = Sandbox(self, "rx7800xt")
        before = tree(box.root)
        code, text, exc = box.cli("--dry-run", "--backend", "hip")
        no_traceback(self, exc, text, "hip dry run")
        self.assertEqual(tree(box.root), before)
        self.assertEqual(box.net.calls, [])


class Consent(unittest.TestCase):
    """2. Without --yes nothing is downloaded; with or without it the same model and quant are chosen."""

    def picks(self, name, *argv, answer=None):
        """(model, quant) from the first planner.plan call of the install, plus the Sandbox and the run result."""
        box = Sandbox(self, name)
        seen = []
        real = planner.plan

        def spy(*a, **k):
            r = real(*a, **k)
            seen.append((r.get("model"), r.get("quant"), r.get("ok")))
            return r
        before = tree(box.root)
        stdin = mock.Mock()
        stdin.isatty.return_value = answer is not None
        with mock.patch.object(planner, "plan", spy), mock.patch.object(sys, "stdin", stdin), \
                mock.patch("builtins.input", return_value=answer or ""):
            code, text, exc = box.cli(*argv)
        return seen, box, code, text, exc, before

    def test_answer_n_downloads_nothing_and_changes_nothing(self):
        for name in ("rx7800xt", "laptop_intel_nvidia", "rtx_12g"):
            seen, box, code, text, exc, before = self.picks(name, answer="n")
            no_traceback(self, exc, text, name)
            self.assertEqual(code, 0, text)
            self.assertIn("Cancelled. Nothing was changed.", text)
            self.assertEqual(box.net.calls, [], f"{name}: network used after 'n': {box.net.calls}")
            self.assertEqual(box.net.count(*Network.DOWNLOADS), 0)
            self.assertEqual(tree(box.root), before, f"{name}: files changed after 'n'")
            OUTCOMES[f"consent n {name}"] = (box.net.count(*Network.DOWNLOADS), len(box.net.calls))

    def test_no_terminal_and_no_yes_downloads_nothing(self):
        seen, box, code, text, exc, before = self.picks("rx7800xt", answer=None)
        no_traceback(self, exc, text, "no tty")
        self.assertEqual(box.net.calls, [])
        self.assertIn("pass --yes to agree", text)
        self.assertEqual(tree(box.root), before)

    def test_same_model_and_quant_with_yes_and_without(self):
        for name in ("rx7800xt", "laptop_intel_nvidia", "rtx_8g", "rtx_16g"):
            no, *_ = self.picks(name, answer="n")
            dry, *_ = self.picks(name, "--dry-run")
            yes, box, code, text, exc, _ = self.picks(name, "--yes")
            no_traceback(self, exc, text, f"{name} --yes")
            self.assertTrue(no and dry and yes, name)
            self.assertEqual(no[0], dry[0], name)
            self.assertEqual(no[0], yes[0], f"{name}: --yes changed the first choice: {no[0]} vs {yes[0]}")
            # with --yes the installer goes on to the engine: the fake network refuses; no model file is ever requested
            self.assertEqual(box.net.count("model_download"), 0, name)
            OUTCOMES[f"choice {name}"] = no[0][:2]


class Matrix(unittest.TestCase):
    """3. The matrix: one test per row; the outcome goes to OUTCOMES (and SHURA_MATRIX_MD)."""

    def dry(self, name, **kw):
        box = Sandbox(self, name, **kw)
        code, text, exc = box.cli("--dry-run")
        return box, code, text, exc

    def test_a_rx7800xt_hip_rocm(self):
        box, code, text, exc = self.dry("rx7800xt")
        no_traceback(self, exc, text, "a")
        hw, cat, plan = plan_of("rx7800xt")
        self.assertTrue(plan["ok"], plan.get("reasons"))
        self.assertIn("rocm", hw["gpus"][0]["backends"])             # detected: the card can run the ROCm/HIP build
        self.assertTrue(plan["mode"])
        # --backend hip: the HIP fork is planned (a plan is made, the dry run says what would be built)
        code2, text2, exc2 = box.cli("--dry-run", "--backend", "hip")
        no_traceback(self, exc2, text2, "a hip")
        self.assertEqual(code2, 0, text2)
        self.assertIn("built on this machine for ROCm", text2)
        self.assertNotIn("no fitting plan", text2)
        self.assertIn("rocm", text2.lower())
        OUTCOMES["a RX 7800 XT 16 GiB"] = (f"model {plan['model']}/{plan['quant']}, mode {plan['mode']}, "
                                           f"GPU backends {hw['gpus'][0]['backends']}, default plan candidates {plan['backend_candidates']}; --backend hip: HIP fork build announced; exit {code}/{code2}")

    def test_b_laptop_intel_nvidia_screen_on_intel(self):
        box, code, text, exc = self.dry("laptop_intel_nvidia")
        no_traceback(self, exc, text, "b")
        hw, cat, plan = plan_of("laptop_intel_nvidia")
        self.assertTrue(plan["ok"], plan.get("reasons"))
        allg = hw["system"]["gpus_all"]
        intel = [g for g in allg if g["vendor"] == "intel"]
        self.assertTrue(intel and intel[0]["display"] is True, allg)
        self.assertEqual(hw["system"]["display"]["display_slot"], intel[0]["slot"])
        nv = [g for g in hw["gpus"] if g["vendor"] == "nvidia"]
        self.assertTrue(nv and nv[0].get("display") is False, hw["gpus"])
        OUTCOMES["b laptop Intel iGPU + NVIDIA"] = (f"model {plan['model']}/{plan['quant']}, mode {plan['mode']}; display GPU = "
                                                   f"{intel[0]['name'][:30]} ({hw['system']['display']['method']}); NVIDIA display={nv[0].get('display')}; exit {code}")

    def test_c_cpu_only(self):
        box, code, text, exc = self.dry("cpu_only")
        no_traceback(self, exc, text, "c")
        hw, cat, plan = plan_of("cpu_only")
        self.assertEqual(hw["gpus"], [])
        if plan["ok"]:
            self.assertEqual(plan["mode"], "cpu")
            self.assertEqual(code, 0, text)
            what = f"CPU path: model {plan['model']}/{plan['quant']}, mode cpu; exit {code}"
        else:
            self.assertTrue(plan["reasons"])
            what = f"no plan: {plan['reasons'][0][:80]}; exit {code}"
        OUTCOMES["c CPU only"] = what

    def test_d_no_free_disk_is_a_clear_error_before_the_model_download(self):
        box = Sandbox(self, "rx7800xt", free=0)
        hw, cat, plan = plan_of("rx7800xt")
        model = next(m for m in cat["models"] if m["id"] == plan["model"])
        name = modelstore.file_name(model, plan["quant"])
        size = 10 * GiB
        engine_calls = []

        def fake_engine(hw_, plan_, root, tag, out, release=None, only_backend=None):
            engine_calls.append(root)
            return "vulkan", Path(root) / "engines" / "fake-server", [{"ok": True, "backend": "vulkan", "tok_s": 1}]
        with mock.patch.object(install, "choose_engine", fake_engine), \
             mock.patch.object(install, "reconcile", lambda hw_, *a, **k: hw_), \
             mock.patch.object(modelstore, "listing", lambda *a, **k: {name: (size, "0" * 64)}):
            code, text, exc = box.cli("--yes")
        no_traceback(self, exc, text, "d")
        self.assertEqual(box.net.count("model_download"), 0, box.net.calls)
        self.assertIn("not enough disk space", text)
        self.assertNotEqual(code, 0)
        self.assertEqual(code, 1)
        # the journal rolled back / nothing is left behind except what the installer itself reports
        OUTCOMES["d no free disk"] = (f"exit {code}; message: 'not enough disk space ... need X GiB ... 0.0 GiB free'; "
                                      f"model downloads: {box.net.count('model_download')}; engine step reached first: {bool(engine_calls)}")

    def test_d2_dry_run_with_no_free_disk_fails_before_any_download(self):
        """INSTALL1: the free space is checked before the download, also in a dry run: exit 1, 'not enough disk space'."""
        box = Sandbox(self, "rx7800xt", free=0)
        code, text, exc = box.cli("--dry-run")
        no_traceback(self, exc, text, "d2")
        self.assertEqual(code, 1, text)
        self.assertIn("not enough disk space", text)
        self.assertEqual(box.net.count("model_download"), 0, box.net.calls)

    def test_e_no_compiler_names_missing_tools_with_install_commands(self):
        tools = {"git", "hipconfig"}                   # cmake and ninja are missing
        box = Sandbox(self, "rx7800xt", tools=tools)
        code, text, exc = box.cli("--dry-run", "--backend", "hip")
        no_traceback(self, exc, text, "e")
        low = text.lower()
        OUTCOMES["e no compiler"] = (f"exit {code}; no traceback; first line: '{text.splitlines()[0][:90]}'; names cmake/ninja: "
                                     f"{'cmake' in low and 'ninja' in low}; install commands: {'dnf' in low or 'apt' in low}; "
                                     f"sudo: {'sudo' in low}")
        self.assertIn("cmake", low)
        self.assertIn("ninja", low)
        self.assertTrue(any(w in low for w in ("install", "dnf", "apt", "pacman", "brew")), text)
        self.assertNotIn("sudo", low)

    def check_rtx(self, name, label):
        box, code, text, exc = self.dry(name)
        no_traceback(self, exc, text, label)
        hw, cat, plan = plan_of(name)
        self.assertTrue(plan["ok"], plan.get("reasons"))
        self.assertTrue({"cuda", "vulkan"} & set(plan["backend_candidates"]), plan["backend_candidates"])
        self.assertEqual(code, 0, text)
        OUTCOMES[label] = (f"model {plan['model']}/{plan['quant']}, mode {plan['mode']}, backends {plan['backend_candidates']}, "
                           f"context {plan['settings']['context']}; exit {code}")

    def test_f_rtx_8g(self):
        self.check_rtx("rtx_8g", "f RTX 8 GiB")

    def test_f_rtx_12g(self):
        self.check_rtx("rtx_12g", "f RTX 12 GiB")

    def test_f_rtx_16g(self):
        self.check_rtx("rtx_16g", "f RTX 16 GiB")

    def test_g_no_cgroup_v2(self):
        box = Sandbox(self, "rx7800xt", drop_files=("/sys/fs/cgroup/cgroup.controllers",))
        code, text, exc = box.cli("--dry-run")
        no_traceback(self, exc, text, "g")
        hw = box.detect()
        self.assertIs(hw["system"]["cgroup_v2"], False)
        self.assertTrue(any("cgroup v2 not found" in n for n in hw["detection_notes"]))
        self.assertEqual(code, 0, text)
        OUTCOMES["g no cgroup v2"] = f"profile cgroup_v2=False, note 'cgroup v2 not found ...'; dry run exit {code}"


def tearDownModule():
    path = os.environ.get("SHURA_MATRIX_MD")
    if not path:
        return
    lines = ["| row | expected | outcome |", "|---|---|---|"]
    for row, exp in EXPECT.items():
        lines.append(f"| {row} | {exp} | {OUTCOMES.get(row, 'not run or failed')} |")
    lines += ["", "| check | values |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in OUTCOMES.items() if k not in EXPECT]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    unittest.main()
