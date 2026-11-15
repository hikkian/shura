"""Engine choice by release-manifest.json: manifest -> source -> upstream. Fake manifest, fake archive, fake network."""
import copy
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
from universal import engine, install, release_manifest as rm  # noqa: E402

BASE = Path(os.environ.get("TMPDIR") or "/dev/shm") / "INSTALL1-tests"
NAME = "shura-cpp-1.0-hip-linux-x86_64.tar.gz"
URL = "https://example.invalid/" + NAME


def make_archive():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        data = b"#!/bin/sh\nexit 0\n"
        ti = tarfile.TarInfo("shura/llama-server")
        ti.size, ti.mode = len(data), 0o755
        t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


ARCHIVE = make_archive()
GOOD = hashlib.sha256(ARCHIVE).hexdigest()


def release(**kw):
    r = {"id": "shura-1.0-hip-gfx1201", "engine": "shura.cpp", "version": "1.0", "family": "HIP",
         "gpu_targets": ["gfx1201", "gfx1200"], "runtime": {"rocm_min": "6.2", "rocm_max": "7.9"}, "glibc_min": "2.31",
         "parts": ["vram_governor", "mtp"], "built_with": {"commit": "0123abc", "cmake_flags": ["-DGGML_HIP=ON"]},
         "assets": [{"name": NAME, "url": URL, "sha256": GOOD, "size": len(ARCHIVE)}]}
    r.update(kw)
    return r


def manifest(*rels):
    return {"schema": 1, "releases": list(rels) or [release()]}


HW = {"schema": 1, "os": {"family": "linux", "arch": "x86_64"}, "cpu": {}, "memory": {}, "disk": {},
      "gpus": [{"name": "RX 9070 XT", "vendor": "amd", "gfx": "gfx1201", "backends": ["rocm", "vulkan"]}],
      "toolchain": {"hipconfig": True, "cmake": True, "ninja": True, "git": True}}
PLAN = {"backend_candidates": ["rocm", "vulkan"], "needs_probe": False}


class Out:
    def __init__(self):
        self.lines = []

    def say(self, text=""):
        self.lines.append(text)

    @property
    def text(self):
        return "\n".join(self.lines)


class Fake:
    """Patches the network (`engine._open`), the self-test, the tiny model, the machine probes and the source build."""

    def __init__(self, tmp, *, served=ARCHIVE, rocm="6.4.1", glibc="2.39", mtext=None, source_ok=True, hw=HW):
        self.tmp, self.hw = Path(tmp), hw
        self.served, self.rocm, self.glibc, self.mtext, self.source_ok = served, rocm, glibc, mtext, source_ok
        self.opened, self.source_calls, self.upstream_calls = [], [], []

    def _open(self, url, **kw):
        self.opened.append(url)
        if url == URL:
            return io.BytesIO(self.served)
        raise engine.EngineError("unexpected url " + url)

    def build(self, home, out, arch=None):
        self.source_calls.append(home)
        if not self.source_ok:
            raise engine.EngineError("build failed")
        Path(home).mkdir(parents=True, exist_ok=True)
        p = Path(home) / "src-server"
        p.write_text("x")
        return p

    def upstream(self, hw, plan, root, tag, out, release=None, only_backend=None, skip_hipfork=False):
        self.upstream_calls.append(skip_hipfork)
        return "vulkan", Path("up"), [{"backend": "vulkan", "ok": True, "tok_s": 1}]

    def __enter__(self):
        mpath = self.tmp / "release-manifest.json"
        mpath.write_text(self.mtext if self.mtext is not None else json.dumps(manifest()))
        self.patches = [
            mock.patch.dict(os.environ, {engine.MANIFEST_ENV: str(mpath)}),
            mock.patch.object(engine, "_open", side_effect=self._open),
            mock.patch.object(engine, "smoke_model", return_value=Path("m.gguf")),
            mock.patch.object(engine, "selftest", return_value={"ok": True, "tok_s": 50.0}),
            mock.patch.object(install, "build_rocm_fork", side_effect=self.build),
            mock.patch.object(install, "_choose_upstream", side_effect=self.upstream),
            mock.patch.object(rm, "_glibc", return_value=self.glibc),
            mock.patch.object(rm, "_read", side_effect=lambda p: self.rocm),
            mock.patch.object(rm, "_run", return_value=""),
        ]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self.patches:
            p.stop()

    def choose(self):
        out = Out()
        return install.choose_engine(self.hw, PLAN, self.tmp / "home", None, out), out


class ManifestCase(unittest.TestCase):
    def setUp(self):
        BASE.mkdir(parents=True, exist_ok=True)
        self._d = tempfile.TemporaryDirectory(dir=BASE)
        self.tmp = Path(self._d.name)

    def tearDown(self):
        self._d.cleanup()


class Chosen(ManifestCase):
    def test_a_fitting_build_is_taken_and_its_sum_matches(self):
        with Fake(self.tmp) as f:
            (backend, server, res), out = f.choose()
        self.assertEqual(backend, "rocm")
        self.assertEqual(res[0]["engine_path"], "manifest")
        self.assertIn("SHA-256 matched", res[0]["engine_why"])
        self.assertTrue(server.is_file())
        self.assertEqual(f.opened, [URL])
        self.assertEqual((f.source_calls, f.upstream_calls), ([], []))
        self.assertIn("prebuilt shura.cpp 1.0", out.text)

    def test_a_wrong_sum_is_refused_and_the_source_build_runs(self):
        bad = manifest(release(assets=[{"name": NAME, "url": URL, "sha256": "0" * 64, "size": 1}]))
        with Fake(self.tmp, mtext=json.dumps(bad)) as f:
            (backend, server, res), out = f.choose()
        self.assertEqual(len(f.source_calls), 1)
        self.assertEqual(res[0]["engine_path"], "source")
        self.assertIn("сумма не сошлась", out.text)
        self.assertIn("building shura.cpp from source", out.text)
        self.assertEqual(f.opened, [URL])                                   # no second prebuilt build was tried
        self.assertFalse(list((self.tmp / "home" / "engines").glob("*shura*")))   # nothing was unpacked

    def test_a_wrong_sum_and_no_toolchain_end_in_upstream_with_an_honest_line(self):
        bad = manifest(release(assets=[{"name": NAME, "url": URL, "sha256": "0" * 64}]))
        hw = {**HW, "toolchain": {}}
        with Fake(self.tmp, mtext=json.dumps(bad), hw=hw) as f:
            (_, _, res), out = f.choose()
        self.assertEqual((f.source_calls, f.upstream_calls), ([], [True]))
        self.assertEqual(res[0]["engine_path"], "upstream")
        self.assertIn("WITHOUT our parts", out.text)
        self.assertIn("no VRAM guard", res[0]["engine_why"])

    def test_no_fitting_build_goes_to_source(self):
        cases = {"other gpu": dict(hw={**HW, "gpus": [{**HW["gpus"][0], "gfx": "gfx1100"}]}),
                 "rocm too old": dict(rocm="6.0"), "rocm too new": dict(rocm="8.0"),
                 "glibc older": dict(glibc="2.28"), "rocm unknown": dict(rocm=""), "glibc unknown": dict(glibc=None)}
        for name, kw in cases.items():
            with self.subTest(name), Fake(self.tmp, **kw) as f:
                (_, _, res), out = f.choose()
                self.assertEqual(res[0]["engine_path"], "source")
                self.assertEqual(f.opened, [])                              # nothing was downloaded
                self.assertIn("no prebuilt shura.cpp build fits", out.text)

    def test_a_different_architecture_does_not_fit(self):
        m = {"os": "linux", "arch": "arm64", "backends": ["rocm"], "gpu_targets": ["gfx1201"], "rocm": "6.4", "glibc": "2.39"}
        rel, why = rm.best_release(manifest(), m)
        self.assertIsNone(rel)
        self.assertIn("x86_64", why)

    def test_a_broken_manifest_is_a_clear_message_not_a_crash(self):
        no_sha = manifest()
        del no_sha["releases"][0]["assets"][0]["sha256"]
        for name, text, needle in (("not json", "{nope", "not valid JSON"), ("no sha256", json.dumps(no_sha), "'sha256'"),
                                   ("wrong shape", "[]", "not a JSON object")):
            with self.subTest(name), Fake(self.tmp, mtext=text) as f:
                (_, _, res), out = f.choose()
                self.assertIn("manifest is unusable", out.text)
                self.assertIn(needle, out.text)
                self.assertEqual(res[0]["engine_path"], "source")
                self.assertEqual(f.opened, [])
        with self.assertRaises(rm.ManifestError) as cm:
            rm.parse(json.dumps(no_sha))
        self.assertIn("sha256", str(cm.exception))

    def test_without_a_manifest_the_old_behaviour_stays(self):
        with Fake(self.tmp) as f, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(engine.MANIFEST_ENV, None)
            (_, _, res), out = f.choose()
        self.assertEqual(f.upstream_calls, [False])                         # the upstream path, hipfork branch left alone
        self.assertEqual((f.source_calls, f.opened), ([], []))
        self.assertNotIn("engine_path", res[0])
        self.assertNotIn("Engine path", out.text)


class Selection(unittest.TestCase):
    M = {"os": "linux", "arch": "x86_64", "backends": ["rocm", "vulkan", "cpu"], "gpu_targets": ["gfx1201"],
         "rocm": "6.4", "vulkan": "1.3", "glibc": "2.39"}

    def test_best_is_the_first_backend_then_the_newest_version(self):
        vk = release(id="vk", family="Vulkan", gpu_targets=[], version="1.0", assets=[{"name": "shura-cpp-1.0-vulkan-linux-x86_64.tar.gz", "url": URL, "sha256": GOOD}], runtime={"vulkan_min": "1.2"})
        new = release(id="new", version="1.1")
        rel, _ = rm.best_release(manifest(vk, release(), new), self.M)
        self.assertEqual(rel["id"], "new")

    def test_a_missing_field_is_not_guessed(self):
        r = release()
        del r["glibc_min"]
        with self.assertRaises(rm.ManifestError):
            rm.validate(manifest(r))
        rel, why = rm.best_release(manifest(release(runtime={})), self.M)
        self.assertIsNone(rel)
        self.assertIn("does not say", why)

    def test_cpu_family_needs_no_gpu_target(self):
        cpu = release(id="cpu", family="CPU", gpu_targets=[], runtime={},
                      assets=[{"name": "shura-cpp-1.0-cpu-linux-x86_64.tar.gz", "url": URL, "sha256": GOOD}])
        rel, _ = rm.best_release(manifest(cpu), {**self.M, "gpu_targets": [], "backends": ["cpu"]})
        self.assertEqual(rel["id"], "cpu")

    def test_http_urls_and_bad_sums_are_rejected(self):
        for assets in ([{"name": NAME, "url": "http://x/a", "sha256": GOOD}], [{"name": NAME, "url": URL, "sha256": "abc"}]):
            with self.assertRaises(rm.ManifestError):
                rm.validate(manifest(release(assets=assets)))


class ReleaseRules(unittest.TestCase):
    RT = {"rocm_min": "7.0", "rocm_max": "7.99"}

    def test_rocm_same_major(self):
        self.assertTrue(rm.rocm_same_major(self.RT, "7.2.4")[0])
        ok, why = rm.rocm_same_major(self.RT, "10.1.0")
        self.assertFalse(ok)
        self.assertIn("ROCm major 10, release needs 7", why)
        self.assertEqual(rm.rocm_same_major(self.RT, None), (False, "ROCm version unknown"))
        ok, why = rm.rocm_same_major({"rocm_min": "7.0"}, "7.2.4")
        self.assertFalse(ok)
        self.assertIn("rocm_max", why)

    def test_glibc_host_ok(self):
        self.assertTrue(rm.glibc_host_ok("2.35", "2.35")[0])
        self.assertFalse(rm.glibc_host_ok("2.35", "2.34")[0])
        self.assertEqual(rm.glibc_host_ok("2.35", None), (False, "host glibc unknown"))
        self.assertTrue(rm.glibc_host_ok("2.35", "2.100")[0])           # numeric, not string, comparison


class AssetNameAndRedirects(unittest.TestCase):
    def test_bad_asset_names_are_rejected(self):
        for name in ("../../evil.tar.gz", "/home/u/.bashrc", "sub/x.tar.gz", "..", ".", ".hidden", "a\\b", "a\0b", "a..b"):
            with self.subTest(name=name), self.assertRaises(rm.ManifestError):
                rm.validate(manifest(release(assets=[{"name": name, "url": URL, "sha256": GOOD}])))

    def test_a_plain_name_passes(self):
        rm.validate(manifest(release(assets=[{"name": NAME, "url": URL, "sha256": GOOD}])))

    def test_install_build_refuses_a_path_name(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(engine, "_open") as op:
            asset = {"name": "../../evil.tar.gz", "url": URL, "sha256": GOOD}
            with self.assertRaises(rm.ManifestError):
                engine.install_build({"tag": "t", "assets": [asset]}, asset, Path(d), label="shura", sha256=GOOD)
            op.assert_not_called()

    def test_download_stays_in_its_folder(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(engine, "_open", return_value=io.BytesIO(ARCHIVE)):
            out = engine.download(URL, Path(d) / "sub" / ".." / ".." / "x.tar.gz", GOOD)
            self.assertEqual(out.parent.resolve(), (Path(d) / "sub" / ".." / "..").resolve())
            self.assertFalse(list(Path(d).rglob("*.part")))

    def test_redirects_only_to_https(self):
        from universal import netsafe
        h = netsafe.HttpsOnlyRedirect()
        req = urllib.request.Request("https://a.example/f")
        with self.assertRaisesRegex(urllib.error.URLError, "redirect to non-HTTPS URL refused"):
            h.redirect_request(req, None, 302, "Found", {}, "http://b.example/f")
        self.assertIsNotNone(h.redirect_request(req, None, 302, "Found", {}, "https://b.example/f"))

    def test_direct_http_is_refused_for_manifest_and_model(self):
        from universal import modelstore
        with self.assertRaises(rm.ManifestError):
            engine.fetch_manifest("http://x.example/m.json")
        with self.assertRaises(modelstore.StoreError):
            modelstore._open("http://x.example/m.gguf")
        with self.assertRaises(engine.EngineError):
            engine._open("http://x.example/m.json")

    def test_loopback_still_works_for_the_model_store(self):
        import http.server
        import threading
        from universal import modelstore

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *a):
                pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with modelstore._open(f"http://127.0.0.1:{srv.server_address[1]}/x", allow_local=True) as r:
                self.assertEqual(r.read(), b"ok")
        finally:
            srv.shutdown()
            srv.server_close()

class ManifestSourceConfig(unittest.TestCase):
    def test_local_path_in_config_is_ignored(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            p = Path(d) / "installer.json"
            p.write_text(json.dumps({"release_manifest_url": "/etc/passwd"}))
            with mock.patch.dict(os.environ), mock.patch.object(engine, "INSTALLER_CONFIG", p):
                os.environ.pop(engine.MANIFEST_ENV, None)
                self.assertIsNone(engine.manifest_source())
                p.write_text(json.dumps({"release_manifest_url": "http://x.invalid/m.json"}))
                self.assertIsNone(engine.manifest_source())
                os.environ[engine.MANIFEST_ENV] = "/tmp/local.json"           # a local path only from the environment
                self.assertEqual(engine.manifest_source(), "/tmp/local.json")


if __name__ == "__main__":
    unittest.main()
