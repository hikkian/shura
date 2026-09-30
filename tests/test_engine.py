"""Engine tests: archive safety, download verification, self-test against a fake server, backend choice. Offline."""
import hashlib
import io
import os
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
from universal import engine as E  # noqa: E402

FAKE = ROOT / "tests/fixtures/fake_llama_server.py"


def tar_with(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class Extract(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", "no POSIX execute bit on Windows")
    def test_extracts_and_finds_the_server(self):
        with tempfile.TemporaryDirectory() as d:
            arc = Path(d) / "b.tar.gz"
            arc.write_bytes(tar_with([("build/bin/llama-server", b"#!/bin/sh\n"), ("build/bin/libx.so", b"x")]))
            E.extract(arc, Path(d) / "out")
            server = E.find_server(Path(d) / "out")
            self.assertEqual(server.name, "llama-server")
            self.assertTrue(os.access(server, os.X_OK))

    def test_path_traversal_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            arc = Path(d) / "evil.tar.gz"
            arc.write_bytes(tar_with([("../escape.txt", b"x")]))
            with self.assertRaises(E.EngineError):
                E.extract(arc, Path(d) / "out")
            self.assertFalse((Path(d) / "escape.txt").exists())
            z = Path(d) / "evil.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("../../escape2.txt", "x")
            with self.assertRaises(E.EngineError):
                E.extract(z, Path(d) / "out2")

    def test_missing_server_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(E.EngineError):
                E.find_server(d)


class Download(unittest.TestCase):
    def test_non_https_is_refused(self):
        with self.assertRaises(E.EngineError):
            E._open("http://example.com/x")

    def test_hash_is_verified_and_a_bad_file_is_removed(self):
        payload = b"model bytes"
        with tempfile.TemporaryDirectory() as d, mock.patch.object(E, "_open", return_value=io.BytesIO(payload)):
            good = E.download("https://x/y", Path(d) / "ok.bin", hashlib.sha256(payload).hexdigest())
            self.assertEqual(good.read_bytes(), payload)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(E, "_open", return_value=io.BytesIO(payload)):
            with self.assertRaises(E.EngineError):
                E.download("https://x/y", Path(d) / "bad.bin", "sha256:" + "0" * 64)
            self.assertEqual(list(Path(d).iterdir()), [])


@unittest.skipIf(sys.platform == "win32", "the fake server is a script started through its shebang")
class SelfTest(unittest.TestCase):
    def run_fake(self, mode="ok", tok_s="50", backend="cpu", timeout=30):
        with mock.patch.dict(os.environ, {"FAKE_MODE": mode, "FAKE_TOK_S": tok_s}):
            return E.selftest(FAKE, "model.gguf", backend=backend, timeout=timeout)

    def test_working_server_reports_speed(self):
        r = self.run_fake(tok_s="42.5")
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["backend"], r["tok_s"], r["tokens"]), ("cpu", 42.5, 24))

    def test_crash_at_start_is_reported_with_the_log(self):
        r = self.run_fake("crash")
        self.assertFalse(r["ok"])
        self.assertIn("exited", r["error"])
        self.assertIn("cannot load the model", r["log_tail"])

    def test_empty_answer_is_a_failure(self):
        r = self.run_fake("empty")
        self.assertFalse(r["ok"])
        self.assertIn("empty", r["error"])


class Choice(unittest.TestCase):
    def r(self, b, tok_s, ok=True):
        return {"backend": b, "ok": ok, "tok_s": tok_s}

    def test_fastest_wins_when_clearly_faster(self):
        self.assertEqual(E.choose_backend([self.r("rocm", 40), self.r("vulkan", 60)], ["rocm", "vulkan", "cpu"])[0],
                         "vulkan")

    def test_near_ties_prefer_the_more_tested_backend(self):
        b, why = E.choose_backend([self.r("rocm", 58), self.r("vulkan", 60)], ["rocm", "vulkan", "cpu"])
        self.assertEqual(b, "rocm")
        self.assertIn("more tested", why)

    def test_failed_backends_are_ignored_and_none_left_is_reported(self):
        self.assertEqual(E.choose_backend([self.r("cuda", 99, ok=False), self.r("cpu", 5)], ["cuda", "cpu"])[0], "cpu")
        self.assertIsNone(E.choose_backend([self.r("cuda", 0, ok=False)], ["cuda"])[0])


if __name__ == "__main__":
    unittest.main()
