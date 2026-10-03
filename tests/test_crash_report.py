"""Crash reports stay small, redact launch secrets and tolerate missing journals."""

import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))
from crash_report import MAX_RECORD_BYTES, write_crash_report


class CrashReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="shura-crash-report-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)

    def test_missing_journalctl_does_not_break_crash_capture(self):
        path = write_crash_report(
            self.tmp.name,
            ["llama-server", "--api-key", "secret-value"],
            -signal.SIGABRT,
            "generating response",
            lambda: {"status": "READY", "vram_free_mib": 321},
            read_journal=lambda: (_ for _ in ()).throw(FileNotFoundError("journalctl")),
            now=1000.25,
        )
        self.assertIsNotNone(path)
        record = json.loads(Path(path).read_text())
        self.assertEqual(record["signal"], "SIGABRT")
        self.assertEqual(record["operation"], "generating response")
        self.assertEqual(record["journal_error"], "FileNotFoundError")
        self.assertNotIn("secret-value", Path(path).read_text())

    def test_twelve_crashes_keep_only_ten_valid_records(self):
        for index in range(12):
            result = write_crash_report(
                self.tmp.name, ["llama-server"], index, "loading",
                lambda: {"status": "LOADING"}, read_journal=lambda: [], now=2000 + index,
            )
            self.assertIsNotNone(result)
        records = sorted((Path(self.tmp.name) / "crashes").glob("crash-*.json"))
        self.assertEqual(len(records), 10)
        self.assertEqual(json.loads(records[-1].read_text())["exit_code"], 11)
        self.assertTrue(all(path.stat().st_size <= MAX_RECORD_BYTES for path in records))

    def test_large_journal_is_trimmed_under_64k(self):
        path = write_crash_report(
            self.tmp.name, ["llama-server", "--password=secret"], 1, "loading",
            lambda: {"status": "LOADING"},
            read_journal=lambda: ["x" * 4096 for _ in range(150)], now=3000,
        )
        self.assertLessEqual(Path(path).stat().st_size, MAX_RECORD_BYTES)
        record = json.loads(Path(path).read_text())
        self.assertLessEqual(len(record["server_log_tail"]), 150)
        self.assertNotIn("--password=secret", record["launch_args"])


if __name__ == "__main__":
    unittest.main()
