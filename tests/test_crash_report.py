"""Crash reports stay small, redact launch secrets and tolerate missing journals."""

import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))
import crash_report  # noqa: E402

from crash_report import MAX_RECORD_BYTES, write_crash_report  # noqa: E402


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

    def test_gateway_journal_unit_is_configurable_only_for_isolated_runner(self):
        with (patch.dict(os.environ, {"AI_GATEWAY_CRASH_JOURNAL_UNIT": "shura-test.service"}),
              patch.object(crash_report.subprocess, "run",
                           return_value=CompletedProcess([], 0, "test journal line\n", "")) as run):
            report = write_crash_report(self.tmp.name, ["llama-server"], -9, "stream", lambda: {})
        self.assertIsNotNone(report)
        self.assertIn("shura-test.service", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
