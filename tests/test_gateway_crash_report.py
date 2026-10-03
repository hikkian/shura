"""The monitor captures an unexpected server exit without depending on journalctl."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_gateway as fixture
sys.path.insert(0, str(fixture.REPO / "gateway"))
import crash_report


class GatewayCrashReport(fixture.GatewayGuard):
    def test_monitor_writes_crash_bundle_when_journalctl_is_missing(self):
        state_dir = Path(self.tmp.name) / "state"
        self.patch_attr("STATE_DIR", state_dir)
        failed = Mock(pid=23456, args=["llama-server", "--api-key", "do-not-store"])
        failed.poll.return_value = -11
        failed.returncode = -11
        self.patch_attr("st", self.gw.State())
        self.gw.st.proc = failed
        self.gw.st.status = "READY"
        self.gw.st.model_id = self.gw.M["defaultModel"]
        self.gw.st.busy = 1
        with patch.object(crash_report, "journal_tail", side_effect=FileNotFoundError("journalctl")):
            self.gw.monitor_tick(100, 400, 20)
        files = list((state_dir / "crashes").glob("crash-*.json"))
        self.assertEqual(len(files), 1)
        text = files[0].read_text()
        self.assertNotIn("do-not-store", text)
        record = json.loads(text)
        self.assertEqual(record["signal"], "SIGSEGV")
        self.assertEqual(record["state"]["busy_requests"], 1)
        self.assertEqual(record["journal_error"], "FileNotFoundError")
        self.assertIsNone(self.gw.st.proc)


if __name__ == "__main__":
    unittest.main()
