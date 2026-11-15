"""SAGECFG1: the user switches the model (ShuraCode /model) between two models of the same kind (Tiel and Sage in the candidate;
here the example config's tiel-coder and tiel-coder-apex stand in). Real gateway, fake server, fake /proc for the RAM ladder.
A switch unloads the running model before it loads the other; with too little RAM it waits a bounded time and is refused
honestly, and the model that was loaded stays and keeps answering."""
import re
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
sys.path.insert(0, str(HERE))

from gateway_harness import GatewayHarness  # noqa: E402
from test_gateway_ram_ladder_e2e import FIXED, write_mem  # noqa: E402

A, B = "tiel-coder", "tiel-coder-apex"


def started_models(gw):
    return [re.search(r" -m (\S+)", line) and line for line in gw.full_log().splitlines() if "Starting llama-server" in line]


class ModelSwitchE2E(unittest.TestCase):
    def setUp(self):
        self.proc = Path(tempfile.mkdtemp(prefix="sagecfg1-proc-", dir="/dev/shm" if Path("/dev/shm").is_dir() else None))
        self.addCleanup(shutil.rmtree, self.proc, True)

    def harness(self, free_mib, wait_s=3):
        write_mem(self.proc, free_mib)
        return GatewayHarness(guardian={"ramPolicy": "honest", "ramWaitSeconds": wait_s}, env={"FAKE_PROC_ROOT": str(self.proc)})

    def test_switching_back_and_forth_unloads_before_it_loads(self):
        with self.harness(30000) as gw:
            for model in (A, B, A):
                code, body = gw.chat("hello", model=model)
                self.assertEqual(code, 200, body)
                self.assertEqual(gw.status()["model"], model)
            self.assertEqual(len(started_models(gw)), 3)
            log = gw.full_log()
            self.assertGreaterEqual(log.count("switching to model="), 2, log[-2000:])

    def test_a_switch_without_room_is_refused_and_the_loaded_model_keeps_answering(self):
        with self.harness(30000) as gw:
            self.assertEqual(gw.chat("hello", model=A)[0], 200)
            write_mem(self.proc, FIXED + 500)                 # far below the other model's need, even with no prompt cache
            started = time.monotonic()
            code, body = gw.chat("switch", model=B)
            waited = time.monotonic() - started
            self.assertEqual(code, 503, body)
            self.assertIn("not enough memory for the model yet", body["error"]["message"])
            self.assertLess(waited, 3 + 10)                   # the wait is bounded (ramWaitSeconds 3), never endless
            status = gw.status()
            self.assertEqual((status["model"], status["status"]), (A, "READY"))
            self.assertEqual(gw.chat("still here", model=A)[0], 200)
            self.assertEqual(len(started_models(gw)), 1)      # the loaded model was never stopped for the refused switch


if __name__ == "__main__":
    unittest.main()
