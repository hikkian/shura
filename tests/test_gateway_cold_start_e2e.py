"""GWSTALL2: a cold start with a tight desk. The model is not loaded, the wanted layout does not pass admission: the gateway
starts the lightest layout of the same model that fits (lite for short) instead of refusing until the client gives up, and
"Load blocked" is one line per reason, not one per retry. Real gateway, fake server, the card's free VRAM from FAKE_NVML_FREE_MIB."""
import re
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
sys.path.insert(0, str(HERE))

from gateway_harness import GatewayHarness

HOOKS = ["--vram-test-hooks", "--vram-warmup-seconds", "0.1"]
LITE_MODEL = {"extraArgs": HOOKS, "liteProfile": {"nCpuMoe": 30, "moeCacheSlots": 2, "vramFreeMinGBToLoad": 0}}
GUARDIAN = {"vramPolicy": "governor", "vramCardReserveMiB": 1536, "vramLiteAfterSeconds": 1, "vramLiteReturnSeconds": 1,
            "vramPauseFloorMiB": 0, "vramPauseAfterSeconds": 0, "vramReloadWaitSeconds": 2}   # MIN without a pause


def wait_for(predicate, timeout=30.0, step=0.5):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(step)
    return False


def starts(gw):
    return [re.search(r"-ncmoe (\d+)", line).group(1) for line in gw.full_log().splitlines() if "Starting llama-server" in line]


def blocked_lines(gw):
    return [line for line in gw.full_log().splitlines() if "Load blocked" in line]


class ColdStartE2E(unittest.TestCase):
    def learned_then_unloaded(self, gw):
        """short runs once (its fixed part is learned: 7470 in the fake), then the server is gone: the next request is a cold start."""
        self.assertEqual(gw.chat("hello")[0], 200, gw.log()[-1500:])
        self.assertTrue(wait_for(lambda: "layout=short fixed_mib" in gw.full_log()), gw.log()[-1500:])
        code, _ = gw.request("POST", "/guardian/unload")
        self.assertEqual(code, 200)
        self.assertTrue(wait_for(lambda: gw.status()["status"] != "READY", timeout=15))

    def test_a_short_refused_on_a_cold_start_lite_starts_and_answers(self):
        # 9000 MiB free: short needs 7470 + 1536 + 256 = 9262 once learned (262 short, as Occamy's 192-223); lite is not learned yet
        with GatewayHarness(guardian=GUARDIAN, model=LITE_MODEL, env={"FAKE_NVML_FREE_MIB": "9000"}) as gw:
            gw.start()
            self.learned_then_unloaded(gw)
            t0 = time.time()
            code, body = gw.chat("cold")
            self.assertEqual(code, 200, (body, gw.log()[-1500:]))
            self.assertLess(time.time() - t0, 25.0)
            self.assertEqual(gw.status()["layout"], "lite")
            self.assertEqual(starts(gw), ["26", "30"], gw.full_log()[-1500:])       # short once, then lite on the cold start
            self.assertIn("layout_upgrade_held", gw.full_log())
            for n in range(3):                                                        # later requests stay on lite, no new refusals
                self.assertEqual(gw.chat(f"more {n}")[0], 200)
            self.assertEqual(gw.status()["layout"], "lite")
            self.assertEqual(len(blocked_lines(gw)), 1, blocked_lines(gw))
            self.assertEqual(len(starts(gw)), 2)

    def test_b_nothing_fits_an_honest_503_at_once_and_one_log_line_per_reason(self):
        model = {**LITE_MODEL, "liteProfile": {**LITE_MODEL["liteProfile"], "vramFreeMinGBToLoad": 40}}   # lite needs 40 GB: refused too
        with GatewayHarness(guardian=GUARDIAN, model=model, env={"FAKE_NVML_FREE_MIB": "9000"}) as gw:
            gw.start()
            self.learned_then_unloaded(gw)
            for n in range(4):                                                        # a client retrying, as the eval harness did
                t0 = time.time()
                code, body = gw.chat(f"cold {n}")
                self.assertEqual(code, 503, (body, gw.log()[-1500:]))
                self.assertIn("Not enough free VRAM", str(body))
                self.assertLess(time.time() - t0, 5.0)
            self.assertEqual(len(starts(gw)), 1)
            lines = blocked_lines(gw)                                                 # short's refusal once, not once per retry;
            self.assertEqual(len(lines), 1, lines)                                    # lite is refused by the quiet first check
            self.assertIn("/short under the governor", lines[0])
            self.assertNotIn("layout_upgrade_held", gw.full_log())                    # nothing started: nothing to hold

    def test_c_short_fits_the_card_a_cold_start_loads_short_as_before(self):
        with GatewayHarness(guardian=GUARDIAN, model=LITE_MODEL) as gw:               # the fake NVML's default: 12000 MiB free
            gw.start()
            self.learned_then_unloaded(gw)
            self.assertEqual(gw.chat("cold")[0], 200, gw.log()[-1500:])
            self.assertEqual(gw.status()["layout"], "short")
            self.assertEqual(starts(gw), ["26", "26"])
            self.assertEqual(blocked_lines(gw), [])
            self.assertNotIn("layout_upgrade_held", gw.full_log())

    def test_d_a_refusal_before_the_start_also_falls_to_lite_without_the_wait(self):
        # short's own threshold (12 GB) refuses it in acquire_model's first check; before GWSTALL2 the request waited
        # vramReloadWaitSeconds and got a 503; lite (threshold 0) fits and starts at once
        model = {**LITE_MODEL, "vramFreeMinGBToLoad": 12}
        with GatewayHarness(guardian=GUARDIAN, model=model, env={"FAKE_NVML_FREE_MIB": "9000"}) as gw:
            gw.start()
            t0 = time.time()
            code, body = gw.chat("first")
            self.assertEqual(code, 200, (body, gw.log()[-1500:]))
            self.assertLess(time.time() - t0, 20.0)
            self.assertEqual(gw.status()["layout"], "lite")
            self.assertEqual(starts(gw), ["30"])
            self.assertEqual(len(blocked_lines(gw)), 0)                               # the first check refuses quietly; only start_llama logs


if __name__ == "__main__":
    unittest.main()
