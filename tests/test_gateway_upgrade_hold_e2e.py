"""GWSTALL1 end to end: the real gateway against the fake llama-server with the VRAM governor. The card's free VRAM as the
gateway sees it comes from the fake NVML (FAKE_NVML_FREE_MIB); the governor's own numbers come from the fake server."""
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
    return sum(1 for line in gw.full_log().splitlines() if "Starting llama-server" in line)


class UpgradeHoldE2E(unittest.TestCase):
    def lite_after_min(self, gw):
        """short runs and its fixed part is learned; the desk takes VRAM for 12 s, the governor sits in MIN, the next request goes lite."""
        self.assertEqual(gw.chat("hello")[0], 200, gw.log()[-1500:])
        self.assertEqual(gw.status()["layout"], "short")
        self.assertTrue(wait_for(lambda: "vram_governor_fixed_learned" in gw.log()), gw.log()[-1500:])
        gw.request("POST", "/vram/simulate", {"free_delta_mib": -6000, "seconds": 12}, port=gw.llama_port, timeout=10)
        self.assertTrue(wait_for(lambda: "MIN" in gw.log()), gw.log()[-1500:])
        time.sleep(1.5)
        self.assertEqual(gw.chat("another")[0], 200, gw.log()[-1500:])
        self.assertEqual(gw.status()["layout"], "lite")
        self.assertEqual(starts(gw), 2)
        # the desk gives the VRAM back: the governor leaves MIN, the relief time passes
        self.assertTrue(wait_for(lambda: "FULL" in gw.log() or "GROW" in gw.log(), timeout=30))
        time.sleep(1.5)

    def test_a_short_does_not_fit_the_card_lite_answers_at_once(self):
        # the card has 1000 MiB free next to lite's 8000: short (fixed 7470 + 1536 + 256 = 9262) is 262 MiB short, as in SAGE1
        with GatewayHarness(guardian=GUARDIAN, model=LITE_MODEL, env={"FAKE_NVML_FREE_MIB": "1000"}) as gw:
            gw.start()
            self.lite_after_min(gw)
            for n in range(3):
                t0 = time.time()
                code, body = gw.chat(f"request {n} after the desk gave the VRAM back")
                self.assertEqual(code, 200, (body, gw.log()[-1500:]))
                self.assertLess(time.time() - t0, 5.0)
                self.assertEqual(gw.status()["layout"], "lite")
            self.assertEqual(starts(gw), 2, "short was tried although it does not fit: " + gw.log()[-1500:])
            self.assertNotIn("Load blocked", gw.full_log())
            self.assertIn("layout_upgrade_held", gw.full_log())
            self.assertEqual(gw.full_log().count("layout_upgrade_held"), 1)   # one event for the hold, not one per request

    def test_c_short_fits_the_card_the_upgrade_happens_as_before(self):
        with GatewayHarness(guardian=GUARDIAN, model=LITE_MODEL) as gw:      # the fake NVML's default: 12000 MiB free
            gw.start()
            self.lite_after_min(gw)
            self.assertEqual(gw.chat("third")[0], 200, gw.log()[-1500:])
            self.assertEqual(gw.status()["layout"], "short")
            self.assertEqual(starts(gw), 3)
            self.assertNotIn("layout_upgrade_held", gw.full_log())

    def test_b_nothing_fits_the_request_an_honest_503_within_the_bounded_wait(self):
        model = {"extraArgs": HOOKS, "longContextSwitchTokens": 1000, "longContextHoldSeconds": 0,
                 "longContextProfile": {"ctxSize": 400000, "nCpuMoe": 30, "moeCacheSlots": 2, "vramFreeMinGBToLoad": 40}}
        with GatewayHarness(guardian=GUARDIAN, model=model) as gw:
            gw.start()
            self.assertEqual(gw.chat("hello")[0], 200, gw.log()[-1500:])
            self.assertEqual(gw.status()["layout"], "short")
            t0 = time.time()
            code, body = gw.chat("x" * 6000)          # ~2400 tokens by the legacy estimate: the long layout, which needs 40 GB
            elapsed = time.time() - t0
            self.assertEqual(code, 503, (body, gw.log()[-1500:]))
            self.assertIn("SHURA_MODEL_UNAVAILABLE", str(body))
            self.assertIn("Not enough free VRAM", str(body))
            self.assertLess(elapsed, 10.0, "the client waited longer than vramReloadWaitSeconds")
            self.assertGreaterEqual(elapsed, 2.0)
            status = gw.status()
            self.assertEqual((status["status"], status["layout"]), ("READY", "short"))   # the refused switch left the model loaded
            self.assertEqual(gw.chat("hello again")[0], 200)
            self.assertEqual(starts(gw), 1)


if __name__ == "__main__":
    unittest.main()
