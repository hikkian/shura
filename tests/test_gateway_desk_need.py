"""MEMV2: the real gateway against the fake llama-server with the VRAM governor: the desk's signals (a test-hook file instead of the
real desktop) move the reserve and the sleep level, the server's journal reaches the daily totals, the status shows it all."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
sys.path.insert(0, str(HERE))

from gateway_harness import GatewayHarness  # noqa: E402
from vram_mode import legacy_only  # noqa: E402

HOOKS = {"extraArgs": ["--vram-test-hooks"]}


def policy_posts(gw):
    log = gw.fake_log() or []
    return [e["body"] for e in log if e.get("method") == "POST" and e.get("path") == "/vram/policy"]


class DeskNeedE2E(unittest.TestCase):
    def guardian(self, signals_path, **extra):
        return {"vramPolicy": "governor", "deskNeed": True, "deskSignalsFile": str(signals_path), "deskPollSeconds": 0.5,
                "deskPushMinSeconds": 1, "deskNeedDownSeconds": 5, "deskSleepOnSeconds": 1, "deskSleepOffSeconds": 3, **extra}

    def test_a_game_signal_raises_the_reserve_and_sleeps_the_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            signals = Path(tmp) / "signals.json"
            signals.write_text(json.dumps({"idle_s": 2.0}))
            with GatewayHarness(guardian=self.guardian(signals), model=HOOKS) as gw:
                gw.start()
                code, _ = gw.chat("hello")
                self.assertEqual(code, 200)
                first = gw.wait_for(lambda: [p for p in policy_posts(gw) if "human_need" in p], timeout=20, what="the first desk push")
                self.assertEqual(first[-1]["human_need"], 0.3)
                self.assertEqual(first[-1]["reserve_mib"], 1536)
                signals.write_text(json.dumps({"game": True}))
                gw.wait_for(lambda: any(p.get("sleep_level") == 1 for p in policy_posts(gw)), timeout=20, what="light sleep")
                body = [p for p in policy_posts(gw) if p.get("sleep_level") == 1][-1]
                self.assertEqual(body["reserve_mib"], 2560)
                self.assertIn("gamemoded", body["note"])
                status = gw.wait_for(lambda: (gw.status().get("desk_need") or {}).get("sleep_level") == 1 and gw.status()["desk_need"], timeout=20,
                                     what="the status")
                self.assertEqual(status["need"], 1.0)
                self.assertTrue(status["status_line"])
                self.assertIn("signals", status)
                signals.write_text(json.dumps({"idle_s": 1.0}))
                gw.wait_for(lambda: any(p.get("sleep_level") == 0 and p["human_need"] <= 0.3 for p in policy_posts(gw)[2:]), timeout=40,
                            what="wake")
                self.assertTrue(any("desk_need" in line for line in [gw.log()]))

    def test_a_long_game_parks_the_model_and_the_next_request_after_it_loads_again(self):
        # MEMV2 L2: light sleep first, then the model leaves the card (game mode's parking) and requests wait; after the game the
        # next request loads it again
        with tempfile.TemporaryDirectory() as tmp:
            signals = Path(tmp) / "signals.json"
            signals.write_text(json.dumps({"idle_s": 2.0}))
            with GatewayHarness(guardian=self.guardian(signals, deskDeepSleepSeconds=3, gameRetrySeconds=1), model=HOOKS) as gw:
                gw.start()
                self.assertEqual(gw.chat("hello")[0], 200)
                signals.write_text(json.dumps({"game": True}))
                gw.wait_for(lambda: any(p.get("sleep_level") == 1 for p in policy_posts(gw)), timeout=20, what="light sleep")
                status = gw.wait_for(lambda: gw.status()["status"] == "PARKED_BY_GAME" and gw.status(), timeout=30, what="parked")
                self.assertEqual(status["desk_need"]["sleep_level"], 2)
                self.assertIn("desk:", status["game_mode"]["reason"])
                self.assertIn("выгружена", status["desk_need"]["status_line"])
                code, body = gw.chat("during the game")
                self.assertEqual(code, 503, body)
                self.assertIn("игры", body["error"]["message"])
                signals.write_text(json.dumps({"idle_s": 1.0}))
                gw.wait_for(lambda: gw.status()["status"] == "UNLOADED", timeout=40, what="unparked after the game")
                self.assertEqual(gw.chat("after the game")[0], 200)
                self.assertIn("game_parked", gw.full_log())

    def test_video_raises_the_working_buffer_and_a_quiet_desk_lowers_it(self):
        # VRAMLOGIC3: the governor's buffer follows the desk without a restart
        with tempfile.TemporaryDirectory() as tmp:
            signals = Path(tmp) / "signals.json"
            signals.write_text(json.dumps({"idle_s": 2.0}))
            with GatewayHarness(guardian=self.guardian(signals), model=HOOKS) as gw:
                gw.start()
                self.assertEqual(gw.chat("hello")[0], 200)
                first = gw.wait_for(lambda: [p for p in policy_posts(gw) if "buffer_mib" in p], timeout=20, what="the first buffer push")
                self.assertEqual(first[-1]["buffer_mib"], 256)
                signals.write_text(json.dumps({"idle_s": 2.0, "video": True}))
                gw.wait_for(lambda: any(p.get("buffer_mib") == 512 for p in policy_posts(gw)), timeout=20, what="the active buffer")
                code, policy = gw.request("GET", "/vram/policy", port=gw.llama_port, timeout=10)
                self.assertEqual((code, policy["buffer_mib"], policy["target_mib"]), (200, 512, policy["reserve_mib"] + 512))
                status = gw.wait_for(lambda: (gw.status().get("desk_need") or {}).get("buffer_mib") == 512 and gw.status()["desk_need"],
                                     timeout=20, what="the status")
                self.assertIn("active desk", status["buffer_reason"])
                signals.write_text(json.dumps({"idle_s": 2.0}))
                gw.wait_for(lambda: policy_posts(gw)[-1].get("buffer_mib") == 256, timeout=40, what="back to the quiet buffer")

    def test_off_by_default(self):
        with GatewayHarness(guardian={"vramPolicy": "governor"}, model=HOOKS) as gw:
            gw.start()
            gw.chat("hello")
            time.sleep(3)
            self.assertFalse(gw.status()["desk_need"]["enabled"])
            self.assertTrue(all("human_need" not in p for p in policy_posts(gw)))

    def test_a_server_before_memv2_gets_the_reserve_and_the_log_stays_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            signals = Path(tmp) / "signals.json"
            signals.write_text(json.dumps({"game": True}))
            hooks = {"extraArgs": ["--vram-test-hooks", "--vram-legacy-api"]}
            with GatewayHarness(guardian=self.guardian(signals), model=hooks) as gw:
                gw.start()
                code, _ = gw.chat("hello")
                self.assertEqual(code, 200)
                gw.wait_for(lambda: any(p.get("reserve_mib") == 2560 and "sleep_level" not in p for p in policy_posts(gw)), timeout=25,
                            what="the reserve without the new fields")
                posts = policy_posts(gw)
                self.assertTrue(any(p.get("max_chunks") == 0 for p in posts), posts)      # light sleep as the old game mode
                time.sleep(8)
                log = gw.full_log()
                self.assertEqual(log.count("desk_need_failed"), 0, log[-1500:])
                self.assertEqual(log.count("desk_need_compat"), 2, log[-1500:])
                status = gw.status()["desk_need"]
                self.assertEqual(status["server_supports"], {"extra": False, "journal": False, "buffer": True})

    @legacy_only("the suite forces the governor policy, so there is no legacy gateway to check")
    def test_not_on_the_legacy_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            signals = Path(tmp) / "signals.json"
            signals.write_text(json.dumps({"game": True}))
            with GatewayHarness(guardian={"deskNeed": True, "deskSignalsFile": str(signals)}) as gw:
                gw.start()
                gw.chat("hello")
                time.sleep(3)
                self.assertFalse((gw.status().get("desk_need") or {}).get("enabled", False))

    def test_a_bad_desk_setting_stops_the_start(self):
        with self.assertRaises(RuntimeError) as caught:
            with GatewayHarness(guardian={"vramPolicy": "governor", "deskNeed": True, "deskReserveFloorMiB": 5}, model=HOOKS) as gw:
                gw.start(timeout=8)
        self.assertIn("deskReserveFloorMiB", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
