"""VRAMLOGIC2: floor + buffer + instant release. The server keeps free VRAM at reserve + buffer (the working target) and trims
by the whole deficit as soon as it drops below the target, so a desk burst up to the buffer never reaches the floor. The gateway
judges admission against the target the server really keeps, sends the buffer only when the config names it, and still works
with a server from before VRAMLOGIC2. Fake server, no GPU."""
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
sys.path.insert(0, str(HERE.parent / "gateway"))
sys.path.insert(0, str(HERE))

import vram_budget
from fake_vram import FakeVram, PolicyError, validate_policy_patch
from gateway_harness import GatewayHarness

HOOKS = ["--vram-test-hooks", "--vram-warmup-seconds", "0.1"]
GUARDIAN = {"vramPolicy": "governor", "vramCardReserveMiB": 1536, "vramPauseFloorMiB": 0, "vramLiteAfterSeconds": 1}
STALL = {"vramPolicy": "governor", "vramCardReserveMiB": 1536, "vramPauseFloorMiB": 0}   # only the stall rule can pause


def wait_for(predicate, timeout=30.0, step=0.5):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(step)
    return False


class BudgetUnit(unittest.TestCase):
    def test_admission_margin_is_floor_plus_buffer_plus_hysteresis(self):
        b = vram_budget.Budget(reserve_mib=1536, hysteresis_mib=256)
        self.assertEqual((b.buffer_mib, b.target_mib, b.admit_margin_mib), (0, 1536, 1792))     # before the server said anything
        b.learn_policy({"buffer_mib": 512, "reserve_mib": 1536})
        self.assertEqual((b.target_mib, b.admit_margin_mib), (2048, 2304))
        b.fixed_learned[("m", "short")] = 12800
        ok, short, _ = b.admission("m", "short", free_before_start_mib=14344)                      # SAGE1's card: 14344 free
        self.assertEqual((ok, short), (False, 760))                                                 # 12800 + 2304 - 14344
        b.learn_policy({"reserve_mib": 1536})                                                       # an older server: no field, keeps 512
        self.assertEqual(b.buffer_mib, 512)
        b.learn_policy({"buffer_mib": 1.5})                                                         # nonsense is ignored
        self.assertEqual(b.buffer_mib, 512)

    def test_the_config_names_the_buffer_and_the_fast_poll_only_when_set(self):
        _policy, values, _lite = vram_budget.check_policy_config({})
        self.assertNotIn("buffer_mib", values)
        self.assertNotIn("poll_fast_ms", values)
        _policy, values, _lite = vram_budget.check_policy_config({"vramBufferMiB": 640, "vramPollFastMs": 40})
        self.assertEqual((values["buffer_mib"], values["poll_fast_ms"]), (640, 40))
        for bad in ({"vramBufferMiB": 4096}, {"vramBufferMiB": 0.5}, {"vramPollFastMs": 5}):
            with self.assertRaises(vram_budget.BudgetError, msg=str(bad)):
                vram_budget.check_policy_config(bad)


class FakeGovernorUnit(unittest.TestCase):
    """The fake mirrors the server's new rules, so the gateway tests see the right behaviour."""

    def ready(self, desk=0, fixed=None):
        """Warm-up done and the cache grown to the target: the fake grows one chunk per poll once grow_seconds is 0."""
        g = FakeVram(warmup_seconds=0.2, policy={"grow_seconds": 0, "buffer_mib": 512}, fixed=fixed)   # the active-desk buffer
        g.set_desk(desk)
        g.tick(15)
        return g

    def test_fills_to_target_plus_hysteresis_and_a_burst_within_the_buffer_never_reaches_the_floor(self):
        g = self.ready()
        self.assertGreaterEqual(g.card_free(), 2048 + 256)
        self.assertLess(g.card_free(), 2048 + 256 + g.chunk_mib)
        full = g.chunks
        g.set_desk(400)                                # +400 MiB: below the target, above the floor
        self.assertGreaterEqual(g.card_free(), 1536)
        g.tick(0.2)                                    # one poll
        self.assertLess(g.chunks, full)
        self.assertGreaterEqual(g.card_free(), 2048)   # the whole deficit in one resize
        self.assertEqual(g.counters["free_below_reserve_events"], 0)
        self.assertEqual(g.counters["trims"], 1)

    def test_a_brave_sized_burst_is_absorbed_in_one_poll_when_the_cache_holds_enough(self):
        # a smaller fixed part: a cache of ~2.5 GiB to give back
        g = self.ready(fixed={"weights": 4000, "kv": 640, "draft": 120, "pool": 410, "sched": 300, "vision": 0})
        self.assertGreaterEqual(g.cache_mib(), 2500)
        g.set_desk(1700)
        self.assertLess(g.card_free(), 1536)
        g.tick(0.2)
        self.assertGreaterEqual(g.card_free(), 2048)
        self.assertEqual(g.counters["pauses"], 0)
        self.assertLessEqual(g.counters["free_below_reserve_max_s"], 0.2 + 1e-9)

    def test_between_floor_and_target_with_an_empty_cache_is_min_without_a_pause(self):
        g = self.ready()
        g.set_desk(g.card_free() + g.cache_mib() - 1800)   # free 1800 with an empty cache
        g.tick(3.0)
        self.assertEqual((g.chunks, g.state, g.counters["pauses"]), (0, "MIN", 0))

    def test_the_policy_knows_the_new_fields_and_a_legacy_server_refuses_them(self):
        g = FakeVram()
        code, policy = g.post_policy({"buffer_mib": 256, "poll_fast_ms": 40, "trim_abort": False})
        self.assertEqual((code, policy["buffer_mib"], policy["poll_fast_ms"], policy["trim_abort"], policy["target_mib"]), (200, 256, 40, False, 1792))
        self.assertEqual(g.post_policy({"buffer_mib": 5000})[0], 400)
        self.assertEqual(g.post_policy({"trim_abort": "yes"})[0], 400)
        with self.assertRaises(PolicyError):
            validate_policy_patch({"buffer_mib": 256}, g.policy, legacy=True)
        g.legacy_api = True
        self.assertNotIn("buffer_mib", g.get_policy())
        self.assertEqual(g.target_mib(), 1536)


class BufferEndToEnd(unittest.TestCase):
    def fake_args(self, gw):
        code, body = gw.request("GET", "/__fake/args", port=gw.llama_port, timeout=10)
        return body if code == 200 else []

    def policy_posts(self, gw):
        return [e["body"] for e in gw.fake_log() if e.get("method") == "POST" and e.get("path") == "/vram/policy"]

    def test_without_a_config_value_the_server_keeps_its_own_buffer_and_the_gateway_learns_it(self):
        with GatewayHarness(guardian=GUARDIAN, model={"extraArgs": HOOKS}) as gw:
            gw.start()
            self.assertEqual(gw.chat("hello")[0], 200, gw.log()[-1500:])
            self.assertTrue(wait_for(lambda: "vram_governor_fixed_learned" in gw.full_log()), gw.log()[-1500:])
            args = self.fake_args(gw)
            self.assertNotIn("--vram-buffer-mib", args)
            self.assertNotIn("--vram-poll-fast-ms", args)
            for body in self.policy_posts(gw):
                self.assertNotIn("buffer_mib", body)
            code, policy = gw.request("GET", "/vram/policy", port=gw.llama_port, timeout=10)
            self.assertEqual((code, policy["buffer_mib"], policy["target_mib"]), (200, 256, 1792))   # the server's own default (VRAMLOGIC3: 256)
            self.assertNotIn("vram_governor_compat", gw.full_log())

    def test_a_config_value_reaches_the_flags_and_the_policy(self):
        with GatewayHarness(guardian={**GUARDIAN, "vramBufferMiB": 384, "vramPollFastMs": 40}, model={"extraArgs": HOOKS}) as gw:
            gw.start()
            self.assertEqual(gw.chat("hello")[0], 200, gw.log()[-1500:])
            self.assertTrue(wait_for(lambda: "vram_governor_fixed_learned" in gw.full_log()), gw.log()[-1500:])
            args = self.fake_args(gw)
            self.assertIn("--vram-buffer-mib", args)
            self.assertEqual(args[args.index("--vram-buffer-mib") + 1], "384")
            self.assertEqual(args[args.index("--vram-poll-fast-ms") + 1], "40")
            posts = self.policy_posts(gw)
            self.assertTrue(posts and posts[0]["buffer_mib"] == 384 and posts[0]["poll_fast_ms"] == 40, posts)
            _code, policy = gw.request("GET", "/vram/policy", port=gw.llama_port, timeout=10)
            self.assertEqual((policy["buffer_mib"], policy["target_mib"]), (384, 1920))

    def test_a_server_from_before_vramlogic2_still_gets_the_rest_of_the_policy(self):
        model = {"extraArgs": HOOKS + ["--vram-legacy-api"]}
        with GatewayHarness(guardian={**GUARDIAN, "vramBufferMiB": 384}, model=model) as gw:
            gw.start()
            self.assertEqual(gw.chat("hello")[0], 200, gw.log()[-1500:])
            self.assertTrue(wait_for(lambda: "vram_governor_fixed_learned" in gw.full_log()), gw.log()[-1500:])
            self.assertIn("vram_governor_compat", gw.full_log())
            self.assertNotIn("vram_governor_unavailable", gw.full_log())
            posts = self.policy_posts(gw)
            self.assertGreaterEqual(len(posts), 2)
            self.assertIn("buffer_mib", posts[0])
            self.assertNotIn("buffer_mib", posts[1])
            self.assertEqual(posts[1]["reserve_mib"], 1536)



class PauseAfterUnit(unittest.TestCase):
    """VRAMLOGIC2: the stall pause in ms, 0.25 s by default, so it comes before a 1 s stand watchdog."""

    def test_config(self):
        self.assertEqual(vram_budget.check_policy_config({})[1]["pause_after_ms"], 250)
        self.assertEqual(vram_budget.check_policy_config({"vramPauseAfterSeconds": 0.5})[1]["pause_after_ms"], 500)
        self.assertEqual(vram_budget.check_policy_config({"vramPauseAfterSeconds": 2})[1]["pause_after_ms"], 2000)
        self.assertEqual(vram_budget.check_policy_config({"vramPauseAfterSeconds": 0})[1]["pause_after_ms"], 0)
        for bad in (-0.1, 601, "0.5", True, None):
            with self.assertRaises(vram_budget.BudgetError, msg=repr(bad)):
                vram_budget.check_policy_config({"vramPauseAfterSeconds": bad})

    def test_an_older_server_gets_whole_seconds_rounded_up(self):
        values = {"reserve_mib": 1536, "buffer_mib": 512, "poll_fast_ms": 50, "pause_after_ms": 250}
        self.assertEqual(vram_budget.older_policy(values), {"reserve_mib": 1536, "pause_after_seconds": 1})
        self.assertEqual(vram_budget.older_policy({"pause_after_ms": 0})["pause_after_seconds"], 0)
        self.assertEqual(vram_budget.older_policy({"pause_after_ms": 2500})["pause_after_seconds"], 3)
        self.assertNotIn("pause_after_seconds", vram_budget.older_policy({"reserve_mib": 1536}))

    def starved(self, pause_after_ms, legacy=False):
        g = FakeVram(warmup_seconds=0.2, policy={"grow_seconds": 0, "pause_floor_mib": 0})
        g.legacy_api = legacy
        if pause_after_ms is not None:
            g.policy["pause_after_ms"] = pause_after_ms
        g.tick(0.4)
        g.set_desk(g.card_free() + g.cache_mib() - 1300)       # free 1300 with an empty cache: 236 below the floor
        return g

    def test_the_fake_pauses_after_the_stall_time_and_not_with_zero(self):
        g = self.starved(250)
        g.tick(0.2)                                             # the poll that trims to zero and starts the stall
        g.tick(0.2)
        self.assertNotEqual(g.state, "PAUSED")                  # 0.2 s < 0.25 s
        g.tick(0.2)
        self.assertEqual((g.state, g.exit_code), ("PAUSED", 75))
        self.assertIn("short_stall", g.reason)
        off = self.starved(0)
        off.tick(10)
        self.assertEqual((off.state, off.counters["pauses"]), ("MIN", 0))

    def test_within_the_slack_is_not_a_stall(self):
        g = self.starved(250)
        g.set_desk(g.card_free() + g.cache_mib() - (1536 - 30))   # 30 MiB below the floor
        g.tick(5)
        self.assertEqual(g.counters["pauses"], 0)

    def test_policy_fields_and_the_legacy_server(self):
        g = FakeVram()
        self.assertEqual(g.post_policy({"pause_after_seconds": 0.5})[1]["pause_after_ms"], 500)
        self.assertEqual(g.post_policy({"pause_after_ms": 300, "pause_after_seconds": 9})[1]["pause_after_ms"], 300)
        self.assertEqual(g.get_policy()["pause_after_seconds"], 0.3)
        g.legacy_api = True
        self.assertEqual(g.post_policy({"pause_after_ms": 300})[0], 400)
        self.assertEqual(g.post_policy({"pause_after_seconds": 0.5})[0], 400)
        code, body = g.post_policy({"pause_after_seconds": 1})
        self.assertEqual((code, body["pause_after_seconds"]), (200, 1))


class PauseAfterEndToEnd(unittest.TestCase):
    def policy_posts(self, gw):
        return [e["body"] for e in gw.fake_log() if e.get("method") == "POST" and e.get("path") == "/vram/policy"]

    def fake_args(self, gw):
        code, body = gw.request("GET", "/__fake/args", port=gw.llama_port, timeout=10)
        return body if code == 200 else []

    def starve(self, gw):
        gw.request("POST", "/vram/simulate", {"free_delta_mib": -6000, "seconds": 60}, port=gw.llama_port, timeout=10)

    def test_the_default_stall_pause_reaches_the_server_and_fires_before_a_second(self):
        with GatewayHarness(guardian=STALL, model={"extraArgs": HOOKS}) as gw:
            gw.start()
            self.assertEqual(gw.chat("hello")[0], 200, gw.log()[-1500:])
            self.assertTrue(wait_for(lambda: self.policy_posts(gw)), gw.log()[-1500:])
            self.assertEqual(self.policy_posts(gw)[0]["pause_after_ms"], 250)
            self.assertFalse([a for a in self.fake_args(gw) if a.startswith("--vram-pause-after")])   # never a flag
            self.starve(gw)
            self.assertTrue(wait_for(lambda: "PAUSED" in gw.full_log(), timeout=15), gw.log()[-2000:])
            code, snap = gw.request("GET", "/vram", port=gw.llama_port, timeout=10)
            if code == 200:                                   # the fake process may already be gone after exit 75
                self.assertIn("short_stall", snap["reason"])

    def test_zero_switches_the_stall_pause_off(self):
        with GatewayHarness(guardian={**STALL, "vramPauseAfterSeconds": 0}, model={"extraArgs": HOOKS}) as gw:
            gw.start()
            self.assertEqual(gw.chat("hello")[0], 200, gw.log()[-1500:])
            self.assertTrue(wait_for(lambda: self.policy_posts(gw)), gw.log()[-1500:])
            self.assertEqual(self.policy_posts(gw)[0]["pause_after_ms"], 0)
            self.starve(gw)
            self.assertTrue(wait_for(lambda: "MIN" in gw.full_log(), timeout=15), gw.log()[-2000:])
            time.sleep(4)
            self.assertNotIn("PAUSED", gw.full_log())

    def test_an_older_server_gets_one_second(self):
        with GatewayHarness(guardian=STALL, model={"extraArgs": HOOKS + ["--vram-legacy-api"]}) as gw:
            gw.start()
            self.assertEqual(gw.chat("hello")[0], 200, gw.log()[-1500:])
            self.assertTrue(wait_for(lambda: len(self.policy_posts(gw)) >= 2), gw.log()[-1500:])
            posts = self.policy_posts(gw)
            self.assertEqual(posts[0]["pause_after_ms"], 250)
            self.assertNotIn("pause_after_ms", posts[1])
            self.assertEqual(posts[1]["pause_after_seconds"], 1)
            self.assertIn("vram_governor_compat", gw.full_log())


if __name__ == "__main__":
    unittest.main()
