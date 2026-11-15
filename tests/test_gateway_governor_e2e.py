"""End-to-end governor scenarios: the real gateway against the fake llama-server with the VRAM governor (VRAMLOGIC1h).

The fake serves the governor contract (reports/vramlogic1/API.md) and advances on wall-clock time; POST /vram/simulate
(needs --vram-test-hooks, passed as an extra model argument) moves the desk load. Nothing touches the GPU or the live
gateway: the harness uses a temporary folder and its own ports.
"""

import json
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
sys.path.insert(0, str(HERE))

from gateway_harness import GatewayHarness  # noqa: E402
from vram_mode import legacy_only


HOOKS = {"extraArgs": ["--vram-test-hooks"]}


def wait_for(predicate, timeout=30.0, step=0.5):
    end = time.time() + timeout
    while time.time() < end:
        value = predicate()
        if value:
            return value
        time.sleep(step)
    return None


def fake_log(gw):
    code, body = gw.request("GET", "/__fake/log", port=gw.llama_port, timeout=10)
    return body if code == 200 else []


def fake_args(gw):
    code, body = gw.request("GET", "/__fake/args", port=gw.llama_port, timeout=10)
    return body if code == 200 else []


def gateway_events(gw):
    return gw.log()


class GovernorStartupTests(unittest.TestCase):
    def test_governor_pushes_policy_and_learns_fixed_part(self):
        with GatewayHarness(guardian={"vramPolicy": "governor"}, model=HOOKS) as gw:
            gw.start()
            code, _ = gw.chat("hello")
            self.assertEqual(code, 200)
            pushed = wait_for(lambda: next((e for e in fake_log(gw)
                                             if e.get("method") == "POST" and e.get("path") == "/vram/policy"), None))
            self.assertIsNotNone(pushed, "the gateway never pushed the policy")
            self.assertEqual(pushed["body"]["reserve_mib"], 1536)
            self.assertEqual(pushed["body"]["hysteresis_mib"], 256)
            learned = wait_for(lambda: "vram_governor_fixed_learned" in gateway_events(gw))
            self.assertTrue(learned, gateway_events(gw)[-2000:])

    def test_governor_server_gets_governor_flags_and_empty_cache(self):
        with GatewayHarness(guardian={"vramPolicy": "governor"}, model=HOOKS) as gw:
            gw.start()
            code, _ = gw.chat("hello")
            self.assertEqual(code, 200)
            args = wait_for(lambda: fake_args(gw)) or []
            self.assertIn("--vram-reserve-mib", args)
            self.assertEqual(args[args.index("--vram-reserve-mib") + 1], "1536")
            self.assertIn("--moe-cache-chunk-slots", args)
            self.assertNotIn("--moe-cache-live-resize", args)
            slots = [args[i + 1] for i, a in enumerate(args) if a == "--moe-cache-slots"]
            self.assertEqual(slots, ["0"], "A2PORT1: under the governor the command line says --moe-cache-slots once, with 0")
            self.assertIn("--moe-cache-profile", args)


class RunningServerTests(unittest.TestCase):
    def test_second_request_on_a_running_server_is_admitted_without_a_new_load(self):
        # VRAMLOGIC1i (Sonnet's GPU finding): with the server already running, admission was re-estimated as a fresh load
        # (free with the server minus its whole fixed part = refused). The governor owns the cache: the second request
        # must go straight to the running server.
        guardian = {"vramPolicy": "governor", "vramCardReserveMiB": 1536}
        with GatewayHarness(guardian=guardian, model=HOOKS) as gw:
            gw.start()
            self.assertEqual(gw.chat("first")[0], 200)
            self.assertTrue(wait_for(lambda: "vram_governor_fixed_learned" in gateway_events(gw)))
            self.assertEqual(gw.chat("second")[0], 200, gateway_events(gw)[-1500:])
            self.assertEqual(gw.chat("third")[0], 200)
            starts = [e for e in gateway_events(gw).splitlines() if "Starting llama-server" in e]
            self.assertEqual(len(starts), 1, "a running server was loaded again")


class PauseDirectoryTests(unittest.TestCase):
    """VRAMLOGIC1 (Sonnet's E2E): the server writes the pause files to the --slot-save-path the gateway passed it. With vramGuard the
    server's directory is the RAM slot folder, not the gateway's slot folder: the continuation must read it from there."""

    def test_pause_files_are_read_from_the_directory_given_to_the_server(self):
        guardian = {"vramPolicy": "governor", "vramGuard": True}
        with GatewayHarness(guardian=guardian, behaviour={"n_tokens": 6, "token_delay_ms": 5, "text": "w", "pause_after_tokens": 2},
                            model=HOOKS) as gw:
            gw.start()
            text, _reasoning, errors = PauseContinuationTests.stream_text(PauseContinuationTests(), gw,
                                                                          {"messages": [{"role": "user", "content": "x"}], "stream": True,
                                                                           "max_tokens": 16})
            self.assertEqual(errors, [], gateway_events(gw)[-1500:])
            self.assertEqual(text, "w " * 6, gateway_events(gw)[-1500:])
            self.assertIn("vram_pause_resumed", gateway_events(gw))


class LayoutTooBigTests(unittest.TestCase):
    """VRAMLOGIC1i (Sonnet's GPU finding): a layout that exits 75 during warm-up does not keep the reserve. The gateway remembers
    it and starts the lite layout, instead of answering 503."""

    def test_exit_75_on_the_short_layout_starts_lite_and_the_next_request_stays_on_lite(self):
        guardian = {"vramPolicy": "governor", "vramCardReserveMiB": 1536}
        model = {**LITE_MODEL, "extraArgs": ["--vram-test-hooks"]}
        behaviour = {"exit75_when_ncmoe_below": 28}      # the short layout (ncmoe 26) exits 75; lite (ncmoe 30) starts
        with GatewayHarness(guardian=guardian, behaviour=behaviour, model=model) as gw:
            gw.start()
            self.assertEqual(gw.chat("first")[0], 200, gateway_events(gw)[-1500:])
            self.assertEqual(gw.status()["layout"], "lite")
            self.assertIn("vram_layout_too_big", gateway_events(gw))
            self.assertEqual(gw.chat("second")[0], 200)
            starts = [e for e in gateway_events(gw).splitlines() if "Starting llama-server" in e]
            self.assertEqual(len(starts), 2, "the marked layout was tried again: " + str(len(starts)) + " starts")


class LegacyTests(unittest.TestCase):
    @legacy_only("this scenario is about the legacy default; under the governor suite the policy is forced")
    def test_legacy_sends_no_governor_traffic_and_no_governor_flags(self):
        with GatewayHarness(model=HOOKS) as gw:
            gw.start()
            code, _ = gw.chat("hello")
            self.assertEqual(code, 200)
            time.sleep(2.5)
            paths = [e.get("path", "") for e in fake_log(gw)]
            self.assertFalse([p for p in paths if p.startswith("/vram")], paths)
            args = fake_args(gw)
            self.assertNotIn("--vram-reserve-mib", args)
            self.assertNotIn("vram_governor_fixed_learned", gateway_events(gw))


class DeskLoadTests(unittest.TestCase):
    def desk_load(self, gw, mib, seconds=60):
        gw.request("POST", "/vram/simulate", {"free_delta_mib": -mib, "seconds": seconds}, port=gw.llama_port, timeout=10)

    def test_desk_load_moves_the_governor_into_min(self):
        # the pause floor and the stall pause are off here, so a load that leaves no free VRAM keeps the governor in MIN
        guardian = {"vramPolicy": "governor", "vramLiteAfterSeconds": 1, "vramPauseFloorMiB": 0, "vramPauseAfterSeconds": 0}
        with GatewayHarness(guardian=guardian, model=HOOKS) as gw:
            gw.start()
            code, _ = gw.chat("hello")
            self.assertEqual(code, 200)
            self.assertTrue(wait_for(lambda: "vram_governor_fixed_learned" in gateway_events(gw)))
            self.desk_load(gw, 6000)
            self.assertTrue(wait_for(lambda: "FULL->MIN" in gateway_events(gw) or "MIN" in gateway_events(gw)),
                            gateway_events(gw)[-2000:])

    def test_desk_load_bigger_than_the_floor_pauses_with_a_checkpoint(self):
        # the governor pauses itself (pause_floor) and exits 75; the gateway logs the state. Continuing the response is a
        # separate step (VRAMLOGIC1h scenario 3) and is not checked here.
        guardian = {"vramPolicy": "governor"}
        with GatewayHarness(guardian=guardian, model=HOOKS) as gw:
            gw.start()
            code, _ = gw.chat("hello")
            self.assertEqual(code, 200)
            self.assertTrue(wait_for(lambda: "vram_governor_fixed_learned" in gateway_events(gw)))
            self.desk_load(gw, 6000)
            self.assertTrue(wait_for(lambda: "PAUSED" in gateway_events(gw)), gateway_events(gw)[-2000:])


LITE_MODEL = {**HOOKS, "liteProfile": {"nCpuMoe": 30, "moeCacheSlots": 2, "vramFreeMinGBToLoad": 11.5}}


class LiteUnderGovernorTests(unittest.TestCase):
    def test_min_longer_than_t_lite_switches_to_lite_and_back_after_relief(self):
        guardian = {"vramPolicy": "governor", "vramCardReserveMiB": 1536, "vramLiteAfterSeconds": 1,
                    "vramLiteReturnSeconds": 1, "vramPauseFloorMiB": 0, "vramPauseAfterSeconds": 0}
        with GatewayHarness(guardian=guardian, model=LITE_MODEL) as gw:
            gw.start()
            code, _ = gw.chat("hello")
            self.assertEqual(code, 200)
            self.assertEqual(gw.status()["layout"], "short")
            self.assertTrue(wait_for(lambda: "vram_governor_fixed_learned" in gateway_events(gw)))
            # the desk takes the free part for a while: the governor goes to MIN and stays there
            gw.request("POST", "/vram/simulate", {"free_delta_mib": -6000, "seconds": 12}, port=gw.llama_port, timeout=10)
            self.assertTrue(wait_for(lambda: "MIN" in gateway_events(gw)), gateway_events(gw)[-1500:])
            time.sleep(1.5)                      # longer than T_lite (1 s) in MIN
            code, _ = gw.chat("another")
            self.assertEqual(code, 200)
            self.assertEqual(gw.status()["layout"], "lite")
            # the desk gives the VRAM back: out of MIN for longer than T_lite, the normal layout comes back
            self.assertTrue(wait_for(lambda: "FULL" in gateway_events(gw) or "GROW" in gateway_events(gw), timeout=30))
            time.sleep(1.5)
            code, _ = gw.chat("third")
            self.assertEqual(code, 200)
            self.assertEqual(gw.status()["layout"], "short")


class SwitchModelsUnderGovernorTests(unittest.TestCase):
    def test_model_a_is_stopped_before_model_b_loads(self):
        with GatewayHarness(guardian={"vramPolicy": "governor"}, model=HOOKS) as gw:
            gw.start()
            code, _ = gw.request("POST", "/v1/chat/completions",
                                 {"model": "tiel-coder", "messages": [{"role": "user", "content": "a"}], "max_tokens": 8})
            self.assertEqual(code, 200)
            code, _ = gw.request("POST", "/v1/chat/completions",
                                 {"model": "tiel-coder-apex", "messages": [{"role": "user", "content": "b"}], "max_tokens": 8})
            self.assertEqual(code, 200)
            self.assertEqual(gw.status()["model"], "tiel-coder-apex")
            lines = gw.full_log().splitlines()
            starts = [i for i, l in enumerate(lines) if "Starting llama-server" in l]
            stops = [i for i, l in enumerate(lines) if "Stopping llama-server" in l]
            self.assertGreaterEqual(len(starts), 2, gw.full_log()[-1500:])
            self.assertTrue(any(starts[0] < i < starts[1] for i in stops),
                            "model A was not stopped between the two loads:\n" + "\n".join(lines[starts[0]:starts[1] + 1])[-1500:])


class PauseContinuationTests(unittest.TestCase):
    """VRAMLOGIC1i: a pause mid-answer is invisible to the client: the answer continues on the restored slot."""

    def stream_text(self, gw, body):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", gw.gw_port, timeout=60)
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
        raw = conn.getresponse().read().decode("utf-8", "replace")
        conn.close()
        content, reasoning, errors = [], [], []
        for line in raw.splitlines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            value = json.loads(line[5:].strip())
            if "error" in value:
                errors.append(value)
                continue
            for choice in value.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
        return "".join(content), "".join(reasoning), errors

    def run_paused(self, behaviour):
        """Same answer with and without a pause: the client's pieces must be equal."""
        with GatewayHarness(guardian={"vramPolicy": "governor"}, behaviour=behaviour, model=HOOKS) as gw:
            gw.start()
            text, reasoning, errors = self.stream_text(gw, {"messages": [{"role": "user", "content": "x"}], "stream": True,
                                                              "max_tokens": 16})
            return text, reasoning, errors, gateway_events(gw)

    def test_pause_in_content_is_continued_without_a_visible_error(self):
        text, reasoning, errors, events = self.run_paused({"n_tokens": 6, "token_delay_ms": 5, "text": "w",
                                                           "pause_after_tokens": 2})
        self.assertEqual(errors, [], "the pause frame reached the client")
        self.assertEqual(text, "w " * 6, events[-1500:])
        self.assertEqual(reasoning, "")
        self.assertIn("vram_pause_resumed", events)

    def test_pause_in_reasoning_is_continued_in_the_same_phase(self):
        text, reasoning, errors, events = self.run_paused({"n_tokens": 4, "reasoning_tokens": 3, "token_delay_ms": 5,
                                                           "text": "w", "pause_after_tokens": 1})
        self.assertEqual(errors, [], "the pause frame reached the client")
        self.assertEqual(reasoning, "think " * 3, events[-1500:])
        self.assertEqual(text, "w " * 4)
        self.assertIn("vram_pause_resumed", events)


if __name__ == "__main__":
    unittest.main()
