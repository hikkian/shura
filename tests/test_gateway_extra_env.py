"""AMD2a: the model's extraEnv reaches the llama-server process; without it nothing changes. Fake llama-server, no GPU."""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
from gateway_harness import GatewayHarness  # noqa: E402

MODERN = {"vramGuard": True, "slotSaveCheckpoints": 1}  # without these the harness runs the legacy gateway, which has no extraEnv
CHAT = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
SECRET_LIKE = "value-that-must-not-reach-the-log-42"


def fake_env(gw):
    return gw.request("GET", "/__fake/env", port=gw.llama_port, timeout=5)[1]


class ExtraEnv(unittest.TestCase):
    def test_extra_env_reaches_the_llama_server_process_and_only_names_reach_the_log(self):
        with GatewayHarness(behaviour={"n_tokens": 2}, guardian=MODERN, model={"extraEnv": {"SHURA_TEST_FLAG": SECRET_LIKE}}) as gw:
            status, _ = gw.request("POST", "/v1/chat/completions", CHAT)
            self.assertEqual(status, 200)
            env = fake_env(gw)
            self.assertEqual(env.get("SHURA_TEST_FLAG"), SECRET_LIKE)
            log = gw.log()
            self.assertIn("extra environment (names only)", log)
            self.assertIn("SHURA_TEST_FLAG", log)
            self.assertNotIn(SECRET_LIKE, log)

    def test_without_extra_env_the_process_gets_no_test_variable(self):
        with GatewayHarness(behaviour={"n_tokens": 2}, guardian=MODERN) as gw:
            status, _ = gw.request("POST", "/v1/chat/completions", CHAT)
            self.assertEqual(status, 200)
            self.assertEqual({k for k in fake_env(gw) if k.startswith("SHURA_TEST_")}, set())

    def test_malformed_extra_env_stops_the_gateway_at_startup(self):
        with self.assertRaises(Exception):
            with GatewayHarness(behaviour={"n_tokens": 2}, guardian=MODERN, model={"extraEnv": {"SHURA_TEST_FLAG": 1}}) as gw:
                gw.request("POST", "/v1/chat/completions", CHAT, timeout=5)
                self.fail("gateway started with a non-string extraEnv value")


if __name__ == "__main__":
    unittest.main()
