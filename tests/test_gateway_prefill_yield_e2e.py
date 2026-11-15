"""Prefill yield through the real gateway handler: the fake llama server records every /moe-cache/slots call."""
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
from gateway_harness import GatewayHarness  # noqa: E402
from vram_mode import legacy_only


YIELD = {"prefillCacheYield": True, "prefillCacheYieldMinTokens": 1000, "vramGuard": True, "slotSaveCheckpoints": 1}


def chat(chars):
    return {"messages": [{"role": "user", "content": "x" * chars}]}


def resize_calls(log):
    return [e["body"]["slots"] for e in log if e.get("path") == "/moe-cache/slots"]


class PrefillYieldEndToEnd(unittest.TestCase):
    @legacy_only("the prefill yield (prefillCacheYield) is not run under the governor")
    def test_large_new_prefill_is_shrunk_then_restored_around_the_answer(self):
        with GatewayHarness(behaviour={"n_tokens": 2}, guardian=YIELD) as gw:
            status, _ = gw.request("POST", "/v1/chat/completions", chat(8000))
            self.assertEqual(status, 200)
            # the restore runs in the handler's finally, after the answer has been sent: wait for it instead of racing it
            deadline = time.monotonic() + 10
            calls = resize_calls(gw.fake_log())
            while len(calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.1)
                calls = resize_calls(gw.fake_log())
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], 0)
        self.assertGreater(calls[1], 0)

    def test_small_request_is_served_without_any_resize(self):
        with GatewayHarness(behaviour={"n_tokens": 2}, guardian=YIELD) as gw:
            status, _ = gw.request("POST", "/v1/chat/completions", chat(200))
            self.assertEqual(status, 200)
            calls = resize_calls(gw.fake_log())
        self.assertEqual(calls, [])

    @legacy_only("the prefill yield (prefillCacheYield) is not run under the governor")
    def test_refused_shrink_still_serves_the_request(self):
        with GatewayHarness(behaviour={"n_tokens": 2, "resize_status": 500}, guardian=YIELD) as gw:
            status, _ = gw.request("POST", "/v1/chat/completions", chat(8000))
            self.assertEqual(status, 200)
            calls = resize_calls(gw.fake_log())
        self.assertEqual(calls, [0])  # the shrink was refused, so there is nothing to restore


    @legacy_only("the prefill yield (prefillCacheYield) is not run under the governor")
    def test_client_abort_mid_answer_still_restores_the_cache(self):
        import json
        import socket
        with GatewayHarness(behaviour={"n_tokens": 400, "token_delay_ms": 20}, guardian=YIELD) as gw:
            body = json.dumps(chat(8000)).encode()
            sock = socket.create_connection(("127.0.0.1", gw.gw_port), timeout=10)
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                         + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            sock.recv(256)  # the answer has started
            sock.close()    # the client leaves before the answer ends
            deadline = time.monotonic() + 15
            calls = resize_calls(gw.fake_log())
            while len(calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.1)
                calls = resize_calls(gw.fake_log())
        self.assertEqual(calls[:1], [0])
        self.assertEqual(len(calls), 2, "the cache must come back even when the client went away")
        self.assertGreater(calls[1], 0)


if __name__ == "__main__":
    unittest.main()
