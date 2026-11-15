"""FABLE1: the calibrated size estimate for the layout switch (gateway/layout_estimate.py) and its use by the gateway.

Unit part: characters of messages and tools, ASCII against Cyrillic, the answer reserve, the usage parser, the feedback.
Gateway part (fake server): with longContextEstimate "calibrated" a request that the legacy estimate sends to the long
layout (big tools, max_tokens 16384) stays short, and the server's own token count keeps an appended request short."""
import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))
from gateway_harness import FAKE_BIN, GatewayHarness  # noqa: E402
import layout_estimate as le  # noqa: E402

try:
    from tests.test_gateway_layout import BASE_MODEL, GUARD, LONG_PROFILE, starts  # noqa: E402
except ImportError:  # run from the tests folder
    from test_gateway_layout import BASE_MODEL, GUARD, LONG_PROFILE, starts  # noqa: E402

DUMMY = Path("/dev/shm") / "layout-test-model.gguf"


class Unit(unittest.TestCase):
    def test_counts_messages_and_tools_ascii_and_other(self):
        body = json.dumps({"messages": [{"role": "system", "content": "abc"}, {"role": "user", "content": [{"type": "text", "text": "привет"}]}],
                           "tools": [{"type": "function", "function": {"name": "read", "description": "xy", "parameters": {"a": "z"}}}]}).encode()
        ascii_chars, other = le.count_chars(body)
        self.assertEqual(other, 6)
        self.assertEqual(ascii_chars, 3 + len("system") + len("user") + len("text") + len("function") + len("read") + 2 + 1)

    def test_not_a_chat_body_counts_as_empty(self):
        self.assertEqual(le.count_chars(b"not json"), (0, 0))
        self.assertEqual(le.count_chars(b"[1, 2]"), (0, 0))

    def test_base_estimate_uses_one_calibration_per_script(self):
        self.assertAlmostEqual(le.base_estimate(3200, 0), 1000)
        self.assertAlmostEqual(le.base_estimate(0, 2500), 1000)

    def test_answer_reserve_is_the_client_limit_capped_by_the_config(self):
        self.assertEqual(le.answer_reserve(16384, None), 16384)
        self.assertEqual(le.answer_reserve(16384, 8192), 8192)
        self.assertEqual(le.answer_reserve(1024, 8192), 1024)
        self.assertEqual(le.answer_reserve(0, 8192), 8192)
        self.assertEqual(le.answer_reserve(0, None), 0)

    def test_usage_is_read_from_a_stream_chunk_and_from_a_plain_answer(self):
        chunk = (b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":12000,"completion_tokens":700,"total_tokens":12700},'
                 b'"timings":{"prompt_n":300,"cache_n":11700,"predicted_n":700}}\n\ndata: [DONE]\n\n')
        self.assertEqual(le.parse_usage(chunk), {"prompt": 12000, "completion": 700})
        plain = json.dumps({"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 5}}).encode()
        self.assertEqual(le.parse_usage(plain), {"prompt": 50, "completion": 5})
        timings_only = b'{"timings":{"prompt_n":40,"cache_n":60,"predicted_n":9}}'
        self.assertEqual(le.parse_usage(timings_only), {"prompt": 100, "completion": 9})
        self.assertIsNone(le.parse_usage(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'))
        self.assertIsNone(le.parse_usage(b'{"usage":{"prompt_tokens":0,"completion_tokens":0}}'))

    def test_feedback_estimates_an_appended_request_from_the_known_point(self):
        fb = le.TokenFeedback()
        self.assertEqual(fb.estimate("s1", 38000, 0)[1], "chars")
        fb.commit("s1", 38000, {"prompt": 9000, "completion": 400})
        tokens, how = fb.estimate("s1", 38000 + 3500, 0)
        self.assertEqual(how, "feedback")
        self.assertAlmostEqual(tokens, 9400 + 1000 + le.TEMPLATE_OVERHEAD_TOKENS)
        # a shorter request is another conversation: characters alone
        self.assertEqual(fb.estimate("s1", 20000, 0)[1], "chars")
        # another session key: characters alone
        self.assertEqual(fb.estimate("s2", 50000, 0)[1], "chars")
        self.assertEqual(fb.snapshot()["how"], "chars")

    def test_feedback_ignores_empty_usage_and_keeps_few_sessions(self):
        fb = le.TokenFeedback()
        fb.commit("s1", 100, None)
        self.assertEqual(fb.known, {})
        for i in range(70):
            fb.commit(f"s{i}", 100, {"prompt": 1, "completion": 1})
        self.assertLessEqual(len(fb.known), 64)


def launch(estimate=None, reserve=None, threshold=28000):
    model = dict(BASE_MODEL)
    model["longContextProfile"] = dict(LONG_PROFILE)
    model["longContextSwitchTokens"] = threshold
    model["longContextHoldSeconds"] = 0
    if estimate:
        model["longContextEstimate"] = estimate
    if reserve:
        model["longContextAnswerReserveTokens"] = reserve
    return {"exePath": str(FAKE_BIN), "defaultModel": "tiel-coder", "cacheRamMB": 0, "models": {"tiel-coder": model}}


TOOLS = [{"type": "function", "function": {"name": f"tool{i}", "description": "d" * 900, "parameters": {"type": "object"}}} for i in range(40)]
SYSTEM = "You are an agent. " * 1200     # about 22 000 ASCII characters, like the live system prompt


def chat(h, conversation, session="", max_tokens=16384):
    import http.client
    msgs = [{"role": "system", "content": SYSTEM}] + conversation
    body = json.dumps({"model": "tiel-coder", "messages": msgs, "tools": TOOLS, "max_tokens": max_tokens, "stream": True}).encode()
    headers = {"Content-Type": "application/json"}
    if session:
        headers["X-Shura-Session"] = session
    conn = http.client.HTTPConnection("127.0.0.1", h.gw_port, timeout=60)
    conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


class Gateway(unittest.TestCase):
    def run_gateway(self, **kw):
        h = GatewayHarness(guardian=dict(GUARD), launch=launch(**kw), behaviour={"n_tokens": 3})
        h.start(timeout=60)
        self.addCleanup(h.stop)
        return h

    def status(self, h):
        import urllib.request
        return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{h.gw_port}/guardian/status", timeout=10).read())

    def test_legacy_estimate_sends_a_live_sized_first_request_to_the_long_layout(self):
        # 22 000 chars / 2.5 + 16 384 = 25 184 is below 28 000, but one tool output of 7 500 characters takes it over
        h = self.run_gateway()
        self.assertEqual(chat(h, [{"role": "user", "content": "read the file"}])[0], 200)
        self.assertEqual(chat(h, [{"role": "user", "content": "read the file"}, {"role": "assistant", "content": "ok"},
                                  {"role": "tool", "content": "x" * 7500}])[0], 200)
        self.assertEqual(starts(h), [(32768, True), (200000, False)])
        self.assertEqual(self.status(h)["layout_estimate"]["how"], "legacy")

    def test_calibrated_estimate_keeps_the_same_conversation_short(self):
        # tools 40 x ~950 = 38 000 chars and the system 22 000: about 18 750 tokens at 3.2 chars per token (CTX1), with
        # 8 192 reserved and a 1 000-character tool output: 27 250 < 28 000, short. (Legacy would say long: 41 000 / 2.5.)
        h = self.run_gateway(estimate="calibrated", reserve=8192)
        self.assertEqual(chat(h, [{"role": "user", "content": "read the file"}])[0], 200)
        self.assertEqual(chat(h, [{"role": "user", "content": "read the file"}, {"role": "assistant", "content": "ok"},
                                  {"role": "tool", "content": "x" * 1000}])[0], 200)
        self.assertEqual(starts(h), [(32768, True)])
        est = self.status(h)["layout_estimate"]
        self.assertEqual(est["reserve"], 8192)
        self.assertIn(est["how"], ("chars", "feedback"))

    def test_calibrated_estimate_counts_the_tools_and_switches_when_the_window_is_really_full(self):
        h = self.run_gateway(estimate="calibrated", reserve=8192)
        self.assertEqual(chat(h, [{"role": "user", "content": "read the file"}])[0], 200)
        # no session header: characters alone; 60 000 characters of tool output: 15 800 + 15 800 + 8 192 > 28 000
        self.assertEqual(chat(h, [{"role": "user", "content": "read the file"}, {"role": "assistant", "content": "ok"},
                                  {"role": "tool", "content": "y" * 60000}], session="")[0], 200)
        self.assertEqual(starts(h), [(32768, True), (200000, False)])

    def test_server_token_count_is_the_base_for_an_appended_request(self):
        # the fake reports usage prompt_tokens 100 and completion 20 whatever the prompt: with the feedback the appended request is
        # 120 + new text, so even a request that the character estimate alone would send long stays short
        h = self.run_gateway(estimate="calibrated", reserve=8192)
        sid = "11111111-2222-4333-8444-555555555555"
        first = [{"role": "user", "content": "read the file"}]
        self.assertEqual(chat(h, first, session=sid)[0], 200)
        self.assertEqual(self.status(h)["layout_estimate"]["how"], "chars")
        grown = first + [{"role": "assistant", "content": "ok"}, {"role": "tool", "content": "z" * 60000}]
        self.assertEqual(chat(h, grown, session=sid)[0], 200)
        est = self.status(h)["layout_estimate"]
        self.assertEqual(est["how"], "feedback")
        self.assertLess(est["tokens"], 20000)
        self.assertEqual(starts(h), [(32768, True)])
        # another session: no feedback point, characters alone (and the same 60 000 characters are long)
        self.assertEqual(chat(h, grown, session="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")[0], 200)
        self.assertEqual(self.status(h)["layout_estimate"]["how"], "chars")
        self.assertEqual(starts(h), [(32768, True), (200000, False)])

    def test_config_validation(self):
        import subprocess, os
        for bad, text in ((("estimate", "magic"), "longContextEstimate"), (("reserve", 40000), "longContextAnswerReserveTokens")):
            h = GatewayHarness(guardian=dict(GUARD), launch=launch(**{bad[0]: bad[1]}))
            with self.assertRaises(RuntimeError) as cm:
                h.start(timeout=20)
            h.stop()
            self.assertIn(text, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
