"""CACHE3: a chat request that does not fit the model's context window is shortened in its tool results, not sent to a 400."""
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gateway"))

import context_fit  # noqa: E402

SYSTEM = "You are a coding agent. " * 80          # ~2.1k characters, the same in every turn
BIG = "line of tool output with some text\n" * 2000   # ~70k characters


def chat(messages, **extra):
    return json.dumps({"model": "tiel-coder", "messages": messages, "stream": True, **extra}).encode()


def turn_messages(n_tools, tool_chars=None):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "fix the test"}]
    for i in range(n_tools):
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"call{i}", "type": "function", "function": {"name": "read", "arguments": "{}"}}]})
        body = BIG if tool_chars is None else BIG[:tool_chars]
        msgs.append({"role": "tool", "tool_call_id": f"call{i}", "content": body})
    msgs.append({"role": "user", "content": "and now the other test"})
    return msgs


class FitBodyTests(unittest.TestCase):
    def test_fits_already_is_returned_byte_for_byte(self):
        raw = chat(turn_messages(1, tool_chars=500))
        out, info = context_fit.fit_body(raw, 200000, 8192)
        self.assertEqual(out, raw)
        self.assertIsNone(info)

    def test_no_context_size_or_not_json_is_left_alone(self):
        raw = chat(turn_messages(3))
        self.assertEqual(context_fit.fit_body(raw, None)[0], raw)
        self.assertEqual(context_fit.fit_body(b"not json", 1000)[0], b"not json")
        self.assertEqual(context_fit.fit_body(b"[1, 2]", 1000)[0], b"[1, 2]")

    def test_oversize_shortens_tool_results_and_nothing_else(self):
        raw = chat(turn_messages(6))
        before = json.loads(raw)
        out, info = context_fit.fit_body(raw, 30000, 4096)
        self.assertTrue(info["fits"])
        self.assertGreater(info["trimmed"], 0)
        self.assertLessEqual(info["after"], info["budget"])
        after = json.loads(out)
        self.assertEqual(after["messages"][0], before["messages"][0])          # system message untouched
        self.assertEqual(after["messages"][1], before["messages"][1])          # first user message untouched
        self.assertEqual(after["messages"][-1], before["messages"][-1])        # last user message untouched
        for a, b in zip(after["messages"], before["messages"]):
            if a.get("role") == "assistant":
                self.assertEqual(a, b)                                          # tool calls kept as they were
            if a.get("role") == "tool":
                self.assertEqual(a["tool_call_id"], b["tool_call_id"])          # pairing kept
                self.assertEqual(len(a["content"]) < len(b["content"]), True)

    def test_shortened_result_keeps_head_tail_and_says_so(self):
        raw = chat(turn_messages(4))
        out, _ = context_fit.fit_body(raw, 20000, 4096)
        tool = next(m for m in json.loads(out)["messages"] if m.get("role") == "tool" and "removed" in m["content"])
        self.assertIn("characters of this tool result were removed by the gateway", tool["content"])
        self.assertTrue(tool["content"].startswith(BIG[:context_fit.KEEP_HEAD]))
        self.assertTrue(tool["content"].endswith(BIG[-context_fit.KEEP_TAIL:]))

    def test_same_request_gives_the_same_bytes(self):
        raw = chat(turn_messages(6))
        self.assertEqual(context_fit.fit_body(raw, 30000, 4096)[0], context_fit.fit_body(raw, 30000, 4096)[0])

    def test_cannot_fit_by_tool_results_is_sent_unchanged(self):
        huge_user = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "x" * 400000}]
        raw = chat(huge_user)
        out, info = context_fit.fit_body(raw, 30000, 4096)
        self.assertEqual(out, raw)
        self.assertFalse(info["fits"])
        self.assertEqual(info["trimmed"], 0)

    def test_the_40k_case_from_ttft1_fits_under_a_40k_window(self):
        # a 41 670-token request (the TTFT1 failure) against a 40 960 window: after shortening it must fit
        msgs = turn_messages(14)
        raw = chat(msgs)
        est = context_fit.estimate_tokens(json.loads(raw))
        self.assertGreater(est, 40960 - 4096)
        out, info = context_fit.fit_body(raw, 40960, 4096)
        self.assertTrue(info["fits"], info)
        self.assertLessEqual(context_fit.estimate_tokens(json.loads(out)), 40960 - 4096)


class GatewayHookTests(unittest.TestCase):
    """fit_to_context in the gateway: the model's ctxSize decides, the switch can turn it off."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.TemporaryDirectory(prefix="shura-fit-import-", dir="/tmp")
        for name in ("guardian", "model-launch"):
            (Path(cls.root.name) / f"{name}.json").write_text((REPO / "config" / f"{name}.example.json").read_text())
        spec = importlib.util.spec_from_file_location("fit_test_gateway", REPO / "gateway" / "ai_gateway.py")
        cls.gw = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, AI_GATEWAY_CONFIG=cls.root.name, AI_GATEWAY_STATE=str(Path(cls.root.name) / "state")):
            spec.loader.exec_module(cls.gw)

    @classmethod
    def tearDownClass(cls):
        cls.root.cleanup()

    def test_hook_shortens_when_the_model_window_is_small(self):
        model = self.gw.M["defaultModel"]
        raw = chat(turn_messages(6))
        with patch.dict(self.gw.M["models"][model], {"ctxSize": 30000}), \
                patch.object(self.gw, "event") as ev:
            out = self.gw.fit_to_context(raw, "/v1/chat/completions", model, False)
        self.assertNotEqual(out, raw)
        self.assertLessEqual(context_fit.estimate_tokens(json.loads(out)), 30000)
        ev.assert_called_once()
        self.assertEqual(ev.call_args.args[0], "context_fit")

    def test_hook_does_nothing_for_other_paths_or_when_switched_off(self):
        model = self.gw.M["defaultModel"]
        raw = chat(turn_messages(6))
        with patch.dict(self.gw.M["models"][model], {"ctxSize": 30000}):
            self.assertEqual(self.gw.fit_to_context(raw, "/completion", model, False), raw)
            with patch.dict(self.gw.G, {"contextFit": False}):
                self.assertEqual(self.gw.fit_to_context(raw, "/v1/chat/completions", model, False), raw)


LONG = {"ctxSize": 32768, "longContextProfile": {"ctxSize": 200000}, "longContextSwitchTokens": 28000,
        "longContextEstimate": "calibrated", "longContextAnswerReserveTokens": 8192}


class LongLayoutTests(GatewayHookTests):
    """A model with a long layout: the fit counts the same answer room as the layout choice, moves a too-big request to the long
    window, and refuses (honestly) a prompt that fits no window at all. The 2026-10-25 SAGE1 stand sent 18 161-token requests with the
    log line "still too large: sent unchanged" because the fit reserved the full 16 384 tokens and the layout choice only 8 192."""

    def model(self):
        return self.gw.M["defaultModel"]

    def prose(self, chars):
        return chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "word " * (chars // 5)}])

    def test_same_answer_room_as_the_layout_choice_so_no_false_too_large(self):
        raw = self.prose(54000)                                    # ~18k tokens by the careful estimate
        with patch.dict(self.gw.M["models"][self.model()], LONG), patch.object(self.gw, "event") as ev:
            out = self.gw.fit_to_context(raw, "/v1/chat/completions", self.model(), False)
        self.assertEqual(out, raw)                                 # fits the short window with the 8192 answer room
        ev.assert_not_called()
        self.assertFalse(getattr(self.gw.REQUEST_CTX, "force_long", False))

    def test_a_request_that_leaves_room_for_an_answer_stays_short_whatever_max_tokens_says(self):
        # ~18k prompt, the client asks for 16 384 answer tokens: the window leaves ~14k, so it is served as it is (no layout switch)
        raw = json.dumps({"model": "tiel-coder", "max_tokens": 16384, "stream": True, "messages": json.loads(self.prose(54000))["messages"]}).encode()
        with patch.dict(self.gw.M["models"][self.model()], {**LONG, "longContextAnswerReserveTokens": 0}), \
                patch.object(self.gw, "pure_request_layout", return_value="short"), patch.object(self.gw, "event") as ev:
            self.gw.REQUEST_CTX.force_long = False
            out = self.gw.fit_to_context(raw, "/v1/chat/completions", self.model(), False)
        self.assertEqual(out, raw)
        self.assertFalse(self.gw.REQUEST_CTX.force_long)
        self.assertFalse(ev.call_args.kwargs.get("fallback"))      # an information line, not a warning
        self.assertIn("tokens for the answer", ev.call_args.args[1])

    def test_too_big_for_the_short_window_moves_to_the_long_layout(self):
        raw = self.prose(99000)                                    # ~33k tokens by the careful estimate: no room for an answer in 32768
        self.gw.REQUEST_CTX.force_long = False
        with patch.dict(self.gw.M["models"][self.model()], LONG), \
                patch.object(self.gw, "pure_request_layout", return_value="short"), patch.object(self.gw, "event") as ev:
            out = self.gw.fit_to_context(raw, "/v1/chat/completions", self.model(), False)
        self.assertEqual(out, raw)                                 # nothing to shorten: the long window has the room
        self.assertTrue(self.gw.REQUEST_CTX.force_long)
        self.assertIn("long layout", ev.call_args.args[1])
        self.gw.REQUEST_CTX.force_long = False

    def test_choose_request_layout_obeys_the_flag(self):
        raw = self.prose(300)
        with patch.dict(self.gw.M["models"][self.model()], LONG):
            self.gw.REQUEST_CTX.force_long = True
            try:
                self.assertEqual(self.gw.choose_request_layout(self.model(), False, raw, 100, 0.0, ""), "long")
            finally:
                self.gw.REQUEST_CTX.force_long = False
            self.assertEqual(self.gw.choose_request_layout(self.model(), False, raw, 100, 0.0, ""), "short")

    def test_a_prompt_that_fits_no_window_is_refused_with_an_honest_error(self):
        import service_requests as sr
        raw = self.prose(800000)                                   # ~270k tokens, the long window is 200 000
        with patch.dict(self.gw.M["models"][self.model()], LONG), patch.object(self.gw, "event"):
            with self.assertRaises(sr.RequestRejected) as caught:
                self.gw.fit_to_context(raw, "/v1/chat/completions", self.model(), False)
        error = caught.exception.payload["error"]
        self.assertEqual((caught.exception.status, error["type"], error["n_ctx"]), (400, "exceed_context_size_error", 200000))
        self.assertGreater(error["n_prompt_tokens"], 200000)

    def test_without_a_long_layout_a_hopeless_prompt_is_refused_too_and_a_borderline_one_is_not(self):
        import service_requests as sr
        small = {"ctxSize": 30000}
        with patch.dict(self.gw.M["models"][self.model()], small), patch.object(self.gw, "event"):
            with self.assertRaises(sr.RequestRejected):
                self.gw.fit_to_context(self.prose(400000), "/v1/chat/completions", self.model(), False)
            # prompt ~ 28k by the careful estimate and ~26k by the calibrated one: both below the window, only the answer room is short
            out = self.gw.fit_to_context(self.prose(84000), "/v1/chat/completions", self.model(), False)
        self.assertIsInstance(out, bytes)

    def test_the_rejection_is_sent_by_the_wrapper_and_the_handler_is_not_called(self):
        import service_requests as sr

        class Base:
            called = False
            sent = None

            def handle_generation(self, path, body):
                Base.called = True

            def send_json(self, status, payload, retry=None):
                Base.sent = (status, payload)
                return "sent"

        def fit(path, body):
            raise sr.RequestRejected(400, {"error": {"type": "exceed_context_size_error"}})

        wrapped = sr.with_title_answer(Base, lambda: False, fit)
        self.assertEqual(wrapped().handle_generation("/v1/chat/completions", b"{}"), "sent")
        self.assertFalse(Base.called)
        self.assertEqual(Base.sent[0], 400)


class SharedWrapperTests(unittest.TestCase):
    """The fit hook sits in the wrapper both gateway paths use: it runs before the handler, titles are not touched."""

    class Base:
        seen = []

        def handle_generation(self, path, body):
            SharedWrapperTests.Base.seen.append(body)
            return "base"

    def setUp(self):
        self.Base.seen.clear()

    def test_fit_runs_before_the_handler_and_titles_skip_it(self):
        import service_requests as sr
        calls = []

        def fit(path, body):
            calls.append(path)
            return b"shortened"

        wrapped = sr.with_title_answer(self.Base, lambda: False, fit)
        self.assertEqual(wrapped().handle_generation("/v1/chat/completions", b"{}"), "base")
        self.assertEqual(self.Base.seen, [b"shortened"])
        self.assertEqual(calls, ["/v1/chat/completions"])

    def test_without_a_fit_hook_the_body_is_passed_unchanged(self):
        import service_requests as sr
        wrapped = sr.with_title_answer(self.Base, lambda: False)
        wrapped().handle_generation("/v1/chat/completions", b"{}")
        self.assertEqual(self.Base.seen, [b"{}"])


if __name__ == "__main__":
    unittest.main()
