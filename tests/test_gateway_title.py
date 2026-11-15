"""Chat titles are answered by the gateway itself: the model is not loaded and its only slot is not taken (REPORT-CACHE1.md).

Offline: no GPU, no model, no live config. The module is loaded from a temporary config folder, like test_gateway.py does.
"""
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "gateway"))

import service_requests as sr  # noqa: E402
TITLE_SYSTEM = "You are a title generator. You output ONLY a thread title. Nothing else.\n\n<task>\nGenerate a brief title."


def title_body(first_message, stream=True, tools=False):
    body = {"model": "tiel-coder", "stream": stream, "messages": [
        {"role": "system", "content": TITLE_SYSTEM},
        {"role": "user", "content": "Generate a title for this conversation:\n"},
        {"role": "user", "content": f'"{first_message}"'},
    ]}
    if tools:
        body["tools"] = [{"type": "function", "function": {"name": "read"}}]
    return json.dumps(body, ensure_ascii=False).encode()


def chat_body(text):
    return json.dumps({"model": "tiel-coder", "stream": True, "messages": [
        {"role": "system", "content": "You are ShuraCode, running fully locally on the Tiel model."},
        {"role": "user", "content": text},
    ]}, ensure_ascii=False).encode()


class DetectTests(unittest.TestCase):
    def test_title_request_is_detected_and_parsed(self):
        parsed = sr.parse_title_request("/v1/chat/completions", title_body("Прочитай app.py"))
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["model"], "tiel-coder")

    def test_normal_chat_and_tool_requests_are_left_alone(self):
        self.assertIsNone(sr.parse_title_request("/v1/chat/completions", chat_body("hello")))
        self.assertIsNone(sr.parse_title_request("/v1/chat/completions", title_body("x", tools=True)))

    def test_other_paths_and_bad_json_are_left_alone(self):
        self.assertIsNone(sr.parse_title_request("/guardian/status", title_body("x")))
        self.assertIsNone(sr.parse_title_request("/v1/chat/completions", b"{not json"))
        self.assertIsNone(sr.parse_title_request("/v1/chat/completions", b""))


class TitleTextTests(unittest.TestCase):
    def title_of(self, message):
        return sr.title_from(json.loads(title_body(message)))

    def test_title_is_the_first_message_without_quotes_and_whitespace_noise(self):
        self.assertEqual(self.title_of("  Прочитай   src/app.py  и скажи. "), "Прочитай src/app.py и скажи")

    def test_long_titles_are_cut_at_a_word_boundary(self):
        text = "Исправь функцию add в калькуляторе: сейчас она вычитает, а должна складывать, и проверь тесты"
        title = self.title_of(text)
        self.assertLessEqual(len(title), sr.MAX_CHARS)
        self.assertTrue(text.startswith(title))
        self.assertFalse(title.endswith(" "))

    def test_empty_first_message_gets_the_fallback(self):
        self.assertEqual(self.title_of("   "), sr.FALLBACK)


class ResponseTests(unittest.TestCase):
    def test_plain_answer_has_the_completion_shape(self):
        obj = json.loads(title_body("Тесты", stream=False))
        ctype, payload = sr.response_bytes(obj, "Тесты", stream=False)
        self.assertEqual(ctype, "application/json")
        out = json.loads(payload)
        self.assertEqual(out["choices"][0]["message"], {"role": "assistant", "content": "Тесты"})
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")

    def test_streamed_answer_sends_the_title_then_done(self):
        obj = json.loads(title_body("Тесты", stream=True))
        ctype, payload = sr.response_bytes(obj, "Тесты", stream=True)
        self.assertEqual(ctype, "text/event-stream")
        lines = [l for l in payload.decode().split("\n") if l.startswith("data: ")]
        self.assertEqual(lines[-1], "data: [DONE]")
        events = [json.loads(l[6:]) for l in lines[:-1]]
        self.assertEqual("".join(e["choices"][0]["delta"].get("content", "") for e in events), "Тесты")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")

    def test_serve_title_writes_headers_and_the_same_payload(self):
        handler = Mock()
        handler.wfile = io.BytesIO()
        obj = json.loads(title_body("Тесты", stream=False))
        sr.serve_title(handler, obj)
        handler.send_response.assert_called_once_with(200)
        written = handler.wfile.getvalue()
        length = [c.args for c in handler.send_header.call_args_list if c.args[0] == "Content-Length"][0][1]
        self.assertEqual(int(length), len(written))
        self.assertEqual(json.loads(written)["choices"][0]["message"]["content"], "Тесты")


class WrapperTests(unittest.TestCase):
    """The wrapper that both gateway paths use, on a plain base class (the legacy handler is never edited)."""

    class Base:
        calls = []

        def __init__(self):
            self.wfile = io.BytesIO()
            self.sent = []

        def send_response(self, code):
            self.sent.append(code)

        def send_header(self, *args):
            pass

        def end_headers(self):
            pass

        def handle_generation(self, path, body):
            WrapperTests.Base.calls.append(path)
            return "base"

    def setUp(self):
        self.Base.calls.clear()

    def test_titles_are_answered_and_everything_else_passes_through(self):
        wrapped = sr.with_title_answer(self.Base, lambda: True)
        self.assertEqual(wrapped.__name__, "Base")
        title_handler = wrapped()
        self.assertEqual(title_handler.handle_generation("/v1/chat/completions", title_body("Тесты")), None)
        self.assertEqual(title_handler.sent, [200])
        self.assertEqual(self.Base.calls, [])
        self.assertEqual(wrapped().handle_generation("/v1/chat/completions", chat_body("hi")), "base")
        self.assertEqual(self.Base.calls, ["/v1/chat/completions"])

    def test_disabled_flag_leaves_titles_to_the_model(self):
        wrapped = sr.with_title_answer(self.Base, lambda: False)
        self.assertEqual(wrapped().handle_generation("/v1/chat/completions", title_body("Тесты")), "base")


class GatewayRoutingTests(unittest.TestCase):
    """handle_generation answers a title before anything can start the model."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.TemporaryDirectory(prefix="shura-title-import-", dir="/tmp")
        for name in ("guardian", "model-launch"):
            (Path(cls.root.name) / f"{name}.json").write_text((REPO / "config" / f"{name}.example.json").read_text())
        spec = importlib.util.spec_from_file_location("title_test_gateway", REPO / "gateway" / "ai_gateway.py")
        cls.gw = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, AI_GATEWAY_CONFIG=cls.root.name, AI_GATEWAY_STATE=str(Path(cls.root.name) / "state")):
            spec.loader.exec_module(cls.gw)

    @classmethod
    def tearDownClass(cls):
        cls.root.cleanup()

    def test_title_request_never_reaches_the_model(self):
        handler = Mock()
        handler.wfile = io.BytesIO()
        with patch.object(self.gw.service_requests, "serve_title") as serve, \
                patch.object(self.gw, "acquire_model") as acquire:
            wrapped = self.gw.service_requests.with_title_answer(self.gw.Handler, lambda: True)
            wrapped.handle_generation(handler, "/v1/chat/completions", title_body("Тесты"))
        serve.assert_called_once()
        acquire.assert_not_called()

    def test_chat_request_still_goes_to_the_model_path(self):
        wrapped = self.gw.service_requests.with_title_answer(self.gw.Handler, lambda: True)
        instance = wrapped.__new__(wrapped)              # a real instance, so super() works; no socket is involved
        with patch.object(self.gw.Handler, "handle_generation", Mock(return_value="model")) as base_call, \
                patch.object(self.gw.service_requests, "serve_title") as serve:
            result = instance.handle_generation("/v1/chat/completions", chat_body("hello"))
        serve.assert_not_called()
        base_call.assert_called_once()
        self.assertEqual(result, "model")

if __name__ == "__main__":
    unittest.main()
