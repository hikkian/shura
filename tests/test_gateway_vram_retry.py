"""VRAMCOR4: vramRetry. If the backend stops (the VRAM sentinel, a crash) before a real token reached the client, the request is replayed
once on the restarted backend into the same open response (it may come back on the lite twin of the layout). After the first token, or
with the flag off, nothing is replayed and the client gets the honest interruption error. Mocked processes and HTTP only."""
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_gateway as fixture
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
from response_recovery import StreamState  # noqa: E402

PROGRESS = b'data: {"choices":[{"delta":{}}],"prompt_progress":{"total":1000,"processed":500}}\n\n'
TOKEN = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
REASONING = b'data: {"choices":[{"delta":{"reasoning_content":"hm"}}]}\n\n'
DONE = b"data: [DONE]\n\n"


class VramRetry(unittest.TestCase):
    setUpClass = classmethod(fixture.GatewayGuard.setUpClass.__func__)
    tearDownClass = classmethod(fixture.GatewayGuard.tearDownClass.__func__)
    setUp = fixture.GatewayGuard.setUp
    patch_attr = fixture.GatewayGuard.patch_attr
    fake_llama = fixture.GatewayGuard.fake_llama

    def handler(self):
        h = object.__new__(self.gw.Handler)
        h.command = "POST"
        h.path = "/v1/chat/completions"
        h.headers = {}
        h.wfile = io.BytesIO()
        for method in ["send_response", "send_header", "end_headers", "send_json"]:
            setattr(h, method, Mock())
        return h

    def conn(self, chunks):
        r = Mock(status=200)
        r.getheader.return_value = "text/event-stream"
        r.getheaders.return_value = [("Content-Type", "text/event-stream")]
        r.read1.side_effect = chunks
        c = Mock()
        c.getresponse.return_value = r
        return c

    def run_proxy(self, conns, retry=True, start_error=""):
        gw = self.gw
        gw.G["vramRetry"] = retry
        gw.st.busy = 1
        self.proc.poll.return_value = -9
        h = self.handler()

        def start(model, vision, layout="short"):
            gw.st.proc = Mock(pid=4321)
            gw.st.proc.poll.return_value = None
            gw.st.status = "READY"
            return start_error

        with (patch.object(gw.http.client, "HTTPConnection", side_effect=conns) as opened,
              patch.object(gw, "wait_fresh_vram", return_value=True),
              patch.object(gw, "start_llama", side_effect=start) as load):
            h.proxy_recoverable(b"{}", gw.st.model_id, False)
        return h, load, opened

    def test_a_stop_after_only_progress_frames_is_replayed_once_into_the_same_response(self):
        h, load, opened = self.run_proxy([self.conn([PROGRESS, ConnectionResetError()]), self.conn([TOKEN, DONE, b""])])
        out = h.wfile.getvalue()
        load.assert_called_once()
        self.assertEqual(opened.call_count, 2)
        self.assertIn(b'"content":"hi"', out)
        self.assertNotIn(b"SHURA_RESPONSE_INTERRUPTED", out)
        self.assertEqual(h.send_response.call_count, 1)      # one response: the replay continues it
        self.assertEqual(self.gw.st.event_counters["response_replayed_before_content"], 1)

    def test_a_clean_eof_before_any_token_is_replayed_too(self):
        h, load, _ = self.run_proxy([self.conn([PROGRESS, b""]), self.conn([TOKEN, DONE, b""])])
        load.assert_called_once()
        self.assertIn(b'"content":"hi"', h.wfile.getvalue())

    def test_after_a_token_nothing_is_replayed(self):
        for first in (TOKEN, REASONING, b'data: {"choices":[{"delta":{"tool_calls":[{"id":"x"}]}}]}\n\n'):
            with self.subTest(first=first[:40]):
                h, load, opened = self.run_proxy([self.conn([PROGRESS, first, ConnectionResetError()])])
                load.assert_not_called()
                self.assertEqual(opened.call_count, 1)
                self.assertIn(b"SHURA_RESPONSE_INTERRUPTED", h.wfile.getvalue())

    def test_without_the_flag_nothing_is_replayed(self):
        h, load, opened = self.run_proxy([self.conn([PROGRESS, ConnectionResetError()])], retry=False)
        load.assert_not_called()
        self.assertEqual(opened.call_count, 1)
        self.assertIn(b"SHURA_RESPONSE_INTERRUPTED", h.wfile.getvalue())

    def test_the_replay_happens_at_most_once(self):
        h, load, opened = self.run_proxy([self.conn([PROGRESS, ConnectionResetError()]), self.conn([PROGRESS, ConnectionResetError()])])
        load.assert_called_once()
        self.assertEqual(opened.call_count, 2)
        self.assertIn(b"SHURA_RESPONSE_INTERRUPTED", h.wfile.getvalue())

    def test_a_backend_that_does_not_come_back_leaves_the_honest_error(self):
        h, load, opened = self.run_proxy([self.conn([PROGRESS, ConnectionResetError()])], start_error="Model load deferred: card VRAM reserve")
        load.assert_called_once()
        self.assertEqual(opened.call_count, 1)
        self.assertIn(b"SHURA_RESPONSE_INTERRUPTED", h.wfile.getvalue())

    def test_a_non_stream_failure_after_headers_is_never_replayed(self):
        c = self.conn([PROGRESS, ConnectionResetError()])
        c.getresponse.return_value.status = 500
        h, load, _ = self.run_proxy([c])
        load.assert_not_called()

    def test_stream_state_tells_real_tokens_from_progress(self):
        for chunk, content in ((PROGRESS, False), (b'data: {"content":"x","stop":false}\n\n', True), (TOKEN, True), (REASONING, True),
                               (b'data: {"choices":[{"delta":{"content":""}}]}\n\n', False)):
            with self.subTest(chunk=chunk[:30]):
                state = StreamState()
                state.feed(chunk)
                self.assertEqual(state.content, content)


if __name__ == "__main__":
    unittest.main()
