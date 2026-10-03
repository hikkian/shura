"""A2 + first-byte recovery integration, mocked owned processes and HTTP only."""

import io
import unittest
from unittest.mock import Mock, patch
import test_gateway as fixture


class RequestRecovery(unittest.TestCase):
    setUpClass = classmethod(fixture.GatewayGuard.setUpClass.__func__)
    tearDownClass = classmethod(fixture.GatewayGuard.tearDownClass.__func__)
    setUp = fixture.GatewayGuard.setUp
    patch_attr = fixture.GatewayGuard.patch_attr
    fake_llama = fixture.GatewayGuard.fake_llama
    make_disk_checkpoint = fixture.GatewayGuard.make_disk_checkpoint

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

    def test_owned_death_before_bytes_restores_a2_then_replays(self):
        gw = self.gw
        mid = gw.st.model_id
        gw.G["slotSaveCheckpoints"] = 1
        self.assertTrue(gw.save_slot_locked(ram_only=True))
        gw.st.busy = 1
        self.proc.poll.return_value = -9
        h = self.handler()
        conns = [
            self.conn([ConnectionResetError()]),
            self.conn([b"data: [DONE]\n\n", b""]),
        ]

        def start(model, vision):
            self.assertEqual(h.wfile.getvalue(), b"")
            h.end_headers.assert_not_called()
            gw.st.proc = Mock(pid=4321)
            gw.st.proc.poll.return_value = None
            gw.st.status = "READY"
            return gw.restore_slot_locked()

        with (
            patch.object(
                gw.http.client, "HTTPConnection", side_effect=conns
            ) as open_conn,
            patch.object(gw, "wait_fresh_vram", return_value=True),
            patch.object(gw, "start_llama", side_effect=start) as load,
        ):
            h.proxy_recoverable(b"{}", mid, False)
        load.assert_called_once_with(mid, False)
        self.assertEqual(open_conn.call_count, 2)
        self.assertEqual(gw.st.event_counters["checkpoint_restored"], 1)
        self.assertIsNone(gw.st.ram_checkpoint)
        h.send_json.assert_not_called()

    def test_no_new_prompt_tokens_keeps_disk_a2_available_for_crash_restore(self):
        gw = self.gw
        model = gw.M["defaultModel"]
        disk = self.make_disk_checkpoint()
        gw.st.model_id = model
        gw.st.ram_checkpoint = None
        gw.st.skip_disk_restore = False
        self.api.side_effect = lambda method, path, *a, **k: (
            (200, [{"n_prompt_tokens": 0}]) if path == "/slots"
            else (200, {"n_restored": 187000})
        )

        self.assertTrue(gw.save_slot_locked(ram_only=True))
        self.assertFalse(gw.st.skip_disk_restore)
        self.assertTrue(disk.exists())
        self.assertEqual(gw.restore_slot_locked(), "")
        self.assertIn(("POST", "/slots/0?action=restore"), [
            (call.args[0], call.args[1]) for call in self.api.call_args_list
        ])

    def test_emitted_tool_delta_prevents_restart(self):
        gw = self.gw
        h = self.handler()
        self.proc.poll.return_value = -9
        c = self.conn(
            [
                b'data: {"choices":[{"delta":{"tool_calls":[{"id":"x"}]}}]}\n\n',
                ConnectionResetError(),
            ]
        )
        with (
            patch.object(gw.http.client, "HTTPConnection", return_value=c),
            patch.object(gw, "start_llama") as load,
        ):
            h.proxy_recoverable(b"{}", gw.st.model_id, False)
        load.assert_not_called()
        self.assertIn(b"SHURA_RESPONSE_INTERRUPTED", h.wfile.getvalue())

    def test_restart_limit_is_not_bypassed(self):
        gw = self.gw
        h = self.handler()
        gw.st.busy = 1
        gw.st.crash_count = gw.G["crashRestartMaxAttempts"] - 1
        self.proc.poll.return_value = -9
        with (
            patch.object(
                gw.http.client,
                "HTTPConnection",
                return_value=self.conn([ConnectionResetError()]),
            ),
            patch.object(gw, "start_llama") as load,
        ):
            h.proxy_recoverable(b"{}", gw.st.model_id, False)
        load.assert_not_called()
        self.assertEqual(gw.st.status, "ERROR")
        self.assertEqual(h.send_json.call_args.args[0], 503)
