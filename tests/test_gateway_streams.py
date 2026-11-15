"""What the client sees when llama-server misbehaves, and what the gateway does about a client that leaves.

Two layers: `forward_request` against hand-made bad upstreams (exact wire behaviour), and the whole gateway against the fake
llama-server (crash, reset, stall, abandoned requests). Both entry points of the gateway are covered.
"""
import http.client
import json
import socket
import struct
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
sys.path.insert(0, str(HERE.parent / "gateway"))
from gateway_harness import GatewayHarness  # noqa: E402
from proxy_core import forward_request, inspect_body  # noqa: E402

LEGACY = {"slotSaveCheckpoints": 0}
FULL = {"slotSaveCheckpoints": 1}


# --- forward_request against hand-made upstreams ----------------------------------------------------------------------

class Upstream:
    """A raw TCP server that answers the first request with exactly `script(conn)`; records whether the peer closed."""

    def __init__(self, script):
        self.script = script
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        self.closed_by_gateway = threading.Event()
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    def handle(self, conn):
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = conn.recv(65536)
            if not chunk:
                return
            data += chunk
        try:
            self.script(conn, self)
        except OSError:
            pass
        finally:
            conn.close()

    def wait_for_peer_close(self, conn, timeout=5):
        """Read and discard whatever the peer still sends (the request body may arrive after the headers) until it closes."""
        conn.settimeout(timeout)
        try:
            while True:
                if conn.recv(65536) == b"":
                    self.closed_by_gateway.set()
                    return
        except OSError:
            self.closed_by_gateway.set()

    def close(self):
        self.sock.close()


def chunk(data):
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


class Front:
    """A client-facing server whose only job is to call forward_request, like the gateway's Handler does."""

    def __init__(self, upstream_port, config=None, observe=None, lease=None):
        outer = self
        self.result = None
        self.done = threading.Event()

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n)
                outer.result = forward_request(self, body, "127.0.0.1", upstream_port, config or {}, observe=observe, lease=lease)
                outer.done.set()

            do_GET = do_HEAD = do_POST

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def fetch(port, path="/v1/chat/completions", body=b"{}", timeout=10, headers=None):
    """(status, headers, body bytes or the exception text, complete?)"""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", path, body, headers or {})
        resp = conn.getresponse()
        try:
            return resp.status, dict(resp.getheaders()), resp.read(), True
        except (http.client.IncompleteRead, OSError) as error:
            return resp.status, dict(resp.getheaders()), repr(error).encode(), False
    finally:
        conn.close()


class RelayWire(unittest.TestCase):
    def setUp(self):
        self.cleanup = []

    def tearDown(self):
        for item in self.cleanup:
            item.close()

    def pair(self, script, **kw):
        upstream = Upstream(script)
        front = Front(upstream.port, **kw)
        self.cleanup += [upstream, front]
        return upstream, front

    def test_a_json_answer_cut_short_is_not_passed_off_as_complete(self):
        def script(conn, _):
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n" + b'{"partial":')
        _, front = self.pair(script)
        status, _, data, complete = fetch(front.port)
        self.assertEqual(status, 502)                       # nothing was sent yet, so the client gets a real error
        self.assertIn(b"SHURA_UPSTREAM_INTERRUPTED", data)
        self.assertTrue(front.done.wait(3))
        self.assertTrue(front.result["error"])

    def test_a_chunked_json_answer_cut_short_is_cut_for_the_client_too(self):
        def script(conn, _):
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n" + chunk(b'{"partial":'))
        _, front = self.pair(script)
        status, _, data, complete = fetch(front.port)
        self.assertEqual(status, 200)
        self.assertFalse(complete)                          # the client's HTTP library reports an incomplete body
        self.assertTrue(front.done.wait(3))
        self.assertTrue(front.result["error"])

    def test_a_stream_cut_short_ends_with_an_error_event_and_a_clean_end(self):
        def script(conn, _):
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                         + chunk(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'))
            # the connection closes in the middle of the stream
        _, front = self.pair(script)
        status, _, data, complete = fetch(front.port)
        self.assertEqual((status, complete), (200, True))
        text = data.decode()
        self.assertIn('"content": "hi"', text.replace('"content":"hi"', '"content": "hi"'))
        error = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {") and '"error"' in line][0]["error"]
        self.assertEqual(error["code"], "SHURA_RESPONSE_INTERRUPTED")
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))
        self.assertTrue(front.done.wait(3))
        self.assertTrue(front.result["error"])

    def test_a_reset_upstream_is_an_error_event_not_a_silent_stop(self):
        def script(conn, _):
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n" + chunk(b"data: {}\n\n"))
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        _, front = self.pair(script)
        status, _, data, complete = fetch(front.port)
        self.assertTrue(complete)
        # The reset can land before the first byte was read (a clean 502) or after it (an error event in the stream).
        self.assertTrue(b"SHURA_RESPONSE_INTERRUPTED" in data or b"SHURA_UPSTREAM_UNREACHABLE" in data, data)

    def test_a_silent_upstream_is_given_up_on_and_reported(self):
        stalls = []

        def script(conn, _):
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n" + chunk(b"data: {}\n\n"))
            time.sleep(8)
        original = forward_request

        upstream = Upstream(script)
        self.cleanup.append(upstream)
        outer = {}

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                outer["result"] = original(self, b"{}", "127.0.0.1", upstream.port, {"upstreamIdleSeconds": 1}, on_stall=lambda: stalls.append(1))

        server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.cleanup.append(type("S", (), {"close": lambda self_: (server.shutdown(), server.server_close())})())
        started = time.monotonic()
        status, _, data, complete = fetch(server.server_address[1])
        self.assertLess(time.monotonic() - started, 5)
        self.assertIn(b"SHURA_UPSTREAM_STALLED", data)
        self.assertEqual(stalls, [1])

    def test_an_upstream_that_refuses_the_connection_is_a_502_with_the_message_in_the_openai_place(self):
        dead = socket.socket()
        dead.bind(("127.0.0.1", 0))
        port = dead.getsockname()[1]
        dead.close()
        front = Front(port)
        self.cleanup.append(front)
        status, _, data, _ = fetch(front.port)
        self.assertEqual(status, 502)
        self.assertIn("message", json.loads(data)["error"])

    def test_a_client_that_leaves_while_the_model_is_thinking_closes_the_upstream(self):
        def script(conn, up):
            up.wait_for_peer_close(conn, timeout=6)       # no answer for a long time
        upstream, front = self.pair(script)
        sock = socket.create_connection(("127.0.0.1", front.port))
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}")
        time.sleep(0.6)
        sock.close()
        self.assertTrue(upstream.closed_by_gateway.wait(3), "the gateway kept the upstream connection open for a client that had gone")
        self.assertTrue(front.done.wait(3))
        self.assertEqual(front.result["status"], 499)
        self.assertFalse(front.result["error"])           # leaving is not a failure

    def test_a_connection_limited_answer_is_streamed_unchanged(self):
        body = b'{"ok": true}'

        def script(conn, _):
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\nAccess-Control-Allow-Origin: https://x.example\r\n"
                         b"Access-Control-Allow-Credentials: true\r\nServer: llama\r\n\r\n" % len(body) + body)
        _, front = self.pair(script)
        status, headers, data, complete = fetch(front.port)
        self.assertEqual((status, data, complete), (200, body, True))
        self.assertFalse([h for h in headers if h.lower().startswith("access-control")])      # llama-server's CORS headers are dropped
        self.assertEqual([h for h in headers if h.lower() == "server"].__len__(), 1)          # and so is the duplicate Server header

    def test_answers_without_a_body_get_no_chunked_framing(self):
        def head_script(conn, _):
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n")

        def no_content(conn, _):
            conn.sendall(b"HTTP/1.1 204 No Content\r\n\r\n")
        for method, script in (("HEAD", head_script), ("GET", no_content)):
            _, front = self.pair(script)
            sock = socket.create_connection(("127.0.0.1", front.port), timeout=5)
            sock.sendall(f"{method} /x HTTP/1.1\r\nHost: x\r\n\r\n".encode())
            time.sleep(0.5)
            data = sock.recv(65536)
            sock.close()
            self.assertNotIn(b"0\r\n\r\n", data.split(b"\r\n\r\n", 1)[1], (method, data))
            self.assertNotIn(b"Transfer-Encoding: chunked", data, (method, data))

    def test_a_path_http_client_refuses_is_a_400(self):
        upstream = Upstream(lambda conn, _: None)
        self.cleanup.append(upstream)
        outer = {}

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.path = "/x\x7f"
                outer["result"] = forward_request(self, b"{}", "127.0.0.1", upstream.port)

        server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.cleanup.append(type("S", (), {"close": lambda self_: (server.shutdown(), server.server_close())})())
        status, _, data, _ = fetch(server.server_address[1])
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(data)["error"]["code"], "SHURA_BAD_REQUEST")

    def test_game_cancel_gives_the_explicit_error_and_never_hangs(self):
        cancel = threading.Event()
        lease = {"cancel": cancel, "conn": None, "retry_after": 5}

        def script(conn, up):
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n" + chunk(b"data: {}\n\n"))
            time.sleep(5)
        _, front = self.pair(script, lease=lease)

        def game_monitor_cuts_the_request():
            time.sleep(0.6)
            cancel.set()
            lease["sock"].shutdown(socket.SHUT_RDWR)      # what game_tick does
        threading.Thread(target=game_monitor_cuts_the_request, daemon=True).start()
        started = time.monotonic()
        status, _, data, complete = fetch(front.port)
        self.assertLess(time.monotonic() - started, 4)
        self.assertTrue(complete)
        self.assertIn(b"game_pause_timeout", data)


class InspectBody(unittest.TestCase):
    MODELS = {"a": {}, "b": {}}

    def test_a_tool_schema_with_an_image_property_is_not_a_vision_request(self):
        body = json.dumps({"model": "a", "messages": [{"role": "user", "content": "hi"}],
                           "tools": [{"type": "function", "function": {"name": "t", "parameters": {
                               "type": "object", "properties": {"image_url": {"type": "string"}}}}}]}).encode()
        self.assertEqual(inspect_body(body, self.MODELS, "a")[:2], (False, "a"))

    def test_an_image_part_in_the_messages_is_a_vision_request(self):
        for part in ({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}, {"type": "input_image", "image_url": "x"}):
            body = json.dumps({"messages": [{"role": "user", "content": [{"type": "text", "text": "what is this"}, part]}]}).encode()
            self.assertTrue(inspect_body(body, self.MODELS, "a")[0])

    def test_text_that_merely_mentions_image_url_is_not(self):
        body = json.dumps({"messages": [{"role": "user", "content": 'put "image_url" and "type": "image" in the JSON'}]}).encode()
        self.assertFalse(inspect_body(body, self.MODELS, "a")[0])

    def test_the_model_is_the_top_level_field_not_a_nested_one(self):
        body = json.dumps({"messages": [{"role": "user", "content": '"model": "b"'}], "metadata": {"model": "b"}, "model": "a"}).encode()
        self.assertEqual(inspect_body(body, self.MODELS, "b")[1], "a")
        self.assertEqual(inspect_body(json.dumps({"model": "unknown"}).encode(), self.MODELS, "b")[1], "b")

    def test_reasoning_effort_is_renamed_and_broken_json_is_left_alone(self):
        body = json.dumps({"messages": [], "reasoningEffort": "high"}).encode()
        _, _, out = inspect_body(body, self.MODELS, "a")
        self.assertEqual(json.loads(out), {"messages": [], "reasoning_effort": "high"})
        self.assertEqual(inspect_body(b"{not json", self.MODELS, "a"), (False, "a", b"{not json"))
        self.assertEqual(inspect_body(b"", self.MODELS, "a"), (False, "a", b""))


# --- the whole gateway ------------------------------------------------------------------------------------------------

def stream_events(lines):
    out = []
    for line in lines:
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


class GatewayStreams:
    GUARDIAN = FULL

    def harness(self, behaviour=None, guardian=None):
        return GatewayHarness(behaviour={"n_tokens": 40, "token_delay_ms": 5, **(behaviour or {})}, guardian={**self.GUARDIAN, **(guardian or {})})

    def monitor_requests(self, gw):
        status, body = gw.request("GET", "/monitor/api/requests")
        return body if status == 200 else {}

    def test_a_server_killed_mid_stream_tells_the_client(self):
        with self.harness({"crash_after_tokens": 5}) as gw:
            code, body = gw.chat("x", stream=True)
            self.assertEqual(code, 200)
            self.assertTrue(body["complete"])
            self.assertIn("error", " ".join(body["lines"]))
            self.assertNotIn("Traceback", gw.log())

    def test_a_connection_drop_mid_stream_tells_the_client(self):
        with self.harness({"drop_after_tokens": 5}) as gw:
            code, body = gw.chat("x", stream=True)
            self.assertTrue(body["complete"])
            self.assertTrue([e for e in stream_events(body["lines"]) if "error" in e])
            self.assertEqual(gw.status()["status"], "READY")          # a dropped connection is not a dead server

    def test_a_reset_mid_stream_tells_the_client(self):
        with self.harness({"rst_after_tokens": 5}) as gw:
            code, body = gw.chat("x", stream=True)
            self.assertTrue([e for e in stream_events(body["lines"]) if "error" in e], body)

    def test_the_interrupted_request_is_recorded_as_an_error_not_a_success(self):
        with self.harness({"crash_after_tokens": 3}) as gw:
            gw.chat("x", stream=True)
            state = json.dumps(gw.request("GET", "/monitor/api/state")[1])
            self.assertRegex(state, r'"request_errors": [1-9]')

    def test_a_client_that_gives_up_frees_the_single_slot_at_once(self):
        with self.harness({"hang_seconds": 8, "n_tokens": 3}) as gw:
            gw.chat("warm")                                              # loads the model (the first request pays the hang too)
        with self.harness({"n_tokens": 3}) as gw:
            gw.chat("warm")
            gw.set_behaviour(hang_seconds=8)
            sock = socket.create_connection(("127.0.0.1", gw.gw_port), timeout=5)
            payload = json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: "
                         + str(len(payload)).encode() + b"\r\n\r\n" + payload)
            gw.wait_for(lambda: gw.status()["busy_requests"] == 1, what="the request to be in flight")
            sock.close()                                                 # Esc in the agent
            started = time.monotonic()
            gw.wait_for(lambda: gw.status()["busy_requests"] == 0, timeout=4, what="the slot to be released")
            self.assertLess(time.monotonic() - started, 3)                # not after the 8 s the model would have taken

    def test_a_streaming_client_that_leaves_stops_the_generation(self):
        with self.harness({"n_tokens": 400, "token_delay_ms": 20}) as gw:
            conn = http.client.HTTPConnection("127.0.0.1", gw.gw_port, timeout=10)
            conn.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "x"}], "stream": True}),
                         {"Content-Type": "application/json"})
            resp = conn.getresponse()
            resp.readline()
            resp.readline()
            conn.close()
            gw.wait_for(lambda: gw.request("GET", "/__fake/state", port=gw.llama_port)[1]["client_closed"] >= 1, timeout=5,
                        what="llama-server to see the abandoned stream")
            gw.wait_for(lambda: gw.status()["busy_requests"] == 0, timeout=3, what="the slot to be released")

    def test_a_server_that_goes_silent_mid_answer_is_replaced(self):
        with self.harness({"stall_after_tokens": 3, "stall_seconds": 300}, {"upstreamIdleSeconds": 2}) as gw:
            started = time.monotonic()
            code, body = gw.chat("x", stream=True)
            self.assertLess(time.monotonic() - started, 12)
            self.assertIn("SHURA_UPSTREAM_STALLED", " ".join(body["lines"]))
            gw.wait_for(lambda: gw.status()["llama_pid"] is None, timeout=15, what="the hung server to be stopped")
            gw.set_behaviour(stall_after_tokens=None)
            self.assertEqual(gw.chat("again")[0], 200)                    # the next request starts a fresh server

    def test_an_image_request_to_a_text_only_model_does_not_restart_the_server_every_time(self):
        with self.harness({"n_tokens": 3}) as gw:
            image = {"messages": [{"role": "user", "content": [{"type": "text", "text": "?"},
                                                               {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]}
            for _ in range(2):
                self.assertEqual(gw.request("POST", "/v1/chat/completions", image)[0], 200)
            self.assertEqual(gw.log().count("Starting llama-server"), 1)

    def test_a_tool_with_an_image_property_does_not_switch_to_vision(self):
        with self.harness({"n_tokens": 3}) as gw:
            body = {"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": {
                "name": "t", "parameters": {"type": "object", "properties": {"image_url": {"type": "string"}}}}}]}
            self.assertEqual(gw.request("POST", "/v1/chat/completions", body)[0], 200)
            self.assertFalse(gw.status()["vision"])


class GatewayStreamsFull(GatewayStreams, unittest.TestCase):
    GUARDIAN = FULL


class GatewayStreamsLegacy(GatewayStreams, unittest.TestCase):
    GUARDIAN = LEGACY


class GameModeStreams(unittest.TestCase):
    """With gameMode on the proxy has its own lease logic; a failure after the headers must still reach the client."""

    def test_a_server_killed_mid_stream_tells_the_client_in_game_mode(self):
        with GatewayHarness(behaviour={"n_tokens": 40, "token_delay_ms": 5},
                            guardian={"gameMode": True, "slotSaveCheckpoints": 1}) as gw:
            gw.wait_for(lambda: gw.chat("warm")[0] == 200, timeout=30, what="game mode to be ready and the model to load")
            gw.set_behaviour(crash_after_tokens=5)
            code, body = gw.chat("x", stream=True)
            self.assertEqual(code, 200, body)
            self.assertTrue(body["complete"], body)
            self.assertIn("error", " ".join(body["lines"]))


class CrashRecoveryQueue(unittest.TestCase):
    """crashRecovery used to answer a second concurrent request with a 503; it now waits its turn."""

    def test_two_overlapping_chats_both_succeed(self):
        with GatewayHarness(behaviour={"n_tokens": 30, "token_delay_ms": 20},
                            guardian={"crashRecovery": True, "slotSaveCheckpoints": 1}) as gw:
            results = []

            def call():
                results.append(gw.chat("x", stream=True)[0])
            first = threading.Thread(target=call)
            first.start()
            gw.wait_for(lambda: gw.status()["busy_requests"] >= 1, what="the first request")
            second = threading.Thread(target=call)
            second.start()
            first.join(30)
            second.join(30)
            self.assertEqual(results, [200, 200], gw.log()[-800:])


class CrashRecoveryRelay(unittest.TestCase):
    """crashRecovery has its own relay (response_recovery.forward): it must get the same protections."""
    GUARDIAN = {"crashRecovery": True, "slotSaveCheckpoints": 1}

    def test_a_client_that_leaves_while_the_model_thinks_frees_the_slot_and_restarts_nothing(self):
        with GatewayHarness(behaviour={"n_tokens": 3}, guardian=self.GUARDIAN) as gw:
            self.assertEqual(gw.chat("warm")[0], 200)
            pid = gw.status()["llama_pid"]
            gw.set_behaviour(hang_seconds=8)
            sock = socket.create_connection(("127.0.0.1", gw.gw_port), timeout=5)
            payload = json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: "
                         + str(len(payload)).encode() + b"\r\n\r\n" + payload)
            gw.wait_for(lambda: gw.status()["busy_requests"] == 1, what="the request in flight")
            sock.close()
            started = time.monotonic()
            gw.wait_for(lambda: gw.status()["busy_requests"] == 0, timeout=4, what="the slot to be released")
            self.assertLess(time.monotonic() - started, 3)
            self.assertEqual(gw.status()["llama_pid"], pid)               # leaving is not a crash: no restart
            self.assertEqual(gw.status()["crash_count"], 0)

    def test_llama_servers_cors_headers_and_the_wrong_error_type_do_not_reach_the_client(self):
        with GatewayHarness(behaviour={"n_tokens": 40, "token_delay_ms": 5, "cors": True, "crash_after_tokens": 5}, guardian=self.GUARDIAN) as gw:
            conn = http.client.HTTPConnection("127.0.0.1", gw.gw_port, timeout=30)
            conn.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "x"}], "stream": True}),
                         {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{gw.gw_port}"})
            resp = conn.getresponse()
            self.assertFalse([h for h, _ in resp.getheaders() if h.lower().startswith("access-control")])
            self.assertEqual([h for h, _ in resp.getheaders() if h.lower() in ("server", "date")].__len__(), 2)   # ours only, not doubled
            text = resp.read().decode()
            self.assertIn("SHURA_RESPONSE_INTERRUPTED", text)
            self.assertIn('"type": "api_error"', text)
            self.assertNotIn("invalid_request_error", text)

    def test_a_server_that_goes_silent_mid_answer_is_given_up_on(self):
        guardian = {**self.GUARDIAN, "upstreamIdleSeconds": 2}
        with GatewayHarness(behaviour={"n_tokens": 40, "token_delay_ms": 5, "stall_after_tokens": 3, "stall_seconds": 300}, guardian=guardian) as gw:
            started = time.monotonic()
            code, body = gw.chat("x", stream=True)
            self.assertLess(time.monotonic() - started, 15)
            self.assertIn("SHURA_RESPONSE_INTERRUPTED", " ".join(body["lines"]))
            gw.wait_for(lambda: gw.status()["llama_pid"] is None, timeout=15, what="the hung server to be stopped")


if __name__ == "__main__":
    unittest.main()
