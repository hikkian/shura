"""A web page open in the user's browser must not be able to drive the gateway.

Each test runs against the real gateway process (both entry points) with a fake llama-server and a fake NVML:
no GPU, no model, no live config. Browser requests are imitated with raw HTTP: the same headers a browser sends.
"""
import http.client
import json
import socket
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from gateway_harness import GatewayHarness  # noqa: E402

LEGACY = {"slotSaveCheckpoints": 0}
FULL = {"slotSaveCheckpoints": 1}
EVIL = {"Host": "evil.example", "Origin": "https://evil.example"}


def raw(gw, method, path, headers=None, body=None, port=None, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", port or gw.gw_port, timeout=timeout)
    try:
        conn.putrequest(method, path, skip_host=bool(headers and "Host" in headers))
        for key, value in (headers or {}).items():
            conn.putheader(key, value)
        if body is not None:
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        resp = conn.getresponse()
        data = resp.read()
        try:
            parsed = json.loads(data.decode() or "null")
        except ValueError:
            parsed = data.decode(errors="replace")
        return resp.status, parsed, dict(resp.getheaders())
    finally:
        conn.close()


class WebGuardMixin:
    GUARDIAN = FULL

    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness(behaviour={"n_tokens": 4, "token_delay_ms": 2, "cors": True}, guardian={**cls.GUARDIAN, "requestHeaderSeconds": 5})
        cls.gw.start()

    @classmethod
    def tearDownClass(cls):
        cls.gw.stop()

    def setUp(self):
        raw(self.gw, "POST", "/guardian/ai-on")
        raw(self.gw, "POST", "/guardian/unload")

    def status(self):
        return raw(self.gw, "GET", "/guardian/status")[1]

    # --- the browser attacks of the audit (S1) -----------------------------------------------------------------
    def test_a_foreign_page_cannot_switch_the_ai_off(self):
        status, body, _ = raw(self.gw, "POST", "/guardian/ai-off", {"Origin": "https://evil.example", "Content-Type": "text/plain"}, b"")
        self.assertEqual(status, 403, body)
        self.assertEqual(self.status()["override"], "AUTO")

    def test_dns_rebinding_is_refused_by_the_host_header(self):
        status, _, _ = raw(self.gw, "POST", "/guardian/ai-off", {**EVIL}, b"")
        self.assertEqual(status, 421)
        self.assertEqual(raw(self.gw, "GET", "/guardian/status", {"Host": "evil.example"})[0], 421)     # the status leaks too
        self.assertEqual(self.status()["override"], "AUTO")

    def test_an_image_tag_cannot_change_state(self):
        # <img src=".../guardian/ai-off"> sends a plain GET with no Origin, only Sec-Fetch-* headers.
        for headers in ({}, {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors"}):
            status, _, _ = raw(self.gw, "GET", "/guardian/ai-off", headers)
            self.assertIn(status, (403, 405))
        self.assertEqual(raw(self.gw, "GET", "/guardian/unload")[0], 405)
        self.assertEqual(self.status()["override"], "AUTO")

    def test_a_cross_site_post_without_an_origin_is_refused_by_fetch_metadata(self):
        status, _, _ = raw(self.gw, "POST", "/guardian/ai-off", {"Sec-Fetch-Site": "cross-site"}, b"")
        self.assertEqual(status, 403)
        self.assertEqual(self.status()["override"], "AUTO")

    def test_the_command_line_and_the_gateways_own_pages_still_work(self):
        self.assertEqual(raw(self.gw, "POST", "/guardian/ai-off", {}, b"")[0], 200)                      # curl: no Origin
        self.assertEqual(self.status()["override"], "OFF")
        own = {"Origin": f"http://127.0.0.1:{self.gw.gw_port}", "Host": f"127.0.0.1:{self.gw.gw_port}", "Sec-Fetch-Site": "same-origin"}
        self.assertEqual(raw(self.gw, "POST", "/guardian/ai-on", own, b"")[0], 200)                      # the monitor page
        self.assertEqual(self.status()["override"], "AUTO")
        self.assertEqual(raw(self.gw, "GET", "/guardian/status", {"Host": f"localhost:{self.gw.gw_port}"})[0], 200)

    def test_a_listed_origin_may_call_but_the_gateway_never_answers_with_a_wildcard(self):
        # Not listed: refused. The gateway of this class has no allowedOrigins, so a page on another port is a stranger too.
        status, _, headers = raw(self.gw, "OPTIONS", "/v1/chat/completions",
                                 {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
        self.assertEqual(status, 403)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    # --- llama-server's own CORS headers (S2) -------------------------------------------------------------------
    def test_llama_servers_open_cors_policy_does_not_reach_the_client(self):
        self.assertEqual(self.gw.chat("hi")[0], 200)
        for headers in ({}, {"Origin": f"http://127.0.0.1:{self.gw.gw_port}"}):
            status, _, got = raw(self.gw, "POST", "/v1/chat/completions", headers,
                                 json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode())
            self.assertEqual(status, 200)
            self.assertFalse([k for k in got if k.lower().startswith("access-control-")], got)

    # --- the model is not woken by strangers (S3, P-07, L-10) -----------------------------------------------------
    def test_options_health_models_and_favicon_do_not_start_the_model(self):
        for method, path in (("OPTIONS", "/v1/chat/completions"), ("GET", "/health"), ("GET", "/v1/models"), ("GET", "/favicon.ico"),
                             ("GET", "/props"), ("GET", "/anything"), ("HEAD", "/")):
            raw(self.gw, method, path)
        self.assertEqual(self.status()["status"], "UNLOADED")
        self.assertEqual(self.status()["llama_pid"], None)
        status, body, _ = raw(self.gw, "GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertTrue(body["data"])                    # the list comes from the configuration, no model needed
        status, body, _ = raw(self.gw, "GET", "/props")
        self.assertEqual(status, 503)
        self.assertEqual(body["error"]["code"], "SHURA_MODEL_UNLOADED")

    def test_passive_requests_reach_a_running_model_but_do_not_count_as_activity(self):
        self.gw.chat("hi")
        time.sleep(2.5)
        status, body, _ = raw(self.gw, "GET", "/props")
        self.assertEqual(status, 200)
        self.assertIn("model_alias", body)
        self.assertGreaterEqual(self.status()["idle_seconds"], 2)      # the GET did not restart the idle clock

    # --- request framing (S7, P-09) -------------------------------------------------------------------------------
    def send_raw(self, data, timeout=5):
        """Send bytes, read one complete answer (headers and a Content-Length body) or until the server closes or `timeout`."""
        sock = socket.create_connection(("127.0.0.1", self.gw.gw_port), timeout=timeout)
        out = b""
        try:
            sock.sendall(data)
            while True:
                head, sep, rest = out.partition(b"\r\n\r\n")
                if sep:
                    length = 0
                    for line in head.split(b"\r\n")[1:]:
                        if line.lower().startswith(b"content-length:"):
                            length = int(line.split(b":", 1)[1])
                    if len(rest) >= length:
                        return out
                chunk = sock.recv(65536)
                if not chunk:
                    return out
                out += chunk
        except OSError:
            return out
        finally:
            sock.close()

    def test_bad_content_length_gets_an_answer_and_does_not_hold_a_thread(self):
        for value, code in (("abc", b" 400 "), ("-1", b" 400 "), ("99999999999999", b" 413 "), ("1e3", b" 400 ")):
            out = self.send_raw(f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {value}\r\n\r\n".encode())
            self.assertIn(code, out.split(b"\r\n", 1)[0] + b" ", (value, out[:80]))
        self.assertEqual(self.status()["status"], "UNLOADED")       # nothing was started for a broken request

    def test_a_chunked_request_body_is_decoded_not_taken_for_the_next_request(self):
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        wire = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n"
                b"Content-Type: application/json\r\n\r\n" + f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n")
        out = self.send_raw(wire, timeout=20)
        self.assertTrue(out.startswith(b"HTTP/1.1 200"), out[:120])
        self.assertEqual(out.count(b"HTTP/1.1"), 1)                   # not a second, phantom request
        sent = [e for e in self.gw.fake_log() if e.get("method") == "POST" and "/chat/completions" in e.get("path", "")][-1]
        self.assertEqual(sent["body"]["messages"][0]["content"], "hi")

    def test_a_malformed_chunked_body_is_refused(self):
        out = self.send_raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\nx\r\n")
        self.assertIn(b" 400 ", out.split(b"\r\n", 1)[0] + b" ")

    def test_a_path_with_a_control_character_is_an_error_not_a_dead_thread(self):
        out = self.send_raw(b"POST /v1/chat/completions\x7f HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}", timeout=15)
        self.assertTrue(out.startswith(b"HTTP/1.1 4") or out.startswith(b"HTTP/1.1 5"), out[:100])
        self.assertEqual(raw(self.gw, "GET", "/guardian/status")[0], 200)                       # the gateway is fine
        self.assertNotIn("Traceback", self.gw.log())

    def test_the_monitor_survives_a_header_that_is_not_ascii(self):
        status, _, _ = raw(self.gw, "GET", "/monitor/api/state", {"Authorization": "Bearer é".encode("latin-1").decode("latin-1")})
        self.assertIn(status, (200, 401))
        self.assertNotIn("Traceback", self.gw.log())

    # --- the error shape (P-10) ------------------------------------------------------------------------------------
    def test_api_errors_carry_the_message_where_openai_clients_look_for_it(self):
        raw(self.gw, "POST", "/guardian/ai-off", {}, b"")
        status, body, _ = raw(self.gw, "POST", "/v1/chat/completions", {"Content-Type": "application/json"},
                              json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode())
        self.assertEqual(status, 503)
        self.assertIn("ai-off", body["error"]["message"])
        self.assertEqual(body["error"]["code"], "SHURA_AI_OFF")

    # --- findings of the independent review ---------------------------------------------------------------------------
    def test_an_absolute_form_target_cannot_slip_past_the_slot_filter(self):
        self.gw.chat("warm")
        for target in (f"http://127.0.0.1:{self.gw.gw_port}/slots/0?action=erase", "http://evil.example/guardian/ai-off"):
            out = self.send_raw(f"POST {target} HTTP/1.1\r\nHost: 127.0.0.1:{self.gw.gw_port}\r\nContent-Length: 0\r\n\r\n".encode())
            self.assertIn(b" 400 ", out.split(b"\r\n", 1)[0] + b" ", out[:100])
        self.assertFalse([e for e in self.gw.fake_log() if "action=erase" in e.get("path", "")])

    def test_a_client_that_half_closes_after_sending_still_gets_its_answer(self):
        # HTTP/1.0 and `Connection: close` clients (nc -N, simple scripts) shut their sending side and still wait for the answer.
        self.gw.chat("warm")
        payload = json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()
        for version, extra in ((b"HTTP/1.0", b""), (b"HTTP/1.1", b"Connection: close\r\n")):
            sock = socket.create_connection(("127.0.0.1", self.gw.gw_port), timeout=20)
            sock.sendall(b"POST /v1/chat/completions " + version + b"\r\nHost: localhost\r\nContent-Type: application/json\r\n" + extra
                         + b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
            sock.shutdown(socket.SHUT_WR)
            out = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                out += chunk
            sock.close()
            self.assertTrue(out.startswith(b"HTTP/1.") and b" 200 " in out.split(b"\r\n", 1)[0], (version, out[:120]))

    def test_headers_that_trickle_in_are_cut_off_by_a_total_time_limit(self):
        sock = socket.create_connection(("127.0.0.1", self.gw.gw_port), timeout=20)
        sock.sendall(b"GET /guardian/status HTTP/1.1\r\nHost: localhost\r\n")
        started = time.monotonic()
        try:
            while time.monotonic() - started < 14:
                time.sleep(0.8)
                sock.sendall(b"X-Slow: 1\r\n")             # a line every 0.8 s: the per-read timeout never fires
        except OSError:
            pass
        finally:
            sock.close()
        self.assertLess(time.monotonic() - started, 9, "the gateway let a trickling request hold its connection")

    def test_endless_trailers_are_refused(self):
        wire = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n" + b"X: 1\r\n" * 100
        out = self.send_raw(wire)
        self.assertIn(b" 400 ", out.split(b"\r\n", 1)[0] + b" ")

    def test_unclear_framing_is_refused_and_never_parsed_as_the_next_request(self):
        body = b'{"messages":[{"role":"user","content":"hi"}]}'
        cases = {
            "gzip then chunked": b"Transfer-Encoding: gzip\r\nTransfer-Encoding: chunked\r\n",
            "identity": b"Transfer-Encoding: identity\r\n",
            "both": b"Transfer-Encoding: chunked\r\nContent-Length: %d\r\n" % len(body),
            "two lengths": b"Content-Length: 5\r\nContent-Length: 6\r\n",
        }
        for name, headers in cases.items():
            out = self.send_raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n" + headers + b"\r\n" + body)
            self.assertEqual(out.count(b"HTTP/1.1 "), 1, (name, out[:200]))
            self.assertRegex(out.split(b"\r\n", 1)[0], rb"HTTP/1.1 (400|501)", name)
        self.assertEqual(self.status()["status"], "UNLOADED")

    def test_a_body_sent_to_an_endpoint_that_ignores_it_is_not_taken_for_the_next_request(self):
        for wire in (b"POST /monitor HTTP/1.1\r\nHost: localhost\r\nContent-Length: 50\r\n\r\n" + b"GET /guardian/status HTTP/1.1\r\n\r\n".ljust(50, b" "),
                     b"OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 40\r\n\r\n" + b"GET /guardian/status HTTP/1.1\r\n\r\n".ljust(40, b" ")):
            out = self.send_raw(wire)
            self.assertEqual(out.count(b"HTTP/1.1 "), 1, out[:300])

    def test_an_absurd_content_length_is_a_clean_400(self):
        out = self.send_raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: " + b"9" * 5000 + b"\r\n\r\n")
        self.assertIn(b" 400 ", out.split(b"\r\n", 1)[0] + b" ")
        self.assertNotIn("Traceback", self.gw.log())

    def test_chunk_sizes_and_hosts_are_parsed_strictly(self):
        for size in (b"0x5", b"+5", b"5_0", b" 5"):
            out = self.send_raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n" + size + b"\r\nhello\r\n0\r\n\r\n")
            self.assertIn(b" 400 ", out.split(b"\r\n", 1)[0] + b" ", size)
        for host in (b"[::1]evil.com", b"[localhost]", b"localhost:80:80", b"localhost:abc"):
            out = self.send_raw(b"GET /guardian/status HTTP/1.1\r\nHost: " + host + b"\r\n\r\n")
            self.assertIn(b" 421 ", out.split(b"\r\n", 1)[0] + b" ", host)

    def test_expect_100_continue_is_not_answered_before_the_policy(self):
        out = self.send_raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: evil.example\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\n")
        self.assertNotIn(b"100 Continue", out)
        self.assertIn(b" 421 ", out.split(b"\r\n", 1)[0] + b" ")


class WebGuardFull(WebGuardMixin, unittest.TestCase):
    GUARDIAN = FULL


class WebGuardLegacy(WebGuardMixin, unittest.TestCase):
    GUARDIAN = LEGACY


class AllowedOrigin(unittest.TestCase):
    """The user can name a web UI that may call the gateway; only that one, and never with a wildcard."""

    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness(behaviour={"n_tokens": 3, "cors": True},
                                guardian={"slotSaveCheckpoints": 1, "allowedOrigins": ["http://localhost:3000"]})
        cls.gw.start()

    @classmethod
    def tearDownClass(cls):
        cls.gw.stop()

    def test_listed_origin_gets_exactly_its_own_origin_back(self):
        status, _, headers = raw(self.gw, "OPTIONS", "/v1/chat/completions",
                                 {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
                                  "Access-Control-Request-Headers": "content-type,authorization"})
        self.assertEqual(status, 204)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "http://localhost:3000")
        self.assertNotEqual(headers["Access-Control-Allow-Origin"], "*")
        status, _, headers = raw(self.gw, "POST", "/v1/chat/completions", {"Origin": "http://localhost:3000", "Content-Type": "application/json"},
                                 b'{"messages":[{"role":"user","content":"x"}]}')
        self.assertEqual(status, 200)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "http://localhost:3000")
        self.assertEqual(raw(self.gw, "OPTIONS", "/v1/chat/completions", {"Origin": "http://localhost:3001"})[0], 403)

    def test_a_listed_origin_still_cannot_use_a_get_to_change_state(self):
        self.assertEqual(raw(self.gw, "GET", "/guardian/ai-off", {"Origin": "http://localhost:3000"})[0], 405)


class ConnectionLimits(unittest.TestCase):
    def test_idle_clients_are_dropped_and_the_number_of_connections_is_capped(self):
        with GatewayHarness(guardian={"slotSaveCheckpoints": 1, "clientIdleSeconds": 2, "maxConnections": 8}) as gw:
            held = []
            for _ in range(8):
                s = socket.create_connection(("127.0.0.1", gw.gw_port), timeout=5)
                s.sendall(b"GET /guardian/status HTTP/1.1\r\nHost: localhost\r\n")      # half a request: a slowloris
                held.append(s)
            time.sleep(0.3)
            extra = socket.create_connection(("127.0.0.1", gw.gw_port), timeout=5)
            extra.sendall(b"GET /guardian/status HTTP/1.1\r\nHost: localhost\r\n\r\n")
            self.assertEqual(extra.recv(10), b"")                       # over the cap: closed at once, no thread, no answer
            extra.close()
            time.sleep(2.8)
            for s in held:
                self.assertEqual(s.recv(10), b"")                       # the idle timeout closed every half-sent request
                s.close()
            self.assertEqual(gw.request("GET", "/guardian/status")[0], 200)     # and the gateway is usable again


class ApiKey(unittest.TestCase):
    """With an apiKey the guardian's switches ask for it, the key is not on the command line, and the gateway's own slot calls carry it."""

    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness(guardian={"slotSaveCheckpoints": 1}, launch={"apiKey": "s3cret-key"})
        cls.gw.start()

    @classmethod
    def tearDownClass(cls):
        cls.gw.stop()

    def test_switches_need_the_key_status_stays_open(self):
        self.assertEqual(raw(self.gw, "POST", "/guardian/ai-off", {}, b"")[0], 401)
        self.assertEqual(raw(self.gw, "POST", "/guardian/ai-off", {"Authorization": "Bearer wrong"}, b"")[0], 401)
        self.assertEqual(raw(self.gw, "GET", "/guardian/status")[0], 200)
        self.assertEqual(raw(self.gw, "POST", "/guardian/ai-off", {"Authorization": "Bearer s3cret-key"}, b"")[0], 200)
        self.assertEqual(raw(self.gw, "POST", "/guardian/ai-on", {"Authorization": "Bearer s3cret-key"}, b"")[0], 200)

    def test_the_key_goes_to_a_private_file_not_to_the_command_line_or_the_log(self):
        raw(self.gw, "POST", "/v1/chat/completions", {"Content-Type": "application/json", "Authorization": "Bearer s3cret-key"},
            json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode())
        argv = self.gw.request("GET", "/__fake/args", port=self.gw.llama_port)[1]
        self.assertNotIn("s3cret-key", " ".join(argv))
        self.assertIn("--api-key-file", argv)
        path = Path(argv[argv.index("--api-key-file") + 1])
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertNotIn("s3cret-key", self.gw.log())
        self.assertIn("--cors-origins", argv)                        # llama-server answers only its own origin on its own port
        self.assertNotIn("*", argv[argv.index("--cors-origins") + 1])



class ApiKeyAndSlots(unittest.TestCase):
    def test_the_gateways_own_slot_calls_carry_the_key(self):
        with GatewayHarness(behaviour={"n_tokens": 3, "prompt_tokens": 30000}, guardian={"slotSaveCheckpoints": 1, "slotSaveMinTokens": 1000},
                            launch={"apiKey": "s3cret-key"}) as gw:
            request = {"messages": [{"role": "user", "content": "x"}]}
            conn = http.client.HTTPConnection("127.0.0.1", gw.gw_port, timeout=30)
            conn.request("POST", "/v1/chat/completions", json.dumps(request), {"Content-Type": "application/json", "Authorization": "Bearer s3cret-key"})
            self.assertEqual(conn.getresponse().status, 200)
            conn.close()
            self.assertEqual(raw(gw, "POST", "/guardian/unload", {"Authorization": "Bearer s3cret-key"}, b"")[0], 200)
            self.assertIn("Saved slot", gw.log())                     # /slots and the save call were not refused with a 401
            self.assertNotIn("checkpoint failed", gw.log())


class PolicyUnits(unittest.TestCase):
    def policy(self, **config):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
        from web_guard import Policy
        return Policy(config, "127.0.0.1", config.pop("key", None))

    def test_the_origin_must_match_scheme_and_authority(self):
        policy = self.policy()
        self.assertTrue(policy.origin_allowed("http://localhost:8080", "localhost:8080"))
        for origin in ("https://localhost:8080", "ftp://localhost:8080", "//localhost:8080", "null"):
            self.assertFalse(policy.origin_allowed(origin, "localhost:8080"), origin)

    def test_a_utf8_key_works_and_a_bare_token_does_not(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
        from web_guard import Policy
        policy = Policy({}, "127.0.0.1", "k\u00e9y")
        sent = "Bearer " + "k\u00e9y".encode("utf-8").decode("latin-1")          # how http.server hands over the header bytes
        self.assertTrue(policy.key_ok({"Authorization": sent}))
        self.assertFalse(policy.key_ok({"Authorization": "k\u00e9y".encode("utf-8").decode("latin-1")}))
        self.assertFalse(policy.key_ok({"Authorization": "Bearer nope"}))
        self.assertFalse(policy.key_ok({}))

    def test_bad_types_in_the_lists_do_not_break_or_weaken_the_guard(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
        from web_guard import Policy
        for value in (None, 5, {"a": 1}, [None, 3]):
            policy = Policy({"allowedHosts": value, "allowedOrigins": value}, "127.0.0.1")
            self.assertTrue(policy.host_allowed("localhost:1"))
            self.assertFalse(policy.host_allowed("evil.example"))
        one = Policy({"allowedHosts": "example.com"}, "127.0.0.1")
        self.assertTrue(one.host_allowed("example.com"))
        self.assertFalse(one.host_allowed("e"))

    def test_the_new_limits_are_sane_defaults(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
        from web_guard import DEFAULTS
        self.assertLessEqual(DEFAULTS["maxRequestBytes"], 64 * 1024 * 1024)
        self.assertLessEqual(DEFAULTS["maxConnections"], 128)


class KeyedGateway(unittest.TestCase):
    """With a gateway key everything that wakes the model or changes something needs it; the status stays readable."""

    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness(behaviour={"n_tokens": 3}, guardian={"slotSaveCheckpoints": 1, "gatewayApiKey": "gw-key"})
        cls.gw.start()

    @classmethod
    def tearDownClass(cls):
        cls.gw.stop()

    def test_an_unauthenticated_request_cannot_wake_the_model(self):
        body = json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()
        self.assertEqual(raw(self.gw, "POST", "/v1/chat/completions", {"Content-Type": "application/json"}, body)[0], 401)
        self.assertEqual(raw(self.gw, "GET", "/guardian/status")[1]["status"], "UNLOADED")
        status, _, _ = raw(self.gw, "POST", "/v1/chat/completions", {"Content-Type": "application/json", "Authorization": "Bearer gw-key"}, body)
        self.assertEqual(status, 200)


class AllowedOriginKeepAlive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness(behaviour={"n_tokens": 3}, guardian={"slotSaveCheckpoints": 1, "allowedOrigins": ["http://localhost:3000"]})
        cls.gw.start()

    @classmethod
    def tearDownClass(cls):
        cls.gw.stop()

    def test_a_refusal_does_not_carry_the_previous_requests_cors_header(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.gw.gw_port, timeout=10)
        conn.request("GET", "/guardian/status", headers={"Origin": "http://localhost:3000"})
        first = conn.getresponse()
        first.read()
        self.assertEqual(first.getheader("Access-Control-Allow-Origin"), "http://localhost:3000")
        conn.request("GET", "/guardian/status", headers={"Origin": "http://evil.example"})
        second = conn.getresponse()
        second.read()
        conn.close()
        self.assertEqual(second.status, 403)
        self.assertIsNone(second.getheader("Access-Control-Allow-Origin"))

    def test_the_preflight_of_a_listed_origin_to_the_control_endpoints_is_answered(self):
        status, _, headers = raw(self.gw, "OPTIONS", "/guardian/ai-on", {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
        self.assertEqual(status, 204)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "http://localhost:3000")


if __name__ == "__main__":
    unittest.main()
