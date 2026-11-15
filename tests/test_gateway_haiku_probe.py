"""QA1 regression probes for the Shura gateway. Fake llama-server, no GPU.

Each test says what a careful gateway must do and why. The probes pin existing behaviour and the four regressions fixed
from the QA1 report (reports/QA1-gateway-probe.md). They run against the new gateway and fake llama-server, without a GPU.
"""
import json
import os
import signal
import socket
import sys
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
from gateway_harness import GatewayHarness  # noqa: E402

FULL = {"slotSaveCheckpoints": 1}          # the new gateway; the default config selects the legacy one
TERSE_MODEL = {"chatTemplateKwargs": {"terse": False}}   # the model has the terse switch, so apply_terse rewrites bodies
CHAT = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"


def exchange(port, data, wait=4.0):
    """Send raw bytes and return everything the server writes until it closes or `wait` seconds pass."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=wait)
    try:
        sock.sendall(data)
        out, end = b"", time.time() + wait
        while time.time() < end:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            out += chunk
        return out
    finally:
        sock.close()


def answers(raw):
    """Each HTTP answer in a raw byte stream, split at its status line (bodies are not framed here)."""
    return [raw[i:] for i in range(len(raw)) if raw.startswith(b"HTTP/1.1 ", i)]


def gateway_traceback(log):
    """True when the log holds a traceback that runs through the gateway's own code. The fake llama-server writes to the same
    log, so a traceback raised inside tests/support is its noise, not the gateway's."""
    return any("/gateway/" in block for block in log.split("Traceback")[1:])


def started(harness, timeout=25):
    """Start the harness, turning a start-up failure into a clear test failure that quotes the gateway's own error."""
    try:
        return harness.start(timeout=timeout)
    except RuntimeError as error:
        last = [line for line in str(error).splitlines() if line.strip()][-1:]
        raise AssertionError(f"the gateway did not start: {last[0] if last else error}") from None


class LoadingClients(unittest.TestCase):
    """Clients that come and go while the model is still loading."""

    def test_a_client_that_leaves_during_the_load_does_not_lose_the_model(self):
        """A chat client that closes its socket 0.7 s into a 3 s load. The load is the expensive part and it is shared
        by everyone, so the gateway should finish it anyway: the model reaches READY, no request stays busy, only one
        server is listening, and the next client is served at once. A gateway that dropped the load would make
        every abandoned request cost a full model load."""
        with GatewayHarness(behaviour={"load_seconds": 3, "n_tokens": 3, "token_delay_ms": 1}, guardian=FULL) as gw:
            body = json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()
            sock = socket.create_connection(("127.0.0.1", gw.gw_port), timeout=5)
            sock.sendall(CHAT + b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
            time.sleep(0.7)
            sock.close()
            gw.wait_for(lambda: gw.status()["status"] == "READY", timeout=20, what="the load to finish after the client left")
            gw.wait_for(lambda: gw.status()["busy_requests"] == 0, timeout=10, what="no request to stay busy")
            self.assertEqual(len(GatewayHarness.pids_on(gw.llama_port)), 1)
            code, _ = gw.chat("next client")
            self.assertEqual(code, 200)
            self.assertFalse(gateway_traceback(gw.log()), gw.log()[-800:])

    def test_two_first_requests_during_the_load_share_one_backend(self):
        """Two chat clients arrive while the model loads (cold start, 2 s). Both must get answers (200), each request must
        reach the backend exactly once (no replay, no double load), and only one llama-server may be listening.
        Two servers would double the GPU memory the desktop is trying to keep free."""
        with GatewayHarness(behaviour={"load_seconds": 2, "n_tokens": 3, "token_delay_ms": 1}, guardian=FULL) as gw:
            codes = []
            first = threading.Thread(target=lambda: codes.append(gw.chat("x")[0]))
            first.start()
            gw.wait_for(lambda: gw.status()["status"] == "LOADING", timeout=10, what="the first request to start the load")
            second = threading.Thread(target=lambda: codes.append(gw.chat("x")[0]))   # arrives while the model is loading
            second.start()
            for t in (first, second):
                t.join(60)
            self.assertEqual(sorted(codes), [200, 200])
            chats = [e for e in gw.fake_log() if e.get("path", "").endswith("/v1/chat/completions")]
            self.assertEqual(len(chats), 2, "each client's request reaches the backend exactly once")
            self.assertEqual(len(GatewayHarness.pids_on(gw.llama_port)), 1)
            self.assertFalse(gateway_traceback(gw.log()), gw.log()[-800:])


class StopAndLimits(unittest.TestCase):
    """A stop signal in the middle of an answer, and a body that is larger than the gateway will take."""

    def test_sigterm_during_a_stream_stops_the_gateway_and_its_backend(self):
        """A streamed answer of 400 tokens at 20 ms is in progress when the gateway gets SIGTERM. The gateway must exit
        within 20 s and must not leave llama-server behind: a backend still holding the GPU after its gateway is gone is
        exactly the state the desktop cannot recover from by itself."""
        gw = GatewayHarness(behaviour={"n_tokens": 400, "token_delay_ms": 20}, guardian=FULL).start()
        try:
            def stream():
                try:
                    gw.request("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "stream": True},
                               timeout=30, stream=True)
                except Exception:
                    pass                                  # the connection is cut by the stop: that is expected
            threading.Thread(target=stream, daemon=True).start()
            gw.wait_for(lambda: gw.status()["busy_requests"] == 1, what="the stream to be in progress")
            # the gateway alone, not its process group: a group signal would also kill the backend and hide an orphan
            os.kill(gw.proc.pid, signal.SIGTERM)
            try:
                gw.proc.wait(timeout=20)
            except Exception:
                self.fail("the gateway did not exit within 20 s of SIGTERM during a stream")
            deadline = time.time() + 5
            while GatewayHarness.pids_on(gw.llama_port) and time.time() < deadline:
                time.sleep(0.2)
            self.assertEqual(GatewayHarness.pids_on(gw.llama_port), [], "llama-server was left behind")
        finally:
            gw.stop()

    def test_a_body_above_the_limit_is_refused_before_the_model_loads(self):
        """A POST that declares a 40 MB body (above maxRequestBytes, 32 MB) and sends no body at all. The gateway must
        answer 413 from the headers alone: it must not wait for 40 MB, and must not start a model for a request it is
        going to refuse. Starting a 20 GB model for a refused request is the cost this protects against."""
        with GatewayHarness(behaviour={"n_tokens": 2}, guardian=FULL) as gw:
            raw = exchange(gw.gw_port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                                       b"Content-Length: 40000000\r\n\r\n", wait=5)
            self.assertTrue(raw.startswith(b"HTTP/1.1 413"), raw[:80])
            self.assertEqual(gw.status()["status"], "UNLOADED", "a refused body must not load the model")
            self.assertEqual(GatewayHarness.pids_on(gw.llama_port), [])


class PipelinedAndPaths(unittest.TestCase):
    """Several requests on one connection, and spellings of the switch paths that are not the switch path."""

    def test_pipelined_requests_are_answered_one_by_one_in_order(self):
        """Three requests written before any answer is read: two health checks and the status. The gateway must answer
        all three, in order, on the same connection. A proxy that drops or mixes pipelined requests lets one client's
        answer land in another's stream."""
        with GatewayHarness(behaviour={"n_tokens": 2}, guardian=FULL) as gw:
            gw.chat("warm")
            health = b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n"
            raw = exchange(gw.gw_port, health + health + b"GET /guardian/status HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            parts = answers(raw)
            self.assertEqual(len(parts), 3, raw[:300])
            self.assertTrue(all(p.startswith(b"HTTP/1.1 200") for p in parts))
            self.assertIn(b'"gateway_status"', parts[0])
            self.assertIn(b'"gateway_status"', parts[1])
            self.assertIn(b'"override"', parts[2], "the third answer must be the status, not another health check")

    def test_a_path_that_is_not_the_exact_switch_path_changes_nothing(self):
        """Spellings of /guardian/ai-off that are not the switch path (trailing slash, doubled or dotted segments,
        percent-encoding, a stray `;`, a NUL, another case). None may switch the AI off, because the route table is an
        exact match and anything else would be a route nobody reviewed. The exact path is the positive control: the
        probe must be able to switch the AI off, or the negative results mean nothing. The leading `//` spelling is
        left out on purpose: the standard library normalises it to `/` before the policy runs, so the same checks apply."""
        with GatewayHarness(behaviour={"n_tokens": 2}, guardian=FULL) as gw:
            gw.chat("warm")
            spellings = ["/guardian/ai-off/", "/guardian//ai-off", "/guardian/./ai-off", "/guardian/ai-off%20",
                         "/guardian/ai%2Doff", "/guardian/ai-off;x", "/guardian/AI-OFF", "/guardian/ai-off%00"]
            for path in spellings:
                exchange(gw.gw_port, f"POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode(), wait=2)
                self.assertNotEqual(gw.status()["override"], "OFF", f"{path!r} switched the AI off")
            code = gw.request("POST", "/guardian/ai-off")[0]
            self.assertEqual(code, 200, "positive control: the exact path must switch the AI off")
            gw.request("POST", "/guardian/ai-on")


class RegressionFindings(unittest.TestCase):
    """Regression tests for the four gateway defects identified by QA1."""

    def test_a_short_body_gets_a_408_not_silence(self):
        """A POST declares 500 bytes but stalls after 10; return a 408 and then keep serving normal clients."""
        with GatewayHarness(behaviour={"n_tokens": 2}, guardian={**FULL, "requestBodySeconds": 2}) as gw:
            raw = exchange(gw.gw_port, CHAT + b"Content-Length: 500\r\n\r\n{\"messages\":", wait=8)
            self.assertTrue(raw.startswith(b"HTTP/1.1 408"), f"expected a 408 answer, got {raw[:60]!r}")
            self.assertIn(b"Connection: close", raw)
            for _ in range(8):
                self.assertEqual(gw.chat("after body timeout")[0], 200)

    def test_slow_headers_get_408_and_release_the_connection(self):
        """A header deadline must produce 408 and free capacity; ordinary requests before and after still work."""
        with GatewayHarness(behaviour={"n_tokens": 2}, guardian={**FULL, "requestHeaderSeconds": 2}) as gw:
            self.assertEqual(gw.chat("fast before timeout")[0], 200)
            partial = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                       b"X-Partial: ")
            raw = exchange(gw.gw_port, partial, wait=8)
            self.assertTrue(raw.startswith(b"HTTP/1.1 408"), f"expected header timeout, got {raw[:60]!r}")
            self.assertIn(b"Connection: close", raw)
            self.assertEqual(gw.request("GET", "/health")[0], 200)
            for _ in range(8):
                self.assertEqual(gw.chat("after header timeout")[0], 200)

    def test_ambiguous_authorization_headers_are_refused(self):
        """A protected POST (`/guardian/ai-on`, with an API key configured) carrying two Authorization headers, one right
        and one wrong, in either order. Reject any duplicate credentials, accept one valid credential, and refuse three
        values too."""
        with GatewayHarness(behaviour={"n_tokens": 2}, guardian={**FULL, "gatewayApiKey": "good-key"}) as gw:
            good, bad = b"Authorization: Bearer good-key\r\n", b"Authorization: Bearer nope\r\n"
            head = b"POST /guardian/ai-on HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n"
            right_first = exchange(gw.gw_port, head + good + bad + b"\r\n", wait=3)
            wrong_first = exchange(gw.gw_port, head + bad + good + b"\r\n", wait=3)
            triple = exchange(gw.gw_port, head + good + bad + bad + b"\r\n", wait=3)
            single = exchange(gw.gw_port, head + good + b"\r\n", wait=3)
            self.assertTrue(right_first.startswith(b"HTTP/1.1 401"), f"right first got {right_first[:20]!r}")
            self.assertTrue(wrong_first.startswith(b"HTTP/1.1 401"), f"wrong first got {wrong_first[:20]!r}")
            self.assertTrue(triple.startswith(b"HTTP/1.1 401"), f"three headers got {triple[:20]!r}")
            self.assertTrue(single.startswith(b"HTTP/1.1 200"), f"one valid header got {single[:20]!r}")

    def test_a_deeply_nested_body_with_the_terse_switch_gets_an_answer(self):
        """With the terse switch on, a 200 KB body of 100 000 nested brackets (valid size, not valid JSON) is sent to
        /v1/chat/completions. Both body inspection and terse rewriting must tolerate RecursionError; the backend can then
        answer 400, with no gateway traceback, while the gateway still answers /health."""
        depth = 100000
        deep = b'{"model":"fake","messages":[{"role":"user","content":"x"}],"x":' + b"[" * depth + b"]" * depth + b"}"
        with GatewayHarness(behaviour={"n_tokens": 2}, guardian=FULL, model=TERSE_MODEL) as gw:
            gw.chat("warm")
            self.assertEqual(gw.request("POST", "/guardian/terse-on")[0], 200)
            raw = exchange(gw.gw_port, CHAT + b"Content-Length: " + str(len(deep)).encode() + b"\r\n\r\n" + deep, wait=8)
            self.assertTrue(raw.startswith(b"HTTP/1.1 "), f"no answer at all: {raw[:40]!r}")
            self.assertFalse(gateway_traceback(gw.log()), gw.log()[-800:])
            self.assertEqual(gw.request("GET", "/health")[0], 200)

    def test_a_corrupt_override_flag_does_not_stop_the_gateway_starting(self):
        """The override flag (`override.flag` in the state folder, written by ai-on/ai-off) holds bytes that are not
        UTF-8, for example after a disk error. It must start with the flag treated as absent (AI on, automatic) and log
        that it ignored the damaged flag."""
        gw = GatewayHarness(guardian=FULL)
        (gw.tmp / "state").mkdir(exist_ok=True)
        (gw.tmp / "state" / "override.flag").write_bytes(b"\xff\xfe\x00bad")
        try:
            started(gw)
            self.assertEqual(gw.status()["override"], "AUTO")
        finally:
            gw.stop()

    def test_a_corrupt_terse_flag_does_not_stop_the_gateway_starting(self):
        """The same damaged UTF-8 case for `terse.flag`: start with the terse override unset and log the ignored flag."""
        gw = GatewayHarness(guardian=FULL)
        (gw.tmp / "state").mkdir(exist_ok=True)
        (gw.tmp / "state" / "terse.flag").write_bytes(b"\xff\xfe\x00bad")
        try:
            started(gw)
            self.assertIsNone(gw.status().get("terse_override"))
        finally:
            gw.stop()

    def test_unreadable_state_flags_are_ignored_when_running_unprivileged(self):
        """Mode 000 must be tolerated by a non-root gateway, with both switches taking their defaults."""
        if os.geteuid() == 0:
            self.skipTest("mode 000 is readable by root; PermissionError cannot be exercised")
        gw = GatewayHarness(guardian=FULL)
        state = gw.tmp / "state"
        state.mkdir(exist_ok=True)
        override = state / "override.flag"
        terse = state / "terse.flag"
        override.write_text("OFF")
        terse.write_text("ON")
        override.chmod(0)
        terse.chmod(0)
        try:
            started(gw)
            self.assertEqual(gw.status()["override"], "AUTO")
            self.assertIsNone(gw.status().get("terse_override"))
            log = gw.log()
            self.assertIn("State flag ignored: override.flag (PermissionError:", log)
            self.assertIn("State flag ignored: terse.flag (PermissionError:", log)
        finally:
            gw.stop()


class BodiesThatAreNotJson(unittest.TestCase):
    """Bodies the backend would reject: they must get an answer and must not crash the gateway."""

    def test_empty_and_non_json_bodies_get_an_answer_and_no_traceback(self):
        """Three chat requests with bodies that are empty, not UTF-8 inside a JSON string, and not JSON at all. The fake
        backend answers each one 200 (it does not validate), so what this pins is the gateway: each request gets a
        complete answer, nothing is logged as a traceback, and the gateway still answers /health afterwards. A
        gateway that crashed on a body it does not understand would take the whole desktop's AI down with it."""
        with GatewayHarness(behaviour={"n_tokens": 2}, guardian=FULL) as gw:
            gw.chat("warm")
            bad_utf8 = b'{"model":"fake","messages":[{"role":"user","content":"\xff\xfe bad"}]}'
            cases = {
                "empty": CHAT + b"Content-Length: 0\r\nConnection: close\r\n\r\n",
                "invalid utf-8": CHAT + b"Content-Length: " + str(len(bad_utf8)).encode() + b"\r\nConnection: close\r\n\r\n" + bad_utf8,
                "not json": CHAT + b"Content-Length: 5\r\nConnection: close\r\n\r\nhello",
            }
            for name, data in cases.items():
                raw = exchange(gw.gw_port, data, wait=5)
                self.assertTrue(raw.startswith(b"HTTP/1.1 "), f"{name}: no answer, got {raw[:40]!r}")
            self.assertFalse(gateway_traceback(gw.log()), gw.log()[-800:])
            self.assertEqual(gw.request("GET", "/health")[0], 200)


if __name__ == "__main__":
    unittest.main()
