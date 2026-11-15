"""Empty turn (gateway/empty_turn.py): a stream that ends by the output limit with no text and no tool call is continued.

The upstream is a fake llama-server that answers each request with the next scripted answer (SSE frames, or a status code);
it records the request bodies. The client side is the same `Front` the stream tests use, so the wire is the real one.
"""
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gateway"))
from test_gateway_streams import Front, fetch  # noqa: E402
import empty_turn  # noqa: E402

REQUEST = json.dumps({"model": "m", "stream": True, "messages": [{"role": "user", "content": "go"}]}).encode()
ON = {"emptyTurnContinue": True}


def reasoning(text):
    return {"choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}]}


def answer(text):
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}


def tool_call():
    return {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                                              "function": {"name": "bash", "arguments": "{}"}}]},
                         "finish_reason": None}]}


def finish(reason):
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}], "usage": {"completion_tokens": 5}}


def sse(*frames):
    return b"".join(b"data: " + json.dumps(f).encode() + b"\n\n" for f in frames) + b"data: [DONE]\n\n"


def empty_length():
    return sse(reasoning("thinking "), reasoning("thinking "), reasoning("thinking "), finish("length"))


class FakeLlama:
    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                outer.requests.append(json.loads(self.rfile.read(n) or b"{}"))
                answer_spec = outer.answers.pop(0) if outer.answers else 500
                if isinstance(answer_spec, int):
                    body = b'{"error":{"message":"forced by the test"}}'
                    self.send_response(answer_spec)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                # 7-byte pieces, so frames are cut in the middle on the way
                for i in range(0, len(answer_spec), 7):
                    piece = answer_spec[i:i + 7]
                    self.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def client_payloads(body):
    """The `data:` payloads the client received, in order (the [DONE] marker included)."""
    out = []
    for frame in body.decode().split("\n\n"):
        for line in frame.split("\n"):
            if line.startswith("data: "):
                out.append(line[6:])
    return out


def finish_reasons(payloads):
    reasons = []
    for p in payloads:
        if p == "[DONE]":
            continue
        for c in json.loads(p).get("choices") or []:
            if c.get("finish_reason"):
                reasons.append(c["finish_reason"])
    return reasons


class EmptyTurn(unittest.TestCase):
    def setUp(self):
        self.cleanup = []
        for key in empty_turn.STATS:
            empty_turn.STATS[key] = 0

    def tearDown(self):
        for item in self.cleanup:
            item.close()

    def run_turn(self, answers, config):
        llama = FakeLlama(answers)
        front = Front(llama.port, config=config)
        self.cleanup += [llama, front]
        status, _, body, complete = fetch(front.port, body=REQUEST)
        self.assertEqual(status, 200)
        self.assertTrue(complete, "client did not get a complete answer")
        return llama.requests, client_payloads(body)

    def test_empty_length_turn_is_continued_on_the_same_answer(self):
        requests, payloads = self.run_turn([empty_length(), sse(answer("the result"), finish("stop"))], ON)
        self.assertEqual(len(requests), 2)
        self.assertEqual(finish_reasons(payloads), ["stop"])          # the length finish was replaced, not forwarded
        self.assertEqual(payloads[-1], "[DONE]")
        self.assertEqual(sum(1 for p in payloads if p == "[DONE]"), 1)
        self.assertTrue(any("the result" in p for p in payloads))
        self.assertTrue(any("thinking" in p for p in payloads))         # the reasoning was not cut from the client
        second = requests[1]
        self.assertTrue(second["continue_final_message"])
        self.assertFalse(second["add_generation_prompt"])
        last = second["messages"][-1]
        self.assertEqual(last["role"], "assistant")
        self.assertEqual(last["reasoning_content"], "thinking thinking thinking ")
        self.assertEqual(second["messages"][0], {"role": "user", "content": "go"})
        self.assertEqual(empty_turn.STATS["continued"], 1)

    def test_turn_with_text_is_not_continued(self):
        requests, payloads = self.run_turn([sse(reasoning("hm"), answer("partial"), finish("length"))], ON)
        self.assertEqual(len(requests), 1)
        self.assertEqual(finish_reasons(payloads), ["length"])
        self.assertEqual(empty_turn.STATS["continued"], 0)

    def test_turn_with_tool_call_is_not_continued(self):
        requests, payloads = self.run_turn([sse(reasoning("hm"), tool_call(), finish("length"))], ON)
        self.assertEqual(len(requests), 1)
        self.assertEqual(finish_reasons(payloads), ["length"])
        self.assertEqual(empty_turn.STATS["continued"], 0)

    def test_continuations_are_capped(self):
        requests, payloads = self.run_turn([empty_length(), empty_length(), empty_length()], ON)
        self.assertEqual(len(requests), 1 + empty_turn.MAX_CONTINUATIONS)
        self.assertEqual(finish_reasons(payloads), ["length"])         # the last empty turn is what the client gets
        self.assertEqual(sum(1 for p in payloads if p == "[DONE]"), 1)
        self.assertEqual(empty_turn.STATS["continued"], empty_turn.MAX_CONTINUATIONS)
        self.assertEqual(empty_turn.STATS["exhausted"], 1)

    def test_failed_continuation_gives_the_empty_turn_its_end(self):
        requests, payloads = self.run_turn([empty_length(), 500], ON)
        self.assertEqual(len(requests), 2)
        self.assertEqual(finish_reasons(payloads), ["length"])
        self.assertEqual(payloads[-1], "[DONE]")
        self.assertEqual(empty_turn.STATS["gave_up"], 1)
        self.assertEqual(empty_turn.STATS["continued"], 0)

    def test_switch_off_leaves_the_stream_alone(self):
        requests, payloads = self.run_turn([empty_length()], {})
        self.assertEqual(len(requests), 1)
        self.assertEqual(finish_reasons(payloads), ["length"])
        self.assertEqual(payloads[-1], "[DONE]")
        self.assertEqual(empty_turn.STATS["continued"], 0)

    def test_events_reach_the_gateway_observer(self):
        events = []
        llama = FakeLlama([empty_length(), empty_length(), empty_length()])
        front = Front(llama.port, config=ON, observe=lambda kind, value: events.append((kind, value)) if kind == "empty_turn" else None)
        self.cleanup += [llama, front]
        fetch(front.port, body=REQUEST)
        self.assertEqual(events, [("empty_turn", "continued"), ("empty_turn", "continued"), ("empty_turn", "exhausted")])

    def test_non_chat_path_is_not_touched(self):
        llama = FakeLlama([empty_length()])
        front = Front(llama.port, config=ON)
        self.cleanup += [llama, front]
        status, _, body, complete = fetch(front.port, path="/v1/completions", body=REQUEST)
        self.assertEqual(status, 200)
        self.assertEqual(len(llama.requests), 1)
        self.assertEqual(finish_reasons(client_payloads(body)), ["length"])


class GatewayEmptyTurn(unittest.TestCase):
    """The whole gateway against the fake llama-server (tests/support), with the switch from the guardian config."""

    def turn(self, guardian):
        from gateway_harness import GatewayHarness
        behaviour = {"empty_length_turns": 1, "n_tokens": 3, "reasoning_tokens": 4, "text": "ok"}
        with GatewayHarness(behaviour=behaviour, guardian=guardian) as h:
            status, out = h.chat(content="hello", stream=True, timeout=120)
            log = h.fake_log()
            requests = [e["body"] for e in log if e.get("path") == "/v1/chat/completions"] if isinstance(log, list) else []
            return status, out, requests

    def test_gateway_continues_the_empty_turn(self):
        status, out, requests = self.turn({"emptyTurnContinue": True})
        self.assertEqual(status, 200)
        self.assertTrue(out["complete"], out)
        reasons = [c.get("finish_reason") for line in out["lines"] if line != "[DONE]"
                   for c in json.loads(line).get("choices") or [] if c.get("finish_reason")]
        self.assertEqual(reasons, ["stop"])
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1]["messages"][-1]["role"], "assistant")

    def test_gateway_without_the_switch_keeps_the_empty_turn(self):
        status, out, requests = self.turn({})
        self.assertEqual(status, 200)
        reasons = [c.get("finish_reason") for line in out["lines"] if line != "[DONE]"
                   for c in json.loads(line).get("choices") or [] if c.get("finish_reason")]
        self.assertEqual(reasons, ["length"])
        self.assertEqual(len(requests), 1)


class PureParts(unittest.TestCase):
    def test_frames_cut_across_reads(self):
        frames = empty_turn.SSEFrames()
        data = sse(reasoning("a"), finish("stop"))
        got = []
        for i in range(0, len(data), 5):
            got += frames.feed(data[i:i + 5])
        self.assertEqual(len(got), 3)
        self.assertEqual(b"".join(got), data)

    def test_turn_state_classifies_the_turn(self):
        cases = {
            "empty length": (sse(reasoning("x"), finish("length")), True),
            "text length": (sse(answer("y"), finish("length")), False),
            "tool length": (sse(tool_call(), finish("length")), False),
            "empty stop": (sse(reasoning("x"), finish("stop")), False),
        }
        for name, (data, expected) in cases.items():
            state, frames = empty_turn.TurnState(), empty_turn.SSEFrames()
            for frame in frames.feed(data):
                state.feed(frame)
            self.assertEqual(state.empty_length(), expected, name)

    def test_continuation_body_only_for_chat_requests(self):
        self.assertIsNone(empty_turn.continuation_body(b"not json", "x"))
        self.assertIsNone(empty_turn.continuation_body(b'{"prompt": "x"}', "x"))
        self.assertIsNone(empty_turn.continuation_body(b'{"messages": []}', "x"))
        out = json.loads(empty_turn.continuation_body(REQUEST, "plan"))
        self.assertEqual(out["messages"][-1], {"role": "assistant", "content": "", "reasoning_content": "plan"})
        self.assertTrue(out["continue_final_message"])


if __name__ == "__main__":
    unittest.main()
