"""The language guard must never lose an answer (GUARDFIX1).

`forward_request` with `lang_guard_lang` set streams the answer through lang_guard.StreamGuard. Three outcomes, all through
that one path:
  1. the guard decides early that the answer is in the right language and passes it through (feed() releases the events);
  2. the guard decides at the end that the answer is in the wrong language and asks the same model for a rewrite;
  3. the answer is too short to decide: the held events go out unchanged at the end.
The client must get the answer text (and the first role chunk) in every case.
"""
import http.client
import json
import socket
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gateway"))
from proxy_core import forward_request  # noqa: E402

RUS_LONG = ("Сервер запускается из конфигурации, затем проверяет, что модель загружена, и только после этого принимает запросы. "
            "Если загрузка не удалась, шлюз сообщает причину клиенту и не оставляет запрос без ответа.")
ENG_LONG = ("The server starts from the configuration, then checks that the model is loaded, and only after that it accepts "
            "requests. If loading fails, the gateway tells the client the reason and never leaves a request without an answer.")
RUS_SHORT = "Готово."
RUS_REWRITE = ("Сервер запускается из конфигурации, проверяет загрузку модели и только потом принимает запросы. "
               "При ошибке загрузки шлюз сообщает клиенту причину.")


def sse_events(text, parts=6, role=True, finish="stop"):
    """Chunks of an OpenAI-style stream: a role chunk, the text in `parts` pieces, then the finish chunk."""
    events = []
    if role:
        events.append({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    size = max(1, len(text) // parts)
    for i in range(0, len(text), size):
        events.append({"choices": [{"index": 0, "delta": {"content": text[i:i + size]}, "finish_reason": None}]})
    events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    return [("data: " + json.dumps(e, ensure_ascii=False) + "\n\n").encode() for e in events] + [b"data: [DONE]\n\n"]


class FakeUpstream:
    """A raw HTTP server: a streaming request gets `stream_events`; a non-streaming request (the rewrite) gets `rewrite`."""

    def __init__(self, stream_events, rewrite_text=None):
        self.stream_events, self.rewrite_text = stream_events, rewrite_text
        self.requests = []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    def handle(self, conn):
        try:
            data = b""
            while b"\r\n\r\n" not in data:
                more = conn.recv(65536)
                if not more:
                    return
                data += more
            head, _, rest = data.partition(b"\r\n\r\n")
            length = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":", 1)[1])
            while len(rest) < length:
                more = conn.recv(65536)
                if not more:
                    break
                rest += more
            req = json.loads(rest[:length] or b"{}")
            self.requests.append(req)
            if req.get("stream") is False:
                payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": self.rewrite_text or ""}}]},
                                     ensure_ascii=False).encode()
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + str(len(payload)).encode()
                             + b"\r\nConnection: close\r\n\r\n" + payload)
            else:
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
                             + b"".join(self.stream_events))
        except OSError:
            pass
        finally:
            conn.close()

    def close(self):
        self.sock.close()


class FrontWithGuard:
    """A client-facing server that calls forward_request with the language guard on, as the gateway's Handler does."""

    def __init__(self, upstream_port):
        port = upstream_port

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n)
                forward_request(self, body, "127.0.0.1", port, {"slotSaveCheckpoints": 0, "emptyTurnContinue": False},
                                lang_guard_lang="Russian")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def ask(port, question):
    body = json.dumps({"model": "tiel", "stream": True, "messages": [{"role": "user", "content": question}]}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    try:
        conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8", "replace")
    finally:
        conn.close()


def client_view(raw):
    """What the client got: (roles seen, the answer text, number of data events)."""
    roles, text, n = [], "", 0
    for line in raw.splitlines():
        if not line.startswith("data:") or line.strip() == "data: [DONE]":
            continue
        n += 1
        obj = json.loads(line[5:].strip())
        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
        if delta.get("role"):
            roles.append(delta["role"])
        text += delta.get("content") or ""
    return roles, text, n


class LanguageGuardKeepsTheAnswer(unittest.TestCase):
    def run_case(self, upstream, question="Расскажи, как работает сервер"):
        front = FrontWithGuard(upstream.port)
        try:
            status, raw = ask(front.port, question)
        finally:
            front.close()
            upstream.close()
        self.assertEqual(status, 200)
        return client_view(raw)

    def test_1_right_language_passes_early_and_the_client_gets_every_word(self):
        up = FakeUpstream(sse_events(RUS_LONG, parts=8))
        roles, text, _ = self.run_case(up)
        self.assertEqual(text, RUS_LONG, "the answer text was lost: the guard released events in feed() and the gateway dropped them")
        self.assertEqual(roles, ["assistant"], "the first role chunk was lost")

    def test_2_wrong_language_is_rewritten_and_the_rewrite_reaches_the_client(self):
        up = FakeUpstream(sse_events(ENG_LONG, parts=8), rewrite_text=RUS_REWRITE)
        roles, text, _ = self.run_case(up)
        self.assertEqual(text, RUS_REWRITE, "the rewritten answer did not reach the client")
        self.assertEqual(roles, ["assistant"], "the role chunk sent before the decision was lost")

    def test_3_short_answer_is_sent_unchanged_at_the_end(self):
        up = FakeUpstream(sse_events(RUS_SHORT, parts=1))
        roles, text, _ = self.run_case(up, "Готов?")
        self.assertEqual(text, RUS_SHORT)
        self.assertEqual(roles, ["assistant"])


if __name__ == "__main__":
    unittest.main()
