"""RSIL0b --fake stand: the shura-qa fake llama-server with a policy for the pilot's end-to-end check. It is not a model.

The policy reads the system text the engine sent. If it carries the candidate marker (the harness the candidate added), the fake writes
the right files (the task's solution) with one write tool call per file; otherwise it writes the wrong files with the same number of
calls, so the cost of the two arms is comparable. Then it answers with text. Every request is logged to RSI_FAKE_LOG.
"""
import importlib.util
import json
import os
import re
import sys
import time

spec = importlib.util.spec_from_file_location("fk", "/home/alisher/Projects/shura-qa/fake_llama/fake_llama_server.py")
fk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fk)

MARKER = os.environ.get("RSI_FAKE_MARKER", "FAKE-MARK-7")
MANIFEST = os.environ.get("RSI_FAKE_MANIFEST", "")
LOG = os.environ.get("RSI_FAKE_LOG", "")
USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
TIMINGS = {"prompt_n": 100, "predicted_n": 20}
WORKDIR = re.compile(r"Working directory: (\S+)")


def send(self, body, item):
    """Answer one request with `item` ({text} and/or {tool_calls}), as a non-streamed or a streamed chat completion."""
    calls = item.get("tool_calls") or []
    text = item.get("text", "")
    if not body.get("stream"):
        msg = {"role": "assistant", "content": text}
        if calls:
            msg["tool_calls"] = [{"id": f"call_{k}", "type": "function",
                                  "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                                 for k, c in enumerate(calls)]
        return self.send_json(200, {"id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()), "model": "fake",
                                    "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                                    "usage": USAGE, "timings": TIMINGS})
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Transfer-Encoding", "chunked")
    self.end_headers()

    def chunk(delta, finish=None, extra=None):
        obj = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": int(time.time()), "model": "fake",
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if extra:
            obj.update(extra)
        data = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    chunk({"role": "assistant", "content": ""})
    if text:
        chunk({"content": text})
    for k, c in enumerate(calls):
        chunk({"tool_calls": [{"index": k, "id": f"call_{k}", "type": "function",
                               "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}]})
    chunk({}, "tool_calls" if calls else "stop", {"usage": USAGE, "timings": TIMINGS})
    end = b"data: [DONE]\n\n"
    self.wfile.write(f"{len(end):x}\r\n".encode() + end + b"\r\n0\r\n\r\n")
    self.wfile.flush()


def is_title(body):
    """The engine's title request has no tools; the answer is a short text."""
    return not body.get("tools") and "title" in json.dumps(body.get("messages", [])[:1], ensure_ascii=False).lower()


def policy(body):
    messages = body.get("messages") or []
    system = " ".join(str(m.get("content", "")) for m in messages if m.get("role") in ("system", "developer"))
    marker = MARKER in system
    done = sum(1 for m in messages if m.get("role") == "tool")
    where = WORKDIR.search(system)
    files = []
    if MANIFEST and os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            files = json.load(f)["right" if marker else "wrong"]
    if LOG:
        with open(LOG, "a") as f:
            f.write(json.dumps({"ts": time.time(), "marker": marker, "tool_results": done, "files": len(files)}) + "\n")
    if where is None or done >= len(files):
        return {"text": "готово"}
    f = files[done]
    if f.get("run"):   # GEVAL1b: a step that runs a command in the project (e.g. the ffmpeg script the previous step wrote)
        return {"tool_calls": [{"name": "bash", "arguments": {"command": f["run"], "description": "run the script"}}]}
    if f.get("tool"):  # AGTOOLS1: a step that calls any tool with given arguments (e.g. render / frames)
        return {"tool_calls": [{"name": f["tool"], "arguments": f.get("args") or {}}]}
    return {"tool_calls": [{"name": "write", "arguments": {"filePath": f"{where.group(1)}/{f['path']}", "content": f["content"]}}]}


def chat_inner(self, u, body, cfg):
    if is_title(body):
        return send(self, body, {"text": "Тесты"})
    if not self.loaded():
        return self.send_json(503, {"error": {"code": 503, "message": "Loading model"}})
    return send(self, body, policy(body))


fk.Handler.chat_inner = chat_inner

if __name__ == "__main__":
    fk.main(sys.argv[1:])
