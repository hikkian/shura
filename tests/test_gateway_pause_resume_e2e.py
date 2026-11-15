"""PAUSEWAIT1 end to end on the fake llama-server: pause -> blocked -> released -> resume.

The desk takes the card's memory while the answer runs (the test writes a smaller free-VRAM value into the file named by
AI_GATEWAY_TEST_VRAM_FREE_FILE, which the gateway reads instead of NVML). The fake server pauses the answer with a checkpoint and
exits; the gateway waits for the memory (SSE keep-alive comments go to the client meanwhile), the test gives the memory back, and
the answer resumes on the restored session with no error for the client.
"""

import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
sys.path.insert(0, str(HERE))

from gateway_harness import GatewayHarness  # noqa: E402

HOOKS = {"extraArgs": ["--vram-test-hooks"], "vramFreeMinGBToLoad": 2}
FREE_HIGH_MIB = 14000
FREE_LOW_MIB = 500


def wait_until(predicate, timeout=60.0, step=0.1):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(step)
    return False


def stream_raw(gw, body, out):
    """POST a streamed chat request; the raw bytes (keep-alive comments included) go to out['raw']."""
    conn = http.client.HTTPConnection("127.0.0.1", gw.gw_port, timeout=120)
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    out["raw"] = conn.getresponse().read().decode("utf-8", "replace")
    conn.close()


def parse(raw):
    content, errors = [], []
    for line in raw.splitlines():
        if not line.startswith("data:") or line.strip() == "data: [DONE]":
            continue
        value = json.loads(line[5:].strip())
        if "error" in value:
            errors.append(value)
            continue
        for choice in value.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
    return "".join(content), errors


class PauseWaitEndToEnd(unittest.TestCase):
    def test_pause_then_memory_comes_back_then_the_answer_resumes(self):
        vram_file = Path(tempfile.mkdtemp(prefix="shura-pausewait-vram-", dir="/dev/shm")) / "free_mib.txt"
        vram_file.write_text(str(FREE_HIGH_MIB))
        guardian = {"vramPolicy": "governor", "pauseResumeWaitSeconds": 60, "pauseResumePollSeconds": 0.5}
        behaviour = {"n_tokens": 40, "token_delay_ms": 100, "text": "w", "pause_after_tokens": 20}
        env = {"AI_GATEWAY_TEST_VRAM_FREE_FILE": str(vram_file)}
        with GatewayHarness(guardian=guardian, behaviour=behaviour, model=HOOKS, env=env) as gw:
            gw.start()
            out = {}
            body = {"messages": [{"role": "user", "content": "x"}], "stream": True, "max_tokens": 64}
            thread = threading.Thread(target=stream_raw, args=(gw, body, out))
            thread.start()

            # the desk takes the card as soon as the server is up, so the pause finds no memory to resume into
            self.assertTrue(wait_until(lambda: "server_started" in gw.log()), gw.log()[-1500:])
            time.sleep(0.2)
            vram_file.write_text(str(FREE_LOW_MIB))

            # the resume waits: the event is the proof that the gateway saw the pause and is refused for memory
            self.assertTrue(wait_until(lambda: "vram_pause_resume_waiting" in gw.log(), timeout=60), gw.log()[-2000:])
            time.sleep(1.5)     # the client gets keep-alive comments while the gateway waits
            vram_file.write_text(str(FREE_HIGH_MIB))

            thread.join(timeout=120)
            self.assertFalse(thread.is_alive(), "the answer did not finish")
            text, errors = parse(out["raw"])
            events = gw.log()
            status = gw.status()

        self.assertEqual(errors, [], "the client saw an error: " + json.dumps(errors)[:300])
        self.assertEqual(text, "w " * 40, events[-1500:])
        self.assertIn(": keep-alive", out["raw"], "no keep-alive reached the client while the answer waited")
        self.assertIn("vram_pause_resume_ready", events)
        self.assertIn("vram_pause_resumed", events)
        self.assertNotIn("SHURA_RESPONSE_INTERRUPTED", out["raw"])
        self.assertEqual(status.get("pause_resume", {}).get("state"), "ready", status.get("pause_resume"))


class PauseWaitExpiresEndToEnd(unittest.TestCase):
    def test_memory_never_comes_back_and_the_client_gets_the_reason(self):
        vram_file = Path(tempfile.mkdtemp(prefix="shura-pausewait-vram-", dir="/dev/shm")) / "free_mib.txt"
        vram_file.write_text(str(FREE_HIGH_MIB))
        guardian = {"vramPolicy": "governor", "pauseResumeWaitSeconds": 3, "pauseResumePollSeconds": 0.5}
        behaviour = {"n_tokens": 40, "token_delay_ms": 100, "text": "w", "pause_after_tokens": 20}
        env = {"AI_GATEWAY_TEST_VRAM_FREE_FILE": str(vram_file)}
        with GatewayHarness(guardian=guardian, behaviour=behaviour, model=HOOKS, env=env) as gw:
            gw.start()
            out = {}
            body = {"messages": [{"role": "user", "content": "x"}], "stream": True, "max_tokens": 64}
            thread = threading.Thread(target=stream_raw, args=(gw, body, out))
            thread.start()
            self.assertTrue(wait_until(lambda: "server_started" in gw.log()), gw.log()[-1500:])
            time.sleep(0.2)
            vram_file.write_text(str(FREE_LOW_MIB))     # the desk keeps the memory
            thread.join(timeout=120)
            self.assertFalse(thread.is_alive(), "the answer did not end")
            _text, errors = parse(out["raw"])
            events = gw.log()
            status = gw.status()

        self.assertIn("vram_pause_resume_expired", events)
        self.assertEqual(status.get("pause_resume", {}).get("state"), "expired", status.get("pause_resume"))
        self.assertTrue(errors, "the answer must end with an error the client can read")
        self.assertEqual(errors[-1]["error"]["code"], "SHURA_RESPONSE_INTERRUPTED")
        self.assertIn("Not enough free VRAM", json.dumps(errors, ensure_ascii=False) + events[-3000:],
                      "the reason of the wait must reach the client or the log")


if __name__ == "__main__":
    unittest.main()
