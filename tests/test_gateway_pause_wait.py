"""PAUSEWAIT1: after a governor pause the resume waits for the memory the desk holds instead of giving up at once.

The admission a load makes is replaced by a scripted sequence of refusals (no GPU, no server): the waiting must end when the memory
comes back, expire after pauseResumeWaitSeconds with the refusal text, stop at once for a refusal that waiting cannot fix, stop when
the client has gone, and keep the client's stream alive with heartbeats while it waits.
"""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


class Refusal(str):
    def __new__(cls, text, code):
        value = super().__new__(cls, text)
        value.code = code
        return value


class Lease(dict):
    pass


def load_gateway(root):
    for name in ("guardian", "model-launch"):
        (Path(root) / f"{name}.json").write_text((REPO / "config" / f"{name}.example.json").read_text())
    spec = importlib.util.spec_from_file_location("pausewait_test_gateway", REPO / "gateway" / "ai_gateway.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, AI_GATEWAY_CONFIG=root, AI_GATEWAY_STATE=str(Path(root) / "state")):
        spec.loader.exec_module(module)
    return module


class WaitForResumeMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.TemporaryDirectory(prefix="shura-pausewait-", dir="/tmp")
        cls.gw = load_gateway(cls.root.name)

    @classmethod
    def tearDownClass(cls):
        cls.root.cleanup()

    def setUp(self):
        self.events = []
        patches = [
            patch.object(self.gw, "event", side_effect=lambda reason, detail, fallback=False: self.events.append((reason, detail))),
            patch.object(self.gw.time, "sleep", side_effect=lambda seconds: None),
            patch.dict(self.gw.G, {"pauseResumeWaitSeconds": 60, "pauseResumePollSeconds": 0.01}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.gw.PAUSE_WAIT.update(state="idle", model="", since=None, reason="", free_mib=None, waited_s=0.0)

    def script(self, answers):
        """resume_refusal returns the scripted answers in order; the last one repeats."""
        seq = list(answers)

        def fake(model_id, vision, layout):
            return seq.pop(0) if len(seq) > 1 else seq[0]
        return patch.object(self.gw, "resume_refusal", side_effect=fake)

    def test_waits_for_the_memory_then_resumes_and_keeps_the_stream_alive(self):
        beats = []
        memory = Refusal("Not enough free VRAM (11.2 GB free, need 11.60 GB)", "vram")
        with self.script([memory, memory, memory, ""]):
            result = self.gw.wait_for_resume_memory("tiel-coder", False, "short", heartbeat=lambda: beats.append(1))
        self.assertEqual(result, "")
        self.assertEqual(self.gw.PAUSE_WAIT["state"], "ready")
        self.assertEqual(len(beats), 3, "one heartbeat per poll while waiting, so the client stream does not go silent")
        reasons = [reason for reason, _ in self.events]
        self.assertIn("vram_pause_resume_waiting", reasons)
        self.assertIn("vram_pause_resume_ready", reasons)
        self.assertNotIn("vram_pause_resume_expired", reasons)

    def test_no_wait_when_memory_is_already_there(self):
        beats = []
        with self.script([""]):
            result = self.gw.wait_for_resume_memory("tiel-coder", False, "short", heartbeat=lambda: beats.append(1))
        self.assertEqual(result, "")
        self.assertEqual(beats, [])
        self.assertEqual(self.events, [], "no waiting event when nothing was waited for")

    def test_expires_after_pauseResumeWaitSeconds_with_the_refusal_text(self):
        memory = Refusal("Not enough free VRAM (11.2 GB free, need 11.60 GB)", "vram")
        clock = iter([0.0, 0.0, 1.0, 61.0, 61.0, 61.0, 61.0])
        with patch.object(self.gw.time, "monotonic", side_effect=lambda: next(clock)), self.script([memory]):
            result = self.gw.wait_for_resume_memory("tiel-coder", False, "short")
        self.assertEqual(result, memory)
        self.assertEqual(self.gw.PAUSE_WAIT["state"], "expired")
        self.assertIn("vram_pause_resume_expired", [reason for reason, _ in self.events])

    def test_a_refusal_waiting_cannot_fix_stops_at_once(self):
        other = Refusal("AI is manually disabled (ai-off)", "ai_off")
        with self.script([other]):
            result = self.gw.wait_for_resume_memory("tiel-coder", False, "short")
        self.assertEqual(result, other)
        self.assertEqual(self.gw.PAUSE_WAIT["state"], "refused")
        self.assertNotIn("vram_pause_resume_waiting", [reason for reason, _ in self.events])

    def test_a_client_that_went_away_stops_the_wait(self):
        import threading
        lease = Lease(cancel=threading.Event())
        memory = Refusal("Not enough free VRAM (11.2 GB free, need 11.60 GB)", "vram")
        calls = []

        def fake(model_id, vision, layout):
            calls.append(1)
            lease["cancel"].set()
            return memory
        with patch.object(self.gw, "resume_refusal", side_effect=fake):
            result = self.gw.wait_for_resume_memory("tiel-coder", False, "short", lease=lease)
        self.assertEqual(result, memory)
        self.assertEqual(self.gw.PAUSE_WAIT["state"], "cancelled")
        self.assertEqual(len(calls), 1, "no further admission checks after the client went away")
        self.assertIn("vram_pause_resume_cancelled", [reason for reason, _ in self.events])

    def test_the_waiting_reason_is_in_the_event_and_the_status_names_the_model(self):
        memory = Refusal("Not enough free VRAM (11.2 GB free, need 11.60 GB)", "vram")
        with self.script([memory, ""]):
            self.gw.wait_for_resume_memory("tiel-coder", False, "short")
        self.assertEqual(self.gw.PAUSE_WAIT["model"], "tiel-coder")
        self.assertEqual(self.gw.PAUSE_WAIT["state"], "ready")
        waiting = [detail for reason, detail in self.events if reason == "vram_pause_resume_waiting"]
        self.assertEqual(len(waiting), 1)
        self.assertIn("need 11.60 GB", waiting[0])
        self.assertIn("free_mib", self.gw.PAUSE_WAIT)


if __name__ == "__main__":
    unittest.main()
