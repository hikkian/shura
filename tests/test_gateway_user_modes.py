"""SWITCHUI1 / THINK2 / LAYOUT1c: the user's context mode (auto | extended) and thinking level (off | low | medium | high)
persist across gateway restarts, reach the requests, and a layout switch can kill the old server once its slot is saved."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gateway"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import user_modes  # noqa: E402
from gateway_harness import GatewayHarness  # noqa: E402
from test_gateway_layout import GUARD, LONG, SHORT, chat, launch, starts  # noqa: E402

TIEL_LEVELS = {"off": {"enable_thinking": False}, "low": {"reasoning_effort": "low"}, "medium": {},
               "high": {"reasoning_effort": "xhigh"}}
OCCAMY = {"thinkLevels": {"low": {"reasoning_effort": "low"}, "medium": {}, "high": {"reasoning_effort": "high"}},
          "thinkFallback": {"off": "low"}}


def body(**extra):
    return json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra}).encode()


class ThinkUnit(unittest.TestCase):
    def test_default_level_and_medium_leave_the_request_untouched(self):
        mc = {"thinkLevels": TIEL_LEVELS}
        self.assertEqual(user_modes.apply_think(body(), None, mc), body())
        self.assertEqual(user_modes.apply_think(body(), "medium", mc), body())

    def test_levels_add_their_template_kwargs(self):
        mc = {"thinkLevels": TIEL_LEVELS}
        out = json.loads(user_modes.apply_think(body(chat_template_kwargs={"terse": False}), "high", mc))
        self.assertEqual(out["chat_template_kwargs"], {"terse": False, "reasoning_effort": "xhigh"})
        out = json.loads(user_modes.apply_think(body(), "off", mc))
        self.assertEqual(out["chat_template_kwargs"], {"enable_thinking": False})

    def test_a_level_the_template_lacks_uses_the_fallback(self):
        out = json.loads(user_modes.apply_think(body(), "off", OCCAMY))
        self.assertEqual(out["chat_template_kwargs"], {"reasoning_effort": "low"})
        self.assertEqual(user_modes.think_snapshot("occamy", OCCAMY, "off")["effective"], "low")

    def test_the_client_choice_wins(self):
        mc = {"thinkLevels": TIEL_LEVELS}
        for b in (body(reasoning_effort="low"), body(chat_template_kwargs={"enable_thinking": True})):
            self.assertEqual(user_modes.apply_think(b, "off", mc), b)

    def test_a_model_without_levels_is_untouched(self):
        self.assertEqual(user_modes.apply_think(body(), "off", {}), body())
        self.assertIsNone(user_modes.think_snapshot("x", {}, "off")["effective"])

    def test_config_validation(self):
        user_modes.validate_think_config({"t": {"thinkLevels": TIEL_LEVELS}, "o": OCCAMY})
        for bad in ({"thinkLevels": {"ultra": {}}}, {"thinkLevels": {"low": "x"}},
                    {"thinkLevels": {"low": {}}, "thinkFallback": {"off": "high"}}, {"thinkFallback": {"off": "low"}}):
            with self.assertRaises(ValueError):
                user_modes.validate_think_config({"m": bad})

    def test_flags_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "think.flag"
            self.assertIsNone(user_modes.read_flag(path, user_modes.THINK_LEVELS, print))
            user_modes.write_flag(path, "high")
            self.assertEqual(user_modes.read_flag(path, user_modes.THINK_LEVELS, print), "high")
            path.write_text("ultra")
            self.assertIsNone(user_modes.read_flag(path, user_modes.THINK_LEVELS, lambda m: None))
            user_modes.write_flag(path, None)
            self.assertFalse(path.exists())


class Gateway(unittest.TestCase):
    def run_gateway(self, guard=None, levels=True, flags=None, **kw):
        cfg = launch(**kw)
        if levels:
            cfg["models"]["tiel-coder"]["thinkLevels"] = TIEL_LEVELS
        h = GatewayHarness(guardian=dict(GUARD, **(guard or {})), launch=cfg)
        for name, value in (flags or {}).items():   # flags left by an earlier gateway run
            (h.tmp / "state").mkdir(parents=True, exist_ok=True)
            (h.tmp / "state" / name).write_text(value)
        h.start(timeout=60)
        self.addCleanup(h.stop)
        return h

    def posts(self, h):
        return [e["body"] for e in h.fake_log() if e.get("method") == "POST" and e.get("path", "").endswith("/chat/completions")]

    def test_think_level_reaches_the_request_and_is_persisted(self):
        h = self.run_gateway()
        self.assertEqual(chat(h, SHORT)[0], 200)
        first = self.posts(h)[-1]
        self.assertNotIn("reasoning_effort", json.dumps(first.get("chat_template_kwargs", {})))
        code, snap = h.request("POST", "/guardian/think-high")
        self.assertEqual((code, snap["level"], snap["effective"]), (200, "high", "high"))
        self.assertEqual(chat(h, SHORT)[0], 200)
        self.assertEqual(self.posts(h)[-1]["chat_template_kwargs"].get("reasoning_effort"), "xhigh")
        self.assertEqual(h.status()["think"]["level"], "high")
        self.assertEqual((h.tmp / "state" / "think.flag").read_text(), "high")
        self.assertEqual(h.request("POST", "/guardian/think-ultra")[0], 404)
        self.assertEqual(h.request("POST", "/guardian/think-default")[1]["default"], True)
        self.assertFalse((h.tmp / "state" / "think.flag").exists())

    def test_extended_mode_starts_on_the_long_layout_and_auto_is_the_default(self):
        h = self.run_gateway(threshold=1000)
        snap = h.status()["context_mode"]
        self.assertEqual((snap["mode"], snap["short_ctx"], snap["long_ctx"]), ("auto", 32768, 200000))
        code, snap = h.request("POST", "/guardian/context-mode-extended")
        self.assertEqual((code, snap["mode"]), (200, "extended"))
        self.assertEqual(chat(h, SHORT)[0], 200)            # a short request, yet the long layout
        self.assertEqual(starts(h), [(200000, False)])
        self.assertEqual(h.status()["context_mode"]["window"], 200000)
        self.assertEqual((h.tmp / "state" / "context-mode.flag").read_text(), "extended")
        h.request("POST", "/guardian/context-mode-auto")
        self.assertFalse((h.tmp / "state" / "context-mode.flag").exists())

    def test_saved_switches_are_read_at_start(self):
        h = self.run_gateway(flags={"context-mode.flag": "extended", "think.flag": "off"})
        status = h.status()
        self.assertEqual((status["context_mode"]["mode"], status["think"]["level"]), ("extended", "off"))
        self.assertEqual(chat(h, SHORT)[0], 200)
        self.assertEqual(self.posts(h)[-1]["chat_template_kwargs"].get("enable_thinking"), False)

    def test_a_switch_waits_for_the_answer_that_is_just_ending(self):
        import threading
        h = self.run_gateway(threshold=1000, hold=0)
        self.assertEqual(chat(h, SHORT)[0], 200)
        h.set_behaviour(token_delay_ms=100, n_tokens=6)        # about 0.6 s: seen as busy, then ends inside the 1 s grace
        first = []
        t = threading.Thread(target=lambda: first.append(h.chat(SHORT, stream=True)[0]))   # streamed: the fake paces its tokens
        t.start()
        h.wait_for(lambda: h.status()["busy_requests"] == 1, timeout=10, what="the first answer in progress")
        code, data = chat(h, LONG)                            # needs the long layout while the first answer still streams
        t.join()
        self.assertEqual((first, code), ([200], 200), data)
        self.assertEqual([c for c, _ in starts(h)], [32768, 200000])

    def test_prompt_progress_is_asked_for_published_and_cleared(self):
        import threading
        h = self.run_gateway(guard={"promptProgress": True})
        h.set_behaviour(prompt_tokens=40000, progress_steps=6, progress_step_ms=300)
        done = []
        t = threading.Thread(target=lambda: done.append(h.chat(SHORT, stream=True)))
        t.start()
        seen = h.wait_for(lambda: (lambda p: p if p.get("state") == "reading" and p.get("processed") else None)(h.status()["prompt_progress"]),
                          timeout=10, what="the progress of the prompt")
        self.assertEqual(seen["total"], 40000)
        self.assertGreater(seen["eta_s"], 0)
        t.join()
        self.assertTrue(done[0][1]["complete"])
        self.assertEqual(h.status()["prompt_progress"], {"state": "idle"})
        sent = [e["body"] for e in h.fake_log() if e.get("method") == "POST" and e.get("path", "").endswith("/chat/completions")]
        self.assertIs(sent[-1].get("return_progress"), True)

    def test_prompt_progress_off_leaves_the_request_alone(self):
        h = self.run_gateway()
        self.assertEqual(h.chat(SHORT, stream=True)[0], 200)
        sent = [e["body"] for e in h.fake_log() if e.get("method") == "POST" and e.get("path", "").endswith("/chat/completions")]
        self.assertNotIn("return_progress", sent[-1])

    def test_a_request_right_after_an_unload_loads_the_model(self):
        h = self.run_gateway()
        self.assertEqual(chat(h, SHORT)[0], 200)
        code, body = h.request("POST", "/guardian/unload")
        self.assertEqual(code, 200, body)
        self.assertIn("vram_free_mib", body)
        self.assertEqual(chat(h, SHORT)[0], 200)
        self.assertEqual(len(starts(h)), 2)

    def test_layout_switch_kills_the_saved_server_only_when_fast_stop_is_on(self):
        for fast in (False, True):
            with self.subTest(fast=fast):
                h = self.run_gateway(guard={"layoutFastStop": fast}, threshold=1000)
                self.assertEqual(chat(h, SHORT)[0], 200)
                self.assertEqual(chat(h, LONG)[0], 200)
                self.assertEqual([c for c, _ in starts(h)], [32768, 200000])
                self.assertEqual("Fast stop:" in h.log(), fast)
                sw = h.status()["layout_switch"]
                self.assertEqual(sw["state"], "done")


if __name__ == "__main__":
    unittest.main()
