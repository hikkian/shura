"""EFFORT1: the effort slider. The user's level (off, low, medium, high, max) is a session switch that survives restarts; each level
carries a real reasoning budget to the server (ceiling with a wrap-up line, soft target), not only a phrase in the template; a
client's reasoning_effort names the level for one request. Fake server, no GPU."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gateway"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import user_modes
import lang_guard
from gateway_harness import GatewayHarness

GUARD = {"vramGuard": True, "slotSaveCheckpoints": 1, "slotSaveMinTokens": 1}   # the new gateway, as the layout tests run it

TIEL_LEVELS = {"off": {"enable_thinking": False}, "low": {"reasoning_effort": "low"}, "medium": {},
               "high": {"reasoning_effort": "xhigh"}, "max": {"reasoning_effort": "xhigh"}}
BUDGETS = {"low": {"targetTokens": 5, "ceilingTokens": 10},
           "medium": {"targetTokens": 15, "ceilingTokens": 30, "message": "Wrapping up."},
           "high": {"targetTokens": 20, "ceilingTokens": 25},
           "max": {"targetTokens": None, "ceilingTokens": 100}}
TIEL = {"thinkLevels": TIEL_LEVELS, "effortBudgets": BUDGETS}
OCCAMY = {"thinkLevels": {"low": {"reasoning_effort": "low"}, "medium": {}, "high": {"reasoning_effort": "high"}},
          "thinkFallback": {"off": "low", "max": "high"}, "effortBudgets": {"low": {"ceilingTokens": 8}, "high": {"ceilingTokens": 40}}}
GEMMA = {"effortBudgets": {"low": {"ceilingTokens": 12}, "medium": {"ceilingTokens": 24}}}   # a template with no levels: only the ceiling


def body(**extra):
    return json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra}).encode()


class EffortUnit(unittest.TestCase):
    def test_the_slider_has_five_levels_and_openai_names_map_onto_them(self):
        self.assertEqual(user_modes.THINK_LEVELS, ("off", "low", "medium", "high", "max"))
        for value, level in (("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "max"), ("max", "max"),
                             ("none", "off"), ("HIGH", "high")):
            self.assertEqual(user_modes.client_effort_level({"reasoning_effort": value}), level)
        self.assertIsNone(user_modes.client_effort_level({"reasoning_effort": "extreme"}))
        self.assertIsNone(user_modes.client_effort_level({}))

    def test_without_the_switch_requests_are_byte_for_byte_as_before(self):
        self.assertEqual(user_modes.apply_think(body(), None, TIEL), body())
        self.assertEqual(user_modes.apply_think(body(), "medium", TIEL), body())
        out = json.loads(user_modes.apply_think(body(), "max", TIEL))
        self.assertEqual(out["chat_template_kwargs"], {"reasoning_effort": "xhigh"})
        self.assertNotIn("reasoning_budget_tokens", out)

    def test_with_the_switch_each_level_sends_its_budget(self):
        out = json.loads(user_modes.apply_think(body(), None, TIEL, budgets=True))          # the default level is medium
        self.assertEqual((out["reasoning_budget_tokens"], out["reasoning_budget_soft_tokens"], out["reasoning_budget_message"]),
                         (30, 15, "Wrapping up."))
        self.assertNotIn("chat_template_kwargs", out)
        out = json.loads(user_modes.apply_think(body(), "low", TIEL, budgets=True))
        self.assertEqual((out["chat_template_kwargs"], out["reasoning_budget_tokens"], out["reasoning_budget_soft_tokens"]),
                         ({"reasoning_effort": "low"}, 10, 5))
        self.assertEqual(out["reasoning_budget_message"], user_modes.DEFAULT_WRAP_UP)
        out = json.loads(user_modes.apply_think(body(), "max", TIEL, budgets=True))
        self.assertEqual(out["reasoning_budget_tokens"], 100)
        self.assertNotIn("reasoning_budget_soft_tokens", out)      # no target: the model is free up to the ceiling
        out = json.loads(user_modes.apply_think(body(), "off", TIEL, budgets=True))
        self.assertEqual(out["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("reasoning_budget_tokens", out)           # nothing to budget when thinking is off

    def test_a_clients_reasoning_effort_names_the_level_for_that_request(self):
        out = json.loads(user_modes.apply_think(body(reasoning_effort="high"), "low", TIEL, budgets=True))
        self.assertEqual(out["reasoning_budget_tokens"], 25)       # the client's level, not the session's
        self.assertNotIn("chat_template_kwargs", out)              # the client's field reaches the template as before
        out = json.loads(user_modes.apply_think(body(reasoning_effort="xhigh"), None, TIEL, budgets=True))
        self.assertEqual(out["reasoning_budget_tokens"], 100)
        kept = body(reasoning_effort="low", reasoning_budget_tokens=7)
        self.assertEqual(user_modes.apply_think(kept, "max", TIEL, budgets=True), kept)   # the client's own budget wins

    def test_a_level_the_template_lacks_keeps_the_budget_of_its_fallback(self):
        out = json.loads(user_modes.apply_think(body(), "max", OCCAMY, budgets=True))
        self.assertEqual((out["chat_template_kwargs"], out["reasoning_budget_tokens"]), ({"reasoning_effort": "high"}, 40))
        out = json.loads(user_modes.apply_think(body(), "off", OCCAMY, budgets=True))
        self.assertEqual((out["chat_template_kwargs"], out["reasoning_budget_tokens"]), ({"reasoning_effort": "low"}, 8))
        self.assertEqual(user_modes.apply_think(body(), "medium", OCCAMY, budgets=True), body())   # no budget set for medium

    def test_a_template_without_levels_still_gets_the_ceiling(self):
        out = json.loads(user_modes.apply_think(body(), "low", GEMMA, budgets=True))
        self.assertEqual(out["reasoning_budget_tokens"], 12)
        self.assertNotIn("chat_template_kwargs", out)
        self.assertEqual(json.loads(user_modes.apply_think(body(), None, GEMMA, budgets=True))["reasoning_budget_tokens"], 24)
        self.assertEqual(user_modes.apply_think(body(), "max", GEMMA, budgets=True), body())
        self.assertEqual(user_modes.apply_think(body(), "max", {}, budgets=True), body())

    def test_snapshot_reports_the_budget_of_the_effective_level(self):
        snap = user_modes.think_snapshot("tiel", TIEL, "high", budgets=True)
        self.assertEqual((snap["level"], snap["effective"], snap["budget"], snap["budgets"]), ("high", "high", {"target": 20, "ceiling": 25}, True))
        self.assertEqual(snap["levels"], ["off", "low", "medium", "high", "max"])
        self.assertEqual(snap["available"], ["off", "low", "medium", "high", "max"])
        self.assertIsNone(user_modes.think_snapshot("tiel", TIEL, "high")["budget"])
        self.assertEqual(user_modes.think_snapshot("occamy", OCCAMY, "max", budgets=True)["budget"], {"target": None, "ceiling": 40})

    def test_config_validation(self):
        user_modes.validate_think_config({"t": TIEL, "o": OCCAMY, "g": GEMMA})
        for bad in ({"effortBudgets": {"ultra": {}}}, {"effortBudgets": {"low": {"ceilingTokens": -1}}},
                    {"effortBudgets": {"low": {"targetTokens": 50, "ceilingTokens": 10}}}, {"effortBudgets": {"low": {"tokens": 5}}},
                    {"effortBudgets": {"low": {"message": 5}}}, {"effortBudgets": {"low": {"ceilingTokens": 1.5}}}):
            with self.assertRaises(ValueError, msg=str(bad)):
                user_modes.validate_think_config({"m": bad})

    def test_budget_ceiling_plus_margin_must_fit_the_output_limit(self):
        user_modes.validate_think_config({"m": {"outputLimitTokens": 16384, "effortBudgetMarginTokens": 1024,
                                                    "effortBudgets": {"high": {"ceilingTokens": 15360}}}})
        with self.assertRaisesRegex(ValueError, "output limit"):
            user_modes.validate_think_config({"m": {"outputLimitTokens": 16384, "effortBudgetMarginTokens": 1024,
                                                       "effortBudgets": {"max": {"ceilingTokens": 24000}}}})
        with self.assertRaisesRegex(ValueError, "outputLimitTokens"):
            user_modes.validate_think_config({"m": {"outputLimitTokens": "16384", "effortBudgets": {"max": {"ceilingTokens": 8}}}})


class LanguageGuardUnit(unittest.TestCase):
    def test_opt_in_detection_and_stream_rewrite(self):
        req = {"messages": [{"role": "user", "content": "Пожалуйста, объясни подробно, как это работает."}]}
        self.assertEqual(lang_guard.guard_for(req, "/v1/chat/completions", {"langGuard": True}), "Russian")
        self.assertEqual(lang_guard.guard_for(req, "/v1/chat/completions", {}), "")
        guard = lang_guard.StreamGuard("Russian", lambda _text: "Это подробное объяснение ответа на русском языке с дополнительными важными деталями." )
        stream = b'data: {"choices":[{"delta":{"content":"This is a detailed English response with many useful details for you."},"finish_reason":null}]}\n\n'
        stream += b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        guard.feed(stream)
        out = guard.finish_bytes().decode()
        self.assertTrue(guard.rewritten)
        self.assertIn("Это подробное объяснение", out)


class EffortEndToEnd(unittest.TestCase):
    """The real gateway against the fake server, which honours reasoning_budget_tokens the way shura.cpp does: the reasoning stops
    at the budget and the wrap-up line is the last reasoning text."""

    def last_chat_body(self, gw):
        posts = [e for e in gw.fake_log() if e.get("path") == "/v1/chat/completions"]
        return posts[-1]["body"]

    def test_levels_budgets_and_the_client_override(self):
        model = {"thinkLevels": TIEL_LEVELS, "effortBudgets": BUDGETS}
        with GatewayHarness(guardian={**GUARD, "effortBudgets": True}, model=model, behaviour={"reasoning_tokens": 40, "n_tokens": 3}) as gw:
            gw.start()
            code, out = gw.chat("hello")                                       # the default level: medium, ceiling 30
            self.assertEqual(code, 200, out)
            sent = self.last_chat_body(gw)
            self.assertEqual((sent["reasoning_budget_tokens"], sent["reasoning_budget_soft_tokens"]), (30, 15))
            reasoning = out["choices"][0]["message"]["reasoning_content"]
            self.assertEqual(reasoning.count("think"), 30)
            self.assertTrue(reasoning.endswith("Wrapping up."), reasoning[-40:])
            status = gw.status()["think"]
            self.assertEqual((status["level"], status["budget"], status["budgets"]), ("medium", {"target": 15, "ceiling": 30}, True))

            code, th = gw.request("POST", "/guardian/effort-low")              # the slider moves: low
            self.assertEqual((code, th["level"], th["effective"], th["budget"]), (200, "low", "low", {"target": 5, "ceiling": 10}))
            code, out = gw.chat("again")
            sent = self.last_chat_body(gw)
            self.assertEqual((sent["chat_template_kwargs"], sent["reasoning_budget_tokens"]), ({"reasoning_effort": "low"}, 10))
            self.assertEqual(out["choices"][0]["message"]["reasoning_content"].count("think"), 10)

            code, th = gw.request("POST", "/guardian/think-max")               # the old name still works
            self.assertEqual((code, th["level"], th["budget"]), (200, "max", {"target": None, "ceiling": 100}))
            code, out = gw.chat("deep")
            self.assertEqual(out["choices"][0]["message"]["reasoning_content"].count("think"), 40)   # below the ceiling: untouched
            self.assertNotIn("Wrapping", out["choices"][0]["message"]["reasoning_content"])

            code, out = gw.chat("one request", reasoning_effort="high")       # the client names the level for one request
            sent = self.last_chat_body(gw)
            self.assertEqual((sent["reasoning_effort"], sent["reasoning_budget_tokens"]), ("high", 25))
            self.assertNotIn("chat_template_kwargs", sent)
            self.assertEqual(out["choices"][0]["message"]["reasoning_content"].count("think"), 25)
            self.assertEqual(gw.status()["think"]["level"], "max")           # the session level did not move

            code, out = gw.chat("streamed", stream=True, reasoning_effort="low")   # the same over a stream: 10 tokens, then the wrap-up
            self.assertEqual(code, 200, out)
            text = "".join(line for line in out["lines"])
            self.assertEqual(text.count('"reasoning_content": "think "'), 10)
            self.assertIn("Thinking time is up", text)

            code, _ = gw.request("POST", "/guardian/effort-off")
            self.assertEqual(code, 200)
            gw.chat("plain")
            sent = self.last_chat_body(gw)
            self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})
            self.assertNotIn("reasoning_budget_tokens", sent)
            code, err = gw.request("POST", "/guardian/effort-ultra")
            self.assertEqual(code, 404, err)

    def test_the_switch_off_keeps_requests_as_before_and_the_level_survives_a_restart(self):
        model = {"thinkLevels": TIEL_LEVELS, "effortBudgets": BUDGETS}
        with GatewayHarness(guardian=GUARD, model=model, behaviour={"reasoning_tokens": 40, "n_tokens": 3}) as gw:
            gw.start()
            gw.request("POST", "/guardian/effort-max")
            _code, out = gw.chat("hello")
            sent = self.last_chat_body(gw)
            self.assertEqual(sent["chat_template_kwargs"], {"reasoning_effort": "xhigh"})
            self.assertNotIn("reasoning_budget_tokens", sent)
            self.assertEqual(out["choices"][0]["message"]["reasoning_content"].count("think"), 40)
            self.assertEqual(gw.status()["think"]["budgets"], False)

    def test_the_max_level_survives_in_the_flag_file(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            flag = Path(tmp) / "think.flag"
            user_modes.write_flag(flag, "max")
            self.assertEqual(user_modes.read_flag(flag, user_modes.THINK_LEVELS, print), "max")

    def test_asking_for_budgets_alone_selects_the_new_gateway(self):
        model = {"thinkLevels": TIEL_LEVELS, "effortBudgets": BUDGETS}
        with GatewayHarness(guardian={"effortBudgets": True}, model=model, behaviour={"reasoning_tokens": 40, "n_tokens": 3}) as gw:
            gw.start()
            gw.chat("hello")
            self.assertEqual(self.last_chat_body(gw)["reasoning_budget_tokens"], 30)


class LanguageGuardEndToEnd(unittest.TestCase):
    def test_enabled_guard_rewrites_high_effort_russian_user_stream(self):
        english = "This is a long English answer with useful explanation and several details."
        russian = "Это длинный ответ на русском языке с полезным объяснением и несколькими важными подробностями."
        with GatewayHarness(guardian=GUARD, model={"langGuard": True},
                            behaviour={"text": english, "rewrite_text": russian, "n_tokens": 12}) as gw:
            status, out = gw.chat("Пожалуйста, объясни подробно, как работает эта система и почему она отвечает так.",
                                  stream=True, reasoning_effort="high")
            self.assertEqual(status, 200, out)
            content = "".join(json.loads(line)["choices"][0]["delta"].get("content", "") for line in out["lines"]
                              if line != "[DONE]")
            self.assertIn("русском языке", content)
            chats = [e for e in gw.fake_log() if e.get("path") == "/v1/chat/completions" and e.get("method") == "POST"]
            self.assertEqual(len(chats), 2)  # original streaming generation and one non-stream rewrite
            rewrite = chats[-1]["body"]
            self.assertFalse(rewrite["stream"])
            self.assertEqual(rewrite["reasoning_budget_tokens"], 0)

    def test_guard_is_off_by_default(self):
        english = "This is a long English answer with useful explanation and several details."
        with GatewayHarness(guardian=GUARD, behaviour={"text": english, "n_tokens": 12}) as gw:
            status, out = gw.chat("Пожалуйста, объясни подробно, как работает эта система и почему она отвечает так.",
                                  stream=True, reasoning_effort="high")
            self.assertEqual(status, 200, out)
            chats = [e for e in gw.fake_log() if e.get("path") == "/v1/chat/completions" and e.get("method") == "POST"]
            self.assertEqual(len(chats), 1)


if __name__ == "__main__":
    unittest.main()
