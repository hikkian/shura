"""ORNFIX0: the script guard. Unit tests on the request rewriter plus the real gateway against the fake server (no GPU)."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gateway"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import script_guard
from gateway_harness import GatewayHarness

ON = {"scriptGuard": True}
CHAT = "/v1/chat/completions"
GUARD = {"vramGuard": True, "slotSaveCheckpoints": 1, "slotSaveMinTokens": 1}


def req(*messages, **extra):
    return json.dumps({"model": "m", "messages": list(messages), **extra}, ensure_ascii=False).encode()


def user(text):
    return {"role": "user", "content": text}


class Unit(unittest.TestCase):
    def bias(self, body, settings=ON, path=CHAT):
        out = json.loads(script_guard.apply(body, path, settings))
        return out.get("logit_bias")

    def test_ids_file_is_sane(self):
        with open(ROOT / "gateway" / "script_guard_ids_qwen36.json") as f:
            data = json.load(f)
        self.assertEqual(data["n_vocab"], 248320)
        self.assertGreater(len(data["ids"]), 50000)
        self.assertEqual(len(set(data["ids"])), len(data["ids"]))
        self.assertTrue(all(0 <= i < data["n_vocab"] for i in data["ids"]))
        for key in ("cyrillic_in_ids", "latin_in_ids", "greek_in_ids", "digit_in_ids"):
            self.assertEqual(data["checks"][key], 0, key)        # Russian, Kazakh (Cyrillic), English, numbers are never biased
        self.assertLessEqual(data["checks"]["emoji_arrows_boxes_in_ids"], 5)

    def test_off_by_default_and_for_other_paths(self):
        body = req(user("Привет, объясни кратко"))
        self.assertEqual(script_guard.apply(body, CHAT, {}), body)
        self.assertEqual(script_guard.apply(body, CHAT, {"scriptGuard": False}), body)
        self.assertEqual(script_guard.apply(body, "/v1/embeddings", ON), body)
        self.assertEqual(script_guard.apply(b"", CHAT, ON), b"")
        self.assertEqual(script_guard.apply(b"not json", CHAT, ON), b"not json")

    def test_russian_english_and_kazakh_requests_get_the_bias(self):
        for text in ("Привет, объясни кратко, как работает индекс.", "Explain briefly how the index works.",
                     "Қайырлы күн, бұл код неге жұмыс істемейді? Әріптер: ә ғ қ ң ө ұ ү һ і"):
            b = self.bias(req(user(text)))
            self.assertIsNotNone(b, text)
            self.assertGreater(len(b), 50000)
            self.assertTrue(all(isinstance(p, list) and len(p) == 2 and p[1] == -10.0 for p in b))

    def test_bias_value_and_ids_file_can_be_set(self):
        b = self.bias(req(user("hi")), {"scriptGuard": {"bias": -6}})
        self.assertEqual(b[0][1], -6.0)
        self.assertIsNone(self.bias(req(user("hi")), {"scriptGuard": {"enabled": False}}))

    def test_conversation_with_a_foreign_script_is_left_alone(self):
        for body in (req(user("Переведи на русский: 你好, мир")),                                                         # the user writes Chinese
                     req({"role": "system", "content": "Ответ: 日本語"}, user("привет")),                                 # system prompt
                     req(user("исправь файл"), {"role": "tool", "content": 'print("覆盖率")'}),                           # a tool result with Unicode strings
                     req(user("ok"), {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "w", "arguments": '{"text": "한국어"}'}}]}),
                     req(user("مرحبا"))):                                                                              # Arabic
            self.assertEqual(script_guard.apply(body, CHAT, ON), body)

    def test_code_with_unicode_strings_is_left_alone_but_plain_code_is_biased(self):
        code_unicode = req(user("Добавь тест:\n```python\nassert t('你好') == 'hello'\n```"))
        self.assertEqual(script_guard.apply(code_unicode, CHAT, ON), code_unicode)
        code_plain = req(user("Добавь тест:\n```python\nassert t('héllo') == 'hello'  # ✓ → ─\n```"))
        self.assertIsNotNone(self.bias(code_plain))           # accented Latin, check mark, arrow and box drawing are not foreign scripts

    def test_existing_logit_bias_is_kept(self):
        b = self.bias(req(user("hi"), logit_bias=[[5, 2.0]]))
        self.assertEqual(b[0], [5, 2.0])
        self.assertEqual(sum(1 for p in b if p[0] == 5), 1)
        b = self.bias(req(user("hi"), logit_bias={"7": -1}))
        self.assertIn([7, -1], b)
        out = json.loads(script_guard.apply(req(user("hi"), logit_bias=[[script_guard._ids(script_guard.DEFAULT_IDS)[0], 3.0]]), CHAT, ON))
        first = script_guard._ids(script_guard.DEFAULT_IDS)[0]
        self.assertEqual([p for p in out["logit_bias"] if p[0] == first], [[first, 3.0]])       # the client's own entry wins

    def test_has_foreign_script(self):
        for t in ("覆盖率", "ひらがな", "한국어", "עברית", "ภาษาไทย", "हिन्दी", "ქართული"):
            self.assertTrue(script_guard.has_foreign_script(t), t)
        for t in ("Привет", "Қазақша ә ғ қ ң ө ұ ү һ і", "héllo wörld", "αβγ", "123 ✓ → ─ │ 😀 ™ €", "naïve café"):
            self.assertFalse(script_guard.has_foreign_script(t), t)


class EndToEnd(unittest.TestCase):
    def posts(self, gw):
        return [e for e in gw.fake_log() if e.get("path") == CHAT and e.get("method") == "POST"]

    def test_gateway_adds_the_bias_only_when_enabled_and_clean(self):
        with GatewayHarness(guardian=GUARD, model={"scriptGuard": True}, behaviour={"text": "Привет!", "n_tokens": 3}) as gw:
            gw.chat("Привет, как дела у тебя сегодня?")
            gw.chat("Переведи: 你好")
            posts = self.posts(gw)
            self.assertGreater(len(posts[0]["body"].get("logit_bias") or []), 50000)
            self.assertNotIn("logit_bias", posts[1]["body"])
        with GatewayHarness(guardian=GUARD, behaviour={"text": "Привет!", "n_tokens": 3}) as gw:
            gw.chat("Привет, как дела у тебя сегодня?")
            self.assertNotIn("logit_bias", self.posts(gw)[0]["body"])


if __name__ == "__main__":
    unittest.main()
