"""LL1: /guardian/memory/* on the real gateway against the fake llama-server. With memoryEnabled false (the default) the paths are
unknown guardian endpoints and nothing else changes; with it true the store answers and no model is ever started by a memory call."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from gateway_harness import GatewayHarness  # noqa: E402

sys.path.insert(0, str(ROOT / "gateway"))
from web_guard import Policy  # noqa: E402

GUARD = {"vramGuard": True, "slotSaveCheckpoints": 1, "slotSaveMinTokens": 1}


def model_requests(gw):
    """What the fake llama-server received; [] when the gateway never even started it (the usual case here)."""
    try:
        return gw.fake_log()
    except ConnectionRefusedError:
        return []


class MemoryOff(unittest.TestCase):
    def test_without_the_flag_the_memory_paths_are_unknown_and_the_status_is_as_before(self):
        with GatewayHarness(guardian=GUARD) as gw:
            code, body = gw.request("GET", "/guardian/memory/stats")
            self.assertEqual(code, 405, "exactly as before the flag existed: a GET under /guardian/ is refused by web_guard")
            code, body = gw.request("POST", "/guardian/memory/save", {"key": "a", "title": "t", "body": "b"})
            self.assertEqual(code, 404)
            self.assertIn("unknown guardian endpoint", body["error"])
            status = gw.status()
            self.assertNotIn("lifelong", json.dumps(status).lower(), "the status payload does not grow a knowledge-store block")
            self.assertEqual(model_requests(gw), [], "no request reached the model server")


class MemoryOn(unittest.TestCase):
    def test_save_search_fix_forget_stats_through_the_gateway_without_touching_the_model(self):
        with tempfile.TemporaryDirectory(prefix="ll1-gw-") as tmp:
            db = str(Path(tmp) / "lifelong.sqlite")
            with GatewayHarness(guardian={**GUARD, "memoryEnabled": True, "memoryPath": db}, env={"SHURACODE_MEMORY_DIR": str(Path(tmp) / "legacy")}) as gw:
                user = {"X-Shura-Confirmation": gw.request("POST", "/guardian/memory/session-token", {})[1]["token"]}   # the user path: LL6b makes every tokenless write proposed
                code, body = gw.request("POST", "/guardian/memory/save", {"key": "prefers-pytest", "title": "tests run with pytest -q", "body": "Run pytest -q.", "kind": "preference"}, headers=user)
                self.assertEqual(code, 200, body)
                self.assertEqual(body["record"]["passport"]["source"], "user_command")
                self.assertRegex(body["record"]["passport_line"], r"^\(preference, you, ")
                code, body = gw.request("GET", "/guardian/memory/search?q=pytest")
                self.assertEqual(code, 200)
                self.assertEqual([r["key"] for r in body["results"]], ["prefers-pytest"])
                code, body = gw.request("POST", "/guardian/memory/fix", {"key": "prefers-pytest", "body": "Run pytest -q -x."}, headers=user)
                self.assertEqual(code, 200)
                self.assertIsNotNone(body["passport"]["supersedes"])
                code, body = gw.request("GET", "/guardian/memory/read?key=prefers-pytest")
                self.assertEqual(body["body"], "Run pytest -q -x.")
                code, body = gw.request("POST", "/guardian/memory/save", {"key": "creds", "title": "login", "body": "password: hunter2hunter2"})
                self.assertEqual(code, 400)
                self.assertIn("refused", body["error"])
                code, body = gw.request("POST", "/guardian/memory/forget", {"key": "prefers-pytest"})
                self.assertEqual(code, 200)
                code, body = gw.request("GET", "/guardian/memory/search?q=pytest")
                self.assertEqual(body["results"], [])
                code, body = gw.request("POST", "/guardian/memory/restore", {"key": "prefers-pytest"}, headers=user)
                self.assertEqual(code, 200)
                code, body = gw.request("GET", "/guardian/memory/stats")
                self.assertEqual(code, 200)
                self.assertEqual(body["records"], 1)
                self.assertIn("written_bytes_since_start", body)
                code, text = gw.request("GET", "/guardian/memory/export")
                self.assertEqual(code, 200)
                self.assertIn('"key": "prefers-pytest"', text if isinstance(text, str) else json.dumps(text))
                code, body = gw.request("POST", "/guardian/memory/nope", {})
                self.assertEqual(code, 404)
                code, body = gw.request("GET", "/guardian/memory/save")
                self.assertEqual(code, 405, "writes stay POST-only")
                self.assertEqual(model_requests(gw), [], "memory calls never start or touch the model server")
                self.assertTrue((Path(tmp) / "legacy" / "MEMORY.md").exists(), "the legacy index keeps being generated for the engine")
                self.assertIn("status", gw.status(), "the status endpoint still answers")
                # bad parameters are a 400, never a dropped connection
                for bad in ({"key": "x", "title": "t", "body": "b", "actor": "bot"}, {"key": ["list"], "title": "t", "body": "b"}, {"key": "x", "title": "t", "body": "b", "tags": "notalist"}):
                    code, body = gw.request("POST", "/guardian/memory/save", bad)
                    self.assertEqual(code, 400, body)
                code, body = gw.request("GET", "/guardian/memory/search?q=pytest&limit=abc")
                self.assertEqual(code, 400)
                code, body = gw.request("POST", "/guardian/memory/fix", {"key": "prefers-pytest", "body": "x", "relation": "maybe"}, headers=user)
                self.assertEqual(code, 400)


class GatewayModule(unittest.TestCase):
    def test_the_gateway_imports_what_its_memory_error_handlers_name(self):
        src = (ROOT / "gateway" / "ai_gateway.py").read_text()
        self.assertIn("\nimport sqlite3\n", src, "except sqlite3.Error needs the import")
        self.assertIn("_MEMORY_LOCK", src)


class GuardPolicy(unittest.TestCase):
    """web_guard: the four memory reads are GETs only with the flag on; nothing else is weakened."""
    HEADERS = {"Host": "127.0.0.1:8080"}

    def test_with_the_flag_off_every_guardian_path_but_status_still_needs_post(self):
        policy = Policy({}, "127.0.0.1")
        for path in ("/guardian/memory/search", "/guardian/memory/stats", "/guardian/memory/export", "/guardian/memory/read", "/guardian/memory/save"):
            self.assertEqual(policy.check("GET", path, self.HEADERS)[1], "SHURA_POST_REQUIRED", path)
        self.assertIsNone(policy.check("GET", "/guardian/status", self.HEADERS))

    def test_with_the_flag_on_only_the_four_reads_are_gets_and_everything_else_holds(self):
        policy = Policy({"memoryEnabled": True}, "127.0.0.1")
        for path in ("/guardian/memory/search", "/guardian/memory/stats", "/guardian/memory/export", "/guardian/memory/read"):
            self.assertIsNone(policy.check("GET", path, self.HEADERS), path)
        for path in ("/guardian/memory/save", "/guardian/memory/forget", "/guardian/memory/search/", "/guardian/unload", "/guardian/ai-off"):
            self.assertEqual(policy.check("GET", path, self.HEADERS)[1], "SHURA_POST_REQUIRED", path)
        self.assertEqual(policy.check("GET", "/guardian/memory/search", {"Host": "evil.example"})[1], "SHURA_BAD_HOST")
        self.assertEqual(policy.check("GET", "/guardian/memory/search", {**self.HEADERS, "Origin": "https://evil.example"})[1], "SHURA_FOREIGN_ORIGIN")
        self.assertEqual(policy.check("GET", "/guardian/memory/search", {**self.HEADERS, "Sec-Fetch-Site": "cross-site"})[1], "SHURA_CROSS_SITE")
        keyed = Policy({"memoryEnabled": True}, "127.0.0.1", api_key="k" * 32)
        self.assertEqual(keyed.check("GET", "/guardian/memory/export", self.HEADERS)[1], "SHURA_KEY_REQUIRED")
        self.assertIsNone(keyed.check("GET", "/guardian/memory/export", {**self.HEADERS, "Authorization": "Bearer " + "k" * 32}))


if __name__ == "__main__":
    unittest.main()
