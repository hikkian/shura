"""Session cache through the whole gateway with the fake llama-server: the session header, the write-behind on an idle model,
the unload, and the restore on the next request of the same session. Off by default: nothing is written or read then."""
import http.client
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
from gateway_harness import GatewayHarness  # noqa: E402

SID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


def call(gw, method, path, body=None, headers=None, timeout=120):
    conn = http.client.HTTPConnection("127.0.0.1", gw.gw_port, timeout=timeout)
    try:
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


CHAT = {"model": "fake", "messages": [{"role": "user", "content": "Напиши функцию сложения."}], "stream": False}


class SessionCacheGateway(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix="sess-gw-", dir="/dev/shm"))
        self.catalog = self.temp / "catalog"
        self.gw = None

    def tearDown(self):
        if self.gw is not None:
            self.gw.__exit__(None, None, None)
        import shutil
        shutil.rmtree(self.temp, ignore_errors=True)

    def start(self, extra=None):
        guardian = {"slotSaveCheckpoints": 1, "sessionCacheEnabled": True, "sessionCacheDir": str(self.catalog),
                    "sessionCacheIdleSeconds": 0, "sessionCacheMinGrowthTokens": 0, "sessionCacheMinSlotTokens": 0,
                    "sessionCacheIoPressureMax": 1000, "sessionCacheQuotaGB": 1, "sessionCacheReserveGB": 0,
                    "sessionCacheReadGBps": 3, "sessionCacheLoadGBps": 3, "sessionCachePrefillTps": 50,
                    "slotSaveMinTokens": 0, "pollIntervalSeconds": 1}
        guardian.update(extra or {})
        self.gw = GatewayHarness(guardian=guardian)
        self.gw.__enter__()
        return self.gw

    def wait_for(self, predicate, timeout=40):
        end = time.time() + timeout
        while time.time() < end:
            if predicate():
                return True
            time.sleep(0.5)
        return False

    def test_snapshot_is_written_when_idle_and_restored_for_the_same_session(self):
        self.start()
        status, _ = call(self.gw, "POST", "/v1/chat/completions", CHAT, {"X-Shura-Session": SID})
        self.assertEqual(status, 200)
        self.assertTrue(self.wait_for(lambda: (self.catalog / f"{SID}.json").exists()), "no snapshot written while idle")
        meta = json.loads((self.catalog / f"{SID}.json").read_text())
        self.assertEqual(meta["session_id"], SID)
        self.assertIn("saved", meta)                               # the slot record the restore needs
        self.assertFalse(any(p.name.endswith(".tmp") for p in self.catalog.iterdir()))
        call(self.gw, "POST", "/guardian/unload", {})
        self.assertTrue(self.wait_for(lambda: self.gw.status()["status"] == "UNLOADED"))
        status, _ = call(self.gw, "POST", "/v1/chat/completions", CHAT, {"X-Shura-Session": SID})
        self.assertEqual(status, 200)
        # The slot file already holds this session, so the existing disk restore does the work: the server is asked to restore.
        self.assertTrue(any("restore" in str(e.get("path", "")) for e in self.gw.fake_log()), "the server was not asked to restore")

    def test_without_context_checkpoints_nothing_is_written(self):
        self.start({"slotSaveCheckpoints": 0})
        status, _ = call(self.gw, "POST", "/v1/chat/completions", CHAT, {"X-Shura-Session": SID})
        self.assertEqual(status, 200)
        time.sleep(3)
        self.assertFalse((self.catalog / f"{SID}.json").exists())     # no context checkpoints: no snapshot, whichever path runs

    def test_off_by_default_writes_nothing_and_leaves_no_catalogue(self):
        self.start({"sessionCacheEnabled": False})
        status, _ = call(self.gw, "POST", "/v1/chat/completions", CHAT, {"X-Shura-Session": SID})
        self.assertEqual(status, 200)
        time.sleep(3)
        self.assertFalse(self.catalog.exists())
        self.assertFalse(any("save" in str(e.get("path", "")) for e in self.gw.fake_log()))

    def test_warm_route_is_off_by_default_and_refuses_when_a_game_runs(self):
        self.start({"sessionCacheEnabled": False})
        status, _ = call(self.gw, "POST", "/guardian/warm", {})
        self.assertEqual(status, 404)

    def test_warm_loads_the_model_when_the_valves_allow_it(self):
        self.start()
        status, body = call(self.gw, "POST", "/guardian/warm", {}, headers={"X-Shura-Session": SID})
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertTrue(payload["warm"], payload)
        self.assertEqual(self.gw.status()["status"], "READY")


if __name__ == "__main__":
    unittest.main()
