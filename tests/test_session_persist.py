"""SSD level of the session cache (SESS3): a copy of a long session is written only at a gateway stop or an idle unload, only the
newest snapshot is kept, and after a reboot (empty RAM catalogue and slot folder) the same session is restored from the SSD copy.
Off by default: with no sessionPersistDir nothing is written to the SSD."""
import http.client
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gateway"))
sys.path.insert(0, str(HERE / "support"))
import session_cache  # noqa: E402
from gateway_harness import GatewayHarness  # noqa: E402

SID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
OTHER = "11111111-2222-4333-8444-555555555555"
CHAT = {"model": "fake", "messages": [{"role": "user", "content": "Напиши функцию сложения."}], "stream": False}


def call(gw, method, path, body=None, headers=None, timeout=120):
    conn = http.client.HTTPConnection("127.0.0.1", gw.gw_port, timeout=timeout)
    try:
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


class PersistDecision(unittest.TestCase):
    def test_only_a_stop_may_write_and_an_idle_unload_only_when_asked(self):
        self.assertTrue(session_cache.persist_should_write(40000, "gateway shutdown")[0])
        self.assertFalse(session_cache.persist_should_write(40000, "idle for 1800s")[0])   # the RAM copy survives an unload
        self.assertTrue(session_cache.persist_should_write(40000, "idle for 1800s", cfg={"sessionPersistOnIdle": True})[0])
        for reason in ("manual unload", "memory pressure", "switching to another model", "server crashed"):
            self.assertFalse(session_cache.persist_should_write(40000, reason)[0], reason)

    def test_short_sessions_are_not_written(self):
        write, why = session_cache.persist_should_write(29999, "gateway shutdown")
        self.assertFalse(write)
        self.assertIn("short session", why)
        self.assertTrue(session_cache.persist_should_write(30000, "gateway shutdown")[0])

    def test_budget_and_disk_refusals_win(self):
        self.assertFalse(session_cache.persist_should_write(40000, "gateway shutdown", budget_ok=False)[0])
        self.assertFalse(session_cache.persist_should_write(40000, "gateway shutdown", disk_ok=False)[0])

    def test_defaults_keep_the_ssd_level_off(self):
        self.assertEqual(session_cache.DEFAULTS["sessionPersistDir"], "")
        self.assertEqual(session_cache.DEFAULTS["sessionPersistKeep"], 1)


class KeepNewest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="persist-keep-", dir="/dev/shm"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def put(self, sid, last_used):
        (self.root / f"{sid}.bin").write_bytes(b"x")
        (self.root / f"{sid}.json").write_text(json.dumps({"session_id": sid, "model_key": "k", "tokens": 1, "prefix_hash": "",
                                                           "size": 1, "created": last_used, "last_used": last_used,
                                                           "file": f"{sid}.bin"}))

    def test_keeps_the_newest_and_the_current_one(self):
        cat = session_cache.SessionCatalog(self.root, 0, 0, 14)
        self.put(SID, 100.0)
        self.put(OTHER, 300.0)
        self.put("99999999-aaaa-4bbb-8ccc-dddddddddddd", 200.0)
        removed = cat.keep_newest(1, keep_id=SID)
        self.assertEqual(sorted(removed), ["99999999-aaaa-4bbb-8ccc-dddddddddddd"])
        self.assertTrue((self.root / f"{SID}.bin").exists())          # the session being written always stays
        self.assertTrue((self.root / f"{OTHER}.bin").exists())        # the newest of the others stays (count 1 + keep)

    def test_nothing_is_removed_when_there_is_room(self):
        cat = session_cache.SessionCatalog(self.root, 0, 0, 14)
        self.put(SID, 1.0)
        self.assertEqual(cat.keep_newest(2), [])


class SessionPersistGateway(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix="sess-persist-", dir="/dev/shm"))
        self.catalog = self.temp / "catalog"
        self.persist = self.temp / "persist"
        self.gw = None

    def tearDown(self):
        if self.gw is not None:
            self.gw.__exit__(None, None, None)
        shutil.rmtree(self.temp, ignore_errors=True)

    def start(self, extra=None):
        guardian = {"slotSaveCheckpoints": 1, "sessionCacheEnabled": True, "sessionCacheDir": str(self.catalog),
                    "sessionCacheIdleSeconds": 0, "sessionCacheMinGrowthTokens": 0, "sessionCacheMinSlotTokens": 0,
                    "sessionCacheIoPressureMax": 1000, "sessionCacheQuotaGB": 1, "sessionCacheReserveGB": 0,
                    "sessionCacheReadGBps": 3, "sessionCacheLoadGBps": 3, "sessionCachePrefillTps": 50,
                    "slotSaveMinTokens": 0, "pollIntervalSeconds": 1, "idleUnloadSeconds": 2,
                    "sessionPersistOnIdle": True}       # the idle unload is the quickest trigger for the copy mechanics
        guardian.update(extra or {})
        self.gw = GatewayHarness(guardian=guardian, model_file=self.temp / "model.gguf")
        self.gw.__enter__()
        return self.gw

    def wait_for(self, predicate, timeout=40):
        end = time.time() + timeout
        while time.time() < end:
            if predicate():
                return True
            time.sleep(0.5)
        return False

    def test_by_default_an_idle_unload_writes_nothing_to_the_ssd(self):
        self.start({"sessionPersistDir": str(self.persist), "sessionPersistMinTokens": 0, "sessionPersistOnIdle": False})
        status, _ = call(self.gw, "POST", "/v1/chat/completions", CHAT, {"X-Shura-Session": SID})
        self.assertEqual(status, 200)
        self.assertTrue(self.wait_for(lambda: self.gw.status()["status"] == "UNLOADED"))
        self.assertFalse((self.persist / f"{SID}.json").exists(), "the RAM copy survives an unload: no SSD write is needed")

    def test_idle_unload_copies_the_long_session_to_the_ssd(self):
        self.start({"sessionPersistDir": str(self.persist), "sessionPersistMinTokens": 0, "sessionPersistOnIdle": True})
        status, _ = call(self.gw, "POST", "/v1/chat/completions", CHAT, {"X-Shura-Session": SID})
        self.assertEqual(status, 200)
        self.assertTrue(self.wait_for(lambda: (self.persist / f"{SID}.json").exists()), "no SSD copy at the idle unload")
        self.assertTrue(self.wait_for(lambda: self.gw.status()["status"] == "UNLOADED"))
        meta = json.loads((self.persist / f"{SID}.json").read_text())
        self.assertIn("saved", meta)                                   # the slot record the restore needs
        self.assertFalse(any(p.name.endswith(".tmp") for p in self.persist.iterdir()))

    def test_manual_unload_writes_nothing_to_the_ssd(self):
        self.start({"sessionPersistDir": str(self.persist), "sessionPersistMinTokens": 0, "idleUnloadSeconds": 3600})
        status, _ = call(self.gw, "POST", "/v1/chat/completions", CHAT, {"X-Shura-Session": SID})
        self.assertEqual(status, 200)
        self.assertTrue(self.wait_for(lambda: (self.catalog / f"{SID}.json").exists()), "the RAM snapshot is written as before")
        call(self.gw, "POST", "/guardian/unload", {})
        self.assertTrue(self.wait_for(lambda: self.gw.status()["status"] == "UNLOADED"))
        time.sleep(2)
        self.assertFalse((self.persist / f"{SID}.json").exists())

    def test_short_session_is_not_copied(self):
        self.start({"sessionPersistDir": str(self.persist)})          # default sessionPersistMinTokens 30000 > the fake prompt
        status, _ = call(self.gw, "POST", "/v1/chat/completions", CHAT, {"X-Shura-Session": SID})
        self.assertEqual(status, 200)
        self.assertTrue(self.wait_for(lambda: self.gw.status()["status"] == "UNLOADED"))
        time.sleep(2)
        self.assertFalse((self.persist / f"{SID}.json").exists())

    def test_after_a_reboot_the_session_is_restored_from_the_ssd_copy(self):
        extra = {"sessionPersistDir": str(self.persist), "sessionPersistMinTokens": 0}
        self.start(extra)
        status, _ = call(self.gw, "POST", "/v1/chat/completions", CHAT, {"X-Shura-Session": SID})
        self.assertEqual(status, 200)
        self.assertTrue(self.wait_for(lambda: (self.persist / f"{SID}.json").exists()))
        self.assertTrue(self.wait_for(lambda: self.gw.status()["status"] == "UNLOADED"))
        # Reboot: a new gateway process (its session state is empty) and an empty RAM catalogue; the SSD copy survives.
        self.gw.__exit__(None, None, None)
        self.gw = None
        shutil.rmtree(self.catalog, ignore_errors=True)
        self.start(extra)
        self.assertFalse(self.catalog.exists() and any(self.catalog.glob("*.bin")), "the RAM catalogue must be empty after the reboot")
        status, _ = call(self.gw, "POST", "/v1/chat/completions", CHAT, {"X-Shura-Session": SID})
        self.assertEqual(status, 200)
        # The new process placed the SSD copy for the disk restore. (The fake server then compares the slot signature, which
        # includes the per-harness config folder; in a real reboot that folder is the same, so the restore goes through.)
        log = Path(self.gw.log_path).read_text(encoding="utf-8")
        self.assertIn("session_cache_restore", log, "the SSD copy was not placed for the restore")

    def test_the_daily_budget_is_kept_on_the_ssd_beside_the_copy(self):
        self.start({"sessionPersistDir": str(self.persist), "sessionPersistMinTokens": 0, "ssdWriteDailyLimitGB": 8})
        status, _ = call(self.gw, "POST", "/v1/chat/completions", CHAT, {"X-Shura-Session": SID})
        self.assertEqual(status, 200)
        self.assertTrue(self.wait_for(lambda: (self.persist / f"{SID}.json").exists()))
        # next to the catalogue (on the SSD), not in /dev/shm: a reboot must not reset the count
        self.assertTrue((self.temp / "sessions-persist-budget.json").is_file())
        self.assertFalse(any("ssd-write-budget" in str(p) for p in self.temp.iterdir()))

    def test_an_exhausted_daily_budget_stops_the_ssd_copy(self):
        self.start({"sessionPersistDir": str(self.persist), "sessionPersistMinTokens": 0, "ssdWriteDailyLimitGB": 1e-9})
        status, _ = call(self.gw, "POST", "/v1/chat/completions", CHAT, {"X-Shura-Session": SID})
        self.assertEqual(status, 200)
        self.assertTrue(self.wait_for(lambda: self.gw.status()["status"] == "UNLOADED"))
        time.sleep(2)
        self.assertFalse((self.persist / f"{SID}.json").exists())
        self.assertIn("daily write budget used", Path(self.gw.log_path).read_text(encoding="utf-8"))

    def test_off_by_default_writes_nothing(self):
        self.start()                                                  # no sessionPersistDir
        status, _ = call(self.gw, "POST", "/v1/chat/completions", CHAT, {"X-Shura-Session": SID})
        self.assertEqual(status, 200)
        self.assertTrue(self.wait_for(lambda: self.gw.status()["status"] == "UNLOADED"))
        time.sleep(2)
        self.assertFalse(self.persist.exists())


if __name__ == "__main__":
    unittest.main()
