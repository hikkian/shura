"""Session cache (gateway/session_cache.py) on a fake filesystem: quota, LRU, TTL, disk reserve, daily budget, atomic writes,
compatibility key, write-behind policy, restore decision, idle I/O command. Nothing here touches the real disk or a real server."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gateway"))
import session_cache as sc  # noqa: E402

SID1 = "11111111-1111-4111-8111-111111111111"
SID2 = "22222222-2222-4222-8222-222222222222"
SID3 = "33333333-3333-4333-8333-333333333333"


def fake_statvfs(free_bytes, total_bytes=500 * 1024 ** 3):
    def stat(_path):
        return SimpleNamespace(f_bavail=free_bytes // 4096, f_frsize=4096, f_blocks=total_bytes // 4096)
    return stat


class Clock:
    def __init__(self, now=1_700_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


class Catalog(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix="sess-", dir="/dev/shm"))
        self.clock = Clock()
        self.root = self.temp / "sessions"
        self.src = self.temp / "src.bin"

    def make_catalog(self, quota=10 * 1024 ** 3, reserve=0, free=400 * 1024 ** 3, ttl=14):
        return sc.SessionCatalog(self.root, quota, reserve, ttl, clock=self.clock, disk_stats=fake_statvfs(free))

    def payload(self, size):
        self.src.write_bytes(b"x" * size)
        return self.src

    def save(self, cat, sid, size, key="k1", tokens=1000):
        self.payload(size)
        return cat.save(sid, key, tokens, "p" * 8, self.src, now=self.clock())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_save_is_atomic_and_one_snapshot_per_session(self):
        cat = self.make_catalog()
        self.assertEqual(self.save(cat, SID1, 1000, tokens=100), (True, "saved"))
        self.clock.now += 10
        self.assertEqual(self.save(cat, SID1, 2000, tokens=200), (True, "saved"))
        files = sorted(p.name for p in self.root.iterdir())
        self.assertEqual(files, [f"{SID1}.bin", f"{SID1}.json"])          # the previous snapshot is gone
        meta = json.loads((self.root / f"{SID1}.json").read_text())
        self.assertEqual((meta["size"], meta["tokens"]), (2000, 200))
        self.assertFalse(any(p.name.endswith(".tmp") for p in self.root.iterdir()))

    def test_failed_write_leaves_nothing_behind(self):
        cat = self.make_catalog()
        self.assertEqual(self.save(cat, SID1, 500), (True, "saved"))
        before = sorted(p.name for p in self.root.iterdir())
        missing = self.temp / "missing.bin"
        ok, reason = cat.save(SID2, "k1", 1, "p", missing, now=self.clock())
        self.assertFalse(ok)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), before)

    def test_invalid_session_id_is_refused(self):
        cat = self.make_catalog()
        self.payload(10)
        self.assertEqual(cat.save("../../etc", "k1", 1, "p", self.src, now=self.clock()), (False, "invalid session id"))
        self.assertFalse(self.root.exists())

    def test_quota_is_respected_and_lru_evicts_the_least_recently_used(self):
        cat = self.make_catalog(quota=3000)
        self.save(cat, SID1, 1000)
        self.clock.now += 1
        self.save(cat, SID2, 1000)
        self.clock.now += 1
        self.save(cat, SID3, 1000)                          # exactly the quota
        cat.touch(SID1, now=self.clock.now)                  # SID1 is used again: SID2 is now the oldest
        self.clock.now += 1
        self.assertFalse(cat.write_allowed(1))              # no room for another byte at the quota
        removed = cat.prune(now=self.clock.now)
        self.assertEqual(removed, [])                        # within the quota: nothing to evict
        cat.quota = 2000
        removed = cat.prune(now=self.clock.now)
        self.assertEqual(removed, [SID2])                    # least recently used goes first
        self.assertTrue((self.root / f"{SID1}.bin").exists())
        self.assertFalse((self.root / f"{SID2}.bin").exists())

    def test_full_quota_admits_a_write_after_lru_eviction(self):
        # AMD2d: the gateway checked write_allowed before any eviction, so a full quota froze the catalogue until TTL.
        cat = self.make_catalog(quota=3000)
        self.save(cat, SID1, 1000)
        self.clock.now += 1
        self.save(cat, SID2, 1000)
        self.clock.now += 1
        self.save(cat, SID3, 1000)
        cat.touch(SID1, now=self.clock.now)                  # SID2 is now the least recently used
        self.clock.now += 1
        self.assertFalse(cat.write_allowed(1000))            # the old admission refuses a full quota
        self.assertTrue(cat.fits_after_eviction(1000))       # the new admission sees that LRU eviction makes room
        removed = cat.make_room(1000, keep="44444444-4444-4444-8444-444444444444")
        self.assertEqual(removed, [SID2])                    # only what is needed, least recently used first
        self.assertTrue((self.root / f"{SID1}.bin").exists())
        self.assertTrue((self.root / f"{SID3}.bin").exists())
        self.assertEqual(self.save(cat, "44444444-4444-4444-8444-444444444444", 1000), (True, "saved"))

    def test_make_room_never_evicts_the_session_being_replaced(self):
        cat = self.make_catalog(quota=2000)
        self.save(cat, SID1, 1000)
        self.clock.now += 1
        self.save(cat, SID2, 1000)
        self.clock.now += 1
        # SID1 is the oldest, but it is the session being replaced: its old snapshot leaves anyway, SID2 stays
        self.assertEqual(cat.make_room(1000, keep=SID1), [])
        self.assertTrue((self.root / f"{SID2}.bin").exists())

    def test_fits_after_eviction_refuses_a_snapshot_larger_than_the_quota_or_below_the_reserve(self):
        cat = self.make_catalog(quota=3000)
        self.assertFalse(cat.fits_after_eviction(3001))
        small_disk = self.make_catalog(quota=0, reserve=3_000_000, free=4_096_000)  # statvfs works in 4096-byte blocks
        self.assertTrue(small_disk.fits_after_eviction(1_000_000))   # 4_096_000 - 1_000_000 >= reserve
        self.assertFalse(small_disk.fits_after_eviction(1_200_000))  # 2_896_000 < reserve

    def test_make_room_without_quota_evicts_nothing(self):
        cat = self.make_catalog(quota=0)
        self.save(cat, SID1, 1000)
        self.assertEqual(cat.make_room(10 ** 9), [])

    def test_ttl_removes_sessions_not_used_for_14_days(self):
        cat = self.make_catalog(ttl=14)
        self.save(cat, SID1, 100)
        self.clock.now += 14 * 86400 + 1
        self.assertEqual(cat.prune(now=self.clock.now), [SID1])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_disk_reserve_refuses_a_write_that_would_leave_too_little_free(self):
        cat = self.make_catalog(reserve=100 * 1024 ** 3, free=100 * 1024 ** 3 + 500)
        self.assertEqual(self.save(cat, SID1, 1000)[0], False)     # 1000 bytes would leave 100 GiB - 500 < reserve
        cat2 = self.make_catalog(reserve=100 * 1024 ** 3, free=200 * 1024 ** 3)
        self.assertEqual(self.save(cat2, SID1, 1000)[0], True)

    def test_a_full_partition_never_takes_a_write(self):
        cat = self.make_catalog(reserve=0, free=0)                 # the fake partition is completely full
        self.assertEqual(self.save(cat, SID1, 1)[0], False)
        self.assertFalse(self.root.exists() and any(self.root.iterdir()))

    def test_key_mismatch_removes_the_snapshot_and_restores_nothing(self):
        cat = self.make_catalog()
        self.save(cat, SID1, 100, key="model-A")
        self.assertIsNone(cat.find(SID1, "model-B"))
        self.assertFalse((self.root / f"{SID1}.bin").exists())
        self.save(cat, SID2, 100, key="model-A")
        self.assertEqual(cat.find(SID2, "model-A")["tokens"], 1000)

    def test_start_check_drops_snapshots_with_a_stale_key(self):
        cat = self.make_catalog()
        self.save(cat, SID1, 100, key="old")
        self.save(cat, SID2, 100, key="new")
        dropped = cat.start_check(lambda e: "new")
        self.assertEqual(dropped, [SID1])
        self.assertTrue((self.root / f"{SID2}.bin").exists())

    def test_temporary_files_are_removed_at_prune(self):
        cat = self.make_catalog()
        self.root.mkdir(parents=True)
        (self.root / f".{SID1}.tmp").write_bytes(b"partial")
        cat.prune(now=self.clock.now)
        self.assertFalse((self.root / f".{SID1}.tmp").exists())


class AdoptCopied(Catalog):
    def test_an_idle_copy_is_adopted_atomically_with_its_slot_record(self):
        cat = self.make_catalog()
        self.root.mkdir(parents=True)
        tmp = self.root / f".{SID1}.tmp"
        tmp.write_bytes(b"y" * 700)
        ok, _ = cat.adopt_copied(SID1, "k1", 900, "p", {"model": "tiel-coder", "tokens": 900}, tmp, now=self.clock.now)
        self.assertTrue(ok)
        self.assertFalse(tmp.exists())
        self.assertEqual(json.loads((self.root / f"{SID1}.json").read_text())["saved"]["tokens"], 900)
        self.assertEqual((self.root / f"{SID1}.bin").read_bytes(), b"y" * 700)

    def test_a_missing_or_foreign_temporary_file_is_refused(self):
        cat = self.make_catalog()
        self.root.mkdir(parents=True)
        self.assertEqual(cat.adopt_copied(SID1, "k1", 1, "p", {}, self.root / "nope.tmp")[0], False)
        elsewhere = self.temp / f".{SID1}.tmp"
        elsewhere.write_bytes(b"z")
        self.assertEqual(cat.adopt_copied(SID1, "k1", 1, "p", {}, elsewhere)[0], False)


class Policy(unittest.TestCase):
    def test_writes_only_when_idle_grown_big_and_quiet(self):
        ok = sc.should_write(idle_s=130, growth_tokens=5000, slot_tokens=9000, io_pressure=2.0, game_or_video=False, ram_level="ok")
        self.assertEqual(ok, (True, "idle, grown, quiet"))

    def test_each_condition_refuses_on_its_own(self):
        base = dict(idle_s=130, growth_tokens=5000, slot_tokens=9000, io_pressure=2.0, game_or_video=False, ram_level="ok")
        for change, expected in (({"idle_s": 60}, "not idle"), ({"growth_tokens": 100}, "too little"),
                                 ({"slot_tokens": 100}, "slot too small"), ({"io_pressure": 15.0}, "I/O pressure"),
                                 ({"game_or_video": True}, "game"), ({"ram_level": "pressing"}, "RAM valve"),
                                 ({"budget_ok": False}, "daily write budget"), ({"disk_ok": False}, "disk reserve")):
            write, reason = sc.should_write(**{**base, **change})
            self.assertFalse(write, change)
            self.assertIn(expected, reason, change)

    def test_idle_copy_command_uses_the_idle_io_class(self):
        cmd = sc.idle_copy_command("/a", "/b")
        self.assertEqual(cmd[:2], ["ionice", "-c3"])

    def test_restore_when_much_cheaper_than_recomputing(self):
        restore, t_restore, t_recompute = sc.should_restore(size_bytes=300 * 1024 ** 2, read_bps=5e9, load_bps=5e9,
                                                            prefill_tps=200.0, tokens=28000)
        self.assertTrue(restore)
        self.assertLess(t_restore, 0.25 * t_recompute)

    def test_recompute_when_the_snapshot_is_not_much_cheaper(self):
        restore, _, _ = sc.should_restore(size_bytes=10 * 1024 ** 3, read_bps=5e9, load_bps=1e9, prefill_tps=5000.0, tokens=1000)
        self.assertFalse(restore)

    def test_bad_speeds_recompute(self):
        self.assertFalse(sc.should_restore(1, 0, 1, 1, 1)[0])


class Keys(unittest.TestCase):
    def test_compatibility_key_changes_with_any_part(self):
        base = sc.compatibility_key("m", False, "t", ["q8_0", "turbo4"])
        self.assertNotEqual(base, sc.compatibility_key("m2", False, "t", ["q8_0", "turbo4"]))
        self.assertNotEqual(base, sc.compatibility_key("m", True, "t", ["q8_0", "turbo4"]))
        self.assertNotEqual(base, sc.compatibility_key("m", False, "t2", ["q8_0", "turbo4"]))
        self.assertNotEqual(base, sc.compatibility_key("m", False, "t", ["q8_0", "q8_0"]))
        self.assertEqual(base, sc.compatibility_key("m", False, "t", ["q8_0", "turbo4"]))

    def test_session_ids_are_random_uuids_and_validated(self):
        a, b = sc.new_session_id(), sc.new_session_id()
        self.assertNotEqual(a, b)
        self.assertTrue(sc.valid_session_id(a))
        self.assertFalse(sc.valid_session_id("/etc/passwd"))
        self.assertFalse(sc.valid_session_id(None))

    def test_quota_is_the_smaller_of_gb_and_percent(self):
        partition = 500 * 1024 ** 3
        self.assertEqual(sc.quota_bytes(partition, 30, 8), 30 * 1024 ** 3)            # 8% = 40 GB, 30 GB smaller
        self.assertEqual(sc.quota_bytes(partition // 10, 30, 8), (partition // 10) * 8 // 100)   # 8% of 50 GB = 4 GB smaller
        self.assertEqual(sc.quota_bytes(0, 30, 8), 30 * 1024 ** 3)

    def test_prefix_hash_depends_only_on_the_prefix(self):
        self.assertEqual(sc.prefix_hash("abcdefghij" + "x" * 50, 10), sc.prefix_hash("abcdefghij" + "y" * 50, 10))
        self.assertNotEqual(sc.prefix_hash("abcdefghij", 10), sc.prefix_hash("abcdefghik", 10))


class DailyBudget(unittest.TestCase):
    def test_counts_per_day_and_refuses_over_the_limit(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            clock = Clock()
            budget = sc.DailyWriteBudget(1, Path(temp) / "state.json", clock=clock)     # 1 GiB
            budget.record(1024 ** 3 - 10)
            self.assertFalse(budget.allowed(100))
            clock.now += 86400
            self.assertTrue(budget.allowed(100))

    def test_no_limit_writes_no_state_file(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp:
            path = Path(temp) / "state.json"
            budget = sc.DailyWriteBudget(0, path)
            budget.record(10 ** 9)
            self.assertTrue(budget.allowed(10 ** 15))
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
