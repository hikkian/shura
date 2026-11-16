"""Page-cache reclaim on a fake cgroup tree and a fake /proc (nothing real is written)."""
import errno
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))
import cache_reclaim as cr  # noqa: E402
import test_gateway as fixture  # noqa: E402
from test_gateway_ram import CALM, MODEL, mem as ram_mem, fresh_ladder  # noqa: E402

SELF = "/user.slice/user-1000.slice/user@1000.service/app.slice/ai-gateway.service"


def meminfo(free_mib=2700, active=4000, inactive=4700, dirty=0, shmem=0):
    kb = lambda mib: int(mib * 1024)  # noqa: E731
    return (f"MemTotal: {kb(32000)} kB\nMemFree: {kb(free_mib)} kB\nActive(file): {kb(active)} kB\n"
            f"Inactive(file): {kb(inactive)} kB\nDirty: {kb(dirty)} kB\nShmem: {kb(shmem)} kB\n")


class Tree:
    def __init__(self, root, writable=("user@1000.service",), cgroup=SELF):
        self.proc, self.cg = Path(root) / "proc", Path(root) / "cg"
        (self.proc / "self").mkdir(parents=True, exist_ok=True)
        (self.proc / "self" / "cgroup").write_text(f"0::{cgroup}\n")
        self.set_mem()
        base = self.cg
        for part in [p for p in cgroup.split("/") if p]:
            base = base / part
            base.mkdir(parents=True, exist_ok=True)
            if part in writable:
                (base / "memory.reclaim").write_text("")
            elif part == "app.slice":
                (base / "memory.reclaim").write_text("")
                (base / "memory.reclaim").chmod(0o444)
        self.reclaim = self.cg / "user.slice/user-1000.slice/user@1000.service/memory.reclaim"

    def set_mem(self, **kw):
        (self.proc / "meminfo").write_text(meminfo(**kw))


class Reclaim(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tree = Tree(self.tmp.name)

    def run_reclaim(self, want, **kw):
        return cr.reclaim_page_cache(want, str(self.tree.proc), str(self.tree.cg), **kw)

    def test_finds_user_service_level_first(self):
        found = cr.find_reclaim_file(str(self.tree.proc), str(self.tree.cg))
        self.assertEqual(found, self.tree.reclaim)

    def test_nearest_writable_ancestor_when_no_user_service(self):
        tree = Tree(self.tmp.name + "/x", writable=("user-1000.slice",))
        found = cr.find_reclaim_file(str(tree.proc), str(tree.cg))
        self.assertEqual(found.parent.name, "user-1000.slice")        # app.slice is read-only in the fake tree
        self.assertIsNone(cr.find_reclaim_file(str(tree.proc) + "/none", str(tree.cg)))

    def test_writes_size_and_swappiness_and_measures_freed(self):
        real_write = cr.os.write

        def fake_write(fd, data):
            real_write(fd, data)
            self.tree.set_mem(free_mib=5200)           # the kernel freed 2500 MiB
            return len(data)
        with patch.object(cr.os, "write", fake_write):
            self.assertEqual(self.run_reclaim(3000), (2500, ""))
        self.assertEqual(self.tree.reclaim.read_text(), "3000M swappiness=0")

    def test_einval_is_returned_and_not_retried_plain(self):
        calls = []

        def fake_write(fd, data):
            calls.append(data)
            raise OSError(errno.EINVAL, "Invalid argument")
        with patch.object(cr.os, "write", fake_write):
            freed, error = self.run_reclaim(3000)
        self.assertEqual(freed, 0)
        self.assertIn("swappiness", error)
        self.assertEqual(len(calls), 1)                  # one attempt, no plain "3000M" fallback

    def test_eagain_is_a_partial_reclaim(self):
        def fake_write(fd, data):
            self.tree.set_mem(free_mib=3200)
            raise OSError(errno.EAGAIN, "Try again")
        with patch.object(cr.os, "write", fake_write):
            self.assertEqual(self.run_reclaim(3000), (500, ""))

    def test_capped_to_reclaimable_clean_cache(self):
        self.tree.set_mem(active=1000, inactive=500, dirty=200, shmem=100)     # 1300 MiB clean
        self.assertEqual(cr.reclaimable_mib(cr.meminfo_mib(meminfo(active=1000, inactive=500, dirty=200, shmem=100))), 1300)
        self.run_reclaim(9000)
        self.assertEqual(self.tree.reclaim.read_text(), "1300M swappiness=0")

    def test_shmem_is_not_subtracted_from_file_cache(self):
        # 2026-11-16 17:00 live numbers: 8.7 GiB file LRU, 5.3 GiB Shmem (tmpfs lives on the anon LRU, not in Active/Inactive(file)).
        mem = cr.meminfo_mib(meminfo(active=3600, inactive=4900, dirty=0, shmem=5200))
        self.assertEqual(cr.reclaimable_mib(mem), 8500)

    def test_nothing_to_reclaim_and_never_raises(self):
        self.tree.set_mem(active=10, inactive=10)
        self.assertEqual(self.run_reclaim(1000)[0], 0)
        self.assertEqual(cr.reclaim_page_cache("x", "/nonexistent", "/nonexistent")[0], 0)
        self.assertEqual(cr.reclaim_page_cache(1000, "/nonexistent", "/nonexistent")[0], 0)


class Admission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.created_import = not hasattr(fixture.GatewayGuard, "gw")
        if cls.created_import:
            fixture.GatewayGuard.setUpClass()
        cls.gw = fixture.GatewayGuard.gw

    @classmethod
    def tearDownClass(cls):
        if cls.created_import:
            fixture.GatewayGuard.tearDownClass()

    def setUp(self):
        for patcher in (patch.dict(self.gw.RAM_LADDER, fresh_ladder()), patch.dict(self.gw.RAM_RECLAIM, {}, clear=True),
                        patch.dict(self.gw.G, {"ramDeskReserveMiB": 2048, "ramWaitSeconds": 20, "ramReclaimCache": True}),
                        patch.dict(self.gw.M, {"cacheRamMB": 0}), patch.object(self.gw, "DESK", None),
                        patch.object(self.gw.ram_budget, "read_pressure", return_value=CALM)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.state = {"free": 9000.0, "cache": 8700.0}
        self.calls = []

        def read_meminfo(*_a):
            m = ram_mem(self.state["free"] / 1024)
            m.update({"Active(file)": self.state["cache"] / 2, "Inactive(file)": self.state["cache"] / 2, "Dirty": 0.0})
            return m

        def reclaim(want, proc_root="/proc", *a, **k):
            self.calls.append(want)
            freed = min(want, self.state["cache"])
            self.state["free"] += freed
            self.state["cache"] -= freed
            return int(freed), ""
        for patcher in (patch.object(self.gw.ram_budget, "read_meminfo", read_meminfo),
                        patch.object(self.gw.cache_reclaim, "reclaim_page_cache", reclaim)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_wait_turns_into_load_after_reclaim(self):
        self.assertEqual(self.gw.ram_admission(MODEL), "")
        self.assertEqual(len(self.calls), 1)
        self.assertGreater(self.calls[0], 512)

    def test_not_enough_cache_to_cover_does_not_reclaim(self):
        self.state["cache"] = 100.0
        self.assertEqual(self.gw.refusal_code(self.gw.ram_admission(MODEL)), "ram_wait")
        self.assertEqual(self.calls, [])

    def test_disabled_by_the_setting(self):
        with patch.dict(self.gw.G, {"ramReclaimCache": False}):
            self.assertEqual(self.gw.refusal_code(self.gw.ram_admission(MODEL)), "ram_wait")
        self.assertEqual(self.calls, [])

    def test_rate_limited_to_once_per_ten_seconds(self):
        self.state["free"], self.state["cache"] = 4000.0, 20000.0
        with patch.object(self.gw.cache_reclaim, "reclaim_page_cache", lambda *a, **k: (10, "")) as fake:
            self.gw.ram_admission(MODEL)
            self.gw.ram_admission(MODEL)
        self.assertEqual(self.calls, [])
        self.assertGreater(self.gw.RAM_RECLAIM["admission"], 0)
        with patch.object(self.gw.time, "monotonic", return_value=self.gw.RAM_RECLAIM["admission"] + 11):
            self.assertTrue(self.gw.ram_try_reclaim(1000, {"Active(file)": 5000.0}, "t"))
        with patch.object(self.gw.time, "monotonic", return_value=self.gw.RAM_RECLAIM["admission"] + 3):
            self.assertFalse(self.gw.ram_try_reclaim(1000, {"Active(file)": 5000.0}, "t"))

    def test_error_from_reclaim_keeps_the_wait(self):
        with patch.object(self.gw.cache_reclaim, "reclaim_page_cache", lambda *a, **k: (0, "boom")):
            self.assertEqual(self.gw.refusal_code(self.gw.ram_admission(MODEL)), "ram_wait")

    def test_the_valve_does_not_reclaim(self):
        # RAMFIX1 review: the valve reads MemAvailable, which already counts clean page cache: a reclaim would not move it and would
        # only evict Brave's and VS Code's hot cache. The valve path has no reclaim; ram_admission keeps it (honest free counts half).
        self.assertFalse(hasattr(self.gw, "ram_valve_try_reclaim"))
        src = Path(self.gw.__file__).read_text()
        handler = src[src.index('grown = len(body) - RAM_LAST["len"]'):src.index('RAM_LAST["len"] = len(body)')]
        self.assertNotIn("reclaim", handler)

    def test_config_flag_must_be_boolean(self):
        self.gw.validate_ram_config({"ramReclaimCache": False})
        with self.assertRaises(ValueError):
            self.gw.validate_ram_config({"ramReclaimCache": "yes"})


if __name__ == "__main__":
    unittest.main()
