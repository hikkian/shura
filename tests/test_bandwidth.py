"""Bandwidth helper tests: parsing and sizing are pure; one tiny real run (skipped without a C compiler)."""
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "installer"))
from universal import bandwidth as BW  # noqa: E402

GiB = 1024 ** 3


class Pure(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(BW.parse("membw_gbs 34.81\n"), 34.81)
        self.assertIsNone(BW.parse("garbage"))
        self.assertIsNone(BW.parse(None))

    def test_buffers_never_exceed_a_quarter_of_free_ram(self):
        for cores, free in ((6, 20 * GiB), (64, 400 * GiB), (4, 3 * GiB), (16, 1 * GiB)):
            threads, per = BW.plan_threads(cores, free)
            self.assertLessEqual(threads * per * 1024 * 1024, free * 0.25 + threads * 1024 * 1024)
            self.assertLessEqual(per, BW.MAX_MIB_PER_THREAD)

    def test_one_cpu_per_physical_core_whatever_the_numbering(self):
        # siblings far apart (0,6 / 1,7 ...) as on many AMD and Intel desktops
        far = {f"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list": f"{i % 3},{i % 3 + 3}" for i in range(6)}
        self.assertEqual(BW.physical_cpu_ids(far.get), [0, 1, 2])
        # siblings adjacent (0,1 / 2,3 ...)
        near = {f"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list": f"{i - i % 2},{i - i % 2 + 1}"
                for i in range(6)}
        self.assertEqual(BW.physical_cpu_ids(near.get), [0, 2, 4])
        self.assertEqual(BW.physical_cpu_ids(lambda p: None), [])

    def test_too_little_free_ram_means_no_measurement(self):
        self.assertIsNone(BW.measure(8, 100 * 1024 * 1024))


@unittest.skipIf(sys.platform == "win32", "the C helper is POSIX-only for now")
@unittest.skipUnless(shutil.which("cc") or shutil.which("gcc") or shutil.which("clang"), "no C compiler")
class RealRun(unittest.TestCase):
    def test_tiny_measurement_returns_a_positive_number(self):
        gbs = BW.measure(1, 2 * GiB, passes=1)        # 1 thread, small buffer: a fraction of a second
        self.assertIsNotNone(gbs)
        self.assertGreater(gbs, 0.5)


if __name__ == "__main__":
    unittest.main()
