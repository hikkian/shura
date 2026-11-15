"""MEMV2 item 6: the dmem protection plan on a fake cgroup tree. Nothing touches the real /sys/fs/cgroup."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
import dmem_guard as dg  # noqa: E402

UID = 1000
REGION = "drm/0000:2d:00.0/vram"


def tree(root, user_slice="cpu io memory pids", user_n="cpu io memory pids", capacity=f"{REGION} 17163091968\n"):
    root = Path(root)
    (root / "dmem.capacity").write_text(capacity)
    us = root / "user.slice"
    un = us / f"user-{UID}.slice"
    mgr = un / f"user@{UID}.service"
    for d in (us, un, mgr, mgr / "session.slice", mgr / "app.slice"):
        d.mkdir(parents=True, exist_ok=True)
    (us / "cgroup.subtree_control").write_text(user_slice + "\n")
    (un / "cgroup.subtree_control").write_text(user_n + "\n")
    (mgr / "cgroup.subtree_control").write_text("cpu io memory pids\n")
    return mgr


class Plan(unittest.TestCase):
    def test_reads_the_vram_region(self):
        with tempfile.TemporaryDirectory() as d:
            tree(d, capacity=f"{REGION} 17163091968\ndrm/0000:2d:00.0/gtt 33554432\n")
            self.assertEqual(dg.vram_regions(d), [REGION])

    def test_without_the_root_part_nothing_is_written(self):
        with tempfile.TemporaryDirectory() as d:
            tree(d)
            writes, blockers = dg.plan(UID, d)
            self.assertTrue(any("root part" in b for b in blockers))
            self.assertFalse(dg.status(UID, d)["root_part"])

    def test_no_dmem_in_the_kernel_is_said(self):
        with tempfile.TemporaryDirectory() as d:
            tree(d, capacity="")
            _writes, blockers = dg.plan(UID, d)
            self.assertTrue(any("6.14" in b for b in blockers))

    def test_with_the_root_part_the_desk_slices_get_max_and_shura_none(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = tree(d, user_slice="cpu io memory pids dmem", user_n="cpu io memory pids dmem")
            (mgr / "shura.slice").mkdir()
            writes, blockers = dg.plan(UID, d)
            self.assertEqual(blockers, [])
            self.assertEqual(writes[0], (mgr / "cgroup.subtree_control", "+dmem"))
            targets = {str(p.relative_to(mgr)): t for p, t in writes[1:]}
            self.assertEqual(targets, {"session.slice/dmem.low": f"{REGION} max", "app.slice/dmem.low": f"{REGION} max"})
            self.assertNotIn("shura.slice", " ".join(targets))

    def test_apply_writes_in_order_and_undo_takes_the_protection_back(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = tree(d, user_slice="dmem", user_n="dmem")
            writes, _ = dg.plan(UID, d)
            seen = []
            self.assertEqual(dg.apply(writes, writer=lambda p, t: seen.append((p, t))), [])
            self.assertEqual(seen, writes)
            self.assertEqual(dg.apply(writes), [])            # plain files in the fake tree
            self.assertEqual((mgr / "app.slice" / "dmem.low").read_text().strip(), f"{REGION} max")
            undo = dg.undo_writes(UID, d)
            self.assertEqual({t for _p, t in undo}, {f"{REGION} 0"})
            self.assertEqual(len(undo), 2)

    def test_apply_reports_errors_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as d:
            errors = dg.apply([(Path(d) / "missing" / "dmem.low", "x")])
            self.assertEqual(len(errors), 1)


class Placement(unittest.TestCase):
    def test_shura_must_be_outside_the_protected_slices(self):
        base = "/user.slice/user-1000.slice/user@1000.service"
        self.assertFalse(dg.shura_outside_desk(base + "/app.slice/ai-gateway.service"))
        self.assertFalse(dg.shura_outside_desk(base + "/session.slice/x.service"))
        self.assertTrue(dg.shura_outside_desk(base + "/shura.slice/ai-gateway.service"))
        self.assertFalse(dg.shura_outside_desk(None))

    def test_where_is_reads_cgroup_v2(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "42").mkdir()
            (Path(d) / "42" / "cgroup").write_text("0::/user.slice/x\n")
            self.assertEqual(dg.where_is(42, d), "/user.slice/x")
            self.assertIsNone(dg.where_is(43, d))

    def test_the_root_dropin_enables_only_the_two_levels(self):
        self.assertIn("user.slice/cgroup.subtree_control", dg.ROOT_DROPIN)
        self.assertIn("user-%i.slice/cgroup.subtree_control", dg.ROOT_DROPIN)
        self.assertEqual(dg.ROOT_DROPIN.count("+dmem"), 2)
        self.assertIn("Slice=shura.slice", dg.GATEWAY_SLICE_DROPIN)


if __name__ == "__main__":
    unittest.main()
