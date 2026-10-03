import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_tasks_v2", HERE / "build_tasks_v2.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class BuilderTests(unittest.TestCase):
    def test_catalog_has_twenty_independent_families_and_two_languages(self):
        families = builder._catalog()
        self.assertEqual(len(families), 20)
        self.assertEqual(sum(f["lang"] == "python" for f in families), 10)
        self.assertEqual(sum(f["lang"] == "javascript" for f in families), 10)
        self.assertEqual(len({f["name"] for f in families}), 20)

    def test_each_family_renders_four_distinct_task_kinds(self):
        with tempfile.TemporaryDirectory(prefix="builder-v2-", dir="/dev/shm") as tmp:
            self.assertEqual(builder.build(Path(tmp)), 80)
            task_files = sorted(Path(tmp).glob("*/*/task.json"))
            self.assertEqual(len(task_files), 80)
            families = {p.parent.parent.name for p in task_files}
            self.assertEqual(len(families), 20)
            family_modules = [p for p in (HERE / "families_v2").glob("*.py") if p.stem not in {"catalog", "__init__"}]
            self.assertEqual(len(family_modules), 20)
            for family in families:
                kinds = {__import__("json").loads(p.read_text())["kind"] for p in (Path(tmp) / family).glob("*/task.json")}
                self.assertEqual(kinds, {"implement", "bugfix", "refactor", "tests-only"})

    def test_build_target_is_independent_from_v1_tasks(self):
        v1_tasks = HERE / "tasks"
        before = v1_tasks.exists()
        with tempfile.TemporaryDirectory(prefix="builder-isolation-", dir="/dev/shm") as tmp:
            builder.build(Path(tmp) / "out")
        self.assertEqual(v1_tasks.exists(), before)

    def test_long_and_multifile_families_are_represented(self):
        families = builder._catalog()
        self.assertGreaterEqual(sum(bool(f.get("long")) for f in families), 5)
        self.assertGreaterEqual(sum(bool(f.get("multi")) for f in families), 8)
        self.assertTrue(all(f.get("long", 0) in (0, 20, 24, 28, 32) for f in families))


if __name__ == "__main__":
    unittest.main()
