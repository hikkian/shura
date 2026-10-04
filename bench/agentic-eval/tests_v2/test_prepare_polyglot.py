import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "external" / "prepare_polyglot_tasks.py"
SPEC = importlib.util.spec_from_file_location("prepare_polyglot_tasks", SOURCE)
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


class PolyglotPreparationTests(unittest.TestCase):
    def test_import_preserves_starter_and_hides_test_files(self):
        with tempfile.TemporaryDirectory(prefix="polyglot-import-", dir="/dev/shm") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "README.md").write_text("fixture")
            ex = source / "python/exercises/practice/demo"
            (ex / ".meta").mkdir(parents=True)
            (ex / ".docs").mkdir()
            (ex / "demo.py").write_text("def value():\n    return None\n")
            (ex / ".meta/example.py").write_text("def value():\n    return 1\n")
            (ex / "demo_test.py").write_text("from demo import value\ndef test_value(): assert value()==1\n")
            (ex / ".meta/config.json").write_text(json.dumps({"files": {"solution": ["demo.py"], "test": ["demo_test.py"]}}))
            (ex / ".docs/instructions.md").write_text("Implement value(); see demo_test.py for examples.\n")
            out = root / "tasks"
            ids = prep.prepare(source, out, ["python"])
            task_dir = out / ids["python"][0] / "implement"
            task = json.loads((task_dir / "task.json").read_text())
            self.assertEqual(task["hidden_files"], ["demo_test.py"])
            self.assertTrue((task_dir / "start/demo.py").is_file())
            self.assertFalse((task_dir / "start/demo_test.py").exists())
            self.assertTrue((task_dir / "hidden/demo_test.py").is_file())
            self.assertTrue((task_dir / "solution/demo.py").is_file())
            self.assertTrue(task["reference_available"])
            self.assertIn("run_aider_tests.py", task["test_cmd"][1])
            self.assertEqual(task["language_test_cmd"], ["pytest", "-q", "demo_test.py"])
            self.assertNotIn("demo_test.py", task["prompt"])

    def test_task_kinds_are_independent_ideas(self):
        self.assertEqual(prep.TEST_COMMANDS["python"], ["pytest", "-q"])
        self.assertEqual(prep.TEST_COMMANDS["javascript"], ["/aider/benchmark/npm-test.sh"])
        self.assertEqual(len(prep.SUPPORTED), 6)

    def test_reference_example_keeps_non_source_manifest_from_starter(self):
        with tempfile.TemporaryDirectory(prefix="polyglot-rust-", dir="/dev/shm") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "README.md").write_text("fixture")
            exercise = source / "rust/exercises/practice/demo"
            (exercise / ".meta").mkdir(parents=True)
            (exercise / "src").mkdir()
            (exercise / "tests").mkdir()
            (exercise / ".meta/config.json").write_text(json.dumps({
                "files": {"solution": ["src/lib.rs", "Cargo.toml"], "test": ["tests/lib.rs"]}
            }))
            (exercise / ".meta/example.rs").write_text("pub fn value() -> u8 { 1 }\n")
            (exercise / "src/lib.rs").write_text("pub fn value() -> u8 { 0 }\n")
            (exercise / "Cargo.toml").write_text("[package]\nname='demo'\n")
            (exercise / "tests/lib.rs").write_text("#[test] fn value() { assert_eq!(demo::value(),1); }\n")
            out = root / "tasks"
            ids = prep.prepare(source, out, ["rust"])
            task_dir = out / ids["rust"][0] / "implement"
            task = json.loads((task_dir / "task.json").read_text())
            self.assertTrue(task["reference_available"])
            self.assertEqual((task_dir / "solution/src/lib.rs").read_text(), "pub fn value() -> u8 { 1 }\n")
            self.assertEqual((task_dir / "solution/Cargo.toml").read_text(), "[package]\nname='demo'\n")


if __name__ == "__main__":
    unittest.main()
