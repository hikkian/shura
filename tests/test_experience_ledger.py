import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gateway"))
import experience_ledger as L


class ExperienceLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = L.Ledger(Path(self.tmp.name) / "experience.sqlite")

    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def test_opt_in_append_types_split_consistency_and_consent(self):
        with self.assertRaises(L.LedgerError): self.ledger.add_task("task")
        self.ledger.set_consent("capture", True)
        task = self.ledger.add_task("Fix the bug", "snapshot:abc", ["pytest"], repo="repo", block="repo@abc", privacy="open")
        self.assertIn(task["split"], ("train", "exam", "sealed_exam"))
        self.assertEqual(self.ledger.assign_split("repo@abc"), task["split"])
        self.ledger.add_outcome(task["id"], "fixed", "edited", {"pytest": "pass"}, privacy="open")
        self.ledger.add_correction(task["id"], "old", "fixed", privacy="open")
        gate = self.ledger.add_gate_decision("pass", {"ci": [0.1, 0.3]}, privacy="open")
        self.ledger.set_consent("exam", True)
        exam = self.ledger.add_exam_answer("e1", task["id"], "raw answer", privacy="personal")
        pack = self.ledger.add_pack_stack([{"name": "base", "verified": True}, {"name": "ru", "verified": True}], privacy="personal")
        self.assertEqual(self.ledger.pack_recipe(pack["id"])["states"][1]["name"], "ru")
        exported = [json.loads(line) for line in self.ledger.export().splitlines()]
        self.assertIn(gate["id"], [r["id"] for r in exported])
        self.assertNotIn(exam["id"], [r["id"] for r in exported])
        self.assertNotIn(pack["id"], [r["id"] for r in exported])

    def test_growth_is_consent_gated_and_observational(self):
        self.ledger.set_consent("capture", True)
        task = self.ledger.add_task("one", block="week1")
        self.ledger.add_outcome(task["id"], "answer", "accepted")
        self.assertEqual(self.ledger.growth()["weekly_acceptance"], [])
        self.ledger.set_consent("weekly_acceptance", True)
        self.assertEqual(self.ledger.growth()["weekly_acceptance"][0]["accepted_without_edits"], 1)
        self.ledger.set_consent("growth", True)
        self.ledger.set_growth_anchor(0.5, 0.6, 0.55, 0.65, 30)
        self.assertEqual(self.ledger.growth()["anchor"]["task_count"], 30)

    def test_forget_erases_payload_and_preserves_hash_tombstone(self):
        self.ledger.set_consent("capture", True)
        task = self.ledger.add_task("private text", block="private-block")
        before_size = (Path(self.tmp.name) / "experience.sqlite").stat().st_size
        forgotten = self.ledger.forget(task["id"])
        entry = self.ledger.get(task["id"])
        self.assertEqual(entry["kind"], "tombstone")
        self.assertEqual(entry["sha256"], forgotten["sha256"])
        self.assertNotIn("private text", self.ledger.path.read_bytes().decode("latin1"))
        self.assertGreaterEqual(before_size, 0)
        with self.assertRaises(L.LedgerError): self.ledger.forget(task["id"])

    def test_pack_recipe_and_consent_revoke(self):
        self.ledger.set_consent("capture", True)
        source = self.ledger.add_task("seed", block="seed-block", privacy="personal")
        item = self.ledger.add_pack_stack([{"recipe": "base", "verified": True, "source_event_ids": [source["id"]]},
                                           {"recipe": "unrelated", "verified": True}], "open")
        self.assertEqual(self.ledger.get(item["id"])["privacy"], "personal")
        self.ledger.forget(source["id"])
        recipe = self.ledger.pack_recipe(item["id"])
        self.assertEqual([state["recipe"] for state in recipe["states"]], ["unrelated"])
        self.assertEqual(recipe["omitted_erased_source_ids"], [source["id"]])
        self.ledger.forget(item["id"])
        with self.assertRaises(L.LedgerError): self.ledger.pack_recipe(item["id"])
        self.ledger.set_consent("capture", False)
        with self.assertRaises(L.LedgerError): self.ledger.add_task("no")

    def test_statistics_are_bounded_and_report_storage_location(self):
        stats = self.ledger.stats()
        self.assertEqual(stats["bytes_per_day_estimate"], 102400)
        self.assertGreaterEqual(stats["database_bytes"], 0)

    def test_authors_gate_training_and_context_cleanliness(self):
        self.ledger.set_consent("capture", True)
        task = self.ledger.add_task("request", block="auth", author="human")
        correction = self.ledger.add_correction(task["id"], "a", "lesson", author="closed_ai")
        self.assertEqual(self.ledger.get(correction["id"])["author"], "closed_ai")
        self.assertEqual(self.ledger.trainable(correction["id"]), (False, "author_closed_ai_or_unknown"))
        self.assertNotIn(correction["id"], [r["id"] for r in self.ledger.training_view()])
        bad = self.ledger.add_outcome(task["id"], "answer", "accepted", context=["human", "closed_ai"], author="local_model")
        good = self.ledger.add_outcome(task["id"], "answer", "accepted", context=["human", "local_model"], author="local_model")
        missing = self.ledger.add_outcome(task["id"], "answer", "accepted", author="local_model")
        self.assertFalse(self.ledger.get(bad["id"])["payload"]["context_clean"])
        self.assertEqual(self.ledger.trainable(bad["id"]), (False, "context_not_clean"))
        self.assertTrue(self.ledger.get(good["id"])["payload"]["context_clean"])
        self.assertFalse(self.ledger.get(missing["id"])["payload"]["context_clean"])

    def test_legacy_schema_reads_unknown_and_forget_works(self):
        path = Path(self.tmp.name) / "old.sqlite"
        self.ledger.close()
        db = __import__("sqlite3").connect(path)
        db.execute("CREATE TABLE events(id INTEGER PRIMARY KEY,ts TEXT NOT NULL,kind TEXT NOT NULL,privacy TEXT NOT NULL,payload TEXT NOT NULL,forgotten_hash TEXT)")
        db.execute("INSERT INTO events VALUES (1,'2020-01-01','correction_pair','work','{\"correction\":\"old text\"}',NULL)")
        db.commit(); db.close()
        self.ledger = L.Ledger(path)
        self.assertEqual(self.ledger.get(1)["author"], "unknown")
        self.assertEqual(self.ledger.trainable(1), (False, "author_closed_ai_or_unknown"))
        self.assertTrue(self.ledger.forget(1)["sha256"])


if __name__ == "__main__": unittest.main()
