from pathlib import Path
import unittest
class ProjectDocs(unittest.TestCase):
 def test_supporting_files_are_completed(self):
  readme=Path("README.md").read_text()
  design=Path("docs/design.md").read_text()
  self.assertIn("headers", readme)
  self.assertNotIn("Starter project", readme)
  self.assertIn("Validation is explicit", design)
