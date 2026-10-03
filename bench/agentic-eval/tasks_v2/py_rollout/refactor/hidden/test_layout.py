from pathlib import Path
import unittest
class Layout(unittest.TestCase):
 def test_public_package_is_split(self):
  self.assertTrue(Path("rollout/__init__.py").is_file())
  self.assertTrue(Path("rollout/core.py").is_file())
  self.assertFalse(Path("rollout.py").exists())
