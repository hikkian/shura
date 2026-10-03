from pathlib import Path
import unittest
class Layout(unittest.TestCase):
 def test_public_package_is_split(self):
  self.assertTrue(Path("retrying/__init__.py").is_file())
  self.assertTrue(Path("retrying/core.py").is_file())
  self.assertFalse(Path("retrying.py").exists())
