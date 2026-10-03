from pathlib import Path
import unittest
class Layout(unittest.TestCase):
 def test_public_package_is_split(self):
  self.assertTrue(Path("config_merge/__init__.py").is_file())
  self.assertTrue(Path("config_merge/core.py").is_file())
  self.assertFalse(Path("config_merge.py").exists())
