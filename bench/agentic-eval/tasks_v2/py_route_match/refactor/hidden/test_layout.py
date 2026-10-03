from pathlib import Path
import unittest
class Layout(unittest.TestCase):
 def test_public_package_is_split(self):
  self.assertTrue(Path("routes/__init__.py").is_file())
  self.assertTrue(Path("routes/core.py").is_file())
  self.assertFalse(Path("routes.py").exists())
