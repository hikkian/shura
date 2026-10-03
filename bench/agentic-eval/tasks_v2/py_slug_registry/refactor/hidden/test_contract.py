import unittest
from slugs import slugify, unique_slug
class Slugs(unittest.TestCase):
 def test_normalize(self):
  self.assertEqual(slugify("Crème brûlée / v2"), "creme-brulee-v2")
  self.assertEqual(slugify("---"), "item")
  self.assertEqual(slugify("A__B"), "a-b")
 def test_unique(self):
  self.assertEqual(unique_slug("Hello world", set()), "hello-world")
  self.assertEqual(unique_slug("Hello world", {"hello-world", "hello-world-2", "hello-world-4"}), "hello-world-3")
 def test_type(self):
  with self.assertRaises(TypeError): slugify(None)
