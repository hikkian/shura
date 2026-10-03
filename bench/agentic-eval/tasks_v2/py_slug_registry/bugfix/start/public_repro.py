import unittest
from slugs import *
class PublicRepro(unittest.TestCase):
 def test_reported_case(self):
  self.assertEqual(slugify("A__B"), "a-b")
