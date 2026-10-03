import unittest
from slugs import *

class PublicContract(unittest.TestCase):
    def test_edge_case(self):
        self.assertEqual(slugify("A__B"), "a-b")
