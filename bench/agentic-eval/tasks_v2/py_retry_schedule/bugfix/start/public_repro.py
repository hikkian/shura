import unittest
from retrying import *
class PublicRepro(unittest.TestCase):
 def test_reported_case(self):
  self.assertEqual(delay_for(3,1,20,.25,1),5)
