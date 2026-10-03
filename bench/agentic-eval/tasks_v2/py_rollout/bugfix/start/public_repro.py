import unittest
from rollout import *
class PublicRepro(unittest.TestCase):
 def test_reported_case(self):
  self.assertTrue(enabled('new-ui','user-17',37,'release'))
