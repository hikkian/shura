import unittest
from rollout import enabled
class Rollout(unittest.TestCase):
 def test_boundaries_and_stability(self):
  self.assertFalse(enabled('new-ui','u1',0)); self.assertTrue(enabled('new-ui','u1',100))
  self.assertTrue(enabled('new-ui','user-17',37,'release'))
 def test_validation(self):
  for args in [('', 'u', 50), ('x','',50), ('x','u',-1), ('x','u',101), ('x','u',True)]:
   with self.assertRaises(ValueError): enabled(*args)
