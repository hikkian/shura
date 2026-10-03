import unittest
from deps import plan
class Plans(unittest.TestCase):
 def test_order(self):
  g={'app':['lib','util'],'lib':['core'],'util':['core'],'core':[],'z':['ghost']}
  self.assertEqual(plan(g),['core','ghost','lib','util','app','z'])
 def test_cycle(self):
  with self.assertRaises(ValueError) as c: plan({'a':['b'],'b':['a']})
  self.assertEqual(c.exception.cycle[0],c.exception.cycle[-1])
  self.assertGreaterEqual(len(c.exception.cycle),2)
