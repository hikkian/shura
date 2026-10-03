import unittest
from journal import replay
class Replay(unittest.TestCase):
 def test_idempotent_and_ordered(self):
  ev=[{'seq':1,'op':'set','key':'x','value':2},{'seq':1,'op':'set','key':'x','value':9},{'seq':2,'op':'add','key':'x','value':3},{'seq':3,'op':'delete','key':'z'}]
  self.assertEqual(replay(ev,{'z':1}),{'x':5})
 def test_atomic_copy_and_invalid(self):
  base={'a':1}
  with self.assertRaises(ValueError): replay([{'seq':2,'op':'set','key':'a','value':2},{'seq':1,'op':'set','key':'b','value':3}],base)
  self.assertEqual(base,{'a':1})
  with self.assertRaises(ValueError): replay([{'seq':1,'op':'wat','key':'x'}])
