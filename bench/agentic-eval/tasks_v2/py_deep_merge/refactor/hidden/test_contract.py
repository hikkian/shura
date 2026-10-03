import unittest
from config_merge import deep_merge
class Merge(unittest.TestCase):
 def test_recursive_and_replace(self):
  a={'db':{'host':'a','ports':[1,2]},'x':4}; b={'db':{'port':9,'ports':[3]},'x':None}
  self.assertEqual(deep_merge(a,b),{'db':{'host':'a','ports':[3],'port':9},'x':None})
  self.assertEqual(a['db']['ports'],[1,2])
 def test_fresh_and_types(self):
  a={'x':{'y':1}}; out=deep_merge(a,{}); out['x']['y']=3; self.assertEqual(a['x']['y'],1)
  with self.assertRaises(TypeError): deep_merge([],{})
