import unittest
from config_merge import *
class PublicRepro(unittest.TestCase):
 def test_reported_case(self):
  base={'x':{'y':1}}; out=deep_merge(base,{}); out['x']['y']=2; self.assertEqual(base['x']['y'],1)
