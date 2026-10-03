import unittest
from config_merge import *

class PublicContract(unittest.TestCase):
    def test_edge_case(self):
        base={'x':{'y':1}}; out=deep_merge(base,{}); out['x']['y']=2; self.assertEqual(base['x']['y'],1)
