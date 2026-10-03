import unittest
from deps import *

class PublicContract(unittest.TestCase):
    def test_edge_case(self):
        self.assertEqual(plan({'app':['lib','util'],'lib':['core'],'util':['core'],'core':[],'z':['ghost']}),['core','ghost','lib','util','app','z'])
