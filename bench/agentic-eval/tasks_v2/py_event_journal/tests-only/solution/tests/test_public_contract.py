import unittest
from journal import *

class PublicContract(unittest.TestCase):
    def test_edge_case(self):
        events=[{'seq':1,'op':'set','key':'x','value':2},{'seq':1,'op':'set','key':'x','value':9},{'seq':2,'op':'add','key':'x','value':3}]; self.assertEqual(replay(events),{'x':5})
