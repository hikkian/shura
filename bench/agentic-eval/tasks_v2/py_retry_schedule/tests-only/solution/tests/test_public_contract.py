import unittest
from retrying import *

class PublicContract(unittest.TestCase):
    def test_edge_case(self):
        self.assertEqual(delay_for(3,1,20,.25,1),5)
