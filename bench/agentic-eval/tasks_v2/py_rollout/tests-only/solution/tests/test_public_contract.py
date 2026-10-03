import unittest
from rollout import *

class PublicContract(unittest.TestCase):
    def test_edge_case(self):
        self.assertTrue(enabled('new-ui','user-17',37,'release'))
