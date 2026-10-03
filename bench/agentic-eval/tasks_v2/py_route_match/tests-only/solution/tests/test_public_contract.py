import unittest
from routes import *

class PublicContract(unittest.TestCase):
    def test_edge_case(self):
        self.assertIsNone(match_route('/users/:id','/users/1/edit'))
