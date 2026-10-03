import unittest
from invoice import *

class PublicContract(unittest.TestCase):
    def test_edge_case(self):
        self.assertEqual(invoice([{'quantity':1,'unit_cents':5}],1000)['tax'],1)
