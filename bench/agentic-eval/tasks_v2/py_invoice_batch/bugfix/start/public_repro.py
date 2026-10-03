import unittest
from invoice import *
class PublicRepro(unittest.TestCase):
 def test_reported_case(self):
  self.assertEqual(invoice([{'quantity':1,'unit_cents':5}],1000)['tax'],1)
