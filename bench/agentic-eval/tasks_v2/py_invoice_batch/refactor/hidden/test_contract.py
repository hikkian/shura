import unittest
from invoice import invoice
class Invoice(unittest.TestCase):
 def test_integer_rounding(self):
  self.assertEqual(invoice([{'quantity':2,'unit_cents':125},{'quantity':1,'unit_cents':51}],1000),{'subtotal':301,'tax':30,'total':331})
  self.assertEqual(invoice([{'quantity':1,'unit_cents':5}],1000)['tax'],1)
 def test_rejects_bad_values(self):
  for rows,rate in [([{'quantity':True,'unit_cents':2}],0),([{'quantity':1,'unit_cents':-1}],0),([],True),([], -1)]:
   with self.assertRaises(ValueError): invoice(rows,rate)
