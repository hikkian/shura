import unittest
from retrying import delay_for
class Delays(unittest.TestCase):
 def test_growth_cap_and_jitter(self):
  self.assertEqual([delay_for(n,1,5) for n in range(1,6)],[1,2,4,5,5])
  self.assertEqual(delay_for(3,1,20,.25,0),3)
  self.assertEqual(delay_for(3,1,20,.25,1),5)
 def test_validation(self):
  for args in [(0,1,4), (True,1,4), (1,-1,4), (1,2,1), (1,1,2,1.1), (1,1,2,0,float('nan'))]:
   with self.assertRaises(ValueError): delay_for(*args)
