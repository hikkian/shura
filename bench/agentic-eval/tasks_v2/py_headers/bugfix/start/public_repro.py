import unittest
from headers import *
class PublicRepro(unittest.TestCase):
 def test_reported_case(self):
  self.assertEqual(parse_headers(['X-Name:  Alpha','\tbeta','x-name: Gamma','Accept: text/plain','']), {'x-name':['Alpha beta','Gamma'],'accept':'text/plain'})
