import unittest
from headers import *

class PublicContract(unittest.TestCase):
    def test_edge_case(self):
        self.assertEqual(parse_headers(['X-Name:  Alpha','\tbeta','x-name: Gamma','Accept: text/plain','']), {'x-name':['Alpha beta','Gamma'],'accept':'text/plain'})
