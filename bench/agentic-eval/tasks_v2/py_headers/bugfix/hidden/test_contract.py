import unittest
from headers import parse_headers
class Headers(unittest.TestCase):
 def test_case_duplicates_and_continuations(self):
  self.assertEqual(parse_headers(['X-Name:  Alpha','\tbeta','x-name: Gamma','Accept: text/plain','']), {'x-name':['Alpha beta','Gamma'],'accept':'text/plain'})
 def test_invalid(self):
  for lines in [[' orphan'],['bad'],[' : x']]:
   with self.assertRaises(ValueError): parse_headers(lines)
