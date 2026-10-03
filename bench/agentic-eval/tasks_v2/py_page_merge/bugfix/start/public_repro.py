import unittest
from pages import *
class PublicRepro(unittest.TestCase):
 def test_reported_case(self):
  self.assertEqual(merge_pages([[{"id":3,"v":"first"},{"id":1}],[{"id":3,"v":"later"},{"id":2}]], "id"), [{"id":1},{"id":2},{"id":3,"v":"first"}])
