import unittest
from pages import merge_pages
class Merge(unittest.TestCase):
 def test_stable_dedup_and_sort(self):
  a=[{"id":3,"v":"first"},{"id":1},{"id":2}]; b=[{"id":3,"v":"later"},{"id":2,"v":"later"},{"id":4}]
  self.assertEqual(merge_pages([a,b], "id"), [{"id":1},{"id":2},{"id":3,"v":"first"},{"id":4}])
  self.assertEqual(a[0]["v"], "first")
 def test_empty_and_missing(self):
  self.assertEqual(merge_pages([], "id"), [])
  with self.assertRaises(KeyError): merge_pages([[{}]], "id")
