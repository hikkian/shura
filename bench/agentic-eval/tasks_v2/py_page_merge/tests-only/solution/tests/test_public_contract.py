import unittest
from pages import *

class PublicContract(unittest.TestCase):
    def test_edge_case(self):
        self.assertEqual(merge_pages([[{"id":3,"v":"first"},{"id":1}],[{"id":3,"v":"later"},{"id":2}]], "id"), [{"id":1},{"id":2},{"id":3,"v":"first"}])
