import unittest
from shop.paging import paginate, page_count

class T(unittest.TestCase):
    def test_first_page(self):
        self.assertEqual(paginate(list(range(10)), 1, 3), [0, 1, 2])
    def test_count(self):
        self.assertEqual(page_count(10, 3), 4)
