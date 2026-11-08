import unittest
from shop.paging import paginate, page_count
class H(unittest.TestCase):
    def test_pages(self):
        it = list(range(10))
        self.assertEqual(paginate(it, 2, 3), [3, 4, 5])
        self.assertEqual(paginate(it, 4, 3), [9])
        self.assertEqual(paginate(it, 5, 3), [])
    def test_count(self):
        self.assertEqual(page_count(9, 3), 3)
        self.assertEqual(page_count(0, 3), 0)
        self.assertEqual(page_count(1, 3), 1)
    def test_err(self):
        with self.assertRaises(ValueError): paginate([1], 0, 3)
