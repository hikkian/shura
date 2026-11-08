import unittest
from users import create_user
from orders import create_order
class T(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(create_user("a", " A@b.co ")["email"], "a@b.co")
        self.assertEqual(create_order("x@y.org", 5)["amount"], 5)
    def test_bad(self):
        for e in ["", "a@b", "a b@c.de", "@c.de"]:
            with self.assertRaises(ValueError): create_user("a", e)
            with self.assertRaises(ValueError): create_order(e, 1)
