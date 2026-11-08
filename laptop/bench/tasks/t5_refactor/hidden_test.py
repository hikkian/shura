import unittest, pathlib, re
from validators import is_valid_email
class H(unittest.TestCase):
    def test_fn(self):
        self.assertTrue(is_valid_email("a@b.co"))
        self.assertTrue(is_valid_email(" a@b.co "))
        for e in ["a@b", "a b@c.de", "@c.de"]:
            self.assertFalse(is_valid_email(e))
        from users import create_user
        from orders import create_order
        for e in ["", None, "a@b"]:
            with self.assertRaises(ValueError): create_user("a", e)
            with self.assertRaises(ValueError): create_order(e, 1)
    def test_dedup(self):
        for f in ["users.py", "orders.py"]:
            s = pathlib.Path(f).read_text()
            self.assertIn("is_valid_email", s)
            self.assertNotIn("fullmatch", s)
