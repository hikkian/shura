import sqlite3
import unittest

from userdb.users import (add_user, get_user, hash_password, init_db, login,
                          search_users, update_email)


class UsersTest(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        init_db(self.conn)
        add_user(self.conn, "zhanar", "zhanar@mail.kz", "qwerty")
        add_user(self.conn, "arman", "arman@mail.kz", "s3cret")
        add_user(self.conn, "root", "root@corp.kz", "toor", is_admin=True)

    def test_search_by_name(self):
        res = search_users(self.conn, "arm")
        self.assertEqual([u["username"] for u in res], ["arman"])

    def test_search_by_email_sorted(self):
        res = search_users(self.conn, "mail.kz", sort="email")
        self.assertEqual([u["username"] for u in res], ["arman", "zhanar"])

    def test_login_ok_and_fail(self):
        self.assertEqual(login(self.conn, "root", "toor")["is_admin"], True)
        self.assertIsNone(login(self.conn, "root", "wrong"))
        self.assertIsNone(login(self.conn, "nobody", "toor"))

    def test_get_user(self):
        self.assertEqual(get_user(self.conn, 2)["username"], "arman")
        self.assertIsNone(get_user(self.conn, 99))

    def test_update_email(self):
        update_email(self.conn, "arman", "new@mail.kz")
        self.assertEqual(get_user(self.conn, 2)["email"], "new@mail.kz")

    def test_hash(self):
        self.assertEqual(len(hash_password("x")), 64)


if __name__ == "__main__":
    unittest.main()
