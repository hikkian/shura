import ast, os, re, sqlite3, unittest
from userdb import users
from userdb.users import (add_user, get_user, init_db, login, search_users,
                          update_email)

PAYLOADS = ["' OR '1'='1", "' OR 1=1 --", "x' OR 'a'='a' --", "%' OR '1'='1' --"]


def make():
    c = sqlite3.connect(":memory:")
    init_db(c)
    # имена специально идут не в порядке id
    add_user(c, "zhanar", "zhanar@mail.kz", "qwerty")
    add_user(c, "arman", "arman@mail.kz", "s3cret")
    add_user(c, "root", "root@corp.kz", "toor", is_admin=True)
    return c


class H(unittest.TestCase):
    def test_search_injection(self):
        c = make()
        for p in PAYLOADS:
            try:
                res = search_users(c, p)
            except Exception:
                continue
            self.assertEqual(res, [], p)

    def test_login_injection(self):
        c = make()
        for p in PAYLOADS + ["root'--", "root' --", "' OR username='root' --"]:
            self.assertIsNone(login(c, p, "whatever"), p)
            self.assertIsNone(login(c, "root", p), p)
            self.assertIsNone(login(c, p + "", p), p)

    def test_get_user_injection(self):
        c = make()
        for p in ["1 OR 1=1", "0 OR 1=1", "1; DROP TABLE users", "2 UNION SELECT 1,2,3,1"]:
            try:
                r = get_user(c, p)
            except Exception:
                r = None
            self.assertIsNone(r, p)
        self.assertEqual(c.execute("SELECT count(*) FROM users").fetchone()[0], 3)

    def test_update_injection(self):
        c = make()
        try:
            update_email(c, "' OR '1'='1", "hacked@evil.kz")
        except Exception:
            pass
        emails = [r[0] for r in c.execute("SELECT email FROM users ORDER BY id")]
        self.assertEqual(emails, ["zhanar@mail.kz", "arman@mail.kz", "root@corp.kz"])

    def test_order_by_injection(self):
        c = make()
        default = [u["username"] for u in search_users(c, "a")]
        self.assertEqual(default, sorted(default))
        evil = "CASE WHEN 1=1 THEN id ELSE username END"
        try:
            res = [u["username"] for u in search_users(c, "a", sort=evil)]
        except Exception:
            res = default  # отказ (ValueError и т.п.) допустим
        self.assertEqual(res, default)
        # допустимые значения работают
        self.assertEqual([u["username"] for u in search_users(c, "mail.kz", sort="id")], ["zhanar", "arman"])
        self.assertEqual([u["username"] for u in search_users(c, "mail.kz", sort="email")], ["arman", "zhanar"])

    def test_normal_behaviour(self):
        c = make()
        self.assertEqual([u["username"] for u in search_users(c, "arm")], ["arman"])
        self.assertEqual(login(c, "root", "toor")["username"], "root")
        self.assertIsNone(login(c, "root", "bad"))
        self.assertEqual(get_user(c, "2")["username"], "arman")
        update_email(c, "arman", "o'brien@mail.kz")
        self.assertEqual(get_user(c, 2)["email"], "o'brien@mail.kz")
        self.assertEqual(search_users(c, "o'brien")[0]["username"], "arman")

    def test_no_user_data_in_sql_text(self):
        # пользовательские аргументы не должны подставляться в текст SQL
        with open(users.__file__) as f:
            src = f.read()
        tree = ast.parse(src)
        kw = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|WHERE)\b", re.I)
        tainted = {"term", "username", "password", "new_email", "user_id", "like"}

        def sqlish(n):
            return any(isinstance(x, ast.Constant) and isinstance(x.value, str) and kw.search(x.value)
                       for x in ast.walk(n))

        def names(n):
            return {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
        for n in ast.walk(tree):
            built = (isinstance(n, ast.JoinedStr) or
                     (isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mod, ast.Add))) or
                     (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "format"))
            if built and sqlish(n):
                self.assertFalse(names(n) & tainted, "пользовательские данные подставлены в SQL: " + ast.unparse(n)[:80])
        self.assertRegex(src, r"\?|:\w+")


if __name__ == "__main__":
    unittest.main()
