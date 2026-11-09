import unittest

from envconf import ConfigError, Field, load, parse_bool


class T(unittest.TestCase):
    def test_basic(self):
        cfg = load({"port": int, "debug": Field(bool, False)}, prefix="app", environ={"APP_PORT": "80"})
        self.assertEqual(cfg.port, 80)
        self.assertFalse(cfg.debug)

    def test_errors(self):
        with self.assertRaises(ConfigError) as cm:
            load({"a": int, "b": int}, prefix="x", environ={"X_A": "q"})
        self.assertEqual(len(cm.exception.errors), 2)

    def test_bool(self):
        self.assertTrue(parse_bool("YES"))
        with self.assertRaises(ValueError):
            parse_bool("")

    def test_secret(self):
        cfg = load({"tok": Field(str, secret=True)}, prefix="p", environ={"P_TOK": "abc"})
        self.assertEqual(cfg.as_dict(), {"tok": "***"})


if __name__ == "__main__":
    unittest.main()
