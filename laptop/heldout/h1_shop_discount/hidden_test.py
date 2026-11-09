import re
import unittest

from shop import Cart, fmt, render, to_cents


def cart_with(cents, name="Товар"):
    c = Cart()
    c.add(name, cents)
    return c


class H(unittest.TestCase):
    def test_to_cents_exact(self):
        for s, v in [("19.99", 1999), ("0.29", 29), ("1.15", 115), ("4.35", 435), ("0.07", 7),
                     ("8.2", 820), ("1 299,50", 129950), ("1 299,5", 129950), ("0,1", 10),
                     ("100", 10000), ("  12,05 ", 1205), ("2 000 000.99", 200000099)]:
            self.assertEqual(to_cents(s), v, s)

    def test_to_cents_invalid(self):
        for s in ["abc", "", "   ", "12.345", "1.005", "1.2.3", "12,3,4", "1 2 x"]:
            with self.assertRaises(ValueError, msg=repr(s)):
                to_cents(s)

    def test_discount_half_up(self):
        self.assertEqual(cart_with(1005).discount(10), 101)   # 100.5 -> 101
        self.assertEqual(cart_with(25).discount(10), 3)       # 2.5 -> 3 (round() даёт 2)
        self.assertEqual(cart_with(5).discount(50), 3)
        self.assertEqual(cart_with(35).discount(10), 4)
        self.assertEqual(cart_with(1001).discount(12.5), 125)  # 125.125
        self.assertEqual(cart_with(4).discount(12.5), 1)       # 0.5 -> 1
        self.assertEqual(cart_with(250).discount(0.7), 2)      # 1.75
        self.assertEqual(cart_with(500).discount(0.1), 1)      # 0.5 -> 1
        self.assertEqual(cart_with(1000).discount(10), 100)

    def test_discount_exact_float_trap(self):
        self.assertEqual(cart_with(5500).discount(0.7), 39)  # 38.5 -> 39 (float floor(x+0.5) gives 38)

    def test_total(self):
        self.assertEqual(cart_with(1005).total(10), 904)
        self.assertEqual(cart_with(1005).total(), 1005)
        self.assertEqual(cart_with(1005).total(0), 1005)
        self.assertEqual(cart_with(1005).total(100), 0)
        c = Cart()
        c.add("a", "0,33", 3)
        c.add("b", "0,05", 1)
        self.assertEqual(c.subtotal(), 104)
        self.assertEqual(c.total(50), 52)

    def test_bad_pct(self):
        for p in (-1, 100.01, 101):
            with self.assertRaises(ValueError, msg=str(p)):
                cart_with(1000).total(p)

    def test_empty_cart(self):
        c = Cart()
        self.assertEqual(c.subtotal(), 0)
        self.assertEqual(c.total(10), 0)
        out = render(c, 10)
        self.assertIsInstance(out, str)
        self.assertIn("0.00", out)

    def test_render_discount(self):
        c = Cart()
        c.add("Чай «Пуэр» ☕", "5,00", 2)
        c.add("Кофе", "0,05")
        self.assertEqual(c.subtotal(), 1005)
        out = render(c, 10)
        lines = out.splitlines()
        self.assertIn("Чай «Пуэр» ☕", out)
        disc = [l for l in lines if re.match(r"\s*скидк", l, re.I)]
        self.assertEqual(len(disc), 1, out)
        self.assertIn("10%", disc[0])
        self.assertRegex(disc[0], r"-?1\.01")
        total = [l for l in lines if l.startswith("Итого")]
        self.assertEqual(len(total), 1, out)
        self.assertIn("9.04", total[0], out)

    def test_render_no_discount_unchanged(self):
        c = Cart()
        c.add("Чай", "10,00", 2)
        out = render(c)
        self.assertIn("Итого: 20.00", out)
        self.assertNotIn("кидк", out)
        self.assertNotIn("кидк", render(c, 0))
