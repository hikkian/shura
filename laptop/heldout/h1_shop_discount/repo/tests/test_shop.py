import unittest

from shop import Cart, fmt, render, to_cents


class T(unittest.TestCase):
    def test_to_cents(self):
        self.assertEqual(to_cents("10.50"), 1050)
        self.assertEqual(to_cents("1 299,50"), 129950)
        self.assertEqual(to_cents("7"), 700)

    def test_fmt(self):
        self.assertEqual(fmt(1999), "19.99")
        self.assertEqual(fmt(5), "0.05")

    def test_cart(self):
        c = Cart()
        c.add("Чай", "10,00", 2)
        c.add("Кофе", 550)
        self.assertEqual(c.subtotal(), 2550)
        self.assertEqual(c.total(10), 2295)

    def test_render(self):
        c = Cart()
        c.add("Чай", "10,00", 2)
        self.assertIn("Итого: 20.00", render(c))


if __name__ == "__main__":
    unittest.main()
