import unittest, importlib
import rates
class H(unittest.TestCase):
    def setUp(self):
        importlib.reload(rates)
    def test_cache(self):
        rates.get_rate("USD", "KZT", now=1000)
        rates.get_rate("USD", "KZT", now=1030)
        rates.get_rate("USD", "EUR", now=1030)
        self.assertEqual(rates.CALLS, [("USD", "KZT"), ("USD", "EUR")])
    def test_expiry(self):
        rates.get_rate("EUR", "KZT", now=0)
        rates.get_rate("EUR", "KZT", now=61)
        self.assertEqual(len(rates.CALLS), 2)
    def test_default_now(self):
        self.assertEqual(rates.get_rate("USD", "KZT"), 480.0)
        self.assertEqual(rates.get_rate("USD", "KZT"), 480.0)
        self.assertEqual(len(rates.CALLS), 1)
