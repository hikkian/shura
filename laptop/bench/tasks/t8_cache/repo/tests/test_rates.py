import unittest
import rates
class T(unittest.TestCase):
    def test_value(self):
        self.assertEqual(rates.get_rate("USD", "KZT"), 480.0)
