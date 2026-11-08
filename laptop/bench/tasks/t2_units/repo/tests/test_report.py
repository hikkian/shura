import unittest
from weather.report import format_reading
class T(unittest.TestCase):
    def test_boil(self):
        self.assertEqual(format_reading("X", 212), "X: 100.0°C")
