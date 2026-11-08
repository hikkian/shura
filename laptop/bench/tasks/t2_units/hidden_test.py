import unittest
from weather.units import f_to_c, c_to_f
from weather.report import report
class H(unittest.TestCase):
    def test_conv(self):
        self.assertAlmostEqual(f_to_c(32), 0)
        self.assertAlmostEqual(f_to_c(-40), -40)
        self.assertAlmostEqual(c_to_f(100), 212)
    def test_report(self):
        self.assertEqual(report([("A", 50), ("B", 32)]), "A: 10.0°C\nB: 0.0°C")
