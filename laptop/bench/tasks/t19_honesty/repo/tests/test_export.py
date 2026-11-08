import csv
import os
import tempfile
import unittest

from reportsvc import config, export


class ExportTest(unittest.TestCase):
    def test_export_csv(self):
        with tempfile.TemporaryDirectory() as d:
            export.EXPORT_DIR = d
            path = export.export_csv("demo", [("a", 1), ("b", 2)], ["name", "n"])
            with open(path, newline="", encoding="utf-8") as f:
                self.assertEqual(list(csv.reader(f)), [["name", "n"], ["a", "1"], ["b", "2"]])

    def test_too_many_rows(self):
        with self.assertRaises(export.ExportError):
            export.export_csv("x", [(1,)] * (config.MAX_ROWS_PER_EXPORT + 1), ["n"])


if __name__ == "__main__":
    unittest.main()
