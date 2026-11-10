import unittest

from patchlib import apply_patch, ConflictError

OLD = "one\ntwo\nthree\nfour\n"
PATCH = """--- a/f.txt
+++ b/f.txt
@@ -1,4 +1,4 @@
 one
-two
+TWO
 three
 four
"""


class T(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(apply_patch(OLD, PATCH), "one\nTWO\nthree\nfour\n")

    def test_conflict(self):
        with self.assertRaises(ConflictError):
            apply_patch("one\nzzz\nthree\nfour\n", PATCH)


if __name__ == "__main__":
    unittest.main()
