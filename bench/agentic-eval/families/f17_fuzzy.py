FAMILY = {
    "name": "f17_fuzzy", "lang": "python",
    "implement": {
        "prompt": """Write the module `fuzzy.py` (standard library only) with three functions.

distance(a, b) -> int : the optimal string alignment distance (Damerau-Levenshtein, restricted): the minimum number of single-character
  insertions, deletions, substitutions and TRANSPOSITIONS OF TWO ADJACENT characters needed to turn a into b, where no substring is
  edited more than once. distance("ca", "abc") is 3 (not 2). distance("ab", "ba") is 1; ("kitten","sitting") is 3; ("", "abc") is 3.
  Comparison is case-sensitive.
closest(word, candidates, max_distance=2) -> list : candidates whose distance to `word` is <= max_distance, sorted by distance, then by
  how many leading characters they share with `word` (more first), then alphabetically. Duplicates in candidates are returned once.
  max_distance must be an int >= 0 (ValueError otherwise).
best_match(word, candidates) -> str or None : the first element of closest(word, candidates, max_distance=len(word) // 3 + 1), or None
  when it is empty. (So short words tolerate 1 typo, 6-8 letter words 3.)
""",
        "start": {"fuzzy.py": '"""Fuzzy string matching (to be written)."""\n\n\ndef distance(a, b):\n    raise NotImplementedError\n\n\ndef closest(word, candidates, max_distance=2):\n    raise NotImplementedError\n\n\ndef best_match(word, candidates):\n    raise NotImplementedError\n'},
        "solution": {"fuzzy.py": '''"""Fuzzy string matching."""


def distance(a, b):
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _shared_prefix(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def closest(word, candidates, max_distance=2):
    if isinstance(max_distance, bool) or not isinstance(max_distance, int) or max_distance < 0:
        raise ValueError("max_distance must be an int >= 0")
    seen, scored = set(), []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        dist = distance(word, c)
        if dist <= max_distance:
            scored.append((dist, -_shared_prefix(word, c), c))
    return [c for _, _, c in sorted(scored)]


def best_match(word, candidates):
    found = closest(word, candidates, len(word) // 3 + 1)
    return found[0] if found else None
'''},
    },
    "hidden": {"test_fuzzy.py": '''import unittest
from fuzzy import best_match, closest, distance


class Distance(unittest.TestCase):
    def test_basics(self):
        self.assertEqual(distance("", ""), 0)
        self.assertEqual(distance("", "abc"), 3)
        self.assertEqual(distance("abc", ""), 3)
        self.assertEqual(distance("kitten", "sitting"), 3)
        self.assertEqual(distance("flaw", "lawn"), 2)
        self.assertEqual(distance("same", "same"), 0)
        self.assertEqual(distance("a", "A"), 1)

    def test_transposition(self):
        self.assertEqual(distance("ab", "ba"), 1)
        self.assertEqual(distance("teh", "the"), 1)
        self.assertEqual(distance("abcd", "acbd"), 1)
        self.assertEqual(distance("abcdef", "badcfe"), 3)

    def test_restricted_not_full_damerau(self):
        self.assertEqual(distance("ca", "abc"), 3)
        self.assertEqual(distance("abc", "ca"), 3)

    def test_symmetry_and_triangle_samples(self):
        words = ["", "a", "ab", "abc", "bca", "cab", "xyz", "abcxyz", "zyxcba"]
        for a in words:
            for b in words:
                self.assertEqual(distance(a, b), distance(b, a))
                self.assertLessEqual(distance(a, b), max(len(a), len(b)))


class Closest(unittest.TestCase):
    def test_order(self):
        cands = ["cart", "cat", "car", "cast", "dog", "cats", "card", "care"]
        # distances from "cat": cat 0, cats 1, car 1, cast 1, cart 1, card 2, care 2, dog 3
        self.assertEqual(closest("cat", cands, 1), ["cat", "cats", "car", "cart", "cast"])
        self.assertEqual(closest("cat", cands, 2), ["cat", "cats", "car", "cart", "cast", "card", "care"])
        self.assertEqual(closest("cat", cands, 0), ["cat"])

    def test_prefix_beats_alphabet_within_the_same_distance(self):
        self.assertEqual(closest("grape", ["frape", "gripe", "grapes"], 1), ["grapes", "gripe", "frape"])

    def test_duplicates_once(self):
        self.assertEqual(closest("a", ["a", "a", "b"], 1), ["a", "b"])

    def test_default_and_validation(self):
        self.assertEqual(closest("abc", ["abcd", "abcdef", "xyz"]), ["abcd"])
        for bad in (-1, 1.5, None, True):
            with self.assertRaises(ValueError):
                closest("a", ["a"], bad)
        self.assertEqual(closest("a", []), [])


class Best(unittest.TestCase):
    def test_threshold_depends_on_length(self):
        self.assertEqual(best_match("helo", ["hello", "world"]), "hello")
        self.assertIsNone(best_match("xyz", ["abc", "def"]))
        self.assertEqual(best_match("recieve", ["receive", "relieve"]), "receive")
        self.assertIsNone(best_match("ab", ["xyz"]))
        self.assertEqual(best_match("abcdefgh", ["abcdexyz"]), "abcdexyz")
        self.assertEqual(best_match("abcd", ["abxy"]), "abxy")           # 4 // 3 + 1 = 2 edits are allowed
        self.assertIsNone(best_match("abcd", ["axyz"]))                  # 3 edits are too many
        self.assertIsNone(best_match("anything", []))
'''},
    "bugfix": {
        "symptom": "A fuzzy-matching library (fuzzy.py) counts a swapped pair of letters as two edits: distance(\"teh\", \"the\") should be 1 (adjacent transposition is a single edit) but returns 2.",
        "bug": [("fuzzy.py", "            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:\n                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)\n", "")],
        "repro": {"test_repro.py": 'import unittest\nfrom fuzzy import distance\n\n\nclass Repro(unittest.TestCase):\n    def test_transposition_is_one_edit(self):\n        self.assertEqual(distance("teh", "the"), 1)\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
