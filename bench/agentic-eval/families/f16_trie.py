FAMILY = {
    "name": "f16_trie", "lang": "python",
    "implement": {
        "prompt": """Write the module `autocomplete.py` (standard library only) with the class `Autocomplete`.

Autocomplete(case_sensitive=False)
  add(word, weight=1) -> None   insert the word (a non-empty str without leading/trailing whitespace, otherwise ValueError). The weight is
                                a number >= 0. Adding a word that already exists ADDS the new weight to its current weight (it counts uses).
  remove(word) -> bool          delete the word; True if it existed. Words that merely share its prefix stay.
  __contains__(word), __len__   membership and the number of distinct words.
  weight(word) -> number        the weight of the word (0 if unknown).
  suggest(prefix, limit=5) -> list of str   words starting with `prefix` (the empty prefix matches every word), best first: by weight
                                descending, ties alphabetical (ordinary string order of the stored form). Returns at most `limit`
                                (limit >= 0, otherwise ValueError). Words are returned in the form they were first added.
  count_prefix(prefix) -> int   how many words start with the prefix.
  longest_common_prefix(prefix) -> str or None   among the words that start with the prefix, the longest string that all of them
                                start with (the empty prefix is allowed and may give ""); None if no word starts with the prefix.
Case-insensitive mode compares by lower-cased text (suggest("AP") finds "apple"); in case-sensitive mode "Apple" and "apple" differ.
""",
        "start": {"autocomplete.py": '"""Prefix autocomplete (to be written)."""\n\n\nclass Autocomplete:\n    def __init__(self, case_sensitive=False):\n        raise NotImplementedError\n'},
        "solution": {"autocomplete.py": '''"""Prefix autocomplete."""


class Autocomplete:
    def __init__(self, case_sensitive=False):
        self.cs = case_sensitive
        self._root = {}              # char -> node; node["$"] = key
        self._words = {}             # key -> [display, weight]

    def _key(self, word):
        return word if self.cs else word.lower()

    def add(self, word, weight=1):
        if not isinstance(word, str) or not word or word != word.strip():
            raise ValueError("word must be a non-empty string without surrounding whitespace")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError("weight must be a number >= 0")
        key = self._key(word)
        if key in self._words:
            self._words[key][1] += weight
            return
        self._words[key] = [word, weight]
        node = self._root
        for ch in key:
            node = node.setdefault(ch, {})
        node["$"] = key

    def remove(self, word):
        key = self._key(word)
        if key not in self._words:
            return False
        del self._words[key]
        path, node = [], self._root
        for ch in key:
            path.append((node, ch))
            node = node[ch]
        node.pop("$", None)
        for parent, ch in reversed(path):
            if parent[ch]:
                break
            del parent[ch]
        return True

    def __contains__(self, word):
        return isinstance(word, str) and self._key(word) in self._words

    def __len__(self):
        return len(self._words)

    def weight(self, word):
        entry = self._words.get(self._key(word))
        return entry[1] if entry else 0

    def _under(self, prefix):
        node = self._root
        for ch in self._key(prefix):
            if ch not in node:
                return []
            node = node[ch]
        out, stack = [], [node]
        while stack:
            n = stack.pop()
            if "$" in n:
                out.append(n["$"])
            stack.extend(v for k, v in n.items() if k != "$")
        return out

    def suggest(self, prefix, limit=5):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be an int >= 0")
        keys = self._under(prefix)
        entries = [self._words[k] for k in keys]
        entries.sort(key=lambda e: (-e[1], e[0]))
        return [e[0] for e in entries[:limit]]

    def count_prefix(self, prefix):
        return len(self._under(prefix))

    def longest_common_prefix(self, prefix):
        keys = self._under(prefix)
        if not keys:
            return None
        first, last = min(keys), max(keys)
        n = 0
        while n < len(first) and n < len(last) and first[n] == last[n]:
            n += 1
        return first[:n]
'''},
    },
    "hidden": {"test_autocomplete.py": '''import unittest
from autocomplete import Autocomplete


class Basics(unittest.TestCase):
    def setUp(self):
        self.a = Autocomplete()
        for w, wt in [("apple", 5), ("application", 3), ("apply", 3), ("app", 10), ("banana", 2), ("band", 2)]:
            self.a.add(w, wt)

    def test_suggest_order(self):
        self.assertEqual(self.a.suggest("app"), ["app", "apple", "application", "apply"])
        self.assertEqual(self.a.suggest("app", limit=2), ["app", "apple"])
        self.assertEqual(self.a.suggest("ban"), ["banana", "band"])
        self.assertEqual(self.a.suggest("zzz"), [])
        self.assertEqual(self.a.suggest("app", limit=0), [])

    def test_empty_prefix_matches_everything(self):
        self.assertEqual(self.a.suggest("", limit=10), ["app", "apple", "application", "apply", "banana", "band"])
        self.assertEqual(self.a.count_prefix(""), 6)

    def test_weights_accumulate(self):
        self.a.add("band", 5)
        self.assertEqual(self.a.weight("band"), 7)
        self.assertEqual(self.a.suggest("ban")[0], "band")
        self.assertEqual(len(self.a), 6)
        self.assertEqual(self.a.weight("unknown"), 0)

    def test_case_insensitive(self):
        self.a.add("Apple", 100)                      # same word as "apple": weights add, first spelling is kept
        self.assertEqual(self.a.weight("APPLE"), 105)
        self.assertEqual(self.a.suggest("AP")[0], "apple")
        self.assertIn("BANANA", self.a)
        self.assertEqual(len(self.a), 6)

    def test_case_sensitive(self):
        a = Autocomplete(case_sensitive=True)
        a.add("Apple")
        a.add("apple")
        self.assertEqual(len(a), 2)
        self.assertEqual(a.suggest("A"), ["Apple"])
        self.assertEqual(a.suggest("a"), ["apple"])

    def test_remove_keeps_neighbours(self):
        self.assertTrue(self.a.remove("app"))
        self.assertFalse(self.a.remove("app"))
        self.assertNotIn("app", self.a)
        self.assertEqual(self.a.suggest("app"), ["apple", "application", "apply"])
        self.assertTrue(self.a.remove("application"))
        self.assertEqual(self.a.count_prefix("app"), 2)
        self.assertEqual(len(self.a), 4)

    def test_count_and_common_prefix(self):
        self.assertEqual(self.a.count_prefix("appl"), 3)
        self.assertEqual(self.a.count_prefix("ban"), 2)
        self.assertEqual(self.a.longest_common_prefix("a"), "app")
        self.assertEqual(self.a.longest_common_prefix("appli"), "application")
        self.assertEqual(self.a.longest_common_prefix("ban"), "ban")
        self.assertEqual(self.a.longest_common_prefix(""), "")
        self.assertIsNone(self.a.longest_common_prefix("x"))

    def test_validation(self):
        for bad in ["", " a", "a ", None, 5]:
            with self.assertRaises(ValueError):
                self.a.add(bad)
        for w in (-1, "1", True, None):
            with self.assertRaises(ValueError):
                self.a.add("ok", w)
        with self.assertRaises(ValueError):
            self.a.suggest("a", limit=-1)

    def test_ties_alphabetical(self):
        a = Autocomplete()
        for w in ["b", "c", "a"]:
            a.add(w, 1)
        self.assertEqual(a.suggest(""), ["a", "b", "c"])
'''},
    "bugfix": {
        "symptom": "An autocomplete library (autocomplete.py) returns suggestions in the wrong order for equal weights: words with the same weight must come out alphabetically (\"a\", \"b\", \"c\") but come out in insertion order.",
        "bug": [("autocomplete.py", "        entries.sort(key=lambda e: (-e[1], e[0]))", "        entries.sort(key=lambda e: -e[1])")],
        "repro": {"test_repro.py": 'import unittest\nfrom autocomplete import Autocomplete\n\n\nclass Repro(unittest.TestCase):\n    def test_ties_are_alphabetical(self):\n        a = Autocomplete()\n        for w in ["b", "c", "a"]:\n            a.add(w, 1)\n        self.assertEqual(a.suggest(""), ["a", "b", "c"])\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
