FAMILY = {
    "name": "f06_intervals", "lang": "python",
    "implement": {
        "prompt": """Write the module `intervals.py` (standard library only) for HALF-OPEN integer intervals [start, end): start is included, end is not.
An interval is a tuple (start, end) with start < end; an empty or reversed interval (start >= end) is invalid and raises ValueError.

merge(intervals) -> list : merges overlapping or TOUCHING intervals ((1,3) and (3,5) become (1,5)); the result is sorted by start
  and does not modify the input; an empty input gives [].
subtract(intervals, holes) -> list : the parts of the (merged) `intervals` not covered by any of the `holes`; sorted, merged result.
  subtract([(0,10)], [(2,4),(6,8)]) -> [(0,2),(4,6),(8,10)].
intersect(a, b) -> list : the parts covered by both lists (each list is merged first); sorted, non-empty pieces only.
total_length(intervals) -> int : the total length covered (overlaps counted once).
free_slots(busy, window, min_length=1) -> list : within the interval `window`, the gaps NOT covered by `busy`, keeping only gaps with
  length >= min_length; sorted. busy intervals outside the window are ignored; those crossing its edge are clipped.
""",
        "start": {"intervals.py": '"""Half-open interval arithmetic (to be written)."""\n\n\ndef merge(intervals):\n    raise NotImplementedError\n\n\ndef subtract(intervals, holes):\n    raise NotImplementedError\n\n\ndef intersect(a, b):\n    raise NotImplementedError\n\n\ndef total_length(intervals):\n    raise NotImplementedError\n\n\ndef free_slots(busy, window, min_length=1):\n    raise NotImplementedError\n'},
        "solution": {"intervals.py": '''"""Half-open interval arithmetic."""


def _check(iv):
    s, e = iv
    if s >= e:
        raise ValueError(f"invalid interval {iv!r}")


def merge(intervals):
    items = [tuple(i) for i in intervals]
    for i in items:
        _check(i)
    out = []
    for s, e in sorted(items):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def subtract(intervals, holes):
    result = []
    hs = merge(holes)
    for s, e in merge(intervals):
        cur = s
        for hs_, he in hs:
            if he <= cur or hs_ >= e:
                continue
            if hs_ > cur:
                result.append((cur, hs_))
            cur = max(cur, he)
            if cur >= e:
                break
        if cur < e:
            result.append((cur, e))
    return result


def intersect(a, b):
    a, b = merge(a), merge(b)
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def total_length(intervals):
    return sum(e - s for s, e in merge(intervals))


def free_slots(busy, window, min_length=1):
    _check(tuple(window))
    clipped = intersect(busy, [tuple(window)]) if merge(busy) else []
    return [g for g in subtract([tuple(window)], clipped) if g[1] - g[0] >= min_length]
'''},
    },
    "hidden": {"test_intervals.py": '''import unittest
from intervals import free_slots, intersect, merge, subtract, total_length


class Merge(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(merge([]), [])
        self.assertEqual(merge([(5, 8), (1, 3), (2, 4)]), [(1, 4), (5, 8)])
        self.assertEqual(merge([(1, 3), (3, 5)]), [(1, 5)])
        self.assertEqual(merge([(1, 10), (2, 3), (4, 5)]), [(1, 10)])

    def test_input_unchanged_and_lists_accepted(self):
        data = [(5, 8), (1, 3)]
        merge(data)
        self.assertEqual(data, [(5, 8), (1, 3)])
        self.assertEqual(merge([[1, 2], [2, 3]]), [(1, 3)])

    def test_invalid(self):
        for bad in [[(3, 3)], [(4, 2)], [(1, 2), (5, 5)]]:
            with self.assertRaises(ValueError):
                merge(bad)


class Subtract(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(subtract([(0, 10)], [(2, 4), (6, 8)]), [(0, 2), (4, 6), (8, 10)])
        self.assertEqual(subtract([(0, 10)], []), [(0, 10)])
        self.assertEqual(subtract([(0, 10)], [(0, 10)]), [])
        self.assertEqual(subtract([(0, 10)], [(-5, 3)]), [(3, 10)])
        self.assertEqual(subtract([(0, 10)], [(8, 20)]), [(0, 8)])
        self.assertEqual(subtract([(0, 10)], [(20, 30)]), [(0, 10)])

    def test_merges_inputs_first(self):
        self.assertEqual(subtract([(0, 5), (4, 10)], [(2, 3), (3, 4)]), [(0, 2), (4, 10)])

    def test_touching_hole_removes_nothing(self):
        self.assertEqual(subtract([(0, 5)], [(5, 9)]), [(0, 5)])

    def test_several_intervals(self):
        self.assertEqual(subtract([(0, 5), (10, 15)], [(3, 12)]), [(0, 3), (12, 15)])


class Intersect(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(intersect([(0, 5)], [(3, 9)]), [(3, 5)])
        self.assertEqual(intersect([(0, 5)], [(5, 9)]), [])
        self.assertEqual(intersect([(0, 10), (20, 30)], [(5, 25)]), [(5, 10), (20, 25)])
        self.assertEqual(intersect([(0, 3), (2, 6)], [(1, 4), (5, 9)]), [(1, 4), (5, 6)])
        self.assertEqual(intersect([], [(1, 2)]), [])


class Length(unittest.TestCase):
    def test_overlaps_counted_once(self):
        self.assertEqual(total_length([(0, 5), (3, 8), (20, 21)]), 9)
        self.assertEqual(total_length([]), 0)


class Free(unittest.TestCase):
    def test_gaps(self):
        self.assertEqual(free_slots([(2, 4), (6, 7)], (0, 10)), [(0, 2), (4, 6), (7, 10)])
        self.assertEqual(free_slots([(2, 4), (6, 7)], (0, 10), min_length=3), [(7, 10)])
        self.assertEqual(free_slots([], (0, 10)), [(0, 10)])
        self.assertEqual(free_slots([(0, 10)], (0, 10)), [])

    def test_clipping_and_outside(self):
        self.assertEqual(free_slots([(-5, 2), (8, 50), (100, 200)], (0, 10)), [(2, 8)])
        self.assertEqual(free_slots([(100, 200)], (0, 10)), [(0, 10)])

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            free_slots([], (5, 5))
'''},
    "bugfix": {
        "symptom": "An interval library (intervals.py) mishandles touching intervals: merge([(1, 3), (3, 5)]) should be [(1, 5)] (half-open intervals that touch are joined) but returns two intervals.",
        "bug": [("intervals.py", "        if out and s <= out[-1][1]:", "        if out and s < out[-1][1]:")],
        "repro": {"test_repro.py": 'import unittest\nfrom intervals import merge\n\n\nclass Repro(unittest.TestCase):\n    def test_touching(self):\n        self.assertEqual(merge([(1, 3), (3, 5)]), [(1, 5)])\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
