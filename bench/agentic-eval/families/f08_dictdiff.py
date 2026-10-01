FAMILY = {
    "name": "f08_dictdiff", "lang": "python",
    "implement": {
        "prompt": """Write the module `jsondiff.py` (standard library only) with two functions working on JSON-like data (dicts, lists, str, int, float, bool, None).

diff(a, b) -> list of operations that turn `a` into `b`. An operation is a dict {"op": "add"|"remove"|"replace", "path": str, "value": ...}
  ("remove" has no "value"). Paths are JSON Pointers: "" is the root, "/a/b/0" goes through keys and list indexes; in a key, "~" is
  written "~0" and "/" is written "~1". Rules:
  * equal values give []. Different types (or a different scalar) at the same place give one "replace" at that path
    (bool and int are DIFFERENT types: True vs 1 is a replace; 1 vs 1.0 is also a replace).
  * dicts: keys only in `a` are removed, keys only in `b` are added, common keys are compared recursively. Emit the operations in
    this exact order: first all "remove" operations (keys sorted), then the operations for the common keys (keys sorted, recursing
    into each), then all "add" operations (keys sorted).
  * lists: compare index by index for the common length (recurse); if `a` is longer, "remove" the extra items from the HIGHEST index
    down to the lowest (so that applying them in order stays valid); if `b` is longer, "add" the extra items at their indexes in
    ascending order. Operations for the common part come first, then the removes or adds.
apply_patch(a, ops) -> new value: applies the operations in order to a deep COPY of `a` (the input is not modified) and returns it.
  "add" on a dict key sets it; "add" on a list inserts at that index (an index equal to the length appends); "remove" deletes;
  "replace" sets (a root replace returns the value). A path that does not exist (missing key/index, wrong container) raises KeyError.
For all JSON-like a, b: apply_patch(a, diff(a, b)) == b.
""",
        "start": {"jsondiff.py": '"""Diff and patch for JSON-like data (to be written)."""\n\n\ndef diff(a, b):\n    raise NotImplementedError\n\n\ndef apply_patch(a, ops):\n    raise NotImplementedError\n'},
        "solution": {"jsondiff.py": '''"""Diff and patch for JSON-like data."""
import copy


def _esc(key):
    return str(key).replace("~", "~0").replace("/", "~1")


def _unesc(part):
    return part.replace("~1", "/").replace("~0", "~")


def _same_type(x, y):
    return type(x) is type(y)


def diff(a, b, _path=""):
    if _same_type(a, b) and a == b and not isinstance(a, (dict, list)):
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        ops = []
        for k in sorted(set(a) - set(b)):
            ops.append({"op": "remove", "path": f"{_path}/{_esc(k)}"})
        for k in sorted(set(a) & set(b)):
            ops += diff(a[k], b[k], f"{_path}/{_esc(k)}")
        for k in sorted(set(b) - set(a)):
            ops.append({"op": "add", "path": f"{_path}/{_esc(k)}", "value": copy.deepcopy(b[k])})
        return ops
    if isinstance(a, list) and isinstance(b, list):
        ops = []
        common = min(len(a), len(b))
        for i in range(common):
            ops += diff(a[i], b[i], f"{_path}/{i}")
        for i in range(len(a) - 1, common - 1, -1):
            ops.append({"op": "remove", "path": f"{_path}/{i}"})
        for i in range(common, len(b)):
            ops.append({"op": "add", "path": f"{_path}/{i}", "value": copy.deepcopy(b[i])})
        return ops
    if _same_type(a, b) and a == b:
        return []
    return [{"op": "replace", "path": _path, "value": copy.deepcopy(b)}]


def _walk(doc, parts):
    cur = doc
    for p in parts:
        if isinstance(cur, dict):
            if p not in cur:
                raise KeyError(p)
            cur = cur[p]
        elif isinstance(cur, list):
            if not p.isdigit() or int(p) >= len(cur):
                raise KeyError(p)
            cur = cur[int(p)]
        else:
            raise KeyError(p)
    return cur


def apply_patch(a, ops):
    doc = copy.deepcopy(a)
    for op in ops:
        path = op["path"]
        if path == "":
            if op["op"] != "replace":
                raise KeyError("root")
            doc = copy.deepcopy(op["value"])
            continue
        parts = [_unesc(p) for p in path.split("/")[1:]]
        parent, last = _walk(doc, parts[:-1]), parts[-1]
        if isinstance(parent, dict):
            if op["op"] in ("remove", "replace") and last not in parent:
                raise KeyError(last)
            if op["op"] == "remove":
                del parent[last]
            else:
                parent[last] = copy.deepcopy(op["value"])
        elif isinstance(parent, list):
            if not last.isdigit():
                raise KeyError(last)
            i = int(last)
            if op["op"] == "add":
                if i > len(parent):
                    raise KeyError(last)
                parent.insert(i, copy.deepcopy(op["value"]))
            else:
                if i >= len(parent):
                    raise KeyError(last)
                if op["op"] == "remove":
                    del parent[i]
                else:
                    parent[i] = copy.deepcopy(op["value"])
        else:
            raise KeyError(last)
    return doc
'''},
    },
    "hidden": {"test_jsondiff.py": '''import copy
import random
import unittest
from jsondiff import apply_patch, diff


class Diff(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(diff({"a": [1, {"b": None}]}, {"a": [1, {"b": None}]}), [])
        self.assertEqual(diff(1, 1), [])

    def test_scalars_and_types(self):
        self.assertEqual(diff(1, 2), [{"op": "replace", "path": "", "value": 2}])
        self.assertEqual(diff(True, 1), [{"op": "replace", "path": "", "value": 1}])
        self.assertEqual(diff(1, 1.0), [{"op": "replace", "path": "", "value": 1.0}])
        self.assertEqual(diff({"a": 1}, [1]), [{"op": "replace", "path": "", "value": [1]}])
        self.assertEqual(diff(None, 0), [{"op": "replace", "path": "", "value": 0}])

    def test_dict_order_is_removes_then_common_then_adds(self):
        a = {"z": 1, "k": 1, "m": 1}
        b = {"k": 2, "m": 1, "n": 5, "b": 6}
        self.assertEqual(diff(a, b), [
            {"op": "remove", "path": "/z"},
            {"op": "replace", "path": "/k", "value": 2},
            {"op": "add", "path": "/b", "value": 6},
            {"op": "add", "path": "/n", "value": 5},
        ])

    def test_lists(self):
        self.assertEqual(diff([1, 2, 3, 4], [1, 9]), [
            {"op": "replace", "path": "/1", "value": 9},
            {"op": "remove", "path": "/3"},
            {"op": "remove", "path": "/2"},
        ])
        self.assertEqual(diff([1], [1, 2, 3]), [
            {"op": "add", "path": "/1", "value": 2},
            {"op": "add", "path": "/2", "value": 3},
        ])

    def test_nested_and_escaping(self):
        a = {"a/b": {"c~d": [{"x": 1}]}}
        b = {"a/b": {"c~d": [{"x": 2}]}}
        self.assertEqual(diff(a, b), [{"op": "replace", "path": "/a~1b/c~0d/0/x", "value": 2}])

    def test_values_are_copies(self):
        b = {"n": {"deep": [1]}}
        ops = diff({}, b)
        ops[0]["value"]["deep"].append(2)
        self.assertEqual(b, {"n": {"deep": [1]}})


class Patch(unittest.TestCase):
    def test_apply_each_op(self):
        self.assertEqual(apply_patch({"a": 1}, [{"op": "add", "path": "/b", "value": 2}]), {"a": 1, "b": 2})
        self.assertEqual(apply_patch({"a": 1}, [{"op": "remove", "path": "/a"}]), {})
        self.assertEqual(apply_patch({"a": 1}, [{"op": "replace", "path": "/a", "value": 5}]), {"a": 5})
        self.assertEqual(apply_patch([1, 2], [{"op": "add", "path": "/1", "value": 9}]), [1, 9, 2])
        self.assertEqual(apply_patch([1, 2], [{"op": "add", "path": "/2", "value": 9}]), [1, 2, 9])
        self.assertEqual(apply_patch([1, 2, 3], [{"op": "remove", "path": "/0"}]), [2, 3])
        self.assertEqual(apply_patch(1, [{"op": "replace", "path": "", "value": [3]}]), [3])

    def test_input_not_modified(self):
        a = {"a": [1, 2]}
        apply_patch(a, [{"op": "add", "path": "/a/0", "value": 0}])
        self.assertEqual(a, {"a": [1, 2]})

    def test_missing_paths(self):
        for ops, doc in [([{"op": "remove", "path": "/x"}], {}), ([{"op": "replace", "path": "/x"}], {}),
                         ([{"op": "add", "path": "/a/b", "value": 1}], {}), ([{"op": "add", "path": "/5", "value": 1}], [1]),
                         ([{"op": "remove", "path": "/1"}], [1]), ([{"op": "add", "path": "/a/b", "value": 1}], {"a": 5})]:
            with self.subTest(ops=ops):
                with self.assertRaises(KeyError):
                    apply_patch(doc, ops)

    def test_round_trip_random(self):
        rng = random.Random(7)

        def gen(depth=0):
            kind = rng.choice(["i", "s", "n", "b", "l", "d"] if depth < 3 else ["i", "s", "n", "b"])
            if kind == "i":
                return rng.randint(0, 3)
            if kind == "s":
                return rng.choice(["x", "y", "a/b", "~"])
            if kind == "n":
                return None
            if kind == "b":
                return rng.choice([True, False])
            if kind == "l":
                return [gen(depth + 1) for _ in range(rng.randint(0, 4))]
            return {rng.choice(["a", "b", "c/d", "e~f", "g"]): gen(depth + 1) for _ in range(rng.randint(0, 4))}
        for _ in range(300):
            a, b = gen(), gen()
            snapshot = copy.deepcopy(a)
            self.assertEqual(apply_patch(a, diff(a, b)), b)
            self.assertEqual(a, snapshot)
'''},
    "bugfix": {
        "symptom": "A JSON diff library (jsondiff.py) produces patches that fail to apply: diff([1, 2, 3, 4], [1, 9]) should remove the extra list items from the highest index down (so applying the patch works), but the removals come in the wrong order and apply_patch raises KeyError.",
        "bug": [("jsondiff.py", "        for i in range(len(a) - 1, common - 1, -1):", "        for i in range(common, len(a)):")],
        "repro": {"test_repro.py": 'import unittest\nfrom jsondiff import apply_patch, diff\n\n\nclass Repro(unittest.TestCase):\n    def test_shrinking_list(self):\n        a, b = [1, 2, 3, 4], [1, 9]\n        self.assertEqual(apply_patch(a, diff(a, b)), b)\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
