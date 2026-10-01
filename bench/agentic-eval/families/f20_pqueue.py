FAMILY = {
    "name": "f20_pqueue", "lang": "python",
    "implement": {
        "prompt": """Write the module `pq.py` (standard library only) with the class `IndexedPQ`, a min-priority queue whose items can be updated and removed by key.

IndexedPQ()
  push(key, priority)     add a new key (any hashable) with a comparable priority; pushing a key that is already present raises KeyError.
  update(key, priority)   change the priority of an existing key (it may go up or down); a missing key raises KeyError.
  push_or_update(key, priority)  push if absent, otherwise update.
  remove(key)             delete a key; returns its priority; a missing key raises KeyError.
  pop()                   remove and return (key, priority) with the smallest priority; among EQUAL priorities the key that was
                          pushed EARLIEST wins (an update keeps the key's original arrival order). Empty queue: IndexError.
  peek()                  the same pair without removing it; IndexError when empty.
  priority(key)           the priority of the key (KeyError if absent).
  __len__, __contains__(key), __bool__
  items_sorted()          a list of (key, priority) in pop order, without changing the queue.
Every operation except items_sorted must take O(log n) time (use a heap; do not scan the whole queue).
""",
        "start": {"pq.py": '"""An indexed priority queue (to be written)."""\n\n\nclass IndexedPQ:\n    def __init__(self):\n        raise NotImplementedError\n'},
        "solution": {"pq.py": '''"""An indexed priority queue."""
import heapq
import itertools


class IndexedPQ:
    def __init__(self):
        self._heap = []            # [priority, order, key, alive]
        self._entries = {}         # key -> entry
        self._order = {}           # key -> arrival number (kept across updates)
        self._counter = itertools.count()

    def push(self, key, priority):
        if key in self._entries:
            raise KeyError(key)
        self._order[key] = next(self._counter)
        self._insert(key, priority)

    def _insert(self, key, priority):
        entry = [priority, self._order[key], key, True]
        self._entries[key] = entry
        heapq.heappush(self._heap, entry)

    def update(self, key, priority):
        if key not in self._entries:
            raise KeyError(key)
        self._entries[key][3] = False
        self._insert(key, priority)

    def push_or_update(self, key, priority):
        if key in self._entries:
            self.update(key, priority)
        else:
            self.push(key, priority)

    def remove(self, key):
        entry = self._entries.pop(key)
        entry[3] = False
        del self._order[key]
        return entry[0]

    def _clean(self):
        while self._heap and not self._heap[0][3]:
            heapq.heappop(self._heap)

    def pop(self):
        self._clean()
        if not self._heap:
            raise IndexError("pop from an empty queue")
        priority, _, key, _ = heapq.heappop(self._heap)
        del self._entries[key]
        del self._order[key]
        return key, priority

    def peek(self):
        self._clean()
        if not self._heap:
            raise IndexError("peek into an empty queue")
        e = self._heap[0]
        return e[2], e[0]

    def priority(self, key):
        return self._entries[key][0]

    def __len__(self):
        return len(self._entries)

    def __contains__(self, key):
        return key in self._entries

    def __bool__(self):
        return bool(self._entries)

    def items_sorted(self):
        live = sorted((e for e in self._entries.values()), key=lambda e: (e[0], e[1]))
        return [(e[2], e[0]) for e in live]
'''},
    },
    "hidden": {"test_pq.py": '''import random
import time
import unittest
from pq import IndexedPQ


class Basics(unittest.TestCase):
    def test_order_and_ties(self):
        q = IndexedPQ()
        for k, p in [("c", 3), ("a", 1), ("b", 1), ("d", 2)]:
            q.push(k, p)
        self.assertEqual(q.peek(), ("a", 1))
        self.assertEqual([q.pop() for _ in range(4)], [("a", 1), ("b", 1), ("d", 2), ("c", 3)])
        self.assertFalse(q)

    def test_empty_errors(self):
        q = IndexedPQ()
        with self.assertRaises(IndexError):
            q.pop()
        with self.assertRaises(IndexError):
            q.peek()

    def test_duplicate_and_missing_keys(self):
        q = IndexedPQ()
        q.push("a", 1)
        with self.assertRaises(KeyError):
            q.push("a", 2)
        for fn in (lambda: q.update("zz", 1), lambda: q.remove("zz"), lambda: q.priority("zz")):
            with self.assertRaises(KeyError):
                fn()

    def test_update_up_and_down_keeps_arrival_order_for_ties(self):
        q = IndexedPQ()
        for k in "abc":
            q.push(k, 5)
        q.update("c", 1)
        self.assertEqual(q.pop(), ("c", 1))
        q.update("a", 7)
        q.update("a", 5)                                            # back to 5: a arrived before b
        self.assertEqual([q.pop()[0] for _ in range(2)], ["a", "b"])

    def test_remove_and_contains(self):
        q = IndexedPQ()
        q.push("a", 1)
        q.push("b", 2)
        self.assertEqual(q.remove("a"), 1)
        self.assertNotIn("a", q)
        self.assertEqual(len(q), 1)
        self.assertEqual(q.pop(), ("b", 2))
        q.push("a", 9)                                              # a removed key can be pushed again, as a newcomer
        q.push("z", 9)
        self.assertEqual(q.pop(), ("a", 9))

    def test_push_or_update_and_priority(self):
        q = IndexedPQ()
        q.push_or_update("a", 3)
        q.push_or_update("a", 1)
        self.assertEqual(q.priority("a"), 1)
        self.assertEqual(len(q), 1)

    def test_items_sorted_does_not_change_the_queue(self):
        q = IndexedPQ()
        for k, p in [("x", 2), ("y", 1), ("z", 2)]:
            q.push(k, p)
        self.assertEqual(q.items_sorted(), [("y", 1), ("x", 2), ("z", 2)])
        self.assertEqual(len(q), 3)
        self.assertEqual(q.pop(), ("y", 1))

    def test_tuple_priorities_and_none_key(self):
        q = IndexedPQ()
        q.push(None, (1, "b"))
        q.push(0, (1, "a"))
        self.assertEqual(q.pop(), (0, (1, "a")))


class Model(unittest.TestCase):
    def test_against_a_simple_model(self):
        rng = random.Random(3)
        q, model, order, n = IndexedPQ(), {}, {}, 0
        for _ in range(3000):
            op = rng.choice(["push", "update", "remove", "pop", "pop"])
            k = rng.randint(0, 40)
            if op == "push" and k not in model:
                p = rng.randint(0, 9)
                q.push(k, p)
                model[k], order[k] = p, n
                n += 1
            elif op == "update" and k in model:
                p = rng.randint(0, 9)
                q.update(k, p)
                model[k] = p
            elif op == "remove" and k in model:
                self.assertEqual(q.remove(k), model.pop(k))
                order.pop(k)
            elif op == "pop" and model:
                best = min(model, key=lambda x: (model[x], order[x]))
                self.assertEqual(q.pop(), (best, model.pop(best)))
                order.pop(best)
            self.assertEqual(len(q), len(model))

    def test_is_fast_enough(self):
        q = IndexedPQ()
        t0 = time.monotonic()
        for i in range(60000):
            q.push(i, (i * 7919) % 1000)
        for i in range(0, 60000, 2):
            q.update(i, (i * 31) % 1000)
        for i in range(1, 60000, 4):
            q.remove(i)
        while q:
            q.pop()
        self.assertLess(time.monotonic() - t0, 8.0)
'''},
    "bugfix": {
        "symptom": "A priority queue (pq.py) breaks the tie rule after an update: equal priorities must pop in order of the key's first arrival, but a key whose priority was changed and then changed back jumps behind keys that arrived later.",
        "bug": [("pq.py", "        entry = [priority, self._order[key], key, True]", "        entry = [priority, next(self._counter), key, True]")],
        "repro": {"test_repro.py": 'import unittest\nfrom pq import IndexedPQ\n\n\nclass Repro(unittest.TestCase):\n    def test_update_keeps_arrival_order(self):\n        q = IndexedPQ()\n        for k in "abc":\n            q.push(k, 5)\n        q.update("a", 7)\n        q.update("a", 5)\n        self.assertEqual([q.pop()[0] for _ in range(3)], ["a", "b", "c"])\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
