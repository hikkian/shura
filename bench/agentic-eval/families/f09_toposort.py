FAMILY = {
    "name": "f09_toposort", "lang": "python",
    "implement": {
        "prompt": """Write the module `deps.py` (standard library only) for resolving build dependencies.

class CycleError(ValueError): raised for circular dependencies; its attribute `cycle` is a list of node names that forms the cycle,
  starting and ending with the same node, e.g. ["a", "b", "c", "a"] meaning a depends on b, b on c, c on a.

build_order(graph) -> list : `graph` maps a node name to the list of nodes it DEPENDS ON. Returns all nodes so that every node comes
  after its dependencies. Nodes that appear only as a dependency (not as a key) are part of the graph without dependencies. Among
  nodes that are ready at the same time, the alphabetically smallest comes first (so the result is deterministic). Duplicate
  entries in a dependency list are ignored. A node depending on itself is a cycle ["x", "x"].
  For a cyclic graph raise CycleError; report the cycle that is found first when nodes are visited in alphabetical order and each
  node's dependencies are followed in alphabetical order (depth-first), rotated to start at the node where the cycle was detected.
parallel_groups(graph) -> list of lists : the same ordering but grouped in waves: wave 0 holds all nodes with no dependencies, wave 1
  those whose dependencies are all in wave 0, and so on; every wave is sorted alphabetically. Raises CycleError like build_order.
affected(graph, changed) -> list : all nodes that depend on any node of `changed` directly or indirectly (not including the changed
  nodes themselves unless another changed node reaches them), sorted alphabetically. Unknown names in `changed` are ignored.
""",
        "start": {"deps.py": '"""Dependency ordering (to be written)."""\n\n\nclass CycleError(ValueError):\n    pass\n\n\ndef build_order(graph):\n    raise NotImplementedError\n\n\ndef parallel_groups(graph):\n    raise NotImplementedError\n\n\ndef affected(graph, changed):\n    raise NotImplementedError\n'},
        "solution": {"deps.py": '''"""Dependency ordering."""
import heapq


class CycleError(ValueError):
    def __init__(self, cycle):
        super().__init__("dependency cycle: " + " -> ".join(cycle))
        self.cycle = cycle


def _normalise(graph):
    deps = {}
    for node, ds in graph.items():
        deps.setdefault(node, set()).update(ds)
        for d in ds:
            deps.setdefault(d, set())
    return deps


def _find_cycle(deps):
    state, stack = {}, []

    def visit(n):
        state[n] = 1
        stack.append(n)
        for d in sorted(deps[n]):
            if state.get(d) == 1:
                i = stack.index(d)
                return stack[i:] + [d]
            if d not in state:
                found = visit(d)
                if found:
                    return found
        stack.pop()
        state[n] = 2
        return None
    for n in sorted(deps):
        if n not in state:
            found = visit(n)
            if found:
                return found
    return None


def build_order(graph):
    deps = _normalise(graph)
    cycle = _find_cycle(deps)
    if cycle:
        raise CycleError(cycle)
    remaining = {n: set(d) for n, d in deps.items()}
    dependents = {n: [] for n in deps}
    for n, ds in deps.items():
        for d in ds:
            dependents[d].append(n)
    ready = [n for n, ds in remaining.items() if not ds]
    heapq.heapify(ready)
    order = []
    while ready:
        n = heapq.heappop(ready)
        order.append(n)
        for m in dependents[n]:
            remaining[m].discard(n)
            if not remaining[m]:
                heapq.heappush(ready, m)
    return order


def parallel_groups(graph):
    deps = _normalise(graph)
    cycle = _find_cycle(deps)
    if cycle:
        raise CycleError(cycle)
    done, groups = set(), []
    pending = set(deps)
    while pending:
        wave = sorted(n for n in pending if deps[n] <= done)
        groups.append(wave)
        done.update(wave)
        pending.difference_update(wave)
    return groups


def affected(graph, changed):
    deps = _normalise(graph)
    result, frontier = set(), [c for c in changed if c in deps]
    reverse = {n: set() for n in deps}
    for n, ds in deps.items():
        for d in ds:
            reverse[d].add(n)
    while frontier:
        n = frontier.pop()
        for m in reverse[n]:
            if m not in result:
                result.add(m)
                frontier.append(m)
    return sorted(result)
'''},
    },
    "hidden": {"test_deps.py": '''import unittest
from deps import CycleError, affected, build_order, parallel_groups


class Order(unittest.TestCase):
    def test_simple(self):
        g = {"app": ["lib", "util"], "lib": ["core"], "util": ["core"], "core": []}
        self.assertEqual(build_order(g), ["core", "lib", "util", "app"])

    def test_alphabetical_among_ready(self):
        g = {"d": [], "a": [], "c": ["a"], "b": ["d"]}
        self.assertEqual(build_order(g), ["a", "c", "d", "b"])

    def test_implicit_nodes_and_duplicates(self):
        self.assertEqual(build_order({"x": ["y", "y", "z"]}), ["y", "z", "x"])
        self.assertEqual(build_order({}), [])

    def test_all_nodes_present_and_valid_order(self):
        g = {"a": ["b", "c"], "b": ["d"], "c": ["d", "e"], "d": [], "f": ["a"]}
        order = build_order(g)
        self.assertEqual(sorted(order), ["a", "b", "c", "d", "e", "f"])
        for n, ds in g.items():
            for d in ds:
                self.assertLess(order.index(d), order.index(n))

    def test_cycles(self):
        with self.assertRaises(CycleError) as cm:
            build_order({"a": ["b"], "b": ["c"], "c": ["a"]})
        self.assertEqual(cm.exception.cycle, ["a", "b", "c", "a"])
        with self.assertRaises(CycleError) as cm:
            build_order({"x": ["x"]})
        self.assertEqual(cm.exception.cycle, ["x", "x"])
        with self.assertRaises(CycleError) as cm:
            build_order({"a": ["b"], "b": ["c"], "c": ["b"]})
        self.assertEqual(cm.exception.cycle, ["b", "c", "b"])
        self.assertIsInstance(cm.exception, ValueError)

    def test_cycle_found_in_alphabetical_visit_order(self):
        g = {"m": ["n"], "n": ["m"], "a": ["z"], "z": ["y"], "y": ["z"]}
        with self.assertRaises(CycleError) as cm:
            build_order(g)
        self.assertEqual(cm.exception.cycle, ["z", "y", "z"])


class Groups(unittest.TestCase):
    def test_waves(self):
        g = {"app": ["lib", "util"], "lib": ["core"], "util": ["core"], "core": [], "docs": []}
        self.assertEqual(parallel_groups(g), [["core", "docs"], ["lib", "util"], ["app"]])

    def test_chain_and_empty(self):
        self.assertEqual(parallel_groups({"c": ["b"], "b": ["a"]}), [["a"], ["b"], ["c"]])
        self.assertEqual(parallel_groups({}), [])

    def test_cycle(self):
        with self.assertRaises(CycleError):
            parallel_groups({"a": ["b"], "b": ["a"]})


class Affected(unittest.TestCase):
    def test_transitive(self):
        g = {"app": ["lib", "util"], "lib": ["core"], "util": ["core"], "core": [], "docs": []}
        self.assertEqual(affected(g, ["core"]), ["app", "lib", "util"])
        self.assertEqual(affected(g, ["lib"]), ["app"])
        self.assertEqual(affected(g, ["app"]), [])
        self.assertEqual(affected(g, ["nope"]), [])
        self.assertEqual(affected(g, ["lib", "util"]), ["app"])

    def test_changed_nodes_reached_by_other_changed_nodes_are_included(self):
        g = {"b": ["a"], "c": ["b"]}
        self.assertEqual(affected(g, ["a", "b"]), ["b", "c"])
'''},
    "bugfix": {
        "symptom": "A dependency resolver (deps.py) orders independent nodes wrongly: build_order({\"d\": [], \"a\": [], \"c\": [\"a\"], \"b\": [\"d\"]}) should list nodes that are ready at the same time alphabetically (a, c, d, b) but it returns a different order.",
        "bug": [("deps.py", "        for m in dependents[n]:\n            remaining[m].discard(n)\n            if not remaining[m]:\n                heapq.heappush(ready, m)\n    return order", "        for m in dependents[n]:\n            remaining[m].discard(n)\n            if not remaining[m]:\n                ready.append(m)\n    return order")],
        "repro": {"test_repro.py": 'import unittest\nfrom deps import build_order\n\n\nclass Repro(unittest.TestCase):\n    def test_alphabetical_among_ready(self):\n        g = {"d": [], "a": [], "c": ["a"], "b": ["d"]}\n        self.assertEqual(build_order(g), ["a", "c", "d", "b"])\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
