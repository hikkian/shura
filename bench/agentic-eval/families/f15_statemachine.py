FAMILY = {
    "name": "f15_statemachine", "lang": "python",
    "implement": {
        "prompt": """Write the module `fsm.py` (standard library only) with the class `StateMachine` and the exception `TransitionError(Exception)`.

StateMachine(initial, transitions, guards=None)
  transitions: a dict {(state, event): target_state}. guards: optional dict {(state, event): callable(context) -> bool}.
  The machine starts in `initial` (which must appear in some transition as a source or target, otherwise ValueError).
  state            the current state (read-only property).
  can(event, context=None) -> bool   True if a transition exists for (state, event) and its guard (if any) returns a truthy value
                                    when called with `context`. Does not change anything and never raises.
  fire(event, context=None) -> str   perform the transition and return the new state. If there is no transition for (state, event)
                                    raise TransitionError("no transition ..."); if the guard returns a falsy value raise
                                    TransitionError("guard ..."). The state does not change when an error is raised.
  history          a list of (from_state, event, to_state) of the successful transitions, oldest first (a copy: modifying it does not
                   change the machine).
  on(event, callback)   register callback(from_state, to_state, context) called after every successful transition caused by `event`,
                   in registration order; an exception inside a callback does NOT undo the transition and does not stop the other
                   callbacks: collect the exceptions and, after all callbacks ran, raise the first one wrapped in
                   CallbackError(Exception) (a class you define; the original is its `__cause__`). The transition stays done.
  reachable(state=None) -> set   all states reachable from `state` (default: the current one) by any chain of transitions, ignoring
                   guards, not including the start state unless it can be reached again.
  dump() -> dict {"state": str, "history": list}; load(data) restores them (history entries as tuples); load raises ValueError if the
                   state is unknown to the machine or an entry is not a valid transition of the machine.
""",
        "start": {"fsm.py": '"""A finite state machine (to be written)."""\n\n\nclass TransitionError(Exception):\n    pass\n\n\nclass CallbackError(Exception):\n    pass\n\n\nclass StateMachine:\n    def __init__(self, initial, transitions, guards=None):\n        raise NotImplementedError\n'},
        "solution": {"fsm.py": '''"""A finite state machine."""


class TransitionError(Exception):
    pass


class CallbackError(Exception):
    pass


class StateMachine:
    def __init__(self, initial, transitions, guards=None):
        states = {s for (s, _), t in transitions.items()} | set(transitions.values())
        if initial not in states:
            raise ValueError(f"unknown initial state {initial!r}")
        self._t = dict(transitions)
        self._g = dict(guards or {})
        self._state = initial
        self._history = []
        self._callbacks = {}

    @property
    def state(self):
        return self._state

    @property
    def history(self):
        return list(self._history)

    def can(self, event, context=None):
        key = (self._state, event)
        if key not in self._t:
            return False
        guard = self._g.get(key)
        try:
            return bool(guard(context)) if guard else True
        except Exception:
            return False

    def fire(self, event, context=None):
        key = (self._state, event)
        if key not in self._t:
            raise TransitionError(f"no transition for event {event!r} in state {self._state!r}")
        guard = self._g.get(key)
        if guard and not guard(context):
            raise TransitionError(f"guard rejected event {event!r} in state {self._state!r}")
        old, new = self._state, self._t[key]
        self._state = new
        self._history.append((old, event, new))
        errors = []
        for cb in self._callbacks.get(event, []):
            try:
                cb(old, new, context)
            except Exception as e:
                errors.append(e)
        if errors:
            err = CallbackError(f"{len(errors)} callback(s) failed")
            raise err from errors[0]
        return new

    def on(self, event, callback):
        self._callbacks.setdefault(event, []).append(callback)

    def reachable(self, state=None):
        start = self._state if state is None else state
        seen, frontier = set(), [start]
        while frontier:
            s = frontier.pop()
            for (src, _), dst in self._t.items():
                if src == s and dst not in seen:
                    seen.add(dst)
                    frontier.append(dst)
        return seen

    def dump(self):
        return {"state": self._state, "history": [list(h) for h in self._history]}

    def load(self, data):
        states = {s for (s, _), t in self._t.items()} | set(self._t.values())
        if data["state"] not in states:
            raise ValueError("unknown state")
        hist = []
        for entry in data["history"]:
            a, e, b = entry
            if self._t.get((a, e)) != b:
                raise ValueError(f"invalid transition in history: {entry!r}")
            hist.append((a, e, b))
        self._state, self._history = data["state"], hist
'''},
    },
    "hidden": {"test_fsm.py": '''import unittest
from fsm import CallbackError, StateMachine, TransitionError

T = {("new", "pay"): "paid", ("paid", "ship"): "shipped", ("shipped", "deliver"): "delivered", ("new", "cancel"): "cancelled",
     ("paid", "cancel"): "cancelled", ("shipped", "return"): "paid"}


class Basics(unittest.TestCase):
    def test_flow_and_history(self):
        m = StateMachine("new", T)
        self.assertEqual(m.state, "new")
        self.assertEqual(m.fire("pay"), "paid")
        self.assertEqual(m.fire("ship"), "shipped")
        self.assertEqual(m.history, [("new", "pay", "paid"), ("paid", "ship", "shipped")])
        h = m.history
        h.append("junk")
        self.assertEqual(len(m.history), 2)

    def test_errors_do_not_change_state(self):
        m = StateMachine("new", T)
        with self.assertRaises(TransitionError):
            m.fire("deliver")
        self.assertEqual(m.state, "new")
        self.assertEqual(m.history, [])
        self.assertFalse(m.can("deliver"))
        self.assertTrue(m.can("pay"))

    def test_bad_initial(self):
        with self.assertRaises(ValueError):
            StateMachine("nowhere", T)

    def test_state_is_read_only(self):
        m = StateMachine("new", T)
        with self.assertRaises(AttributeError):
            m.state = "paid"


class Guards(unittest.TestCase):
    def test_guard_receives_context(self):
        m = StateMachine("new", T, guards={("new", "pay"): lambda ctx: bool(ctx) and ctx["amount"] > 0})
        self.assertFalse(m.can("pay"))
        self.assertFalse(m.can("pay", {"amount": 0}))
        self.assertTrue(m.can("pay", {"amount": 5}))
        with self.assertRaises(TransitionError) as cm:
            m.fire("pay", {"amount": 0})
        self.assertIn("guard", str(cm.exception))
        self.assertEqual(m.state, "new")
        self.assertEqual(m.fire("pay", {"amount": 5}), "paid")

    def test_can_never_raises(self):
        m = StateMachine("new", T, guards={("new", "pay"): lambda ctx: ctx["missing"]})
        self.assertFalse(m.can("pay", {}))
        self.assertFalse(m.can("pay", None))

    def test_no_transition_message(self):
        m = StateMachine("new", T)
        with self.assertRaises(TransitionError) as cm:
            m.fire("ship")
        self.assertIn("no transition", str(cm.exception))


class Callbacks(unittest.TestCase):
    def test_called_in_order_with_arguments(self):
        calls = []
        m = StateMachine("new", T)
        m.on("pay", lambda a, b, c: calls.append(("1", a, b, c)))
        m.on("pay", lambda a, b, c: calls.append(("2", a, b, c)))
        m.on("ship", lambda a, b, c: calls.append("ship"))
        m.fire("pay", {"x": 1})
        self.assertEqual(calls, [("1", "new", "paid", {"x": 1}), ("2", "new", "paid", {"x": 1})])

    def test_failing_callback_does_not_undo_or_stop_others(self):
        calls = []

        def boom(a, b, c):
            raise RuntimeError("first")

        def boom2(a, b, c):
            raise KeyError("second")
        m = StateMachine("new", T)
        m.on("pay", boom)
        m.on("pay", lambda a, b, c: calls.append("ran"))
        m.on("pay", boom2)
        with self.assertRaises(CallbackError) as cm:
            m.fire("pay")
        self.assertIsInstance(cm.exception.__cause__, RuntimeError)
        self.assertEqual(calls, ["ran"])
        self.assertEqual(m.state, "paid")
        self.assertEqual(m.history, [("new", "pay", "paid")])


class Graph(unittest.TestCase):
    def test_reachable(self):
        m = StateMachine("new", T)
        self.assertEqual(m.reachable(), {"paid", "shipped", "delivered", "cancelled"})
        self.assertEqual(m.reachable("shipped"), {"delivered", "paid", "cancelled", "shipped"})
        self.assertEqual(m.reachable("delivered"), set())
        self.assertEqual(m.reachable("cancelled"), set())

    def test_dump_load(self):
        m = StateMachine("new", T)
        m.fire("pay")
        m.fire("ship")
        data = m.dump()
        m2 = StateMachine("new", T)
        m2.load(data)
        self.assertEqual(m2.state, "shipped")
        self.assertEqual(m2.history, [("new", "pay", "paid"), ("paid", "ship", "shipped")])
        self.assertEqual(m2.fire("deliver"), "delivered")

    def test_load_validates(self):
        m = StateMachine("new", T)
        with self.assertRaises(ValueError):
            m.load({"state": "nowhere", "history": []})
        with self.assertRaises(ValueError):
            m.load({"state": "paid", "history": [["new", "ship", "paid"]]})
        self.assertEqual(m.state, "new")
'''},
    "bugfix": {
        "symptom": "A state machine (fsm.py) breaks its own history: when a transition callback raises an exception, the machine must stay in the new state but the failed event is missing from the history list.",
        "bug": [("fsm.py", '''        self._state = new
        self._history.append((old, event, new))
        errors = []''', '''        self._state = new
        errors = []'''), ("fsm.py", '''        if errors:
            err = CallbackError(f"{len(errors)} callback(s) failed")
            raise err from errors[0]
        return new''', '''        if errors:
            err = CallbackError(f"{len(errors)} callback(s) failed")
            raise err from errors[0]
        self._history.append((old, event, new))
        return new''')],
        "repro": {"test_repro.py": 'import unittest\nfrom fsm import CallbackError, StateMachine\n\n\nclass Repro(unittest.TestCase):\n    def test_history_survives_a_failing_callback(self):\n        m = StateMachine("a", {("a", "go"): "b"})\n\n        def boom(x, y, z):\n            raise RuntimeError("x")\n        m.on("go", boom)\n        with self.assertRaises(CallbackError):\n            m.fire("go")\n        self.assertEqual(m.state, "b")\n        self.assertEqual(m.history, [("a", "go", "b")])\n\n\nif __name__ == "__main__":\n    unittest.main()\n'},
    },
}
