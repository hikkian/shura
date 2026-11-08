import itertools

_counters = {}


def next_id(kind):
    c = _counters.setdefault(kind, itertools.count(1))
    return next(c)


def reset_ids():
    _counters.clear()
