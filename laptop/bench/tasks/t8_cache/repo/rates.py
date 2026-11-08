import time

CALLS = []


def fetch_rate(base, quote):
    """Pretend network call."""
    CALLS.append((base, quote))
    return {"USD": {"KZT": 480.0, "EUR": 0.92}, "EUR": {"KZT": 522.0}}[base][quote]


def get_rate(base, quote, now=None):
    return fetch_rate(base, quote)
