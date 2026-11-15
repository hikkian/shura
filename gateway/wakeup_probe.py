"""Independent, low-cost wakeup probe. Parent supplies a pipe; no files or GPU."""
import json
import math
import time

WINDOW_SECONDS = 5  # p99 needs more than ~100 samples to differ from the maximum


def percentile(values, q):
    """Nearest-rank percentile of an unsorted, non-empty list."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


class Window:
    """Rolling per-second sample lists: one second of samples is ~97 at 10 ms, far too few for p99."""

    def __init__(self, seconds=WINDOW_SECONDS):
        self.seconds, self.chunks = seconds, []

    def add_second(self, samples):
        self.chunks = (self.chunks + [list(samples)])[-self.seconds:]
        every = [x for chunk in self.chunks for x in chunk]
        return {"p99_ms": percentile(every, 0.99), "max_ms": max(samples)}


def main():
    samples = []
    window = Window()
    started = time.monotonic()
    while True:
        before = time.monotonic()
        time.sleep(0.01)
        samples.append(max(0, (time.monotonic() - before - 0.01) * 1000))
        if time.monotonic() - started >= 1:
            print(json.dumps(window.add_second(samples)), flush=True)
            samples.clear()
            started = time.monotonic()


if __name__ == "__main__":
    main()
