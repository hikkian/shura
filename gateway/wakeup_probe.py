"""Independent, low-cost wakeup probe. Parent supplies a pipe; no files or GPU."""
import json
import time


def main():
    samples = []
    window = time.monotonic()
    while True:
        before = time.monotonic()
        time.sleep(0.01)
        samples.append(max(0, (time.monotonic() - before - 0.01) * 1000))
        if time.monotonic() - window >= 1:
            ordered = sorted(samples)
            print(json.dumps({"p99_ms": ordered[min(len(ordered)-1, int(len(ordered)*0.99))],
                              "max_ms": max(ordered)}), flush=True)
            samples.clear()
            window = time.monotonic()


if __name__ == "__main__":
    main()
