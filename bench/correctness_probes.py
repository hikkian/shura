#!/usr/bin/env python3
"""Quick output-correctness check - run after changing KV-cache type, patches or kernels.

A broken attention kernel can produce fluent but wrong answers without crashing (the TurboQuant
head_dim-256 bug did exactly that), so speed numbers alone are not enough.

    python3 bench/correctness_probes.py
"""
import argparse
import sys

from _client import DEFAULT_URL, post_json

PROBES = [
    ("What is 17 + 26?", "43"),
    ("What is 123 * 4?", "492"),
    ("What is 100 - 37?", "63"),
    ("What is 9 * 9?", "81"),
    ("What is 256 / 4?", "64"),
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What color do you get mixing blue and yellow?", "green"),
    ("How many days are in a week?", "7"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--n-predict", type=int, default=256,
                    help="Output token budget; increase for models that emit reasoning before the answer")
    args = ap.parse_args()
    if args.n_predict < 1:
        ap.error("--n-predict must be positive")

    passed = 0
    for question, expected in PROBES:
        out = post_json(f"{args.url}/completion", {"prompt": f"Q: {question}\nA:", "n_predict": args.n_predict, "temperature": 0})
        content = out.get("content", "")
        ok = expected.lower() in content.lower()
        passed += ok
        print(f"[{'OK  ' if ok else 'FAIL'}] {question} -> {content.strip()[:60]!r}")
    print(f"\n{passed}/{len(PROBES)} correct")
    sys.exit(0 if passed == len(PROBES) else 1)


if __name__ == "__main__":
    main()
