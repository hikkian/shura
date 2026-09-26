#!/usr/bin/env python3
"""Decode speed at real long-context depth - the benchmark behind every number in docs/BENCHMARKS.md.

Request 1 prefills the whole context; requests 2..N reuse the cached prefix and measure pure decode.
Each request generates exactly --max-tokens (ignore_eos) so runs are comparable.

    python3 bench/deep_context_bench.py --context /tmp/ctx_187k.txt
"""
import argparse
import statistics

from _client import DEFAULT_URL, big_prompt, chat, load_context


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", required=True, help="filler file from make_context.py")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    messages = big_prompt(load_context(args.context), "Write a short function to reverse a linked list in Python.")
    speeds = []
    for i in range(1, args.runs + 1):
        _, t, wall = chat(args.url, messages, args.max_tokens, ignore_eos=True)
        tps = t.get("predicted_per_second", 0.0)
        speeds.append(tps)
        print(f"request {i}: prompt_n={t.get('prompt_n'):>6}  prompt_ms={t.get('prompt_ms', 0):>9.0f}  "
              f"decode={tps:6.2f} tok/s  wall={wall:6.1f}s", flush=True)
    print(f"\ndecode tok/s: mean {statistics.mean(speeds):.2f}  min {min(speeds):.2f}  max {max(speeds):.2f}")


if __name__ == "__main__":
    main()
