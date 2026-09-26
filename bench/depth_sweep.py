#!/usr/bin/env python3
"""Decode speed as a function of context depth (looks for a cliff vs. gradual decline).

    python3 bench/depth_sweep.py --context /tmp/ctx_187k.txt
"""
import argparse

from _client import DEFAULT_URL, big_prompt, chat, load_context


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", required=True)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--fractions", default="0.05,0.2,0.43,0.65,0.8,1.0",
                    help="comma-separated fractions of the context file to use (default spans ~4k..187k tokens)")
    ap.add_argument("--max-tokens", type=int, default=100)
    args = ap.parse_args()

    ctx = load_context(args.context)
    print(f"{'prompt tokens':>14}  {'decode tok/s':>12}")
    for frac in (float(x) for x in args.fractions.split(",")):
        messages = big_prompt(ctx[:int(len(ctx) * frac)], "Write a short function to reverse a linked list in Python.")
        _, t, _ = chat(args.url, messages, args.max_tokens, ignore_eos=True)
        print(f"{t.get('prompt_n', 0) + t.get('cache_n', 0):>14}  {t.get('predicted_per_second', 0):>12.2f}", flush=True)


if __name__ == "__main__":
    main()
