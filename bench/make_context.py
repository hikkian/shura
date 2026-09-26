#!/usr/bin/env python3
"""Build a realistic long-context filler file (real C/C++/CUDA source) of roughly N tokens.

Token counts are measured with the running server's own tokenizer (/tokenize), so the result
matches exactly what the model sees. Example:

    python3 bench/make_context.py --source ~/ai/llama.cpp-perf --tokens 187000 -o /tmp/ctx_187k.txt
"""
import argparse
import sys
from pathlib import Path

from _client import DEFAULT_URL, post_json


def count_tokens(base, text):
    return len(post_json(f"{base}/tokenize", {"content": text})["tokens"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="directory with source code to concatenate (e.g. a llama.cpp checkout)")
    ap.add_argument("--tokens", type=int, default=187000, help="target token count (default 187000)")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"gateway or llama-server base URL (default {DEFAULT_URL})")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    files = sorted(p for sub in ("src", "ggml/src", "common", "tools")
                   for ext in ("*.cpp", "*.h", "*.cu") for p in (Path(args.source) / sub).rglob(ext))
    if not files:
        sys.exit(f"no source files found under {args.source}")
    corpus = "".join(p.read_text(errors="ignore") for p in files)

    lo, hi = 0, len(corpus)
    while hi - lo > 2000:  # binary search on characters until within ~2k chars of the target
        mid = (lo + hi) // 2
        if count_tokens(args.url, corpus[:mid]) < args.tokens:
            lo = mid
        else:
            hi = mid
    text = corpus[:lo]
    Path(args.output).write_text(text)
    print(f"wrote {args.output}: {len(text)} chars, {count_tokens(args.url, text)} tokens")


if __name__ == "__main__":
    main()
