#!/usr/bin/env python3
"""Measure how fast a long session resumes after switching away from it.

Default mode: session A (long) -> two unrelated sessions -> back to A. Exercises llama-server's
RAM prompt cache (--cache-ram); no disk writes.

--unload mode: session A -> POST /guardian/unload (gateway saves the slot to disk and stops the
model) -> continue A (gateway reloads the model and restores the slot).

    python3 bench/session_cache_test.py --context /tmp/ctx_187k.txt [--unload]
"""
import argparse

from _client import DEFAULT_URL, chat, load_context, post_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", required=True)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--unload", action="store_true", help="test gateway unload + slot restore instead of RAM cache")
    args = ap.parse_args()

    turn1 = [{"role": "user", "content": "Session A. Here is a large codebase:\n\n" + load_context(args.context)
              + "\n\nReply with just: OK"}]
    msg, t, w = chat(args.url, turn1, 64)
    print(f"A, first request : processed {t.get('prompt_n'):>6} tokens  wall {w:6.1f}s", flush=True)

    if args.unload:
        post_json(f"{args.url}/guardian/unload", {}, timeout=300)
        print("unloaded model via gateway (slot saved to disk if large enough)", flush=True)
    else:
        for i in range(2):
            _, tb, wb = chat(args.url, [{"role": "user", "content": f"Session B{i}: what is a hash map, in one sentence?"}], 40)
            print(f"B{i}, other session: processed {tb.get('prompt_n'):>6} tokens  wall {wb:6.1f}s", flush=True)

    assistant = {"role": "assistant", "content": msg.get("content", "")}
    if msg.get("reasoning_content"):
        assistant["reasoning_content"] = msg["reasoning_content"]
    turn2 = turn1 + [assistant, {"role": "user", "content": "Now reply with just: BYE"}]
    _, t2, w2 = chat(args.url, turn2, 32)
    print(f"A, resumed       : processed {t2.get('prompt_n'):>6} tokens (cached {t2.get('cache_n')})  wall {w2:6.1f}s")


if __name__ == "__main__":
    main()
