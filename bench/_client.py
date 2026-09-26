"""Tiny stdlib-only HTTP helpers shared by the benchmark scripts."""
import json
import time
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8080"


def post_json(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def chat(base, messages, max_tokens, ignore_eos=False, **extra):
    """Returns (message, timings, wall_seconds) for one /v1/chat/completions call."""
    t0 = time.time()
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.6,
               "top_k": 20, "top_p": 0.95, **extra}
    if ignore_eos:
        payload["ignore_eos"] = True
    r = post_json(f"{base}/v1/chat/completions", payload)
    return r["choices"][0]["message"], r.get("timings", {}), time.time() - t0


def load_context(path):
    with open(path, errors="ignore") as f:
        return f.read()


def big_prompt(context, question):
    return [{"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Here is a large C++/CUDA codebase for context:\n\n" + context +
             f"\n\nIgnoring the above codebase, just answer this: {question}"}]
