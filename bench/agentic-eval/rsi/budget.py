"""RSIL0: the size budget of a harness edit. Every rule costs prompt prefill on every request, so an edit may add at most BUDGET
tokens and MAX_CHARS characters (the editors are told "about 7 000 characters"; both limits hold). The token count is an estimate
(characters / 3.5 with a 15 % margin): the model's own tokenizer is not run here (no GPU)."""
import math

BUDGET_TOKENS = 2500
MAX_CHARS = 7000
CHARS_PER_TOKEN = 3.5
MARGIN = 1.15


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN * MARGIN)


def added_lines(diff_text: str) -> str:
    """The text an edit adds: the `+` lines of a unified diff, without the file headers."""
    return "\n".join(line[1:] for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))


def check(addition: str, limit: int = BUDGET_TOKENS, max_chars: int = MAX_CHARS) -> dict:
    n = estimate_tokens(addition)
    chars = len(addition)
    return {"tokens_estimate": n, "limit": limit, "chars": chars, "max_chars": max_chars, "ok": n <= limit and chars <= max_chars}
