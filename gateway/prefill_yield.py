"""Pure policy for the opt-in prefill yield: shrink the expert cache to 0 slots for one large new prefill, then restore.

The gateway owns the side effects (the resize calls, the busy count, the events). This module only decides and counts,
so the rules can be tested without a server. Measured in PREFILL2 and PREFILL2b: 30 -> 0 for a 7k prefill gave +14..+18%
prefill speed; the outputs at temperature 0 differ from the cached path (KL 0.140 against -ub noise 0.104), which is why the
switch stays off by default.
"""

DEFAULTS = {
    "prefillCacheYield": False,
    "prefillCacheYieldMinTokens": 4000,
}

# Same estimate as the RAM valve's large-prompt deferral: about four characters per token.
CHARS_PER_TOKEN = 4

REASON_OK = "large new prefill"


def validate(config):
    """Raise ValueError for a malformed switch or threshold; absent keys keep the defaults."""
    if "prefillCacheYield" in config and type(config["prefillCacheYield"]) is not bool:
        raise ValueError("Invalid guardian config: prefillCacheYield must be true or false")
    minimum = config.get("prefillCacheYieldMinTokens", DEFAULTS["prefillCacheYieldMinTokens"])
    if type(minimum) is not int or minimum <= 0:
        raise ValueError("prefillCacheYieldMinTokens must be a positive integer")


def new_tokens_estimate(body_len, previous_len):
    """Estimated new prefill tokens for this request.

    The previous request's body length is the baseline. A body that is not longer than the previous one (or a first request)
    starts a different conversation, so the whole body counts as new.
    """
    if previous_len is None or body_len <= previous_len:
        grown = body_len
    else:
        grown = body_len - previous_len
    return grown // CHARS_PER_TOKEN


def should_yield(enabled, new_tokens, minimum, slots_now, server_ready, busy, vision, game_active, slot_gears):
    """Return (True, reason) only when the cache can be shrunk safely for this request."""
    if not enabled:
        return False, "disabled"
    if vision:
        return False, "vision request"
    if game_active:
        return False, "game mode active"
    if slot_gears:
        return False, "slotGears owns the slot count"
    if not server_ready:
        return False, "server not ready"
    if busy != 1:
        return False, "other requests in flight"
    if slots_now <= 0:
        return False, "cache already empty"
    if new_tokens < minimum:
        return False, "small request"
    return True, REASON_OK


class YieldState:
    """Counters, the last body length, and a restore that has not been acknowledged yet."""

    COUNTERS = ("yielded", "restored", "resize_failed", "restore_failed", "skipped")

    def __init__(self):
        self.counters = {name: 0 for name in self.COUNTERS}
        self.last_len = None
        self.pending = None  # {"model": id, "slots": n} while a restore is still owed

    def count(self, name):
        self.counters[name] += 1

    def remember_len(self, body_len):
        previous = self.last_len
        self.last_len = body_len
        return previous

    def snapshot(self, enabled):
        return {"enabled": bool(enabled), "counters": dict(self.counters), "pending": dict(self.pending) if self.pending else None}
