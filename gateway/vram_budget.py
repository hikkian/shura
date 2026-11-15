"""Gateway side of the VRAM governor (VRAMLOGIC1g, contract: reports/vramlogic1/API.md).

The gateway owns the VRAM budget in MiB: it decides whether a layout may start (admission by the fixed part that the
server measured after warmup), when to switch to the lite layout (governor state MIN for longer than T_lite), and how
to resume a response after the server paused with a checkpoint. It never touches expert-cache slots: the server's
governor owns them, and the gateway only reads GET /vram and sets POST /vram/policy.

All sizes are MiB, all times are seconds. Pure logic: no HTTP, no processes, so it can be tested without a GPU.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_RESERVE_MIB = 1536        # 1.5 GiB: the desktop's reserve (the floor), never occupied by the cache
DEFAULT_BUFFER_MIB = 256          # the server's working buffer above the floor (VRAMLOGIC3: 256 base, desk_need raises it to 512); the gateway learns the live value from GET /vram
DEFAULT_POLL_FAST_MS = 50
# VRAMLOGIC2: the cache is at its minimum and free VRAM stays below the floor this long -> pause with a checkpoint. 0.25 s, so that
# the decision (+ the running decode step, the slot save and the server's 150 ms flush) comes before a 1 s stand watchdog.
DEFAULT_PAUSE_AFTER_SECONDS = 0.25
DEFAULT_HYSTERESIS_MIB = 256
DEFAULT_GROW_SECONDS = 30
DEFAULT_PAUSE_FLOOR_MIB = 768
DEFAULT_MIN_CHUNKS = 0
DEFAULT_LITE_AFTER_SECONDS = 60   # T_lite: how long the governor may stay in MIN before the gateway goes lite
DEFAULT_POLICY = "governor"
POLICIES = ("governor", "legacy")

# Same ranges as the server's POST /vram/policy (API.md); the gateway refuses values the server would reject.
RANGES = {
    "reserve_mib": (256, 4096),
    "buffer_mib": (0, 2048),
    "poll_fast_ms": (20, 1000),
    "pause_after_ms": (0, 600000),
    "hysteresis_mib": (0, 1024),
    "grow_seconds": (0, 600),
    "pause_floor_mib": (0, 2048),
    "min_chunks": (0, 64),
}
SNAPSHOT_KEYS = ("state", "reason", "ready", "pool_frozen", "card", "fixed_mib", "cache", "policy", "counters")
VRAMLOGIC2_KEYS = ("buffer_mib", "poll_fast_ms", "pause_after_ms")   # policy fields a server from before VRAMLOGIC2 refuses


def older_policy(values):
    """The policy for a server from before VRAMLOGIC2: without the new fields; the stall pause in its whole seconds
    (pause_after_seconds, rounded up, at least 1 unless off) - the closest it can do."""
    older = {k: v for k, v in values.items() if k not in VRAMLOGIC2_KEYS}
    if "pause_after_ms" in values:
        ms = values["pause_after_ms"]
        older["pause_after_seconds"] = 0 if ms == 0 else max(1, -(-ms // 1000))
    return older
PAUSE_EXIT_CODE = 75


class BudgetError(ValueError):
    pass


def check_policy_config(config):
    """Read the vram* keys from guardian config. Returns the policy as the server's field names."""
    policy = config.get("vramPolicy", DEFAULT_POLICY)
    if policy not in POLICIES:
        raise BudgetError(f"vramPolicy must be one of {POLICIES}, got {policy!r}")
    values = {
        "reserve_mib": config.get("vramReserveMiB", DEFAULT_RESERVE_MIB),
        "hysteresis_mib": config.get("vramHysteresisMiB", DEFAULT_HYSTERESIS_MIB),
        "grow_seconds": config.get("vramGrowSeconds", DEFAULT_GROW_SECONDS),
        "pause_floor_mib": config.get("vramPauseFloorMiB", DEFAULT_PAUSE_FLOOR_MIB),
        "min_chunks": config.get("vramMinChunks", DEFAULT_MIN_CHUNKS),
    }
    # VRAMLOGIC2: the buffer and the fast poll are sent only when the config names them; otherwise the server's own defaults
    # stand (a server from before VRAMLOGIC2 would refuse the fields and the flags)
    for key, config_key in (("buffer_mib", "vramBufferMiB"), ("poll_fast_ms", "vramPollFastMs")):
        if config.get(config_key) is not None:
            values[key] = config[config_key]
    # the stall pause always travels (in the policy, never as a flag): the server's own default is 2 s, later than the stand watchdog
    pause_after = config.get("vramPauseAfterSeconds", DEFAULT_PAUSE_AFTER_SECONDS)
    if type(pause_after) not in (int, float) or isinstance(pause_after, bool) or not 0 <= pause_after <= 600:
        raise BudgetError(f"vramPauseAfterSeconds must be a number 0..600 (0 = off), got {pause_after!r}")
    values["pause_after_ms"] = round(pause_after * 1000)
    for key, value in values.items():
        lo, hi = RANGES[key]
        if type(value) is not int or not lo <= value <= hi:
            raise BudgetError(f"{key} must be an integer {lo}..{hi} MiB (or seconds/chunks), got {value!r}")
    lite_after = config.get("vramLiteAfterSeconds", DEFAULT_LITE_AFTER_SECONDS)
    if type(lite_after) not in (int, float) or lite_after < 0:
        raise BudgetError("vramLiteAfterSeconds must be a non-negative number")
    return policy, values, float(lite_after)


def policy_body(values):
    """Body for POST /vram/policy from the checked values (only the fields the gateway owns)."""
    return dict(values)


def parse_snapshot(data):
    """Check that a GET /vram answer has the fields the gateway uses. Returns the same dict."""
    if not isinstance(data, dict):
        raise BudgetError("GET /vram answer must be a JSON object")
    missing = [key for key in SNAPSHOT_KEYS if key not in data]
    if missing:
        raise BudgetError(f"GET /vram answer misses fields: {', '.join(missing)}")
    fixed = data["fixed_mib"]
    if not isinstance(fixed, dict) or type(fixed.get("total")) is not int:
        raise BudgetError("fixed_mib.total must be an integer MiB")
    card = data["card"]
    if type(card.get("free_mib")) is not int or type(card.get("total_mib")) is not int:
        raise BudgetError("card.free_mib and card.total_mib must be integer MiB")
    return data


@dataclass
class Budget:
    """Gateway-owned VRAM budget for one card."""
    reserve_mib: int = DEFAULT_RESERVE_MIB
    buffer_mib: int = 0                                 # VRAMLOGIC2: working buffer above the floor (learned from the server's policy)
    hysteresis_mib: int = DEFAULT_HYSTERESIS_MIB
    lite_after_s: float = DEFAULT_LITE_AFTER_SECONDS
    fixed_learned: dict = field(default_factory=dict)   # (model, layout) -> fixed MiB measured after warmup
    _min_since: float = None                            # when the governor entered MIN (None if not in MIN)
    _left_min_at: float = None                          # when the governor last left MIN (None if never)

    @property
    def target_mib(self):
        """VRAMLOGIC2: the working target the server keeps free: floor + buffer."""
        return self.reserve_mib + self.buffer_mib

    @property
    def admit_margin_mib(self):
        return self.target_mib + self.hysteresis_mib

    def learn_policy(self, policy):
        """Take the buffer the server really runs with (GET /vram -> policy.buffer_mib); a server without the field keeps 0."""
        if isinstance(policy, dict) and type(policy.get("buffer_mib")) is int and 0 <= policy["buffer_mib"] <= 4096:
            self.buffer_mib = policy["buffer_mib"]

    def learn(self, model, layout, snapshot):
        """Store the fixed part measured after warmup (only when the server reports ready)."""
        snap = parse_snapshot(snapshot)
        if not snap["ready"]:
            raise BudgetError("fixed part is only trusted after warmup (ready=true)")
        self.fixed_learned[(model, layout)] = snap["fixed_mib"]["total"]
        return self.fixed_learned[(model, layout)]

    def admission(self, model, layout, free_before_start_mib, estimate_mib=None):
        """May this layout start? free_before_start is the card's free MiB before loading (nothing of ours loaded).

        A learned fixed part is authoritative. Without one, the caller's estimate is used and the result is marked
        as an estimate so the caller can re-check after the first warmup.
        Returns (ok, need_mib, source).
        """
        learned = self.fixed_learned.get((model, layout))
        if learned is not None:
            fixed, source = learned, "learned"
        elif estimate_mib is not None:
            fixed, source = int(estimate_mib), "estimate"
        else:
            raise BudgetError(f"no fixed part known for {model}/{layout}: learn it or give an estimate")
        need = fixed + self.admit_margin_mib
        return free_before_start_mib >= need, need - free_before_start_mib if free_before_start_mib < need else 0, source

    def observe(self, state, now):
        """Track the governor state (call every poll). Keeps when MIN started and when it was last left."""
        if state == "MIN":
            if self._min_since is None:
                self._min_since = now
        elif self._min_since is not None:
            self._min_since = None
            self._left_min_at = now

    def governor_says_lite(self, state, now):
        """True once the governor has stayed in MIN for longer than T_lite (the gateway then switches on a request boundary)."""
        self.observe(state, now)
        return self._min_since is not None and now - self._min_since >= self.lite_after_s

    def lite_relief_ok(self, state, now):
        """True when the governor has been out of MIN for at least T_lite: the normal layout may come back (no ping-pong)."""
        self.observe(state, now)
        if state == "MIN":
            return False
        return self._left_min_at is None or now - self._left_min_at >= self.lite_after_s

    def pause_frame(self, chunk):
        """Parse the server's pause frame: an ordinary error of type vram_pause (API.md).

        The stream frame is {"error": {"code": 503, "type": "vram_pause", "message": "<JSON string>"}}; the message holds
        checkpoint, slot, reason, n_past, generated_tokens, saved and tokens_file: the name of the
        file next to the checkpoint that carries the exact token ids
        (tokens_file, API.md «Файл-спутник»). Returns None for any other chunk.
        """
        if not isinstance(chunk, dict) or not isinstance(chunk.get("error"), dict):
            return None
        error = chunk["error"]
        if error.get("type") != "vram_pause":
            return None
        try:
            inner = json.loads(error.get("message") or "{}")
        except (TypeError, ValueError):
            raise BudgetError("vram_pause message is not JSON") from None
        if not isinstance(inner, dict) or type(inner.get("n_past")) is not int or type(inner.get("saved")) is not bool:
            raise BudgetError("vram_pause message needs an integer n_past and a boolean saved")
        checkpoint = inner.get("checkpoint", "")
        tokens_file = inner.get("tokens_file", "")
        if not isinstance(checkpoint, str) or not isinstance(tokens_file, str):
            raise BudgetError("vram_pause checkpoint and tokens_file must be file name strings")
        return {"checkpoint": checkpoint, "tokens_file": tokens_file, "slot": inner.get("slot", 0),
                "reason": str(inner.get("reason", "")), "n_past": inner["n_past"],
                "generated_tokens": int(inner.get("generated_tokens", 0)), "saved": inner["saved"]}

    def resume_plan(self, frame, next_layout_admitted):
        """After a pause: restart the server on the admitted layout, restore the checkpoint, continue the same response.

        Without a saved checkpoint there is nothing to continue from: the request fails explicitly. If the next layout is not
        admitted, the plan is to wait and retry admission (the caller decides how long).
        """
        if frame is None:
            raise BudgetError("no pause frame to resume from")
        if not frame["saved"] or not frame["checkpoint"]:
            return {"action": "fail", "reason": "checkpoint was not saved"}
        if not next_layout_admitted:
            return {"action": "wait_and_retry", "checkpoint": frame["checkpoint"]}
        return {"action": "restart_and_restore", "checkpoint": frame["checkpoint"], "n_past": frame["n_past"]}


def load_pause_tokens(path, frame_n_past=None):
    """Read <slot-save-path>/<checkpoint>.tokens.json written next to a checkpoint (API.md, «Файл-спутник»).

    The continuation reuses these exact ids: restore the checkpoint, then POST /completion with prompt = tokens and
    n_predict = n_predict_left. Refuses anything it cannot reproduce exactly: a multimodal slot (its images are not in the ids),
    and token counts that do not match n_past.
    Returns {"tokens", "n_past", "n_predict", "n_predict_left", "params", "processing"}.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BudgetError(f"tokens file unreadable: {error}") from None
    if not isinstance(data, dict):
        raise BudgetError("tokens file must be a JSON object")
    if data.get("multimodal") is not False:
        raise BudgetError("the slot had images or audio: no exact continuation")
    tokens = data.get("tokens")
    if not isinstance(tokens, list) or not all(type(t) is int and t >= 0 for t in tokens):
        raise BudgetError("tokens must be a list of non-negative integers")
    if type(data.get("n_past")) is not int or data["n_past"] != len(tokens):
        raise BudgetError("n_past must equal the number of tokens")
    if frame_n_past is not None and frame_n_past != len(tokens):
        raise BudgetError("tokens file does not match the frame's n_past")
    processing = data.get("processing")
    if type(processing) is not bool:
        raise BudgetError("processing must be true or false")
    if not processing:
        return {"tokens": tokens, "n_past": len(tokens), "n_predict": None, "n_predict_left": None, "params": None,
                "processing": False}
    left = data.get("n_predict_left")
    if type(left) is not int or left < -1:
        raise BudgetError("n_predict_left must be -1 or a non-negative integer")
    params = data.get("params")
    if not isinstance(params, dict):
        raise BudgetError("params must be the request's sampling parameters as an object")
    return {"tokens": tokens, "n_past": len(tokens), "n_predict": data.get("n_predict"), "n_predict_left": left,
            "params": params, "processing": True}
