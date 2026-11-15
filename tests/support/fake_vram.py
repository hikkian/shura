"""Fake VRAM governor for gateway tests (VRAMLOGIC1g).

Implements the server side of reports/vramlogic1/API.md on a virtual clock: GET /vram, GET and POST /vram/policy,
POST /vram/pause, POST /moe-cache/chunks, POST /vram/simulate. No GPU, no real process: the test calls tick(seconds)
and the state machine moves the same way the real governor is specified to move.

All sizes are MiB, all times are seconds.
"""

import copy
import json

POLICY_RANGES = {
    "reserve_mib": (256, 4096),
    "buffer_mib": (0, 2048),
    "poll_fast_ms": (20, 1000),
    "pause_after_ms": (0, 600000),
    "hysteresis_mib": (0, 1024),
    "grow_seconds": (0, 600),
    "pause_floor_mib": (0, 2048),
    "min_chunks": (0, 64),
    "poll_ms": (50, 1000),
}
POLICY_DEFAULTS = {
    "reserve_mib": 1536,
    "buffer_mib": 256,        # VRAMLOGIC2: working target = reserve + buffer (VRAMLOGIC3: 256 base, the gateway raises it)
    "poll_fast_ms": 50,
    "trim_abort": True,
    "pause_after_ms": 2000,   # the server's own default; the gateway sends its own (vramPauseAfterSeconds, 0.25 s)
    "hysteresis_mib": 256,
    "grow_seconds": 30,
    "pause_floor_mib": 768,
    "min_chunks": 0,
    "max_chunks": None,
    "prefill_yield": False,
    "poll_ms": 200,
    "sleep_level": 0,
}
POLL_SECONDS = 0.2  # the governor samples sysfs every 200 ms by default
STALL_SLACK_MIB = 48  # a shortfall up to this much below the floor does not count as a stall (as the server)


class PolicyError(ValueError):
    pass


def validate_policy_patch(patch, current, legacy=False):
    """Return the new policy dict, or raise PolicyError. Nothing is applied on error (API.md: atomic).
    legacy: a server before MEMV2 (VRAMLOGIC1 only): it does not know sleep_level, human_need, note."""
    if not isinstance(patch, dict):
        raise PolicyError("policy body must be a JSON object")
    new = dict(current)
    for key, value in patch.items():
        if legacy and key in ("sleep_level", "human_need", "note", "buffer_mib", "poll_fast_ms", "trim_abort", "pause_after_ms"):
            raise PolicyError(f"unknown field {key}")
        if key == "pause_after_seconds":
            # VRAMLOGIC1 field, whole seconds there; VRAMLOGIC2 servers take a fraction too. Stored as pause_after_ms.
            ok_type = type(value) is int if legacy else (type(value) in (int, float) and not isinstance(value, bool))
            if not ok_type or not 0 <= value <= 600:
                raise PolicyError("pause_after_seconds must be " + ("an integer" if legacy else "a number") + " 0..600")
            if "pause_after_ms" not in patch:
                new["pause_after_ms"] = round(value * 1000)
            continue
        if key == "max_chunks":
            if value is not None and (type(value) is not int or not 0 <= value <= 64):
                raise PolicyError("max_chunks must be null or an integer 0..64")
        elif key in ("prefill_yield", "trim_abort"):
            if type(value) is not bool:
                raise PolicyError(f"{key} must be true or false")
        elif key == "sleep_level":
            if type(value) is not int or not 0 <= value <= 1:
                raise PolicyError("sleep_level must be an integer 0..1")
        elif key == "human_need":
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise PolicyError("human_need must be a number 0..1")
            continue
        elif key == "note":
            if type(value) is not str or len(value) > 200:
                raise PolicyError("note must be a string up to 200 characters")
            continue
        elif key in POLICY_RANGES:
            lo, hi = POLICY_RANGES[key]
            if type(value) is not int or not lo <= value <= hi:
                raise PolicyError(f"{key} must be an integer {lo}..{hi}")
        else:
            raise PolicyError(f"unknown policy field: {key}")
        new[key] = value
    return new


class FakeVram:
    """Virtual-clock governor. Use tick(seconds) to advance time; desk_mib models other processes on the card."""

    def __init__(self, total_mib=12282, fixed=None, chunk_slots=4, chunk_mib=104, start_chunks=0,
                 policy=None, warmup_seconds=4.2, checkpoint_dir="/tmp/shura-vram-checkpoints"):
        self.total = total_mib
        # default: a layout that fits a 12 GiB card with the 1.5 GiB reserve (the API.md example, 10570, does not: admission would refuse it)
        self.fixed = dict(fixed or {"weights": 6000, "kv": 640, "draft": 120, "pool": 410, "sched": 300, "vision": 0})
        self.chunk_slots = chunk_slots
        self.chunk_mib = chunk_mib
        self.chunks = start_chunks
        self.desk_mib = 0
        self.policy = dict(POLICY_DEFAULTS, **(policy or {}))
        self.state = "WARMUP"
        self.reason = "starting"
        self.ready = False
        self.pool_frozen = False
        self.warmup_left = float(warmup_seconds)
        self.warmup_seconds = float(warmup_seconds)
        self.uptime = 0.0
        self.counters = {"trims": 0, "chunks_dropped": 0, "grows": 0, "chunks_grown": 0, "pauses": 0,
                         "pool_overflow": 0, "free_below_reserve_events": 0, "free_below_reserve_max_s": 0.0}
        self.checkpoint_dir = checkpoint_dir
        self.frames = []          # pause frames the server would send, for assertions
        self.exit_code = None     # 75 after a pause
        self._below_since = None  # time since free went below reserve (for the 1 s invariant)
        self._grow_clock = 0.0    # seconds free has stayed above the grow threshold
        self._floor_polls = 0     # consecutive polls with MIN and free below pause floor
        self._stall_since = None  # uptime when MIN with free below the floor by more than the slack began (stall pause)
        self._window_min = None   # minimum free since the last GET /vram
        self._episode_counted = False
        self._sim_restore_at = None
        self._sim_saved = 0
        self.legacy_api = False   # True: behave like a server before MEMV2 (no sleep_level/human_need/note, no /vram/journal)
        self.journal = []         # MEMV2: GET /vram/journal events
        self.human_need = None

    # --- derived values -------------------------------------------------------------------------------------------
    def fixed_total(self):
        return sum(self.fixed.values())

    def cache_mib(self):
        return self.chunks * self.chunk_mib

    def card_free(self):
        return max(0, self.total - self.fixed_total() - self.cache_mib() - self.desk_mib)

    def max_chunks_allowed(self):
        if self.policy["sleep_level"] >= 1:
            return 0
        cap = self.policy["max_chunks"]
        return 10**6 if cap is None else cap

    def target_mib(self):
        """VRAMLOGIC2: the working target: the floor (reserve) plus the buffer; trimming starts below it, growth above it + hysteresis."""
        return self.policy["reserve_mib"] + (0 if self.legacy_api else self.policy.get("buffer_mib", 0))

    def chunk_budget(self):
        """How many chunks fit above target + hysteresis right now (the chunk budget, never negative)."""
        room = self.card_free() - self.target_mib() - self.policy["hysteresis_mib"]
        return max(0, room // self.chunk_mib)

    # --- time -----------------------------------------------------------------------------------------------------
    def tick(self, seconds):
        step = POLL_SECONDS
        t = 0.0
        while t < seconds - 1e-9:
            dt = min(step, seconds - t)
            self._poll(dt)
            t += dt

    def _poll(self, dt):
        self.uptime += dt
        free = self.card_free()
        if self._window_min is None or free < self._window_min:
            self._window_min = free
        if self.state == "WARMUP":
            self.warmup_left -= dt
            if self.warmup_left <= 0:
                self.state = "FULL"
                self.ready = True
                self.pool_frozen = True
                self.reason = "warmup done: pool frozen"
            return
        if self.state == "PAUSED":
            return
        self._restore_simulated_desk()
        free = self.card_free()
        reserve = self.policy["reserve_mib"]
        if free < reserve:
            if self._below_since is None:
                self._below_since = 0.0
                self._episode_counted = False
            self._below_since += dt
            self.counters["free_below_reserve_max_s"] = max(self.counters["free_below_reserve_max_s"], round(self._below_since, 3))
            # an episode counts once, when it outlives a single poll (API.md: events are "held longer than one poll")
            if self._below_since > POLL_SECONDS + 1e-9 and not self._episode_counted:
                self.counters["free_below_reserve_events"] += 1
                self._episode_counted = True
        else:
            if self._below_since is not None:
                for e in reversed(self.journal):
                    if e["kind"] == "trim" and e["recovered_s"] is None:
                        e["recovered_s"] = round(self._below_since, 2)
                    else:
                        break
            self._below_since = None
        cap = self.max_chunks_allowed() if self.policy["sleep_level"] >= 1 else self.policy["max_chunks"]
        if cap is not None and self.chunks > cap:
            # policy cap (game mode uses 0): give the chunks back at once, never grow above the cap
            self.counters["chunks_dropped"] += self.chunks - cap
            self._jadd("trim", "policy_cap", self.chunks, cap)
            self.chunks = cap
            self.counters["trims"] += 1
            self.state = "TRIM"
            self.reason = f"max_chunks={cap} from policy"
            self._grow_clock = 0.0
            free = self.card_free()
        target = self.target_mib()
        if free < target and self.chunks > self.policy["min_chunks"]:
            # VRAMLOGIC2: the whole deficit to the target in one resize
            need_chunks = -(-(target - free) // self.chunk_mib)
            drop = min(need_chunks, self.chunks - self.policy["min_chunks"])
            self._jadd("trim", f"short_{target - free}mib", self.chunks, self.chunks - drop)
            self.chunks -= drop
            self.counters["trims"] += 1
            self.counters["chunks_dropped"] += drop
            self.state = "TRIM"
            self.reason = f"desk_pressure need={target - free}"
            self._grow_clock = 0.0
            free = self.card_free()
        if free < target and self.chunks <= self.policy["min_chunks"]:
            self.state = "MIN"
            self.reason = f"min_chunks={self.policy['min_chunks']} reached, target still low"
            # the pause rules look at the floor (reserve), not at the target
            self._floor_polls = self._floor_polls + 1 if free < self.policy["pause_floor_mib"] else 0
            if reserve - free > STALL_SLACK_MIB:
                self._stall_since = self.uptime if self._stall_since is None else self._stall_since   # as the server: from the first poll that saw it
            else:
                self._stall_since = None
            after = self.policy.get("pause_after_ms", 0) / 1000.0
            if self._floor_polls >= 2:
                self._pause("pause_floor")
            elif after > 0 and self._stall_since is not None and self.uptime - self._stall_since >= after - 1e-9:
                self._pause("short_stall")
            return
        self._floor_polls = 0
        self._stall_since = None
        if self.state == "MIN" and free >= target:
            self.state = "FULL"
            self.reason = "target restored at minimum chunks"
        # grow in steps, only above target + hysteresis + one chunk, held for grow_seconds
        threshold = target + self.policy["hysteresis_mib"] + self.chunk_mib
        if free >= threshold and self.chunks < self.max_chunks_allowed() and self.state != "TRIM":
            self._grow_clock += dt
            if self._grow_clock >= self.policy["grow_seconds"]:
                self._grow_clock = 0.0
                if self.chunk_budget() >= 1:
                    self._jadd("grow", "room", self.chunks, self.chunks + 1)
                    self.chunks += 1
                    self.counters["grows"] += 1
                    self.counters["chunks_grown"] += 1
                    self.state = "GROW"
                    self.reason = "grew one chunk above reserve+hysteresis"
        else:
            self._grow_clock = 0.0
        if self.state in ("TRIM", "GROW") and self.chunks >= self.max_chunks_allowed():
            self.state = "FULL"

    def _jadd(self, kind, reason, before=0, after=0, note=""):
        self.journal.append({"id": len(self.journal) + 1, "t": 1761390000.0 + self.uptime, "kind": kind, "reason": reason, "state": self.state,
                             "chunks_before": before, "chunks_after": after, "mib": (after - before) * self.chunk_mib,
                             "free_mib": self.card_free(), "recovered_s": None, "note": note})

    def journal_json(self, since=0, limit=200):
        events = [e for e in self.journal if e["id"] > since][-limit:]
        trims = [e for e in self.journal if e["kind"] == "trim"]
        done = [e["recovered_s"] for e in trims if e["recovered_s"] is not None]
        return {"events": copy.deepcopy(events), "last_id": len(self.journal),
                "totals": {"yields": len(trims), "yielded_mib": sum(-e["mib"] for e in trims), "recovered": len(done),
                           "recovered_mean_s": round(sum(done) / len(done), 2) if done else 0.0, "recovered_max_s": max(done, default=0.0),
                           "pauses": self.counters["pauses"], "sleeps": sum(1 for e in self.journal if e["kind"] == "sleep")},
                "human_need": self.human_need, "uptime_s": round(self.uptime, 1)}

    def _pause(self, reason):
        self.state = "PAUSING"
        self.counters["pauses"] += 1
        self.reason = f"pause:{reason}"
        self._jadd("pause", reason)
        path = f"{self.checkpoint_dir}/slot-pause-{self.counters['pauses']}.bin"
        name = path.rsplit("/", 1)[-1]
        message = json.dumps({"checkpoint": name, "slot": 0, "reason": reason, "n_past": 0, "generated_tokens": 0, "saved": True})
        self.frames.append({"error": {"code": 503, "type": "vram_pause", "message": message}})
        self.state = "PAUSED"
        self.exit_code = 75

    # --- HTTP-shaped surface (what the gateway sees) --------------------------------------------------------------
    def snapshot(self):
        """GET /vram. Resets the min-window like the real endpoint."""
        free = self.card_free()
        window_min = self._window_min if self._window_min is not None else free
        self._window_min = None
        return {
            "state": self.state,
            "reason": self.reason,
            "ready": self.ready,
            "pool_frozen": self.pool_frozen,
            "card": {"total_mib": self.total, "free_mib": free, "min_free_mib_window": window_min, "source": "sysfs"},
            "fixed_mib": dict(self.fixed, total=self.fixed_total()),
            "cache": {"enabled": True, "chunk_slots": self.chunk_slots, "chunk_mib": self.chunk_mib, "chunks": self.chunks,
                      "slots": self.chunks * self.chunk_slots, "mib": self.cache_mib(),
                      "target_chunks": self.chunks, "max_chunks": self.policy["max_chunks"]},
            "policy": self.get_policy(),
            "counters": dict(self.counters),
            "warmup": {"done": self.ready, "seconds": self.warmup_seconds, "shapes": ["prefill_ub", "decode_draft", "vision"]},
            "uptime_s": round(self.uptime, 3),
        }

    def get_policy(self):
        policy = copy.deepcopy(self.policy)
        if not self.legacy_api:
            policy["target_mib"] = self.target_mib()     # as the real server reports it
            policy["pause_after_seconds"] = policy.get("pause_after_ms", 2000) / 1000.0
        else:
            for key in ("buffer_mib", "poll_fast_ms", "trim_abort"):
                policy.pop(key, None)
            policy["pause_after_seconds"] = policy.pop("pause_after_ms", 2000) // 1000
        return policy

    def post_policy(self, body):
        """POST /vram/policy: atomic. Returns (http_code, json_body)."""
        before = self.policy
        try:
            self.policy = validate_policy_patch(body, self.policy, self.legacy_api)
        except PolicyError as exc:
            return 400, {"error": str(exc)}
        note = body.get("note", "") if isinstance(body, dict) else ""
        if isinstance(body, dict) and "human_need" in body:
            self.human_need = body["human_need"]
        if self.policy["sleep_level"] != before["sleep_level"]:
            self._jadd("sleep" if self.policy["sleep_level"] > before["sleep_level"] else "wake", f"sleep_l{self.policy['sleep_level']}", note=note)
        if self.policy["reserve_mib"] != before["reserve_mib"]:
            self._jadd("policy", f"reserve={self.policy['reserve_mib']}", note=note)
        return 200, self.get_policy()

    def post_pause(self, body):
        """POST /vram/pause: 202 with state PAUSING; the frame is queued in self.frames."""
        if self.state in ("PAUSING", "PAUSED"):
            return 409, {"error": "already pausing"}
        self._pause(str(body.get("reason", "gateway")))
        return 202, {"state": "PAUSING"}

    def post_chunks(self, body):
        """POST /moe-cache/chunks (bench-only): set the chunk count directly, respecting min/max."""
        target = body.get("chunks")
        if type(target) is not int or target < 0:
            return 400, {"error": "chunks must be a non-negative integer"}
        target = max(self.policy["min_chunks"], min(target, self.max_chunks_allowed()))
        self.chunks = target
        return 200, self.snapshot()["cache"]

    def simulate(self, body):
        """POST /vram/simulate: test hook. Takes free_delta_mib (negative = desk uses more) for `seconds`, then gives it back."""
        delta = int(body.get("free_delta_mib", 0))
        seconds = float(body.get("seconds", 0))
        self._sim_saved = self.desk_mib if self._sim_restore_at is None else self._sim_saved
        self.desk_mib = max(0, self.desk_mib - delta)
        self._sim_restore_at = self.uptime + seconds
        return 200, {"desk_mib": self.desk_mib, "until": self._sim_restore_at}

    def _restore_simulated_desk(self):
        if self._sim_restore_at is not None and self.uptime >= self._sim_restore_at:
            self.desk_mib = self._sim_saved
            self._sim_restore_at = None

    def set_desk(self, mib):
        self.desk_mib = max(0, int(mib))
