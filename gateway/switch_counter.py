"""SHADOW1 day layer: switch counters on live work (MASTER §3 "Выключатель", §5а "Выключатель и счётчик").

A switch is the condition under which a candidate change acts. While a candidate is in the shadow, the gateway counts only that
condition: the candidate itself never acts and its weights are never loaded. One record per pair (independent unit, switch); the
unit of live work is a conversation, identified by a salted hash, never by its text. The share of units where the switch fired at
least once is the ceiling of the harm the candidate could do on work like the recent one.

Two sources feed the counter:
  * the thinking brake of the server (THINK3b): with a shadow candidate configured, the gateway adds `reasoning_brake_shadow: true`
    and the candidate's soft target / ceiling to a chat request and reads `timings.reasoning_budget` from the answer (plain JSON or
    SSE). Nothing else in the request changes, and a request that already carries its own reasoning budget (effort levels) is left
    alone: the server's shadow mode would switch its real ceiling off;
  * the gateway's own conditional interventions, through `record(switch_id, unit, fired, ...)` (the memory note: lifelong_request).

Storage: in RAM for the session whenever `switchCounters` is on; in the experience ledger (kind "switch": numbers and labels only)
only with the `capture` consent, at most two rows per (unit, switch) (first seen, first fired) to keep SSD writes small.

Export (read by SWCOUNT1; do not change silently), one line per unit:
  {"switch": "...", "mode": "shadow|live", "unit": "<hash>", "slice": {"lang": "...", "project": "...", "kind": "..."},
   "fired": true, "first_fire_token": 1234, "requests": 7, "source": "live"}

The fuse (live mode only): at most once a day per switch a decider gets the counters and the budget and answers keep / remove. The
default decider is an exact one-sided Clopper-Pearson bound per foreign slice with the significance level split in advance over the
number of checks in the switch's lifetime. "remove" puts the change back into shadow mode (it stops acting, counting goes on),
writes a gate decision and a morning-summary line. Rules the person set through /teach are never removed: a summary line only.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import threading
import time

BRAKE_SWITCH = "think-brake"
MEMORY_SWITCH = "memory-inject"
SLICE_KEYS = ("lang", "project", "kind")
DAY = 86400.0
DEFAULT_BUDGET = 0.03          # δ of the quality rows (decision 2026-11-12)
DEFAULT_ALPHA = 0.05
DEFAULT_CHECKS = 28            # one check a day over the longest shadow life (4 weeks)
DEFAULT_BRAKE = {"soft": 2048, "ceiling": 8192, "margin": 0.1}
_OWN_BUDGET_FIELDS = ("reasoning_budget_tokens", "thinking_budget_tokens", "reasoning_budget_soft_tokens",
                      "reasoning_brake_shadow", "think_brake_margin", "reasoning_budget_grace_tokens")
_DATA_LINE = re.compile(rb"\{.*\}")
_LOG = []


def _log(text):
    """Swallowed problems stay visible (GET /guardian/switches shows the last ones)."""
    _LOG.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}")
    del _LOG[:-20]


# ----------------------------------------------------------------------------------------------- statistics (no scipy here)

def _binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p), in log space (n up to many thousands)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k, n + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def clopper_pearson_lower(k, n, alpha):
    """Exact one-sided lower confidence bound of a binomial share at level 1 - alpha (0 when k = 0)."""
    if n <= 0 or k <= 0:
        return 0.0
    lo, hi = 0.0, k / n
    for _ in range(100):
        mid = (lo + hi) / 2
        if _binom_sf(k, n, mid) > alpha:
            hi = mid
        else:
            lo = mid
    return lo


def clopper_pearson_upper(k, n, alpha):
    """Exact one-sided upper bound (the 'row closed by the counter' test: upper bound <= δ)."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return 1.0 - clopper_pearson_lower(n - k, n, alpha)


def default_decider(counters, budget):
    """counters: {"slices": [{"slice": {...}, "units": n, "fired": k, "foreign": bool}], "alpha": per-check level}.
    'remove' when on some foreign slice the share is proven above the budget."""
    alpha = counters["alpha"]
    for row in counters["slices"]:
        if not row["foreign"]:
            continue
        if clopper_pearson_lower(row["fired"], row["units"], alpha) > budget:
            return {"action": "remove", "slice": row["slice"], "units": row["units"], "fired": row["fired"],
                    "lower_bound": clopper_pearson_lower(row["fired"], row["units"], alpha)}
    return {"action": "keep"}


def close_row(units, budget=DEFAULT_BUDGET, alpha=DEFAULT_ALPHA, checks=DEFAULT_CHECKS, look=1):
    """Close an exam row when the family-level union firing rate is below its shared budget.

    ``units`` is an iterable of (unit_id, fired) pairs after OR-ing all counter switches for
    each independent family.  Alpha is spent in advance across all planned looks.
    """
    k = max(1, int(checks))
    look = int(look)
    if look < 1 or look > k:
        raise ValueError("look must be between 1 and checks")
    merged = {}
    for unit, fired in units:
        merged[str(unit)] = merged.get(str(unit), False) or bool(fired)
    n = len(merged)
    fired = sum(merged.values())
    upper = clopper_pearson_upper(fired, n, float(alpha) / k)
    return {"closed": n > 0 and upper <= float(budget), "units": n, "fired": fired,
            "upper_bound": upper, "budget": float(budget), "alpha_per_look": float(alpha) / k,
            "checks": k, "look": look}


# ----------------------------------------------------------------------------------------------- units and slices

def unit_id(obj, session, salt):
    """The conversation: the client's session header, else the first user message with the system messages before it (the same
    root lifelong_request uses). Salted hash: the text never leaves this function."""
    base = ""
    if session:
        base = "s:" + str(session)
    else:
        messages = obj.get("messages") if isinstance(obj, dict) else None
        if isinstance(messages, list):
            users = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"]
            cut = messages[:users[0] + 1] if users else messages[:1]
            base = "m:" + json.dumps(cut, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            base += "\0" + str(obj.get("model", ""))
    return hashlib.sha256((salt + "\0" + base).encode("utf-8", "surrogatepass")).hexdigest()[:24]


def _label(value):
    if not isinstance(value, str):
        return ""
    value = value.strip()[:40]
    return value if re.fullmatch(r"[A-Za-z0-9_.:\-]*", value) else hashlib.sha256(value.encode()).hexdigest()[:12]


def request_slice(obj, project="", kind=""):
    """Language of the request (same rule as the language guard), plus the client's project and task-kind labels if it sends them.
    Labels are short identifiers; anything else is hashed, so free text never lands in the counter."""
    lang = ""
    try:
        import lang_guard
        lang = {"Russian": "ru", "Kazakh": "kk", "mixed": "mixed"}.get(lang_guard.detect_language(obj.get("messages")), "")
    except Exception:
        lang = ""
    if not lang:
        lang = "en" if isinstance(obj.get("messages"), list) and obj["messages"] else ""
    return {"lang": lang, "project": _label(project), "kind": _label(kind)}


def _slice_key(s):
    return tuple(s.get(k, "") for k in SLICE_KEYS)


# ----------------------------------------------------------------------------------------------- the counter

class SwitchCounter:
    MAX_UNITS = 200000                # RAM cap: the oldest units are dropped (their ledger rows, if any, stay)

    def __init__(self, ledger_factory=None, clock=time.time, decider=default_decider, salt=None, ledger_owned=False):
        self.lock = threading.Lock()
        self.fuse_lock = threading.Lock()
        self.ledger_lock = threading.Lock()
        self.ledger_owned = ledger_owned  # the factory opens a fresh ledger per call (sqlite connections are per thread); close it
        self.reported = set()
        self.skipped = {}                 # requests the brake candidate left alone, by reason
        self.registry = {}            # switch -> {"mode", "budget", "alpha", "checks", "teach", "own", "last_check", "registered"}
        self.units = {}               # (switch, unit) -> record
        self.ledger_factory = ledger_factory
        self.clock = clock
        self.decider = decider
        self.salt = salt or secrets.token_hex(16)
        self.summary = []             # morning-summary lines
        self.decisions = []

    # registry -------------------------------------------------------------------------------------------------------------
    def register(self, switch, mode="shadow", budget=DEFAULT_BUDGET, alpha=DEFAULT_ALPHA, checks=DEFAULT_CHECKS, teach=False,
                 own=None, title=None):
        if mode not in ("shadow", "live"):
            raise ValueError("mode must be shadow or live")
        with self.lock:
            entry = self.registry.setdefault(switch, {"registered": self.clock(), "last_check": None})
            entry.update({"mode": mode, "budget": float(budget), "alpha": float(alpha), "checks": max(1, int(checks)),
                          "teach": bool(teach), "own": dict(own or {}), "title": title or switch})
        return entry

    def mode(self, switch):
        return self.registry.get(switch, {}).get("mode", "shadow")

    # counting -------------------------------------------------------------------------------------------------------------
    def record(self, switch, unit, fired, first_fire_token=None, slice=None, source="live", requests=1):
        """One observed request of `unit` for `switch`. Returns the unit record (a copy)."""
        if switch not in self.registry:
            self.register(switch)
        now = self.clock()
        new_unit = first_fire = False
        with self.lock:
            key = (switch, unit)
            rec = self.units.get(key)
            if rec is None:
                if len(self.units) >= self.MAX_UNITS:
                    for old in sorted(self.units, key=lambda k: self.units[k]["last_seen"])[: self.MAX_UNITS // 10]:
                        del self.units[old]
                rec = {"switch": switch, "unit": unit, "slice": dict(slice or {k: "" for k in SLICE_KEYS}), "fired": False,
                       "first_fire_token": None, "requests": 0, "source": source, "first_seen": now, "last_seen": now}
                self.units[key] = rec
                new_unit = True
            rec["requests"] += int(requests)
            rec["last_seen"] = now
            if fired and not rec["fired"]:
                rec["fired"] = True
                rec["first_fire_token"] = int(first_fire_token) if isinstance(first_fire_token, int) and first_fire_token >= 0 else None
                first_fire = True
            out = dict(rec)
        if new_unit or first_fire:
            self._ledger_write(out)
        return out

    def _with_ledger(self, fn):
        """Run fn(ledger) under the ledger lock; a fresh ledger (sqlite connection) per call when the factory owns them."""
        if self.ledger_factory is None:
            return None
        with self.ledger_lock:
            ledger = None
            try:
                ledger = self.ledger_factory()
                return fn(ledger) if ledger is not None else None
            except Exception as error:      # the counter never breaks a request, but says why it could not write
                _log(f"switch counter: ledger write skipped: {error}")
                return None
            finally:
                if self.ledger_owned and ledger is not None:
                    try:
                        ledger.close()
                    except Exception:
                        pass

    def _ledger_write(self, rec):
        line = self._line(rec)
        self._with_ledger(lambda led: led._append("switch", line, "work", "capture") if led.consent("capture") else None)

    def _line(self, rec):
        return {"switch": rec["switch"], "mode": self.mode(rec["switch"]), "unit": rec["unit"],
                "slice": {k: rec["slice"].get(k, "") for k in SLICE_KEYS}, "fired": bool(rec["fired"]),
                "first_fire_token": rec["first_fire_token"], "requests": int(rec["requests"]), "source": rec["source"]}

    # output ---------------------------------------------------------------------------------------------------------------
    def export_lines(self):
        with self.lock:
            recs = [dict(r) for r in self.units.values()]
        return [json.dumps(self._line(r), ensure_ascii=False, sort_keys=False) for r in sorted(recs, key=lambda r: (r["switch"], r["first_seen"]))]

    def snapshot(self):
        """Per switch and slice: units, fired units, period."""
        with self.lock:
            recs = [dict(r) for r in self.units.values()]
            reg = {k: dict(v) for k, v in self.registry.items()}
        out = {}
        for sw in sorted(set(reg) | {r["switch"] for r in recs}):
            rows = {}
            for r in (x for x in recs if x["switch"] == sw):
                k = _slice_key(r["slice"])
                row = rows.setdefault(k, {"slice": dict(zip(SLICE_KEYS, k)), "units": 0, "fired": 0, "requests": 0,
                                          "from": r["first_seen"], "to": r["last_seen"]})
                row["units"] += 1
                row["fired"] += int(r["fired"])
                row["requests"] += r["requests"]
                row["from"], row["to"] = min(row["from"], r["first_seen"]), max(row["to"], r["last_seen"])
            total = [x for x in recs if x["switch"] == sw]
            info = reg.get(sw, {})
            out[sw] = {"mode": info.get("mode", "shadow"), "budget": info.get("budget"), "teach": info.get("teach", False),
                       "units": len(total), "fired": sum(int(x["fired"]) for x in total),
                       "period": [min((x["first_seen"] for x in total), default=None), max((x["last_seen"] for x in total), default=None)],
                       "slices": sorted(rows.values(), key=lambda r: tuple(r["slice"].values()))}
        return {"switches": out, "summary": list(self.summary)}

    def shared_row(self, switches, slice, budget=DEFAULT_BUDGET, alpha=DEFAULT_ALPHA, checks=DEFAULT_CHECKS, look=1):
        """Evaluate a row's shared firing budget across switches, counting each family once."""
        wanted, target = set(switches), _slice_key(slice)
        with self.lock:
            rows = [(r["unit"], r["fired"]) for r in self.units.values()
                    if r["switch"] in wanted and _slice_key(r["slice"]) == target]
        return close_row(rows, budget, alpha, checks, look)

    # the fuse -------------------------------------------------------------------------------------------------------------
    def check_fuse(self, ledger=None):
        """Run the decider for every live switch whose last check is a day old (at most `checks` checks in its life; the
        significance level is split over those checks and over the foreign slices tested). Returns the decisions taken."""
        if not self.fuse_lock.acquire(blocking=False):
            return []                 # another thread is checking right now
        try:
            now = self.clock()
            taken = []
            for sw, info in list(self.registry.items()):
                if info.get("mode") != "live":
                    continue
                if info.get("last_check") is not None and now - info["last_check"] < DAY:
                    continue
                done = info.get("checks_done", 0)
                if done >= info["checks"]:
                    key = (sw, "checks_spent")
                    if key not in self.reported:
                        self.reported.add(key)
                        self.summary.append(f"«{info['title']}»: запланированные проверки предохранителя закончились; "
                                            f"нужна новая предрегистрация")
                    continue
                info["last_check"] = now
                info["checks_done"] = done + 1
                snap = self.snapshot()["switches"].get(sw, {"slices": []})
                own = info.get("own") or {}
                slices = [{"slice": r["slice"], "units": r["units"], "fired": r["fired"],
                           "foreign": not own or any(r["slice"].get(k) != v for k, v in own.items())} for r in snap["slices"]]
                n_foreign = max(1, sum(1 for r in slices if r["foreign"]))
                counters = {"switch": sw, "slices": slices, "alpha": info["alpha"] / info["checks"] / n_foreign}
                verdict = self.decider(counters, info["budget"])
                if verdict.get("action") != "remove":
                    continue
                s = verdict.get("slice") or {}
                where = ", ".join(f"{k}={v}" for k, v in s.items() if v) or "все срезы"
                share = f"{verdict.get('fired')}/{verdict.get('units')}"
                if info.get("teach"):
                    line = (f"Правило «{info['title']}» (из /teach) срабатывает в чужих задачах ({where}: {share}); "
                            f"оставляю как есть — решаете вы")
                    decision = {**verdict, "switch": sw, "action": "report_only", "reason": "teach rule"}
                    key = (sw, "teach", _slice_key(s))
                    if key in self.reported:
                        continue          # said once; the person decides
                    self.reported.add(key)
                else:
                    info["mode"] = "shadow"
                    line = (f"Навык «{info['title']}» стал срабатывать в чужих задачах ({where}: {share}); "
                            f"поставила на паузу и перепроверяю")
                    decision = {**verdict, "switch": sw, "action": "live_to_shadow"}
                self.summary.append(line)
                del self.summary[:-50]
                self.decisions.append(decision)
                taken.append(decision)
                if ledger is not None:
                    try:
                        ledger.add_gate_decision("switch_fuse", decision)
                    except Exception as error:
                        _log(f"switch counter: gate decision not written: {error}")
                else:
                    self._with_ledger(lambda led: led.add_gate_decision("switch_fuse", decision))
            return taken
        finally:
            self.fuse_lock.release()


# ----------------------------------------------------------------------------------------------- gateway glue

COUNTER = None
_COUNTER_LOCK = threading.Lock()


def _ledger_salt(ledger_factory):
    """A salt that survives restarts when the ledger exists (derived from its own secret salt, never stored in the clear)."""
    if ledger_factory is None:
        return None
    try:
        led = ledger_factory()
        if led is None:
            return None
        try:
            row = led.db.execute("SELECT value FROM meta WHERE key='salt'").fetchone()
            return hashlib.sha256(b"switch-counter\0" + row[0].encode()).hexdigest()[:32] if row else None
        finally:
            led.close()
    except Exception as error:
        _log(f"switch counter: no ledger salt: {error}")
        return None


def counter(ledger_factory=None):
    """The process's counter. `ledger_factory` must open a fresh ledger per call (it is closed after use)."""
    global COUNTER
    with _COUNTER_LOCK:
        if COUNTER is None:
            COUNTER = SwitchCounter(ledger_factory=ledger_factory, salt=_ledger_salt(ledger_factory), ledger_owned=True)
        elif COUNTER.ledger_factory is None and ledger_factory is not None:
            COUNTER.ledger_factory = ledger_factory     # a GET came first: the counter learns the ledger at the first request
    return COUNTER


def _brake_candidate(G):
    """The configured brake candidate, validated; None when absent or invalid (an invalid one is logged, never raised)."""
    cand = G.get("shadowCandidates")
    if cand is True:
        return dict(DEFAULT_BRAKE)
    if isinstance(cand, dict) and isinstance(cand.get(BRAKE_SWITCH), dict):
        merged = dict(DEFAULT_BRAKE)
        merged.update({k: v for k, v in cand[BRAKE_SWITCH].items() if k in DEFAULT_BRAKE})
        try:
            soft, ceiling, margin = int(merged["soft"]), int(merged["ceiling"]), float(merged["margin"])
        except (TypeError, ValueError):
            _log("switch counter: shadowCandidates.think-brake needs integer soft/ceiling and a numeric margin; ignored")
            return None
        if not (0 <= soft <= ceiling) or margin < 0:
            _log("switch counter: shadowCandidates.think-brake needs 0 <= soft <= ceiling and margin >= 0; ignored")
            return None
        return {"soft": soft, "ceiling": ceiling, "margin": margin}
    return None


def _skip_reason(obj, path, G, settings):
    """Requests whose real behaviour the server's shadow mode would change: never add the shadow fields to them."""
    if any(k in obj for k in _OWN_BUDGET_FIELDS):
        return "own budget"              # the client's or an effort level's budget: shadow mode would turn its ceiling off
    if G.get("effortBudgets"):
        return "effort budgets"          # apply_think adds the level's budget after us and would take ours for the client's
    if obj.get("stream") is True and settings:
        try:
            import lang_guard
            if lang_guard.guard_for(obj, path, settings):
                return "language guard"  # its rewrite request sets a thinking budget of 0, which shadow mode would not force
        except Exception:
            return "language guard"
    return None


def prepare(path, body, ctx, G, ledger_factory=None, settings=None):
    """Before the request goes upstream. With switchCounters off: returns `body` itself (the same bytes) and touches nothing.
    With it on: notes the unit and slice in the thread-local `ctx`; with a brake shadow candidate also adds the shadow fields.
    Never raises: any problem leaves the request as it was."""
    ctx.switch = None
    if G.get("switchCounters") is not True or not body or not path.endswith("/chat/completions"):
        return body
    try:
        obj = json.loads(body)
        if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
            return body
        c = counter(ledger_factory)
        unit = unit_id(obj, getattr(ctx, "session", ""), c.salt)
        sl = request_slice(obj, getattr(ctx, "project", ""), getattr(ctx, "task_kind", ""))
        state = {"unit": unit, "slice": sl, "brake": False, "done": False, "tail": b""}
        ctx.switch = state
        c.check_fuse()                   # cheap; acts at most once a day per live switch
        cand = _brake_candidate(G)
        if cand is None:
            return body
        if BRAKE_SWITCH not in c.registry:
            c.register(BRAKE_SWITCH, mode="shadow", title="тормоз мышления")
        why = _skip_reason(obj, path, G, settings)
        if why:
            c.skipped[why] = c.skipped.get(why, 0) + 1
            return body
        obj["reasoning_brake_shadow"] = True
        obj["reasoning_budget_soft_tokens"] = cand["soft"]
        obj["reasoning_budget_tokens"] = cand["ceiling"]
        obj["think_brake_margin"] = cand["margin"]
        out = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()
        state["brake"] = True
        # the request counts now, so an answer that never finishes (aborted, disconnected) is still a unit that did not report
        c.record(BRAKE_SWITCH, unit, False, slice=sl)
        return out
    except Exception as error:
        _log(f"switch counter: request left unchanged: {error}")
        ctx.switch = None
        return body


def memory_report(ctx, G):
    """A callback for lifelong_request: fired = a memory note was appended to this request."""
    state = getattr(ctx, "switch", None)
    if G.get("switchCounters") is not True or not state:
        return None
    c = counter()
    if MEMORY_SWITCH not in c.registry:
        # the memory note acts whenever it is on (live); its records are the person's explicit requests, so the fuse only reports
        c.register(MEMORY_SWITCH, mode="live", teach=True, title="подстановка памяти")

    def report(fired):
        try:
            c.record(MEMORY_SWITCH, state["unit"], bool(fired), slice=state["slice"])
        except Exception as error:
            _log(f"switch counter: memory record skipped: {error}")
    return report


def _find_budget(raw):
    """The last timings.reasoning_budget object in complete lines of a JSON body or SSE text; None if absent."""
    found = None
    for line in raw.splitlines():
        if b"reasoning_budget" not in line:
            continue
        m = _DATA_LINE.search(line)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except (ValueError, UnicodeDecodeError, RecursionError):
            continue
        t = obj.get("timings") if isinstance(obj, dict) else None
        if isinstance(t, dict) and isinstance(t.get("reasoning_budget"), dict):
            found = (t["reasoning_budget"], t.get("reasoning_tokens"))
    return found


MAX_LINE = 1 << 20


def observe(ctx, value):
    """Every relayed piece of the answer (SSE chunk or whole JSON). Records the brake switch once per request.
    Only the unfinished last line is kept between chunks (a line longer than 1 MB is dropped). Never raises."""
    state = getattr(ctx, "switch", None)
    if not state or not state.get("brake") or state.get("done"):
        return
    try:
        raw = value if isinstance(value, (bytes, bytearray)) else (value.encode() if isinstance(value, str) else json.dumps(value).encode())
        buf = state["tail"] + bytes(raw)
        cut = buf.rfind(b"\n")
        complete, rest = (buf[:cut + 1], buf[cut + 1:]) if cut >= 0 else (b"", buf)
        state["tail"] = rest if len(rest) <= MAX_LINE else b""
        hit = _find_budget(complete) if b"reasoning_budget" in complete else None
        if hit is None and b"reasoning_budget" in rest:
            hit = _find_budget(rest)                     # a whole JSON body often arrives without a line end
        if hit is None:
            return
        rb, _tokens = hit
        state["done"] = True
        state["tail"] = b""
        fired = bool(rb.get("brake_fired") or rb.get("budget_hit"))
        first = rb.get("first_fire_token")
        counter().record(BRAKE_SWITCH, state["unit"], fired, first if isinstance(first, int) else None, slice=state["slice"],
                         requests=0)                     # the request itself was counted in prepare()
    except Exception as error:
        _log(f"switch counter: answer not read: {error}")


def guardian(handler, path, G):
    """GET /guardian/switches (JSON) and /guardian/switches.jsonl (the export, one line per unit). Read only."""
    c = counter()
    if path.endswith(".jsonl"):
        lines = c.export_lines()
        data = ("\n".join(lines) + "\n").encode() if lines else b""
        handler.send_response(200)
        handler.send_header("Content-Type", "application/x-ndjson")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
        return None
    snap = c.snapshot()
    snap["skipped_requests"] = dict(c.skipped)
    snap["problems"] = list(_LOG)
    return handler.send_json(200, snap)
