"""SHADOW1 part A: the switch counter (gateway/switch_counter.py) and its gateway hookup.

Unit tests run anywhere. The gateway tests start the real gateway against the shura-qa fake llama-server (no GPU, no model),
wrapped so that the fake answers like a THINK3b server: `timings.reasoning_budget` when the request asks for the shadow brake;
the brake "would fire" when the last user message contains the word FIRE. They are skipped when shura-qa is not installed.
"""
import http.client
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gateway"))

import experience_ledger  # noqa: E402
import lifelong_request  # noqa: E402
import switch_counter as S  # noqa: E402

QA = Path(os.environ.get("SHURA_QA_ROOT", str(Path.home() / "Projects" / "shura-qa")))
HARNESS = QA / "harness" / "gateway_harness.py"
FAKE_DIR = QA / "fake_llama"

SECRET = "Секретный текст запроса 7731"


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class StatsTests(unittest.TestCase):
    def test_clopper_pearson_matches_master_rule(self):
        # MASTER §5а: for δ 0.03 a row is closed by 0 firings in 100 units, 1 in 157 or 2 in 208 (upper 95 % bound <= δ)
        self.assertLessEqual(S.clopper_pearson_upper(0, 100, 0.05), 0.03)
        self.assertLessEqual(S.clopper_pearson_upper(1, 157, 0.05), 0.0301)
        self.assertLessEqual(S.clopper_pearson_upper(2, 208, 0.05), 0.0301)
        self.assertGreater(S.clopper_pearson_upper(0, 90, 0.05), 0.03)
        self.assertAlmostEqual(S.clopper_pearson_upper(0, 100, 0.05), 1 - 0.05 ** (1 / 100), places=6)
        # lower bound: k/n inside, monotone in k
        lo = S.clopper_pearson_lower(30, 200, 0.05 / 28)
        self.assertTrue(0.05 < lo < 0.15)
        self.assertEqual(S.clopper_pearson_lower(0, 50, 0.05), 0.0)
        self.assertLess(S.clopper_pearson_lower(5, 200, 0.05), S.clopper_pearson_lower(9, 200, 0.05))

    def test_shared_row_counts_family_once_and_spends_alpha_for_peeks(self):
        one_look = S.close_row([(str(i), False) for i in range(100)], checks=1)
        repeated = S.close_row([(str(i), False) for i in range(100)], checks=28)
        self.assertTrue(one_look["closed"])
        # Spending alpha over 28 looks is intentionally more conservative than the
        # MASTER's single-look 0/100 threshold; the report records this tradeoff.
        self.assertFalse(repeated["closed"])
        self.assertEqual(repeated["alpha_per_look"], 0.05 / 28)
        self.assertFalse(S.close_row([(str(i), i < 10) for i in range(100)], checks=1)["closed"])
        self.assertEqual(S.close_row([("family", False), ("family", True)], checks=1)["fired"], 1)


class CounterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="swc-", dir="/dev/shm" if os.path.isdir("/dev/shm") else None))
        self.clock = Clock()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_planted_share_and_no_text(self):
        c = S.SwitchCounter(clock=self.clock, salt="x")
        fired_units = {5, 17, 33}                                 # 3 conversations of 40
        for u in range(40):
            obj = {"messages": [{"role": "user", "content": f"{SECRET} {u}"}]}
            unit = S.unit_id(obj, "", c.salt)
            for r in range(3):                                    # three requests per conversation
                c.record(S.BRAKE_SWITCH, unit, u in fired_units and r == 1, 1234 if r == 1 else None,
                         slice=S.request_slice(obj))
        snap = c.snapshot()["switches"][S.BRAKE_SWITCH]
        self.assertEqual((snap["units"], snap["fired"]), (40, 3))
        self.assertEqual(snap["slices"][0]["slice"]["lang"], "ru")
        lines = c.export_lines()
        self.assertEqual(len(lines), 40)
        rows = [json.loads(x) for x in lines]
        self.assertEqual(sum(r["fired"] for r in rows), 3)
        self.assertEqual(set(rows[0]), {"switch", "mode", "unit", "slice", "fired", "first_fire_token", "requests", "source"})
        self.assertTrue(all(r["requests"] == 3 for r in rows))
        self.assertEqual(sorted(r["first_fire_token"] for r in rows if r["fired"]), [1234, 1234, 1234])
        blob = "\n".join(lines) + json.dumps(c.snapshot(), ensure_ascii=False)
        self.assertNotIn("Секрет", blob)
        self.assertNotIn("7731", blob)

    def test_unit_is_the_conversation(self):
        a = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q1"}]}
        b = {"messages": a["messages"] + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "q2"}]}
        self.assertEqual(S.unit_id(a, "", "salt"), S.unit_id(b, "", "salt"))
        self.assertNotEqual(S.unit_id(a, "", "salt"), S.unit_id(a, "", "other"))
        self.assertNotEqual(S.unit_id(a, "sess-1", "salt"), S.unit_id(a, "sess-2", "salt"))

    def test_free_text_labels_are_hashed(self):
        sl = S.request_slice({"messages": [{"role": "user", "content": "hello there, write code"}]}, project="my secret project",
                             kind="bugfix")
        self.assertEqual(sl["kind"], "bugfix")
        self.assertNotIn("secret", sl["project"])
        self.assertEqual(len(sl["project"]), 12)

    def test_ledger_only_with_capture_and_at_most_two_rows(self):
        led = experience_ledger.Ledger(self.tmp / "exp.sqlite")
        c = S.SwitchCounter(ledger_factory=lambda: led, clock=self.clock, salt="x")
        c.record(S.BRAKE_SWITCH, "u1", False)
        self.assertEqual([r for r in led.list() if r["kind"] == "switch"], [])      # no consent: nothing written
        led.set_consent("capture", True)
        for i in range(10):
            c.record(S.BRAKE_SWITCH, "u2", i == 4, 77 if i == 4 else None, slice={"lang": "ru", "project": "", "kind": ""})
        rows = [r for r in led.list() if r["kind"] == "switch"]
        self.assertEqual(len(rows), 2)                                              # first seen, first fired
        self.assertEqual(rows[-1]["payload"]["fired"], True)
        self.assertEqual(rows[-1]["payload"]["first_fire_token"], 77)
        for r in rows:
            self.assertEqual(set(r["payload"]), {"switch", "mode", "unit", "slice", "fired", "first_fire_token", "requests", "source"})
        led.close()

    def _live(self, c, fired, units, teach=False):
        c.register("skill-x", mode="live", teach=teach, own={"project": "mine"}, title="навык X")
        for u in range(units):
            c.record("skill-x", f"f{u}", u < fired, slice={"lang": "en", "project": "other", "kind": ""})
        for u in range(50):                                       # its own slice fires a lot: that is the point of the change
            c.record("skill-x", f"o{u}", True, slice={"lang": "en", "project": "mine", "kind": ""})

    def test_fuse_removes_when_share_proven_above_budget(self):
        led = experience_ledger.Ledger(self.tmp / "exp.sqlite")
        c = S.SwitchCounter(ledger_factory=lambda: led, clock=self.clock, salt="x")
        self._live(c, fired=30, units=200)
        taken = c.check_fuse()
        self.assertEqual(c.mode("skill-x"), "shadow")
        self.assertEqual(taken[0]["action"], "live_to_shadow")
        self.assertIn("поставила на паузу и перепроверяю", c.summary[-1])
        self.assertTrue(any(r["kind"] == "gate_decision" for r in led.list()))
        led.close()

    def test_fuse_keeps_within_budget_and_runs_once_a_day(self):
        c = S.SwitchCounter(clock=self.clock, salt="x")
        self._live(c, fired=2, units=200)                         # 1 % on the foreign slice, budget 3 %
        self.assertEqual(c.check_fuse(), [])
        self.assertEqual(c.mode("skill-x"), "live")
        for u in range(200, 260):                                 # now the share jumps, but the day is not over
            c.record("skill-x", f"f{u}", True, slice={"lang": "en", "project": "other", "kind": ""})
        self.assertEqual(c.check_fuse(), [])
        self.assertEqual(c.mode("skill-x"), "live")
        self.clock.t += S.DAY + 1
        self.assertEqual(c.check_fuse()[0]["action"], "live_to_shadow")

    def test_fuse_never_removes_teach_rules(self):
        c = S.SwitchCounter(clock=self.clock, salt="x")
        self._live(c, fired=60, units=200, teach=True)
        taken = c.check_fuse()
        self.assertEqual(taken[0]["action"], "report_only")
        self.assertEqual(c.mode("skill-x"), "live")
        self.assertIn("решаете вы", c.summary[-1])

    def test_shared_row_unions_switches_by_family(self):
        c = S.SwitchCounter(clock=self.clock, salt="x")
        sl = {"lang": "en", "project": "other", "kind": ""}
        for i in range(100):
            c.record("a", str(i), i == 0, slice=sl)
            c.record("b", str(i), i == 1, slice=sl)
        row = c.shared_row(["a", "b"], sl, checks=1)
        self.assertEqual((row["units"], row["fired"]), (100, 2))

    def test_fuse_ignores_shadow_switches_and_custom_decider(self):
        calls = []
        c = S.SwitchCounter(clock=self.clock, salt="x", decider=lambda counters, budget: calls.append((counters, budget)) or {"action": "keep"})
        c.register("shadowy", mode="shadow")
        c.record("shadowy", "u", True)
        c.register("live1", mode="live", budget=0.05)
        c.record("live1", "u", True)
        c.check_fuse()
        self.assertEqual(len(calls), 1)
        counters, budget = calls[0]
        self.assertEqual((counters["switch"], budget), ("live1", 0.05))
        self.assertAlmostEqual(counters["alpha"], 0.05 / 28)

    def test_prepare_is_identity_when_off(self):
        class Ctx: pass
        ctx = Ctx(); ctx.session = ""
        body = json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()
        self.assertIs(S.prepare("/v1/chat/completions", body, ctx, {"switchCounters": False, "shadowCandidates": True}), body)
        self.assertIs(S.prepare("/v1/chat/completions", body, ctx, {"switchCounters": True}), body)     # counting only
        out = S.prepare("/v1/chat/completions", body, ctx, {"switchCounters": True, "shadowCandidates": {"think-brake": {"soft": 100}}})
        o = json.loads(out)
        self.assertEqual({k: o[k] for k in o if k != "messages"}, {"reasoning_brake_shadow": True, "reasoning_budget_soft_tokens": 100,
                                                                    "reasoning_budget_tokens": 8192, "think_brake_margin": 0.1})
        own = json.dumps({"messages": [{"role": "user", "content": "x"}], "reasoning_budget_tokens": 900}).encode()
        self.assertIs(S.prepare("/v1/chat/completions", own, ctx, {"switchCounters": True, "shadowCandidates": True}), own)

    def test_observe_parses_stream_split_across_chunks(self):
        S.COUNTER = S.SwitchCounter(clock=self.clock, salt="x")
        class Ctx: pass
        ctx = Ctx(); ctx.session = "s1"
        body = json.dumps({"messages": [{"role": "user", "content": "x"}], "stream": True}).encode()
        S.prepare("/v1/chat/completions", body, ctx, {"switchCounters": True, "shadowCandidates": True})
        last = ('data: {"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"predicted_n":9,"reasoning_tokens":7,'
                '"reasoning_budget":{"brake_fired":false,"budget_hit":true,"first_fire_token":5,"shadow":true}}}\n\n').encode()
        S.observe(ctx, b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n')
        S.observe(ctx, last[:40])
        S.observe(ctx, last[40:])
        S.observe(ctx, b"data: [DONE]\n\n")
        rows = [json.loads(x) for x in S.COUNTER.export_lines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["fired"], rows[0]["first_fire_token"], rows[0]["requests"]), (True, 5, 1))
        S.COUNTER = None


class ReviewFixTests(unittest.TestCase):
    """Cases found by the independent review (SHADOW1)."""

    class Ctx:
        session = ""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="swc-rev-", dir="/dev/shm" if os.path.isdir("/dev/shm") else None))
        S.COUNTER = None

    def tearDown(self):
        S.COUNTER = None
        shutil.rmtree(self.tmp, ignore_errors=True)

    def body(self, **kw):
        return json.dumps({"messages": [{"role": "user", "content": "x"}], **kw}).encode()

    def test_effort_budgets_and_own_budgets_are_left_alone(self):
        G = {"switchCounters": True, "shadowCandidates": True, "effortBudgets": True}
        b = self.body()
        self.assertIs(S.prepare("/v1/chat/completions", b, self.Ctx(), G), b)     # apply_think adds the level's budget later
        self.assertEqual(S.counter().skipped.get("effort budgets"), 1)
        b2 = self.body(thinking_budget_tokens=10)
        self.assertIs(S.prepare("/v1/chat/completions", b2, self.Ctx(), {"switchCounters": True, "shadowCandidates": True}), b2)

    def test_bad_candidate_config_never_breaks_a_request(self):
        for cand in ({"think-brake": {"soft": None}}, {"think-brake": {"soft": 9000, "ceiling": 100}}, {"think-brake": {"margin": "x"}}):
            b = self.body()
            self.assertIs(S.prepare("/v1/chat/completions", b, self.Ctx(), {"switchCounters": True, "shadowCandidates": cand}), b)
        self.assertIs(S.prepare("/v1/chat/completions", b"{not json", self.Ctx(), {"switchCounters": True, "shadowCandidates": True}),
                      b"{not json")

    def test_aborted_answer_still_counts_the_request(self):
        ctx = self.Ctx()
        S.prepare("/v1/chat/completions", self.body(stream=True), ctx, {"switchCounters": True, "shadowCandidates": True})
        S.observe(ctx, b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n')        # the stream stops here
        rows = [json.loads(x) for x in S.counter().export_lines()]
        self.assertEqual((len(rows), rows[0]["requests"], rows[0]["fired"]), (1, 1, False))

    def test_non_stream_body_without_line_end(self):
        ctx = self.Ctx()
        S.prepare("/v1/chat/completions", self.body(), ctx, {"switchCounters": True, "shadowCandidates": True})
        S.observe(ctx, b'{"choices":[],"timings":{"reasoning_budget":{"brake_fired":true,"budget_hit":false,"first_fire_token":9,"shadow":true}}}')
        rows = [json.loads(x) for x in S.counter().export_lines()]
        self.assertEqual((rows[0]["fired"], rows[0]["first_fire_token"], rows[0]["requests"]), (True, 9, 1))

    def test_ledger_writes_from_other_threads(self):
        import threading
        path = self.tmp / "exp.sqlite"
        led = experience_ledger.Ledger(path)
        led.set_consent("capture", True)
        led.close()
        c = S.SwitchCounter(ledger_factory=lambda: experience_ledger.Ledger(path), ledger_owned=True, salt="x")
        threads = [threading.Thread(target=c.record, args=(S.BRAKE_SWITCH, f"u{i}", i % 2 == 0)) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        led = experience_ledger.Ledger(path)
        rows = [r for r in led.list() if r["kind"] == "switch"]
        led.close()
        self.assertEqual(len(rows), 8)                  # first-seen rows (fired is set on the same first row)
        self.assertEqual(S._LOG[-1:] and "skipped" in S._LOG[-1], False)

    def test_guardian_get_is_read_only_and_fuse_runs_from_requests(self):
        c = S.counter()
        c.register("skill-x", mode="live", own={"project": "mine"}, title="навык X")
        for u in range(200):
            c.record("skill-x", f"f{u}", u < 40, slice={"lang": "en", "project": "other", "kind": ""})

        class H:
            def send_json(self, code, obj):
                self.out = (code, obj)
        h = H()
        S.guardian(h, "/guardian/switches", {})
        self.assertEqual(c.mode("skill-x"), "live")     # reading changes nothing
        S.prepare("/v1/chat/completions", self.body(), self.Ctx(), {"switchCounters": True})
        self.assertEqual(c.mode("skill-x"), "shadow")   # the next request ran the daily fuse


class MemorySwitchTests(unittest.TestCase):
    """The memory note as a switch: lifelong_request reports whether it appended a note (FrozenMemory.apply(report=...))."""

    class Store:
        def __init__(self, hits):
            self.hits = hits

        def search(self, text, limit=3):
            if "WANT" not in text:
                return []
            return [{"score": 10.0, "passport_line": "p", "key": "k", "title": "t", "body": "remembered fact"}][:self.hits]

        def note_context_hits(self, n):
            pass

    def test_report_follows_the_note(self):
        fm = lifelong_request.FrozenMemory()
        got = []
        store = self.Store(1)
        b1 = json.dumps({"messages": [{"role": "user", "content": "plain question"}]}).encode()
        b2 = json.dumps({"messages": [{"role": "user", "content": "WANT the fact"}]}).encode()
        out1 = fm.apply(b1, "s1", lambda: store, report=got.append)
        out2 = fm.apply(b2, "s2", lambda: store, report=got.append)
        self.assertIs(out1, b1)
        self.assertIn("remembered fact", out2.decode())
        self.assertEqual(got, [False, True])
        self.assertIs(fm.apply(b1, "s1", lambda: store), b1)        # without report: unchanged behaviour


# ------------------------------------------------------------------------------------------------- gateway end to end

WRAPPER = r'''#!/usr/bin/env python3
"""The shura-qa fake, answering like a THINK3b server (SHADOW1 test)."""
import json, os, sys, threading
sys.path.insert(0, os.environ["SWC_FAKE_DIR"])
import fake_llama_server as F
_tl = threading.local()
_orig_chat = F.Handler.chat
def chat(self, u, body, cfg):
    _tl.body = body
    return _orig_chat(self, u, body, cfg)
F.Handler.chat = chat
class _J:
    def __getattr__(self, name):
        return getattr(json, name)
    @staticmethod
    def loads(s, *a, **k):
        with open(os.environ["SWC_RAW_LOG"], "ab") as f:
            f.write((s if isinstance(s, bytes) else s.encode()) + b"\n")
        return json.loads(s, *a, **k)
    @staticmethod
    def dumps(obj, *a, **k):
        body = getattr(_tl, "body", None) or {}
        t = obj.get("timings") if isinstance(obj, dict) else None
        if isinstance(t, dict) and "predicted_n" in t and body.get("reasoning_brake_shadow") is True:
            msgs = [m for m in body.get("messages", []) if m.get("role") == "user"]
            fire = bool(msgs) and "FIRE" in str(msgs[-1].get("content"))
            t["reasoning_tokens"] = 300
            t["reasoning_budget"] = {"brake_fired": fire, "budget_hit": False, "first_fire_token": 123 if fire else -1, "shadow": True}
        return json.dumps(obj, *a, **k)
F.json = _J()
sys.exit(F.main(sys.argv[1:]))
'''


@unittest.skipUnless(HARNESS.is_file(), "shura-qa harness is not installed")
class GatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(HARNESS.parent))
        global GatewayHarness
        from gateway_harness import GatewayHarness

    def harness(self, guardian):
        tmp = Path(tempfile.mkdtemp(prefix="swc-gw-", dir="/dev/shm" if os.path.isdir("/dev/shm") else None))
        self.addCleanup(shutil.rmtree, tmp, True)
        wrapper = tmp / "fake_think3b.py"
        wrapper.write_text(WRAPPER)
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
        raw = tmp / "raw.log"
        os.environ["SWC_FAKE_DIR"] = str(FAKE_DIR)
        os.environ["SWC_RAW_LOG"] = str(raw)
        gw = GatewayHarness(behaviour={"n_tokens": 4, "token_delay_ms": 1, "reasoning_tokens": 2}, guardian=guardian)
        m = json.loads((gw.cfg / "model-launch.json").read_text())
        m["exePath"] = str(wrapper)
        (gw.cfg / "model-launch.json").write_text(json.dumps(m))
        gw.raw = raw
        return gw

    @staticmethod
    def post(gw, body, session=None, project=None, stream=False):
        conn = http.client.HTTPConnection("127.0.0.1", gw.gw_port, timeout=60)
        headers = {"Content-Type": "application/json"}
        if session:
            headers["X-Shura-Session"] = session
        if project:
            headers["X-Shura-Project"] = project
        body = dict(body, stream=stream)
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(), headers=headers)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, data

    @staticmethod
    def get(gw, path):
        conn = http.client.HTTPConnection("127.0.0.1", gw.gw_port, timeout=30)
        conn.request("GET", path)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, data

    def chat_bodies(self, gw):
        return [json.loads(x) for x in gw.raw.read_bytes().splitlines() if b'"messages"' in x], \
               [x for x in gw.raw.read_bytes().splitlines() if b'"messages"' in x]

    REQ = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": SECRET}], "temperature": 0.3}

    def test_flags_off_requests_are_byte_identical(self):
        raws = {}
        for name, guardian in (("off", {}), ("count", {"switchCounters": True})):
            with self.harness(guardian) as gw:
                for stream in (False, True):
                    code, _ = self.post(gw, self.REQ, session="s1", stream=stream)
                    self.assertEqual(code, 200)
                raws[name] = self.chat_bodies(gw)[1]
                if name == "off":
                    code, _ = self.get(gw, "/guardian/switches")
                    self.assertEqual(code, 405)          # not readable while the flag is off (the web guard asks for POST)
        self.assertEqual(len(raws["off"]), 2)
        self.assertEqual(raws["off"], raws["count"])     # counting alone never changes a byte of the request

    def test_candidate_adds_only_its_fields(self):
        with self.harness({}) as gw:
            self.post(gw, self.REQ, session="s1")
            base = self.chat_bodies(gw)[0][-1]
        with self.harness({"switchCounters": True, "shadowCandidates": {"think-brake": {"soft": 1000, "ceiling": 4096}}}) as gw:
            self.post(gw, self.REQ, session="s1")
            sent = self.chat_bodies(gw)[0][-1]
            effort = dict(self.REQ, reasoning_budget_tokens=2000)
            self.post(gw, effort, session="s2")
            own = self.chat_bodies(gw)[0][-1]
        added = {k: sent[k] for k in sent if k not in base}
        self.assertEqual(added, {"reasoning_brake_shadow": True, "reasoning_budget_soft_tokens": 1000,
                                 "reasoning_budget_tokens": 4096, "think_brake_margin": 0.1})
        self.assertEqual({k: v for k, v in sent.items() if k not in added}, base)
        self.assertNotIn("reasoning_brake_shadow", own)            # an effort-level request keeps its own (real) budget
        self.assertEqual(own["reasoning_budget_tokens"], 2000)

    def test_planted_share_through_the_gateway(self):
        with self.harness({"switchCounters": True, "shadowCandidates": True}) as gw:
            fired = {3, 19, 36}
            for u in range(40):
                text = f"{SECRET} conversation {u}" + (" FIRE" if u in fired else "")
                for turn in range(2):
                    msgs = [{"role": "user", "content": f"{SECRET} conversation {u} start"}]
                    if turn:
                        msgs += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": text}]
                    code, _ = self.post(gw, {"messages": msgs}, stream=(u % 2 == 0),
                                        project="proj-a" if u < 20 else "proj-b")
                    self.assertEqual(code, 200)
            code, data = self.get(gw, "/guardian/switches")
            self.assertEqual(code, 200)
            sw = json.loads(data)["switches"]["think-brake"]
            self.assertEqual((sw["units"], sw["fired"], sw["mode"]), (40, 3, "shadow"))
            by_project = {r["slice"]["project"]: (r["units"], r["fired"]) for r in sw["slices"]}
            self.assertEqual(by_project, {"proj-a": (20, 2), "proj-b": (20, 1)})
            code, data = self.get(gw, "/guardian/switches.jsonl")
            rows = [json.loads(x) for x in data.decode().splitlines()]
            self.assertEqual(len(rows), 40)
            self.assertEqual(sum(r["fired"] for r in rows), 3)
            self.assertTrue(all(r["requests"] == 2 for r in rows))
            self.assertEqual({r["first_fire_token"] for r in rows if r["fired"]}, {123})
            self.assertNotIn("Секрет", data.decode())
            self.assertNotIn("Секрет", json.dumps(json.loads(self.get(gw, "/guardian/switches")[1]), ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
