"""MEMV2: the desk runtime against the fake VRAM governor on a virtual clock: signals in, policy pushes and the journal out."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gateway"))
sys.path.insert(0, str(HERE / "support"))
import desk_need
import desk_runtime
from desk_need import Signals
from fake_vram import FakeVram


class Sources:
    def __init__(self):
        self.signals = Signals(idle_s=2.0)
        self.calls = 0

    def collect(self, model_pids=()):
        self.calls += 1
        return self.signals


class Rig:
    def __init__(self, config=None, vram=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.vram = vram or FakeVram(warmup_seconds=0.0)
        self.sources = Sources()
        self.events = []
        self.requests = []
        self.t = 0.0
        self.wall = 1_761_390_000.0
        self.path = str(Path(self.tmp.name) / "ledger.json")
        self.desk = None
        self.config = {"deskNeed": True, "deskPollSeconds": 1, "deskPushMinSeconds": 1, "deskNeedDownSeconds": 20, "deskSleepOnSeconds": 3,
                       "deskSleepOffSeconds": 10, **(config or {})}
        self.make()

    def make(self):
        self.desk = desk_runtime.DeskRuntime(self.config, self.http, self.sources, self.path, lambda k, t: self.events.append((k, t)),
                                             wall=lambda: self.wall, persist_seconds=30)

    def http(self, method, path, body=None):
        self.requests.append((method, path, body))
        if path.startswith("/vram/journal") and self.vram.legacy_api:
            return 404, {"error": {"code": 404, "message": "File Not Found"}}
        if path.startswith("/vram/journal"):
            q = dict(p.split("=") for p in path.split("?")[1].split("&"))
            return 200, self.vram.journal_json(int(q["since"]), int(q["limit"]))
        if path == "/vram/policy" and method == "POST":
            return self.vram.post_policy(body)
        return 404, None

    def run(self, seconds, pid=1234, ready=True):
        for _ in range(int(seconds)):
            self.t += 1.0
            self.wall += 1.0
            self.vram.tick(1.0)
            self.desk.step(self.t, pid, ready)

    def cleanup(self):
        self.tmp.cleanup()


class Runtime(unittest.TestCase):
    def rig(self, **kw):
        r = Rig(**kw)
        self.addCleanup(r.cleanup)
        r.vram.set_desk(1300)
        return r

    def test_ordinary_use_pushes_the_floor_reserve_once(self):
        r = self.rig()
        r.run(10)
        posts = [b for m, p, b in r.requests if m == "POST"]
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0]["reserve_mib"], 1536)
        self.assertEqual(r.vram.policy["reserve_mib"], 1536)

    def test_a_game_raises_the_reserve_and_puts_the_cache_to_sleep_then_wakes(self):
        r = self.rig()
        r.run(130)                                           # the cache has grown
        grown = r.vram.chunks
        self.assertGreater(grown, 0)
        r.sources.signals = Signals(game=True)
        r.run(6)
        self.assertEqual(r.vram.policy["reserve_mib"], 2560)
        self.assertEqual(r.vram.policy["sleep_level"], 1)
        self.assertEqual(r.vram.chunks, 0)                   # the model's cache was given back
        kinds = [e["kind"] for e in r.vram.journal]
        self.assertIn("sleep", kinds)
        notes = [e["note"] for e in r.vram.journal if e["kind"] in ("sleep", "policy")]
        self.assertTrue(any("gamemoded" in n for n in notes), notes)
        r.sources.signals = Signals(idle_s=1.0)
        r.run(120)
        self.assertEqual(r.vram.policy["sleep_level"], 0)
        self.assertEqual(r.vram.policy["reserve_mib"], 1536)
        self.assertGreater(r.vram.chunks, 0)                 # the cache grows back after the calm period

    def test_the_ledger_counts_what_the_server_gave_back(self):
        r = self.rig()
        r.run(130)
        r.vram.set_desk(r.vram.desk_mib + 2500)              # the desk jumps: the server trims
        r.run(20)
        today = r.desk.snapshot()["today"]
        self.assertGreaterEqual(today["yields"], 1)
        self.assertGreater(today["yielded_mib"], 0)
        self.assertIn("уступила", r.desk.snapshot()["status_line"])

    def test_signal_errors_do_not_lower_the_reserve(self):
        r = self.rig()
        r.sources.signals = Signals(errors=["logind: failed", "pipewire: failed"])
        r.run(100)
        self.assertGreaterEqual(r.vram.policy["reserve_mib"], 1536)

    def test_a_raising_source_never_breaks_the_loop(self):
        r = self.rig()

        def boom(model_pids=()):
            raise RuntimeError("dbus exploded")
        r.sources.collect = boom
        r.run(5)
        self.assertIn("dbus exploded", " ".join(r.desk.snapshot()["signals"]["errors"]))

    def test_nothing_is_pushed_while_the_server_is_not_ready(self):
        r = self.rig()
        r.run(5, ready=False)
        self.assertEqual([x for x in r.requests if x[0] == "POST"], [])
        r.run(3, pid=None)
        self.assertEqual([x for x in r.requests if x[0] == "POST"], [])

    def test_a_restarted_server_is_told_again(self):
        r = self.rig()
        r.run(10, pid=1)
        r.vram.policy["reserve_mib"] = 1536                   # a new process starts with the configured policy
        r.run(10, pid=2)
        self.assertEqual(len([x for x in r.requests if x[0] == "POST"]), 2)

    def test_a_refused_push_is_retried_and_reported_once(self):
        r = self.rig()
        original = r.http
        state = {"fail": True}

        def flaky(method, path, body=None):
            if method == "POST" and state["fail"]:
                return 400, {"error": "no"}
            return original(method, path, body)
        r.desk.http = flaky
        r.run(10)
        self.assertEqual(len([e for e in r.events if e[0] == "desk_need_failed"]), 1)
        state["fail"] = False
        r.run(10)
        self.assertEqual(r.vram.policy["reserve_mib"], 1536)
        self.assertEqual(r.desk.snapshot()["error"], "")

    def test_the_ledger_survives_a_restart_and_is_written_rarely(self):
        r = self.rig()
        r.run(130)
        r.vram.set_desk(r.vram.desk_mib + 2500)
        r.run(40)
        writes = r.desk.persisted_at
        r.desk.flush()
        saved = json.loads(Path(r.path).read_text())
        self.assertGreaterEqual(saved["days"][r.desk.today()]["yields"], 1)
        again = desk_runtime.DeskRuntime(r.config, r.http, r.sources, r.path, lambda k, t: None, wall=lambda: r.wall)
        self.assertGreaterEqual(again.snapshot()["today"]["yields"], 1)
        # without a change nothing is written: the persisted time stays put
        r.desk.persisted_at = -1e9
        r.desk._persist(r.t + 1000)
        self.assertEqual(r.desk.dirty, False)

    def test_signals_file_hook(self):
        r = self.rig()
        path = Path(r.tmp.name) / "signals.json"
        r.desk.cfg["deskSignalsFile"] = str(path)
        r.run(3)
        self.assertEqual(r.desk.signals.game, False)          # no file: neutral
        path.write_text(json.dumps({"game": True, "unknown_key": 1}))
        r.run(5)
        self.assertTrue(r.desk.signals.game)

    def test_floor_can_be_lowered_only_by_config(self):
        r = self.rig(config={"deskReserveFloorMiB": 768})
        r.sources.signals = Signals(locked=True)
        r.run(40)
        self.assertEqual(r.vram.policy["reserve_mib"], 768)


class OldServer(unittest.TestCase):
    """A server before MEMV2 (the stand of SAGE1 ran one): it refuses sleep_level, human_need, note (400, nothing applied) and has no
    /vram/journal (404). The gateway must still deliver the reserve, say so once, and not fill the log."""

    def rig(self, **kw):
        r = Rig(**kw)
        self.addCleanup(r.cleanup)
        r.vram.set_desk(1300)
        r.vram.legacy_api = True
        return r

    def test_the_reserve_still_arrives_and_the_log_is_not_flooded(self):
        r = self.rig()
        r.sources.signals = Signals(game=True)
        r.run(60)
        self.assertEqual(r.vram.policy["reserve_mib"], 2560)                 # not left on the configured reserve
        self.assertEqual(r.vram.chunks, 0)                                   # light sleep arrived as the old game mode
        self.assertEqual(r.vram.policy["max_chunks"], 0)
        kinds = [k for k, _ in r.events]
        self.assertEqual(kinds.count("desk_need_failed"), 0, r.events)
        self.assertEqual(kinds.count("desk_need_compat"), 2, r.events)      # once for the policy fields, once for the journal
        posts = [b for m, p, b in r.requests if m == "POST"]
        self.assertEqual(len([b for b in posts if "sleep_level" in b]), 1)   # the new fields were tried once, then dropped
        journal_calls = [x for x in r.requests if x[1].startswith("/vram/journal")]
        self.assertEqual(len(journal_calls), 1)                              # asked once, not every 5 s
        snap = r.desk.snapshot()
        self.assertEqual(snap["server_supports"], {"extra": False, "journal": False, "buffer": True})
        self.assertEqual(snap["error"], "")

    def test_wake_releases_the_old_game_mode_and_a_later_reserve_change_still_goes(self):
        r = self.rig()
        r.sources.signals = Signals(game=True)
        r.run(20)
        self.assertEqual(r.vram.policy["max_chunks"], 0)
        r.sources.signals = Signals(idle_s=1.0)
        r.run(120)
        self.assertIsNone(r.vram.policy["max_chunks"])                       # woke up: no limit again
        self.assertEqual(r.vram.policy["reserve_mib"], 1536)
        self.assertGreater(r.vram.chunks, 0)

    def test_a_restarted_server_that_knows_the_fields_gets_them(self):
        r = self.rig()
        r.run(10, pid=1)
        self.assertFalse(r.desk.caps["extra"])
        r.vram.legacy_api = False
        r.run(10, pid=2)                                                     # a new process: ask again with everything
        self.assertTrue(r.desk.caps["extra"] and r.desk.caps["journal"])
        self.assertTrue(any("sleep_level" in b for m, p, b in r.requests[-6:] if m == "POST"))

    def test_other_400_errors_are_not_taken_for_an_old_server(self):
        r = self.rig()
        r.vram.legacy_api = False
        original = r.http

        def refuse(method, path, body=None):
            if method == "POST":
                return 400, {"error": "reserve_mib out of range"}
            return original(method, path, body)
        r.desk.http = refuse
        r.run(10)
        self.assertTrue(r.desk.caps["extra"])                                # "unknown" is not in the message: still a new server
        self.assertEqual(len([e for e in r.events if e[0] == "desk_need_failed"]), 1)   # reported once, not every second

    def test_a_failing_post_and_a_failing_journal_do_not_take_turns_logging(self):
        r = self.rig()
        r.vram.legacy_api = False
        original = r.http

        def broken(method, path, body=None):
            if method == "POST":
                return 500, {"error": "boom"}
            return 500, None
        r.desk.http = broken
        r.run(60)
        self.assertLessEqual(len([e for e in r.events if e[0] == "desk_need_failed"]), 2)


class Config(unittest.TestCase):
    def test_defaults_are_off_and_valid(self):
        desk_runtime.validate_desk_config({})
        self.assertFalse(desk_need.DEFAULTS["deskNeed"])

    def test_bad_values(self):
        for bad in ({"deskNeed": "yes"}, {"deskReserveFloorMiB": 100}, {"deskReserveFloorMiB": "1536"}, {"deskPollSeconds": 0},
                    {"deskIdleSeconds": -1}, {"deskSignalsFile": 5}, {"deskGpuAppBusyPercent": 0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                desk_runtime.validate_desk_config(bad)



class BufferCompat(unittest.TestCase):
    """VRAMLOGIC3: a server that knows MEMV2 (sleep_level, human_need, note) but not buffer_mib (before VRAMLOGIC2)."""

    def test_the_buffer_is_dropped_the_rest_stays_and_a_new_process_is_asked_again(self):
        r = Rig()
        self.addCleanup(r.cleanup)
        r.vram.set_desk(1300)
        inner = r.http

        def http(method, path, body=None):
            if path == "/vram/policy" and method == "POST" and isinstance(body, dict) and "buffer_mib" in body and r.old:
                r.requests.append((method, path, body))
                return 400, {"error": "unknown field buffer_mib"}
            return inner(method, path, body)
        r.old = True
        r.desk.http = http
        r.sources.signals = Signals(video=True)
        r.run(10, pid=1)
        self.assertFalse(r.desk.caps["buffer"])
        self.assertTrue(r.desk.caps["extra"])                                  # MEMV2 fields still go
        self.assertEqual(r.vram.policy["reserve_mib"], 1856)                   # video: need 0.6 -> 1856
        kinds = [k for k, _ in r.events]
        self.assertEqual(kinds.count("desk_need_compat"), 1, r.events)
        self.assertEqual(kinds.count("desk_need_failed"), 0, r.events)
        r.old = False
        r.run(10, pid=2)                                                       # a new process: try the buffer again
        self.assertTrue(r.desk.caps["buffer"])
        self.assertEqual(r.vram.policy["buffer_mib"], 512)
        self.assertEqual(r.vram.target_mib(), r.vram.policy["reserve_mib"] + 512)

    def test_a_new_server_follows_the_desk(self):
        r = Rig()
        self.addCleanup(r.cleanup)
        r.vram.set_desk(1300)
        r.sources.signals = Signals(idle_s=2.0, browser_vram_mib=900)
        r.run(5)
        self.assertEqual(r.vram.policy["buffer_mib"], 256)
        r.sources.signals = Signals(idle_s=2.0, browser_vram_mib=1200)          # +300 MiB: the browser is busy
        r.run(5)
        self.assertEqual(r.vram.policy["buffer_mib"], 512)
        self.assertTrue(any(k == "desk_need" and "buffer=512" in t for k, t in r.events), r.events)

if __name__ == "__main__":
    unittest.main()
