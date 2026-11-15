"""MEMV2: human_need from the desk's signals, its hysteresis, the reserve and sleep level it asks for, the daily ledger and the
status line. Pure logic: scripted signals and a scripted clock, no desktop, no GPU."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
import desk_need
from desk_need import DEFAULTS, DeskNeed, Ledger, Signals, reserve_for, status_line, target_need


def run(desk, script, step=1.0, t=0.0):
    """Feed (seconds, Signals) pairs; returns the need after each and the clock."""
    out = []
    for seconds, sig in script:
        for _ in range(int(seconds / step)):
            t += step
            out.append(desk.update(t, sig))
    return out, t


GAME, VIDEO, CALL = Signals(game=True), Signals(video=True), Signals(call=True)
TYPING, AWAY, LOCKED = Signals(idle_s=3.0), Signals(idle_s=900.0), Signals(locked=True)


class TargetNeed(unittest.TestCase):
    def test_levels(self):
        cfg = dict(DEFAULTS)
        self.assertEqual(target_need(GAME, cfg)[0], 1.0)
        self.assertEqual(target_need(Signals(gpu_app="game.exe", gpu_app_detail="gfx 80%"), cfg)[0], 1.0)
        self.assertEqual(target_need(CALL, cfg)[0], 0.6)
        self.assertEqual(target_need(VIDEO, cfg)[0], 0.6)
        self.assertEqual(target_need(Signals(browser_vram_mib=3000), cfg)[0], 0.6)
        self.assertEqual(target_need(TYPING, cfg)[0], 0.3)
        self.assertEqual(target_need(AWAY, cfg)[0], 0.0)
        self.assertEqual(target_need(LOCKED, cfg)[0], 0.0)

    def test_unknown_idle_is_ordinary_use_not_away(self):
        need, why = target_need(Signals(errors=["logind: show-session failed"]), dict(DEFAULTS))
        self.assertEqual(need, 0.3)
        self.assertIn("unknown", why)

    def test_game_beats_a_locked_screen_and_idle(self):
        self.assertEqual(target_need(Signals(game=True, locked=True, idle_s=9999), dict(DEFAULTS))[0], 1.0)


class Reserve(unittest.TestCase):
    def test_formula_and_floor(self):
        # 0.75 GiB + 1.75 GiB * need, rounded up to 64 MiB; the floor wins below it
        self.assertEqual(reserve_for(0.0, 768), 768)
        self.assertEqual(reserve_for(1.0, 768), 2560)
        self.assertEqual(reserve_for(0.3, 768), 1344)
        self.assertEqual(reserve_for(0.0, 1536), 1536)
        self.assertEqual(reserve_for(0.3, 1536), 1536)
        self.assertEqual(reserve_for(0.6, 1536), 1856)
        self.assertEqual(reserve_for(1.0, 1536), 2560)

    def test_never_below_the_floor_by_default(self):
        desk = DeskNeed()
        for sig in (AWAY, LOCKED, TYPING, VIDEO, GAME):
            desk.need = target_need(sig, desk.cfg)[0]
            self.assertGreaterEqual(desk.reserve_mib(), 1536)

    def test_lower_floor_only_by_config(self):
        desk = DeskNeed({"deskReserveFloorMiB": 768})
        desk.need = 0.0
        self.assertEqual(desk.reserve_mib(), 768)

    def test_server_range(self):
        self.assertEqual(reserve_for(1.0, 9000), 4096)
        self.assertEqual(reserve_for(0.0, 10), 768)


class Hysteresis(unittest.TestCase):
    def test_rises_at_once_falls_after_the_hold(self):
        desk = DeskNeed({"deskNeedDownSeconds": 60})
        needs, t = run(desk, [(3, TYPING), (1, GAME)])
        self.assertEqual(needs[-1], 1.0)
        needs, t2 = run(DeskNeed({"deskNeedDownSeconds": 60}), [(1, GAME), (59, TYPING)])
        self.assertEqual(needs[-1], 1.0)                 # 59 s: still held
        needs, _ = run(DeskNeed({"deskNeedDownSeconds": 60}), [(1, GAME), (62, TYPING)])
        self.assertEqual(needs[-1], 0.3)

    def test_a_short_dip_in_a_video_does_not_give_the_cache_back(self):
        desk = DeskNeed()
        needs, _ = run(desk, [(5, VIDEO), (10, TYPING), (5, VIDEO), (10, TYPING), (5, VIDEO)])
        self.assertEqual(min(needs[5:]), 0.6)

    def test_falls_to_the_highest_need_seen_during_the_hold(self):
        desk = DeskNeed({"deskNeedDownSeconds": 30})
        needs, _ = run(desk, [(1, GAME), (10, AWAY), (10, VIDEO), (11, AWAY)])
        self.assertEqual(needs[-1], 0.6)                 # the hold window saw a video (0.6) and no more than that

    def test_night_after_the_hold(self):
        desk = DeskNeed({"deskNeedDownSeconds": 30})
        needs, _ = run(desk, [(2, TYPING), (40, LOCKED)])
        self.assertEqual(needs[-1], 0.0)


class Sleep(unittest.TestCase):
    def test_sleep_after_a_game_holds_and_wakes_after_the_game_ends(self):
        desk = DeskNeed({"deskSleepOnSeconds": 5, "deskSleepOffSeconds": 60, "deskNeedDownSeconds": 10})
        _, t = run(desk, [(4, GAME)])
        self.assertEqual(desk.sleep_level, 0)             # 4 s of game: not yet
        _, t = run(desk, [(2, GAME)], t=t)
        self.assertEqual(desk.sleep_level, 1)
        desk2 = desk
        for _ in range(9):                                # the need falls after 10 s, then wake needs 60 s more
            t += 1
            desk2.update(t, TYPING)
        self.assertEqual(desk2.sleep_level, 1)
        for _ in range(80):
            t += 1
            desk2.update(t, TYPING)
        self.assertEqual(desk2.sleep_level, 0)
        self.assertGreater(desk2.sleeping_s, 60)

    def test_deep_sleep_after_a_long_game_and_the_server_still_hears_level_1(self):
        # MEMV2 L2: a game that goes on parks the model; the server itself only knows levels 0 and 1
        desk = DeskNeed({"deskSleepOnSeconds": 5, "deskDeepSleepSeconds": 30, "deskSleepOffSeconds": 60, "deskNeedDownSeconds": 10})
        _, t = run(desk, [(10, GAME)])
        self.assertEqual(desk.sleep_level, 1)
        _, t = run(desk, [(19, GAME)], t=t)
        self.assertEqual(desk.sleep_level, 1)               # 29 s of game: still light
        _, t = run(desk, [(2, GAME)], t=t)
        self.assertEqual(desk.sleep_level, 2)
        self.assertIn("for 30 s", desk.deep_reason)
        self.assertEqual(desk.wanted()["sleep_level"], 1)
        self.assertEqual(desk.snapshot()["sleep_level"], 2)
        _, t = run(desk, [(100, TYPING)], t=t)                # wakes with light sleep: the need falls, then 60 s more
        self.assertEqual((desk.sleep_level, desk.deep_reason), (0, ""))

    def test_a_pause_in_light_sleep_deepens_it_at_once(self):
        desk = DeskNeed({"deskSleepOnSeconds": 5, "deskDeepSleepSeconds": 600})
        self.assertFalse(desk.deepen("the server paused"))   # awake: a pause is the governor's ordinary business
        run(desk, [(6, GAME)])
        self.assertEqual(desk.sleep_level, 1)
        self.assertTrue(desk.deepen("the server paused"))
        self.assertEqual((desk.sleep_level, desk.deep_reason), (2, "the server paused"))
        self.assertFalse(desk.deepen("again"))

    def test_deep_sleep_can_be_switched_off(self):
        desk = DeskNeed({"deskSleepOnSeconds": 5, "deskDeepSleepSeconds": 10, "deskDeepSleep": False})
        run(desk, [(60, GAME)])
        self.assertEqual(desk.sleep_level, 1)
        self.assertFalse(desk.deepen("the server paused"))

    def test_the_status_line_says_parked(self):
        self.assertIn("выгружена", status_line({"need": 1.0, "sleep_level": 2}, {}))
        self.assertIn("parked", status_line({"need": 1.0, "sleep_level": 2}, {}, language="en"))
        self.assertIn("спит", status_line({"need": 1.0, "sleep_level": 1}, {}))

    def test_a_short_game_signal_does_not_sleep(self):
        desk = DeskNeed({"deskSleepOnSeconds": 5})
        run(desk, [(2, GAME), (20, TYPING)])
        self.assertEqual(desk.sleep_level, 0)


class Push(unittest.TestCase):
    def test_first_push_carries_everything_and_the_reason(self):
        desk = DeskNeed()
        desk.update(0, GAME)
        body = desk.push_due(1.0)
        self.assertEqual(body["reserve_mib"], 2560)
        self.assertEqual(body["sleep_level"], 0)
        self.assertEqual(body["human_need"], 1.0)
        self.assertIn("gamemoded", body["note"])

    def test_nothing_new_nothing_pushed_and_the_number_alone_never_pushes(self):
        desk = DeskNeed()
        desk.update(0, TYPING)
        desk.pushed_ok(0, desk.push_due(1.0))
        self.assertIsNone(desk.push_due(100.0))
        desk.need = 0.0                                   # reserve stays 1536 (floor): only human_need moved
        self.assertIsNone(desk.push_due(200.0))

    def test_min_interval_and_step(self):
        desk = DeskNeed({"deskPushMinSeconds": 5})
        desk.update(0, TYPING)
        desk.pushed_ok(0, desk.push_due(0.0))
        desk.update(1, GAME)
        self.assertIsNone(desk.push_due(2.0))             # 2 s after the last push: wait
        self.assertIsNotNone(desk.push_due(6.0))

    def test_a_restarted_server_gets_everything_again(self):
        desk = DeskNeed()
        desk.update(0, TYPING)
        desk.pushed_ok(0, desk.push_due(0.0))
        desk.server_restarted()
        self.assertIsNotNone(desk.push_due(1.0))


class LedgerTests(unittest.TestCase):
    TRIM = {"id": 3, "kind": "trim", "mib": -312, "recovered_s": None}

    def test_counts_each_event_once(self):
        ledger = Ledger()
        events = [{"id": 1, "kind": "grow", "mib": 100}, dict(self.TRIM), {"id": 4, "kind": "pause"}, {"id": 5, "kind": "sleep"}]
        ledger.add_events("d", events)
        ledger.add_events("d", events)
        d = ledger.day("d")
        self.assertEqual((d["yields"], d["yielded_mib"], d["pauses"], d["sleeps"]), (1, 312, 1, 1))

    def test_recovery_time_arrives_later_and_counts_once(self):
        ledger = Ledger()
        ledger.add_events("d", [dict(self.TRIM)])
        self.assertEqual(ledger.day("d")["recovered"], 0)
        filled = dict(self.TRIM, recovered_s=0.24)
        ledger.add_events("d", [filled])
        ledger.add_events("d", [filled])
        d = ledger.day("d")
        self.assertEqual((d["yields"], d["recovered"], d["recovered_max_s"]), (1, 1, 0.24))

    def test_a_restarted_server_starts_its_ids_again(self):
        ledger = Ledger()
        ledger.add_events("d", [dict(self.TRIM)])
        ledger.server_restarted()
        ledger.add_events("d", [dict(self.TRIM)])
        self.assertEqual(ledger.day("d")["yields"], 2)

    def test_bad_events_are_ignored(self):
        ledger = Ledger()
        ledger.add_events("d", [None, "x", {"kind": "trim"}, {"id": "7", "kind": "trim"}])
        self.assertEqual(ledger.day("d")["yields"], 0)

    def test_save_and_load(self):
        ledger = Ledger()
        ledger.add_events("2026-10-25", [dict(self.TRIM)])
        other = Ledger()
        other.load(ledger.to_json())
        self.assertEqual(other.day("2026-10-25")["yielded_mib"], 312)
        other.load({"days": {"x": {"yields": "no", "unknown": 1}}})       # garbage must not break loading
        other.load("not a dict")


class StatusLine(unittest.TestCase):
    def test_russian_and_english(self):
        snap = {"need": 0.6, "sleep_level": 0}
        today = {"yields": 3, "yielded_mib": 1040, "recovered": 3, "recovered_max_s": 0.31}
        ru = status_line(snap, today)
        self.assertIn("уступила 3×", ru)
        self.assertIn("0.31", ru)
        self.assertIn("0.6", ru)
        self.assertIn("gave way 3x", status_line(snap, today, "en"))
        self.assertIn("не уступала", status_line(snap, {}))
        self.assertIn("спит", status_line({"need": 1.0, "sleep_level": 1}, {}))



class Buffer(unittest.TestCase):
    """VRAMLOGIC3: the governor's working buffer from instant signals: 256 quiet, 512 while the desk is active."""

    def feed(self, desk, pairs, t=0.0):
        for seconds, sig in pairs:
            for _ in range(int(seconds)):
                t += 1.0
                desk.update(t, sig)
        return t

    def test_quiet_desk_and_the_need_levels(self):
        d = DeskNeed()
        self.feed(d, [(3, TYPING)])
        self.assertEqual((d.buffer_mib, d.wanted()["buffer_mib"]), (256, 256))
        self.feed(d, [(2, VIDEO)])
        self.assertEqual(d.buffer_mib, 512)
        self.assertIn("active desk", d.buffer_reason)
        d2 = DeskNeed(); self.feed(d2, [(2, CALL)]); self.assertEqual(d2.buffer_mib, 512)
        d3 = DeskNeed(); self.feed(d3, [(2, Signals(browser_vram_mib=3000))]); self.assertEqual(d3.buffer_mib, 512)   # the 2 GiB rule (need 0.6)
        d4 = DeskNeed(); self.feed(d4, [(2, AWAY)]); self.assertEqual(d4.buffer_mib, 256)

    def test_browser_growth_holds_the_larger_buffer_for_the_hold_time(self):
        d = DeskNeed({"deskBufferHoldSeconds": 120})
        t = self.feed(d, [(3, Signals(idle_s=2.0, browser_vram_mib=900))])
        self.assertEqual(d.buffer_mib, 256)
        t = self.feed(d, [(1, Signals(idle_s=2.0, browser_vram_mib=1100))], t)      # +200 MiB between two polls
        self.assertEqual(d.buffer_mib, 512)
        self.assertIn("browser VRAM grew by 200 MiB", d.buffer_reason)
        self.assertEqual(d.need, 0.3)                                                # the need itself did not move
        t = self.feed(d, [(118, Signals(idle_s=2.0, browser_vram_mib=1100))], t)
        self.assertEqual(d.buffer_mib, 512)
        t = self.feed(d, [(2, Signals(idle_s=2.0, browser_vram_mib=1100))], t)
        self.assertEqual(d.buffer_mib, 256)                                          # the hold is over

    def test_slow_browser_growth_and_shrinking_do_not_count(self):
        d = DeskNeed()
        browser = 100                                                                 # stays below the 2 GiB rule (need 0.6) throughout
        t = 0.0
        for _ in range(30):
            browser += 60                                                             # 60 MiB per poll: below the 128 step
            t = self.feed(d, [(1, Signals(idle_s=2.0, browser_vram_mib=browser))], t)
        self.assertEqual(d.buffer_mib, 256)
        t = self.feed(d, [(1, Signals(idle_s=2.0, browser_vram_mib=browser - 1000))], t)   # a big drop is not activity
        self.assertEqual(d.buffer_mib, 256)

    def test_a_buffer_change_is_pushed_and_recorded(self):
        d = DeskNeed({"deskPushMinSeconds": 1})
        t = self.feed(d, [(3, TYPING)])
        body = d.push_due(t)
        self.assertEqual(body["buffer_mib"], 256)
        d.pushed_ok(t, body)
        self.assertIsNone(d.push_due(t + 2))
        t = self.feed(d, [(1, Signals(idle_s=2.0, browser_vram_mib=0)), (1, Signals(idle_s=2.0, browser_vram_mib=300))], t)
        body = d.push_due(t)
        self.assertIsNotNone(body)                                                   # only the buffer moved, and it is pushed
        self.assertEqual((body["buffer_mib"], body["reserve_mib"]), (512, 1536))
        d.pushed_ok(t, body)
        self.assertEqual(d.pushed["buffer_mib"], 512)
        snap = d.snapshot()
        self.assertEqual((snap["buffer_mib"], snap["buffer_reason"].startswith("browser VRAM grew")), (512, True))

    def test_config_validation(self):
        import desk_runtime
        desk_runtime.validate_desk_config({"deskBufferQuietMiB": 0, "deskBufferActiveMiB": 768})
        for bad in ({"deskBufferQuietMiB": 600, "deskBufferActiveMiB": 512}, {"deskBufferActiveMiB": 4096}, {"deskBufferQuietMiB": 1.5},
                    {"deskBrowserGrowthMiB": 0}, {"deskBufferHoldSeconds": -1}):
            with self.assertRaises(ValueError, msg=str(bad)):
                desk_runtime.validate_desk_config(bad)

if __name__ == "__main__":
    unittest.main()
