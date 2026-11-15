"""MEMPROF1: the planned restart of the server when its heap + /dev/zero grow over the limit (a temporary measure for the HIP runtime
leak). Between turns the session goes to RAM, the server stops, and the next request loads it again. In-process, no GPU."""
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_gateway as fixture

SMAPS = """\
7f0000000000-7f0000100000 rw-p 00000000 00:00 0
Rss:                4096 kB
7f0000100000-7f0000200000 rw-p 00000000 00:00 0                          [heap]
Rss:                1048576 kB
7f0000200000-7f0000300000 rw-s 00000000 00:06 1234                       /dev/zero
Rss:                2097152 kB
7f0000300000-7f0000400000 r-xp 00000000 08:01 99                         /usr/lib64/libc.so.6
Rss:                 524288 kB
"""


class GatewayServerMem(unittest.TestCase):
    patch_attr = fixture.GatewayGuard.patch_attr
    fake_llama = fixture.GatewayGuard.fake_llama

    @classmethod
    def setUpClass(cls):
        cls.created_import = not hasattr(fixture.GatewayGuard, "gw")
        if cls.created_import:
            fixture.GatewayGuard.setUpClass()
        cls.gw = fixture.GatewayGuard.gw

    @classmethod
    def tearDownClass(cls):
        if cls.created_import:
            fixture.GatewayGuard.tearDownClass()

    def setUp(self):
        self.gw = type(self).gw
        fixture.GatewayGuard.setUp(self)
        self.gw = type(self).gw
        self.gw.G["serverMemRestartGiB"] = 6
        self.gw.G["serverMemRestartIdleSeconds"] = 20
        self.gw.G["serverMemCheckSeconds"] = 30
        proc = Mock()
        proc.pid = 4242
        proc.poll.return_value = None
        self.gw.st.proc, self.gw.st.status, self.gw.st.busy = proc, "READY", 0
        self.gw.st.mem_checked_at = -math.inf
        self.gw.st.mem_baseline = (4242, 0.5)      # the first reading of this server process was 0.5 GiB
        self.addCleanup(lambda: self.gw.G.update(serverMemRestartGiB=0))

    def events(self):
        return [e["reason"] for e in self.gw.st.events]

    def tick(self, mem, idle=60, saved=True, stopped=True, now=1000.0):
        with patch.object(self.gw, "server_mem_gib", return_value=mem), \
                patch.object(self.gw, "save_slot_locked", return_value=saved) as save, \
                patch.object(self.gw, "stop_llama_locked", return_value=stopped) as stop:
            self.gw.server_mem_restart_tick(now, idle)
        return save, stop

    def test_the_smaps_reader_sums_only_the_heap_and_dev_zero(self):
        self.assertEqual(self.gw.parse_smaps_gib(SMAPS), 3.0)       # 1 GiB heap + 2 GiB /dev/zero; the library and the anonymous map are ignored
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "7").mkdir()
            (Path(tmp) / "7" / "smaps").write_text(SMAPS, encoding="utf-8")
            self.assertEqual(self.gw.server_mem_gib(7, proc=tmp), 3.0)
            self.assertIsNone(self.gw.server_mem_gib(8, proc=tmp))

    def test_over_the_limit_between_turns_saves_the_session_and_stops_the_server(self):
        save, stop = self.tick(6.6)                      # grown by 6.1 GiB since the first reading
        save.assert_called_once_with(ram_only=True)
        stop.assert_called_once_with("server memory growth", save=False)
        self.assertIn("server_mem_restart", self.events())
        self.assertEqual(self.gw.st.last_yield_reason, "server_mem")

    def test_under_the_limit_nothing_happens(self):
        save, stop = self.tick(6.4)                      # grown by 5.9 GiB
        save.assert_not_called()
        stop.assert_not_called()

    def test_off_by_default_busy_and_short_idle_never_restart(self):
        self.gw.G["serverMemRestartGiB"] = 0
        _, stop = self.tick(9.5)
        stop.assert_not_called()
        self.gw.G["serverMemRestartGiB"] = 6
        self.gw.st.busy = 1
        _, stop = self.tick(9.5)
        stop.assert_not_called()
        self.gw.st.busy = 0
        _, stop = self.tick(9.5, idle=5)
        stop.assert_not_called()

    def test_a_failed_save_keeps_the_server_loaded_and_the_next_check_follows(self):
        save, stop = self.tick(6.6, saved=False)
        stop.assert_not_called()
        self.assertIn("server_mem_restart_skipped", self.events())
        self.assertEqual(self.gw.st.mem_checked_at, 1000.0)          # the check is throttled: serverMemCheckSeconds

    def test_a_big_absolute_size_is_not_growth_the_first_reading_is_the_baseline(self):
        # FINAL2: the pinned expert cache puts 6-7 GiB into /dev/zero at once; the old absolute rule restarted such a server after every request
        self.gw.st.mem_baseline = (None, 0.0)
        save, stop = self.tick(7.5)
        save.assert_not_called()
        stop.assert_not_called()
        self.assertEqual(self.gw.st.mem_baseline, (4242, 7.5))
        save, stop = self.tick(8.0, now=1100.0)          # +0.5 GiB: nothing
        stop.assert_not_called()
        save, stop = self.tick(13.6, now=1200.0)         # +6.1 GiB: now it restarts
        stop.assert_called_once()

    def test_a_new_server_process_gets_a_new_baseline(self):
        self.gw.st.mem_baseline = (1111, 0.5)            # the baseline belongs to the previous process
        _, stop = self.tick(9.0)
        stop.assert_not_called()
        self.assertEqual(self.gw.st.mem_baseline, (4242, 9.0))

    def test_the_reading_is_throttled(self):
        self.tick(5.0, now=1000.0)
        with patch.object(self.gw, "server_mem_gib", side_effect=AssertionError("read too early")):
            self.gw.server_mem_restart_tick(1010.0, 60)              # 10 s later: under serverMemCheckSeconds

    def test_the_monitor_tick_runs_the_check(self):
        with patch.object(self.gw, "server_mem_restart_tick") as tick:
            self.gw.monitor_tick(500.0, 12000, 24)
        tick.assert_called_once()

    CANDIDATE = Path.home() / "Desktop/AI/reports/amd2c/guardian.candidate.json"

    @unittest.skipUnless(CANDIDATE.exists(), "the candidate guardian is not on this machine")
    def test_the_candidate_guardian_passes_the_validator_with_the_restart_on(self):
        import json
        config = json.loads(self.CANDIDATE.read_text(encoding="utf-8"))
        self.gw.validate_server_mem_config(config)
        self.assertEqual(config["serverMemRestartGiB"], 3)          # FINAL2: growth since the first reading, measured on a 2-hour soak

    def test_config_validation(self):
        good = {"serverMemRestartGiB": 6, "vramGuard": True, "slotSaveCheckpoints": 1}
        self.gw.validate_server_mem_config(good)
        self.gw.validate_server_mem_config({})                                # off: nothing required
        for name, config in {"negative": {"serverMemRestartGiB": -1}, "text": {"serverMemRestartGiB": "6"}, "bool": {"serverMemRestartGiB": True},
                             "tiny": {"serverMemRestartGiB": 0.5}, "no guard": {"serverMemRestartGiB": 6},
                             "bad idle": dict(good, serverMemRestartIdleSeconds=0), "bad check": dict(good, serverMemCheckSeconds="x")}.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.gw.validate_server_mem_config(config)


if __name__ == "__main__":
    unittest.main()
