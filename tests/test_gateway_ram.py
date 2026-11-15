"""RAMGUARD1 (R0) in the gateway: the load check uses honest free memory when ramPolicy is "honest"; off by default.
RAMLOGIC2: the reserve follows the desk, and a ladder (smaller prompt cache, short wait, guarded load) comes before a refusal."""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))      # ram_budget, also when run alone
import test_gateway as fixture  # noqa: E402
import ram_budget  # noqa: E402

MODEL = {"ctxSize": 200000, "nCpuMoe": 26}
FIXED = ram_budget.resident_fixed_estimate_mib(26, 0)          # the server without its prompt cache in RAM: 9736 MiB


def mem(free_gb, swap_used_gb=0.0):
    kb = lambda gb: int(gb * 1024 * 1024)  # noqa: E731
    return {"MemFree": free_gb * 1024, "Inactive(file)": 0.0, "SwapTotal": 16384.0, "SwapFree": 16384.0 - swap_used_gb * 1024,
            "Shmem": 0.0, "MemTotal": 32768.0}


CALM = {"some": {"avg10": 0.0, "avg60": 0.0}, "full": {"avg10": 0.0}}


def fresh_ladder():
    return {"cache_cap": None, "state": "idle", "since": 0.0, "until": 0.0, "free_mib": None, "need_mib": None,
            "reserve_mib": None, "reason": ""}


class FakeDesk:
    def __init__(self, need):
        self.need = type("Need", (), {"need": need})()


class RamAdmission(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(self.gw.RAM_LADDER, fresh_ladder())
        patcher.start()
        self.addCleanup(patcher.stop)

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

    def test_a_load_with_room_is_admitted(self):
        with patch.dict(self.gw.G, {"ramDeskReserveMiB": 6144}), \
                patch.object(self.gw.ram_budget, "read_meminfo", return_value=mem(24.0)), \
                patch.object(self.gw.ram_budget, "read_pressure", return_value=CALM):
            self.assertEqual(self.gw.ram_admission(MODEL), "")

    def test_a_load_that_would_eat_the_desktop_reserve_waits_first(self):
        with patch.dict(self.gw.G, {"ramDeskReserveMiB": 6144, "ramWaitSeconds": 20}), \
                patch.object(self.gw.ram_budget, "read_meminfo", return_value=mem(9.0)), \
                patch.object(self.gw.ram_budget, "read_pressure", return_value=CALM):
            message = self.gw.ram_admission(MODEL)
            self.assertTrue(message.startswith("Model load waiting: RAM waiting for memory: "), message)
            self.assertEqual(self.gw.refusal_code(message), "ram_wait")
            self.assertIn("room for the desktop 6.0 GiB", message)
            self.assertEqual(self.gw.RAM_LADDER["state"], "waiting")
            self.assertGreater(self.gw.RAM_LADDER["until"], time.time() + 15)

    def test_the_reserve_follows_the_desk_unless_it_is_configured(self):
        with patch.dict(self.gw.G, {"ramDeskReserveMiB": None}):
            with patch.object(self.gw, "DESK", None):
                self.assertEqual(self.gw.ram_desk_reserve(), 2048)
            with patch.object(self.gw, "DESK", FakeDesk(0.0)):
                self.assertEqual(self.gw.ram_desk_reserve(), 1024)
            with patch.object(self.gw, "DESK", FakeDesk(0.6)):
                self.assertEqual(self.gw.ram_desk_reserve(), 3072)
        with patch.dict(self.gw.G, {"ramDeskReserveMiB": 5120}), patch.object(self.gw, "DESK", FakeDesk(0.0)):
            self.assertEqual(self.gw.ram_desk_reserve(), 5120)

    def test_a_light_step_caps_the_prompt_cache_of_the_next_start(self):
        free = (FIXED + 2048 + 1300) / 1024                          # room for 1280 MiB of the 4096 MiB prompt cache
        with patch.dict(self.gw.G, {"ramDeskReserveMiB": None}), patch.object(self.gw, "DESK", None), \
                patch.dict(self.gw.M, {"cacheRamMB": 4096}), \
                patch.object(self.gw.ram_budget, "read_meminfo", return_value=mem(free)), \
                patch.object(self.gw.ram_budget, "read_pressure", return_value=CALM):
            self.assertEqual(self.gw.ram_admission(MODEL), "")
            self.assertEqual(self.gw.RAM_LADDER["state"], "light")
            self.assertEqual(self.gw._ram_cache_budget(MODEL)[0], 1280)
            self.assertEqual(self.gw._ram_cache_budget(MODEL, ladder=False)[0], 4096)
        with patch.dict(self.gw.G, {"ramDeskReserveMiB": None}), patch.object(self.gw, "DESK", None), \
                patch.dict(self.gw.M, {"cacheRamMB": 4096}), \
                patch.object(self.gw.ram_budget, "read_meminfo", return_value=mem(24.0)), \
                patch.object(self.gw.ram_budget, "read_pressure", return_value=CALM):
            self.assertEqual(self.gw.ram_admission(MODEL), "")           # room again: the next start gets the whole cache
            self.assertEqual(self.gw._ram_cache_budget(MODEL)[0], 4096)

    def test_after_the_wait_a_guarded_load_or_an_honest_refusal(self):
        guarded, short = (FIXED + 1500) / 1024, (FIXED + 500) / 1024
        with patch.dict(self.gw.G, {"ramDeskReserveMiB": None, "ramWaitSeconds": 20, "ramHardMinMiB": 1024}), \
                patch.object(self.gw, "DESK", None), patch.dict(self.gw.M, {"cacheRamMB": 4096}), \
                patch.object(self.gw.ram_budget, "read_pressure", return_value=CALM):
            with patch.object(self.gw.ram_budget, "read_meminfo", return_value=mem(guarded)):
                self.assertEqual(self.gw.refusal_code(self.gw.ram_admission(MODEL)), "ram_wait")
                self.gw.RAM_LADDER["until"] = time.time() - 1                # the window has passed
                self.assertEqual(self.gw.ram_admission(MODEL), "")
                self.assertEqual((self.gw.RAM_LADDER["state"], self.gw.RAM_LADDER["cache_cap"]), ("guarded", 0))
                self.assertEqual(self.gw.ram_admission(MODEL), "")              # start_llama asks again: still guarded
            with patch.object(self.gw.ram_budget, "read_meminfo", return_value=mem(short)):
                message = self.gw.ram_admission(MODEL)                           # the window is spent: no second wait
                self.assertEqual(self.gw.refusal_code(message), "ram")
                self.assertTrue(message.startswith("Model load deferred: RAM not enough memory for the model yet: "), message)
                self.assertNotIn("close", message.lower())

    def test_a_wait_of_zero_goes_straight_to_the_guarded_step(self):
        with patch.dict(self.gw.G, {"ramDeskReserveMiB": None, "ramWaitSeconds": 0}), patch.object(self.gw, "DESK", None), \
                patch.dict(self.gw.M, {"cacheRamMB": 4096}), \
                patch.object(self.gw.ram_budget, "read_meminfo", return_value=mem((FIXED + 1500) / 1024)), \
                patch.object(self.gw.ram_budget, "read_pressure", return_value=CALM):
            self.assertEqual(self.gw.ram_admission(MODEL), "")
            self.assertEqual(self.gw.RAM_LADDER["state"], "guarded")

    def test_an_old_wait_window_is_forgotten(self):
        with patch.dict(self.gw.G, {"ramDeskReserveMiB": None, "ramWaitSeconds": 20}), patch.object(self.gw, "DESK", None), \
                patch.dict(self.gw.M, {"cacheRamMB": 4096}), \
                patch.object(self.gw.ram_budget, "read_meminfo", return_value=mem((FIXED + 1500) / 1024)), \
                patch.object(self.gw.ram_budget, "read_pressure", return_value=CALM):
            self.gw.RAM_LADDER.update(since=time.time() - 400, until=time.time() - 380)
            self.assertEqual(self.gw.refusal_code(self.gw.ram_admission(MODEL)), "ram_wait")

    def test_a_switch_credits_the_running_server(self):
        with patch.dict(self.gw.G, {"ramDeskReserveMiB": None}), patch.object(self.gw, "DESK", None), \
                patch.dict(self.gw.M, {"cacheRamMB": 4096}), \
                patch.object(self.gw.ram_budget, "read_meminfo", return_value=mem((FIXED + 1000) / 1024)), \
                patch.object(self.gw.ram_budget, "read_pressure", return_value=CALM):
            self.assertEqual(self.gw.refusal_code(self.gw.ram_admission(MODEL)), "ram_wait")
            self.assertEqual(self.gw.ram_admission(MODEL, reclaim_gb=6.0), "")

    def test_the_status_shows_the_ladder(self):
        snap = self.gw.ram_ladder_snapshot()
        for key in ("policy", "reserve_mib", "state", "free_mib", "need_mib", "until", "cache_cap"):
            self.assertIn(key, snap)

    def test_under_the_honest_policy_memavailable_does_not_decide(self):
        with patch.dict(self.gw.G, {"ramPolicy": "honest", "ramFreeMinGBToLoad": 14.0}), \
                patch.object(self.gw, "ram_admission", return_value=""), patch.object(self.gw, "mem_available_gb", return_value=8.0):
            self.assertNotIn("Not enough free RAM", self.gw.preload_check(MODEL))
        with patch.dict(self.gw.G, {"ramPolicy": "off", "ramFreeMinGBToLoad": 14.0}), \
                patch.object(self.gw, "mem_available_gb", return_value=8.0):
            self.assertIn("Not enough free RAM", self.gw.preload_check(MODEL))

    def test_a_swap_growth_refuses_the_next_loads_for_a_while(self):
        with patch.dict(self.gw.G, {"ramDeskReserveMiB": 6144}), \
                patch.dict(self.gw.RAM_SWAP_REFUSAL, {"until": time.time() + 60, "reason": "swap grew 1536 MiB while m (short) loaded"}), \
                patch.object(self.gw.ram_budget, "read_meminfo", return_value=mem(24.0)), \
                patch.object(self.gw.ram_budget, "read_pressure", return_value=CALM):
            message = self.gw.ram_admission(MODEL)
            self.assertTrue(message.startswith("Model load deferred: RAM swap grew"), message)

    def test_in_flight_check_records_a_big_swap_growth(self):
        with patch.dict(self.gw.RAM_SWAP_REFUSAL, {"until": 0.0, "reason": ""}), \
                patch.object(self.gw.ram_budget, "read_meminfo", return_value=mem(24.0, swap_used_gb=3.5)):
            self.gw.ram_swap_check_after_load("m", "short", mem(24.0, swap_used_gb=2.0))
            self.assertGreater(self.gw.RAM_SWAP_REFUSAL["until"], time.time())
            self.assertIn("swap grew 1536 MiB", self.gw.RAM_SWAP_REFUSAL["reason"])

    def test_in_flight_check_ignores_a_small_growth(self):
        with patch.dict(self.gw.RAM_SWAP_REFUSAL, {"until": 0.0, "reason": ""}), \
                patch.object(self.gw.ram_budget, "read_meminfo", return_value=mem(24.0, swap_used_gb=2.5)):
            self.gw.ram_swap_check_after_load("m", "short", mem(24.0, swap_used_gb=2.0))
            self.assertEqual(self.gw.RAM_SWAP_REFUSAL["until"], 0.0)

    def test_in_flight_check_is_off_unless_honest(self):
        with patch.dict(self.gw.G, {"ramPolicy": "off"}):
            self.assertIsNone(self.gw.ram_swap_before())
        with patch.dict(self.gw.G, {"ramPolicy": "honest"}), \
                patch.object(self.gw.ram_budget, "read_meminfo", return_value=mem(24.0)):
            self.assertIsNotNone(self.gw.ram_swap_before())

    def test_unreadable_memory_refuses(self):
        with patch.dict(self.gw.G, {"ramDeskReserveMiB": 6144}), \
                patch.object(self.gw.ram_budget, "read_meminfo", return_value=None), \
                patch.object(self.gw.ram_budget, "read_pressure", return_value=CALM):
            self.assertTrue(self.gw.ram_admission(MODEL).startswith("Model load deferred: RAM"))

    def test_preload_check_returns_the_ram_refusal_only_when_honest(self):
        with patch.dict(self.gw.G, {"ramPolicy": "honest"}), patch.object(self.gw, "ram_admission", return_value="Model load deferred: RAM x"):
            self.assertEqual(self.gw.preload_check(MODEL), "Model load deferred: RAM x")
        with patch.dict(self.gw.G, {"ramPolicy": "off"}), patch.object(self.gw, "ram_admission", return_value="Model load deferred: RAM x"):
            self.assertNotEqual(self.gw.preload_check(MODEL), "Model load deferred: RAM x")

    def test_config_validation(self):
        self.gw.validate_ram_config({"ramPolicy": "off"})
        self.gw.validate_ram_config({"ramPolicy": "honest", "ramDeskReserveMiB": 6144})
        self.gw.validate_ram_config({"ramPolicy": "honest", "ramDeskReserveMiB": None, "ramWaitSeconds": 0, "ramHardMinMiB": 512})
        for bad in ({"ramWaitSeconds": -1}, {"ramWaitSeconds": 301}, {"ramWaitSeconds": 2.5}, {"ramHardMinMiB": 100},
                    {"ramHardMinMiB": 9000}):
            with self.assertRaises(ValueError):
                self.gw.validate_ram_config({"ramPolicy": "honest", **bad})
        with self.assertRaises(ValueError):
            self.gw.validate_ram_config({"ramPolicy": "bogus"})
        with self.assertRaises(ValueError):
            self.gw.validate_ram_config({"ramPolicy": "honest", "ramDeskReserveMiB": 100})
        with self.assertRaises(ValueError):
            self.gw.validate_ram_config({"ramPolicy": "honest", "ramDeskReserveMiB": 6144.5})


if __name__ == "__main__":
    unittest.main()
