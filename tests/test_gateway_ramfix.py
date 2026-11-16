"""RAMFIX1: the gateway's own files in RAM never block a model load.

On 2026-11-16 the live gateway refused the model ("RAM not enough memory: 7.4 GiB free, the model needs 7.2 GiB and 1.0 GiB must
stay with the desktop") while its own idle-unload snapshot (2 GiB) sat in /dev/shm. Now a load first takes such files back:
catalogue copies are evicted, a saved session or the RAM checkpoint moves to a disk folder and is restored from there."""

import json
import os
import sys
import time
import unittest
import unittest.mock
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))
import ram_budget
import test_gateway as fixture

MODEL = {"ctxSize": 262144, "nCpuMoe": 18}
FIXED = ram_budget.resident_fixed_estimate_mib(18, 0)        # 7336 MiB, the "7.2 GiB" of the incident
CALM = {"some": {"avg10": 0.0, "avg60": 0.0}, "full": {"avg10": 0.0}}
SNAP = 4 * 1048576                                            # the test's "large" snapshot: 4 MiB stands for the 2 GiB one


def mem(free_mib):
    return {"MemFree": float(free_mib), "Inactive(file)": 0.0, "SwapTotal": 16384.0, "SwapFree": 16384.0, "Shmem": 0.0,
            "MemTotal": 32768.0}


def fresh_ladder():
    return {"cache_cap": None, "state": "idle", "since": 0.0, "until": 0.0, "free_mib": None, "need_mib": None,
            "reserve_mib": None, "reason": ""}


class ReleaseOrder(unittest.TestCase):
    def test_cache_copies_go_first_then_the_oldest_saved_session(self):
        entries = [{"kind": "slot", "mib": 2048, "inode": 1, "mtime": 5}, {"kind": "cache", "mib": 300, "inode": 2, "mtime": 9},
                   {"kind": "checkpoint", "mib": 900, "inode": 3, "mtime": 1}, {"kind": "slot", "mib": 100, "inode": 4, "mtime": 1}]
        order = ram_budget.release_order(entries, 10_000)
        self.assertEqual([g[0]["inode"] for g in order], [2, 4, 1, 3])
        self.assertEqual([g[0]["inode"] for g in ram_budget.release_order(entries, 200)], [2])      # stops once the deficit is covered
        self.assertEqual(ram_budget.release_order(entries, 0), [])

    def test_hard_links_are_one_group_handled_as_the_saved_session(self):
        entries = [{"kind": "cache", "mib": 2048, "inode": 7, "mtime": 1}, {"kind": "slot", "mib": 2048, "inode": 7, "mtime": 2},
                   {"kind": "cache", "mib": 500, "inode": 8, "mtime": 3}]
        order = ram_budget.release_order(entries, 10_000)
        self.assertEqual([len(g) for g in order], [1, 2])          # the lone copy first; the linked pair is moved, not dropped
        self.assertEqual(ram_budget.releasable_mib(entries), 2548)   # a hard link frees its memory once


class OwnRamFiles(unittest.TestCase):
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
        self.case = fixture.GatewayGuard("test_idle_pressure_saves_only_in_ram_then_unloads")
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        gw = self.gw
        self.slots = self.case.shm_root / "slots"           # slotSaveDir in RAM, as on the live machine
        self.slots.mkdir()
        self.spill = self.case.root / "spill"                # "the disk": the tests decide what is tmpfs
        self.catalog = self.case.shm_root / "sessions"
        shm = str(self.case.shm_root)
        self.case.patch_attr("SLOT_DIR", self.slots)
        self.case.patch_attr("is_tmpfs", lambda p: str(Path(p)).startswith(shm) or str(Path(p)).startswith(str(self.case.ram)))
        self.case.patch_attr("G", {**gw.G, "ramPolicy": "honest", "ramWaitSeconds": 20, "ramDeskReserveMiB": 2048,
                                   "slotSpillDir": str(self.spill), "ssdWriteDailyLimitGB": 0, "sessionCacheEnabled": False,
                                   "sessionCacheDir": str(self.catalog)})
        # live: --cache-ram 0; the prefill speed is the fallback 1000 tok/s (a re-read longer than 60 s = more than 60000 tokens)
        for patcher in (patch.dict(gw.RAM_LADDER, fresh_ladder()), patch.dict(gw.M, {"cacheRamMB": 0}),
                        patch.dict(gw.G, {"spillPrefillTps": 1000.0, "spillMinReprefillSeconds": 60}),
                        patch.object(gw.MONITOR, "requests", return_value=[]),
                        patch.dict(gw.SERVER_SLOT, {"dir": self.case.shm_root / "server", "owner": None}),
                        patch.dict(gw.PAUSE_STATE, {"resuming": 0, "owner": None, "recent": []})):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.events = []
        real_event = gw.event
        self.case.patch_attr("event", lambda r, d, fallback=False: (self.events.append((r, d)), real_event(r, d, fallback)))
        self.model = gw.M["defaultModel"]
        gw.st.model_id = self.model
        gw.st.proc = None
        gw.st.status = "UNLOADED"

    # --- helpers
    def saved_session(self, folder, payload=b"x" * SNAP, tokens=150000):
        gw = self.gw
        path = folder / gw.slot_filename(self.model, False)
        path.write_bytes(payload)
        saved = {"model": self.model, "vision": False, "tokens": tokens, "bytes": len(payload), "sha256": gw.file_hash(path),
                 "reload_min_gb": 0.0, "signature": gw.checkpoint_signature(self.model, False), "saved_at": time.time()}
        gw.write_saved_slot_metadata(path, saved)
        return path, saved

    def meminfo_without(self, base_free, *files):
        """The fake /proc: honest free grows by the size of each own RAM file once it is gone (as on tmpfs)."""
        def read(*_a):
            gone = sum(size for path, size in files if not Path(path).exists())
            return mem(base_free + gone)
        return read

    def admit(self, read):
        with patch.object(self.gw.ram_budget, "read_meminfo", side_effect=read), \
                patch.object(self.gw.ram_budget, "read_pressure", return_value=CALM):
            return self.gw.ram_admission(MODEL)

    # --- the incident
    def test_the_incident_the_own_snapshot_moves_to_disk_and_the_model_loads(self):
        path, _saved = self.saved_session(self.slots)
        # 7.4 GiB honest free with the 2 GiB snapshot in RAM; the model needs 7.2 + the 2 GiB reserve
        read = self.meminfo_without(7.4 * 1024, (path, 2048))
        self.assertEqual(self.admit(read), "")
        self.assertFalse(path.exists(), "the snapshot left RAM")
        moved = self.spill / path.name
        self.assertTrue(moved.is_file())
        self.assertTrue(self.gw.checkpoint_metadata_path(moved).is_file())
        self.assertEqual(self.gw.RAM_LADDER["state"], "idle")         # a plain load, no wait, no guarded step
        self.assertTrue(any(r == "ram_spill" and "moved to the disk" in d for r, d in self.events), self.events)
        # the session is not lost: the restore finds it on the disk and the server is asked to restore it
        self.assertEqual(self.gw.saved_slot_path(self.model, False), moved)
        self.gw.st.status = "READY"
        self.gw.st.ram_backend = True
        calls = []
        self.case.patch_attr("llama_call", lambda m, p, body=None, timeout=10: (calls.append((p, body)), (200, {"n_restored": 150000}))[1])
        self.assertEqual(self.gw.restore_slot_locked(), "")
        self.assertTrue(any("action=restore" in p for p, _b in calls))
        self.assertFalse(moved.exists(), "read back from the SSD: deleted there at once")
        self.assertTrue(self.gw.st.slot_dirty, "the next unload saves it again where it fits")
        self.assertFalse((self.case.ram / "restore.bin").exists())     # the staging copy in RAM is gone after the restore

    def test_without_the_fix_the_same_reading_was_a_wait_then_a_refusal(self):
        path, _saved = self.saved_session(self.slots)
        read = self.meminfo_without(7.4 * 1024, (path, 2048))
        with patch.dict(self.gw.G, {"shmSpill": False}):
            self.assertEqual(self.gw.refusal_code(self.admit(read)), "ram_wait")
            self.gw.RAM_LADDER["until"] = time.time() - 1                  # the wait is over
            message = self.admit(read)
        self.assertEqual(self.gw.refusal_code(message), "ram")
        self.assertIn("must stay with the desktop", message)
        self.assertTrue(path.exists())

    def test_a_real_shortage_still_climbs_the_ladder_and_touches_nothing_when_nothing_is_own(self):
        read = self.meminfo_without(FIXED + 1300)
        self.assertEqual(self.gw.refusal_code(self.admit(read)), "ram_wait")
        self.gw.RAM_LADDER["until"] = time.time() - 1
        self.assertEqual(self.admit(read), "")                            # guarded load above the hard minimum
        self.assertEqual(self.gw.RAM_LADDER["state"], "guarded")
        self.assertFalse(any(r == "ram_spill" for r, _d in self.events))

    def test_plenty_of_memory_moves_nothing(self):
        path, _saved = self.saved_session(self.slots)
        self.assertEqual(self.admit(self.meminfo_without(20 * 1024, (path, 2048))), "")
        self.assertTrue(path.exists())
        self.assertFalse(self.spill.exists())

    def test_catalogue_copies_are_evicted_before_the_saved_session_moves(self):
        gw = self.gw
        gw.G["sessionCacheEnabled"] = True
        catalog = gw.session_catalog()
        sid = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        (catalog.root / f"{sid}.bin").write_bytes(b"c" * SNAP)
        (catalog.root / f"{sid}.json").write_text(json.dumps({"session_id": sid, "size": SNAP, "last_used": 1}))
        path, _saved = self.saved_session(self.slots)
        need = FIXED + 2048
        read = self.meminfo_without(need - 2, (catalog.root / f"{sid}.bin", 4), (path, 4))   # 2 MiB short: the copy is enough
        self.assertEqual(self.admit(read), "")
        self.assertFalse((catalog.root / f"{sid}.bin").exists())
        self.assertTrue(path.exists(), "the saved session stays in RAM when the cache copy was enough")

    def test_a_hard_linked_catalogue_copy_moves_with_the_saved_session(self):
        gw = self.gw
        gw.G["sessionCacheEnabled"] = True
        catalog = gw.session_catalog()
        sid = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        path, saved = self.saved_session(self.slots)
        os.link(path, catalog.root / f"{sid}.bin")
        (catalog.root / f"{sid}.json").write_text(json.dumps({"session_id": sid, "size": SNAP, "last_used": 1}))
        entries = gw.own_tmpfs_entries()
        self.assertEqual(len(entries), 2)
        self.assertAlmostEqual(ram_budget.releasable_mib(entries), SNAP / 1048576, delta=0.1)
        read = self.meminfo_without(7.4 * 1024, (path, 2048))
        self.assertEqual(self.admit(read), "")
        self.assertFalse(path.exists())
        self.assertFalse((catalog.root / f"{sid}.bin").exists())
        self.assertEqual(gw.validate_saved_slot(self.spill / path.name, self.model, False)["tokens"], saved["tokens"])

    def test_the_ram_checkpoint_of_a_pressure_unload_moves_to_disk_and_is_restored(self):
        gw = self.gw
        active = self.case.ram / "active.bin"
        active.write_bytes(b"k" * SNAP)
        checkpoint = {"model": self.model, "vision": False, "tokens": 90000, "bytes": SNAP, "sha256": gw.file_hash(active),
                      "reload_min_gb": 0.0, "signature": gw.checkpoint_signature(self.model, False), "saved_at": time.time(),
                      "path": active}
        gw.st.ram_checkpoint = checkpoint
        read = self.meminfo_without(7.4 * 1024, (active, 2048))
        self.assertEqual(self.admit(read), "")
        self.assertIsNone(gw.st.ram_checkpoint)
        self.assertFalse(active.exists())
        moved = self.spill / gw.slot_filename(self.model, False)
        self.assertEqual(gw.validate_saved_slot(moved, self.model, False)["tokens"], 90000)

    def test_over_the_daily_ssd_budget_the_session_is_dropped_and_the_model_still_loads(self):
        path, _saved = self.saved_session(self.slots)
        self.gw.G["ssdWriteDailyLimitGB"] = 1e-9
        read = self.meminfo_without(7.4 * 1024, (path, 2048))
        self.assertEqual(self.admit(read), "")
        self.assertFalse(path.exists())
        self.assertFalse((self.spill / path.name).exists())
        self.assertTrue(any(r == "ram_spill" and "daily SSD write limit" in d for r, d in self.events), self.events)

    def test_a_spill_folder_in_ram_is_refused_the_session_is_dropped_and_the_model_loads(self):
        # moving there would free nothing: the session is dropped (re-read later) rather than leaving the model unloaded
        path, _saved = self.saved_session(self.slots)
        self.gw.G["slotSpillDir"] = str(self.case.shm_root / "not-a-disk")
        read = self.meminfo_without(7.4 * 1024, (path, 2048))
        self.assertEqual(self.admit(read), "")
        self.assertFalse(path.exists())
        self.assertFalse((self.case.shm_root / "not-a-disk" / path.name).exists())
        self.assertTrue(any(r == "ram_spill_failed" and "in RAM" in d for r, d in self.events), self.events)

    # --- the save side
    def test_a_large_save_goes_to_disk_when_a_reload_with_it_in_ram_would_not_fit(self):
        gw = self.gw
        gw.G["shmSpillMinMiB"] = 1
        mc = gw.model_config(self.model, False, gw.st.layout)
        FIXED = ram_budget.resident_fixed_estimate_mib(mc.get("nCpuMoe", 0), 0)      # the configured model's own number
        with patch.object(gw.ram_budget, "read_meminfo", return_value=mem(FIXED + 2048)), \
                patch.object(gw.ram_budget, "read_pressure", return_value=CALM), \
                patch.object(gw, "running_footprint", return_value=(0.0, 0.0)):
            self.assertEqual(gw.slot_folder_for(SNAP, 150000), self.spill)
            self.assertEqual(gw.slot_folder_for(SNAP // 8, 150000), self.slots)        # below shmSpillMinMiB: stays in RAM
            self.assertEqual(gw.slot_folder_for(SNAP, 30000), self.slots)   # a 30 s re-read is not worth an SSD write
        with patch.object(gw.ram_budget, "read_meminfo", return_value=mem(FIXED + 2048 + 64)), \
                patch.object(gw.ram_budget, "read_pressure", return_value=CALM), \
                patch.object(gw, "running_footprint", return_value=(0.0, 0.0)):
            self.assertEqual(gw.slot_folder_for(SNAP, 150000), self.slots)              # it fits: RAM, no SSD write
        with patch.object(gw.ram_budget, "read_meminfo", return_value=mem(2048)), \
                patch.object(gw.ram_budget, "read_pressure", return_value=CALM), \
                patch.object(gw, "running_footprint", return_value=((FIXED + 64) / 1024, 0.0)):
            self.assertEqual(gw.slot_folder_for(SNAP, 150000), self.slots)   # the running server's memory comes back at the unload

    def test_an_idle_save_to_disk_and_a_later_save_to_ram_keep_one_copy(self):
        gw = self.gw
        gw.G["shmSpillMinMiB"] = 0
        gw.st.status = "READY"
        gw.st.ram_backend = True
        gw.st.slot_dirty = True
        gw.st.proc = self.case.proc
        name = gw.slot_filename(self.model, False)
        with patch.object(gw, "slot_folder_for", return_value=self.spill):
            self.spill.mkdir()
            self.assertTrue(gw.save_slot_locked())
        self.assertTrue((self.spill / name).is_file())
        self.assertFalse((self.slots / name).exists())
        self.assertEqual(gw.saved_slot_path(self.model, False), self.spill / name)
        gw.st.slot_dirty = True
        with patch.object(gw, "slot_folder_for", return_value=self.slots):
            self.assertTrue(gw.save_slot_locked())
        self.assertTrue((self.slots / name).is_file())
        self.assertFalse((self.spill / name).exists(), "the older disk copy is removed")
        self.assertEqual(gw.saved_slot_path(self.model, False), self.slots / name)

    def test_without_a_ram_backend_a_spilled_session_is_staged_back_for_the_server(self):
        gw = self.gw
        self.spill.mkdir()
        moved, _saved = self.saved_session(self.spill)
        gw.st.status = "READY"
        gw.st.ram_backend = False
        calls = []
        self.case.patch_attr("llama_call", lambda m, p, body=None, timeout=10: (calls.append(body), (200, {"n_restored": 150000}))[1])
        self.assertEqual(gw.restore_slot_locked(), "")
        self.assertEqual(calls[-1]["filename"], moved.name)
        self.assertTrue((self.slots / moved.name).is_file())
        self.assertFalse(moved.exists())


    # --- SSD writes kept to the minimum
    def test_a_session_quicker_to_re_read_than_a_minute_is_dropped_not_written(self):
        path, _saved = self.saved_session(self.slots, tokens=20000)       # 20 s at 1000 tok/s
        read = self.meminfo_without(7.4 * 1024, (path, 2048))
        self.assertEqual(self.admit(read), "")
        self.assertFalse(path.exists())
        self.assertFalse(self.spill.exists(), "no SSD write")
        self.assertTrue(any(r == "ram_spill" and "cheaper than an SSD write" in d for r, d in self.events), self.events)

    def test_the_measured_prefill_speed_decides(self):
        gw = self.gw
        long_prompt = {"prompt_per_second": 3000.0, "prompt_tokens": 90000, "cached_tokens": 10000}
        with patch.object(gw.MONITOR, "requests", return_value=[{"prompt_per_second": 50.0, "prompt_tokens": 900}, long_prompt]):
            self.assertEqual(gw.prefill_tps(), 3000.0)                      # a short prompt says nothing about a re-read
            self.assertFalse(gw.worth_the_disk(150000)[0])                  # 50 s at the measured 3000 tok/s
        self.assertTrue(gw.worth_the_disk(150000)[0])                       # 150 s at the fallback 1000 tok/s

    # --- what a governor pause leaves behind (2026-11-16 14:43)
    def pause_files(self, tokens=89468):
        ram = self.case.shm_root / "server"          # the server's --slot-save-path: the same tmpfs as the slot folder
        ram.mkdir(exist_ok=True)
        checkpoint = ram / "vram-pause.bin"
        checkpoint.write_bytes(b"p" * SNAP)
        (ram / "vram-pause.bin.tokens.json").write_text(json.dumps({"n_past": tokens, "tokens": [1] * 3, "patch_set": "", "patch_hash": ""}))
        (ram / "vram-exit.json").write_text('{"exit_code": 75}')
        return checkpoint

    def test_an_unresumed_pause_checkpoint_becomes_the_saved_session_without_a_write(self):
        gw = self.gw
        self.pause_files()
        gw.PAUSE_STATE["owner"] = (self.model, False, gw.st.layout)
        gw.settle_pause_leftovers("test")
        self.assertEqual(gw.pause_leftovers(), [], "nothing of the pause stays in the server folder")
        saved = gw.validate_saved_slot(self.slots / gw.slot_filename(self.model, False), self.model, False)
        self.assertEqual(saved["tokens"], 89468)
        self.assertFalse(self.spill.exists(), "a rename inside tmpfs: no SSD write")
        # it is now an ordinary saved session in RAM: the first thing a short load takes back
        path = self.slots / gw.slot_filename(self.model, False)
        self.assertEqual(self.admit(self.meminfo_without(7.4 * 1024, (path, 2048))), "")
        self.assertFalse(path.exists())
        self.assertTrue((self.spill / path.name).is_file())                 # 89k tokens: a re-read of about 89 s, worth the disk

    def test_a_pause_checkpoint_of_an_unknown_owner_is_deleted(self):
        self.pause_files()
        self.gw.PAUSE_STATE["owner"] = None                                  # e.g. the gateway restarted since
        self.gw.settle_pause_leftovers("test")
        self.assertEqual(self.gw.pause_leftovers(), [])
        self.assertEqual(list(self.slots.iterdir()), [])
        self.assertFalse(self.spill.exists())
        self.assertTrue(any(r == "pause_leftover" and "owner is unknown" in d for r, d in self.events), self.events)

    def test_pause_files_next_to_a_running_server_are_spent_and_deleted(self):
        gw = self.gw
        self.pause_files()
        gw.PAUSE_STATE["owner"] = (self.model, False, gw.st.layout)
        gw.st.proc = self.case.proc
        self.case.proc.poll.return_value = None
        gw.st.proc_wall_start = time.time() - 60                             # the files were written after this server started:
        gw.settle_pause_leftovers("test")                                    # its own pause in progress
        self.assertEqual(len(gw.pause_leftovers()), 3, "a pausing server's files are not touched while it runs")
        gw.st.proc_wall_start = time.time() + 60                             # a newer server: the files are an earlier, spent pause
        gw.settle_pause_leftovers("test")
        self.assertEqual(gw.pause_leftovers(), [])
        self.assertEqual(list(self.slots.iterdir()), [], "the running server holds the newer state")

    def test_a_resume_in_progress_keeps_its_checkpoint(self):
        self.pause_files()
        self.gw.PAUSE_STATE["resuming"] = 1
        self.gw.settle_pause_leftovers("test")
        self.assertEqual(len(self.gw.pause_leftovers()), 3)

    def test_the_load_takes_the_pause_checkpoint_into_account(self):
        gw = self.gw
        self.pause_files(tokens=30000)                                       # a 30 s re-read: dropped when RAM is short
        gw.PAUSE_STATE["owner"] = (self.model, False, gw.st.layout)
        name = gw.slot_filename(self.model, False)
        # the load settles the pause first (the checkpoint becomes the saved session), then reads /proc: 2 GiB short until it goes
        self.assertEqual(self.admit(self.meminfo_without(7.4 * 1024, (self.slots / name, 2048))), "")
        self.assertFalse((self.slots / name).exists())
        self.assertFalse(self.spill.exists(), "no SSD write")
        self.assertEqual(gw.pause_leftovers(), [])

    def test_an_exit_75_seen_by_the_monitor_is_a_pause_not_a_crash(self):
        gw = self.gw
        self.pause_files()
        gw.PAUSE_STATE["owner"] = (self.model, False, gw.st.layout)
        gw.st.proc = self.case.proc
        self.case.proc.poll.return_value = 75
        gw.st.expected_exit = False
        gw.st.busy = 0
        gw.st.status = "READY"
        gw.PAUSE_STATE["owner"] = None
        gw.SERVER_SLOT["owner"] = (self.model, False, gw.st.layout)       # the server that just exited 75
        with patch.object(gw, "governor_on", return_value=True), patch.object(gw, "note_crash") as crash, \
                patch.object(gw, "record_server_crash") as report:
            gw.monitor_tick(10, 9000, 24)
        self.wait_for_file(self.slots / gw.slot_filename(self.model, False))
        crash.assert_not_called()
        report.assert_not_called()
        self.assertEqual(gw.st.status, "UNLOADED")
        self.assertTrue(any(r == "vram_pause_exit" for r, _d in self.events), self.events)
        self.assertFalse(any(r == "server_crashed" for r, _d in self.events))
        self.assertEqual(gw.pause_leftovers(), [])
        self.assertTrue((self.slots / gw.slot_filename(self.model, False)).is_file(), "the 89k-token session is kept")

    def wait_for_file(self, path, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end and not (path.is_file() and self.gw.checkpoint_metadata_path(path).is_file()):
            time.sleep(0.01)

    def join_release(self):
        worker = self.gw.OWN_WORK["thread"]
        if worker is not None:
            worker.join(10)

    # --- the valve's tight level and the desktop's floor (4 GiB MemAvailable)
    def readings(self, *values):
        seq = list(values)
        return patch.object(self.gw.ram_valve, "read_readings", side_effect=lambda *_a: (seq.pop(0) if len(seq) > 1 else seq[0], 0.0, 0.0))

    def test_a_large_prompt_at_tight_goes_on_while_the_floor_holds(self):
        with self.readings(5.4):                                             # tight (< 6) but 5.4 - 0.12 >= 4
            self.assertTrue(self.gw.room_for_prompt(122100))
        with self.readings(4.05):
            self.assertFalse(self.gw.room_for_prompt(122100))                # nothing own to give back

    def test_below_the_floor_the_own_snapshot_goes_first_then_the_prompt_goes_on(self):
        path, _saved = self.saved_session(self.slots)
        with self.readings(3.9):
            self.assertFalse(self.gw.room_for_prompt(122100))               # the release runs in the background, no st.lock held
        self.join_release()
        self.assertFalse(path.exists())
        with self.readings(5.9):
            self.assertTrue(self.gw.room_for_prompt(122100))

    def test_memory_pressure_or_zram_also_keep_the_prompt_waiting(self):
        gw = self.gw
        with patch.object(gw.ram_valve, "read_readings", return_value=(8.0, 30.0, 0.0)):
            self.assertFalse(gw.room_for_prompt(122100))                    # PSI full 30 % over the tight 5 %
        with patch.object(gw.ram_valve, "read_readings", return_value=(8.0, 0.0, 75.0)):
            self.assertFalse(gw.room_for_prompt(122100))                    # zram 75 % over the tight 60 %

    def test_a_waiting_prompt_counts_as_busy(self):
        gw = self.gw
        seen = []
        with self.readings(3.5, 3.5, 5.0), patch.object(gw.time, "sleep", side_effect=lambda _s: seen.append(gw.st.busy)):
            self.assertTrue(gw.wait_room_for_prompt(122100))
        self.assertTrue(seen and all(b == 1 for b in seen), seen)          # no idle unload while it waits
        self.assertEqual(gw.st.busy, 0)

    def test_the_gateway_waits_itself_instead_of_a_503(self):
        gw = self.gw
        with self.readings(3.5, 3.5, 3.5, 5.0), patch.object(gw.time, "sleep"):
            self.assertTrue(gw.wait_room_for_prompt(122100))
        self.assertTrue(any(r == "ram_valve_defer_wait" for r, _d in self.events))
        self.assertTrue(any(r == "ram_valve_defer_done" for r, _d in self.events))
        gw.G["ramValveDeferWaitSeconds"] = 0
        with self.readings(3.5):
            self.assertFalse(gw.wait_room_for_prompt(122100))                # never held within the wait: the honest 503

    def test_a_loaded_model_below_the_floor_gives_back_own_files_first(self):
        gw = self.gw
        path, _saved = self.saved_session(self.slots)
        gw.st.proc = self.case.proc
        self.case.proc.poll.return_value = None
        gw.st.status = "READY"
        with self.readings(3.6, 5.6), patch.dict(gw.RAM_NEXT, {"at": 0.0, "own_at": 0.0}):
            gw.ram_valve_tick(100.0, 0)
        self.join_release()
        self.assertFalse(path.exists())
        self.assertTrue((self.spill / path.name).is_file())

    # --- a kernel without swappiness= in memory.reclaim (the laptop): a clear event for the monitor, never a silent refusal
    def test_a_kernel_without_the_swappiness_key_says_so_in_the_events(self):
        import errno
        gw = self.gw
        target = self.case.root / "memory.reclaim"
        target.write_text("")
        writes = []

        def refuse(fd, data):
            writes.append(data)
            raise OSError(errno.EINVAL, "Invalid argument")
        mem = {"MemFree": 500.0, "Active(file)": 3000.0, "Inactive(file)": 6000.0, "Dirty": 0.0}
        with patch.object(gw.cache_reclaim, "find_reclaim_file", return_value=target), \
                patch.object(gw.cache_reclaim.os, "write", side_effect=refuse), \
                patch.object(gw.cache_reclaim, "meminfo_mib", return_value=mem), \
                patch.dict(gw.RAM_RECLAIM, {}, clear=True):
            self.assertFalse(gw.ram_try_reclaim(2000, mem, "shortfall 2000 MiB for the model"))
        self.assertEqual(len(writes), 1, "one try, no plain reclaim afterwards (it may swap)")
        self.assertIn(b"swappiness=0", writes[0])
        reasons = [(e["reason"], e["detail"]) for e in gw.st.events]
        self.assertTrue(any(r == "ram_cache_reclaim" and "does not accept swappiness=" in d for r, d in reasons), reasons)

    # --- the review (Sonnet), 2026-11-16
    def test_the_owner_is_the_server_that_exited_75_not_the_next_one(self):
        gw = self.gw
        self.pause_files()
        a, b = (self.model, False, "short"), (self.model, True, "short")
        gw.SERVER_SLOT["owner"] = a
        dead = unittest.mock.Mock(pid=4321)
        dead.poll.return_value = 75
        with patch.object(gw, "governor_on", return_value=True):
            self.assertTrue(gw.paused_exit(dead))
        gw.SERVER_SLOT["owner"] = b                                          # B starts and dies at once
        gw.settle_pause_leftovers("test")
        self.assertTrue((self.slots / gw.slot_filename(self.model, False)).is_file(), "A's session")
        self.assertFalse((self.slots / gw.slot_filename(self.model, True)).exists())

    def test_an_exit_75_without_the_pause_marker_or_again_and_again_is_a_failure(self):
        gw = self.gw
        dead = unittest.mock.Mock(pid=4321)
        dead.poll.return_value = 75
        with patch.object(gw, "governor_on", return_value=True):
            self.assertFalse(gw.paused_exit(dead), "no vram-exit.json, no vram-pause.bin: not a pause")
            self.pause_files()
            self.assertTrue(gw.paused_exit(dead))
            self.assertTrue(gw.paused_exit(dead))
            self.assertFalse(gw.paused_exit(dead), "the third pause within 300 s counts as a failure")
        self.assertTrue(any(r == "vram_pause_repeated" for r, _d in self.events))

    def test_a_failed_adoption_leaves_no_file_behind(self):
        gw = self.gw
        self.pause_files()
        gw.PAUSE_STATE["owner"] = (self.model, False, gw.st.layout)
        with patch.object(gw, "file_hash", side_effect=OSError("read error")):
            gw.settle_pause_leftovers("test")
        self.assertEqual(sorted(p.name for p in self.slots.iterdir()), [], "neither the private nor a final file")
        self.assertEqual(gw.pause_leftovers(), [])
        self.assertTrue(any(r == "pause_leftover_failed" for r, _d in self.events))

    def test_the_ssd_budget_is_reserved_before_the_write_and_once(self):
        budget = self.gw.ram_valve.SsdDailyBudget(1e-9 * 0 + 4 / 1024, self.case.root / "budget.json")    # 4 MiB a day
        self.assertTrue(budget.reserve(3 * 1048576))
        self.assertFalse(budget.reserve(2 * 1048576), "the second writer sees the first one's reservation")
        self.assertEqual(self.gw.ram_valve.DEFAULTS["ssdWriteDailyLimitGB"], 8)

    def test_a_restore_through_the_link_that_fails_is_tried_once_more_with_a_copy(self):
        gw = self.gw
        self.spill.mkdir()
        moved, _saved = self.saved_session(self.spill)
        gw.st.status, gw.st.ram_backend = "READY", True
        answers = [(500, {}), (200, {"n_restored": 150000})]
        kinds = []

        def call(m, p, body=None, timeout=10):
            kinds.append((self.case.ram / "restore.bin").is_symlink())
            return answers.pop(0)
        self.case.patch_attr("llama_call", call)
        cross_device = patch.object(gw.os, "link", side_effect=OSError(18, "Invalid cross-device link"))   # ext4 -> tmpfs
        with patch.object(gw, "is_tmpfs", return_value=False), self.readings(20.0), cross_device:
            result = gw.restore_slot_locked()
        self.assertEqual(result, "", f"{kinds} {self.events}")
        self.assertEqual(kinds, [True, False], f"first the link, then a copy {self.events}")
        self.assertFalse(moved.exists(), "restored: the SSD copy goes")

    def test_no_room_for_the_retry_copy_keeps_the_ssd_session(self):
        gw = self.gw
        self.spill.mkdir()
        moved, _saved = self.saved_session(self.spill)
        gw.st.status, gw.st.ram_backend = "READY", True
        self.case.patch_attr("llama_call", lambda m, p, body=None, timeout=10: (500, {}))
        shm = str(self.case.shm_root)
        ram = str(self.case.ram)
        with patch.object(gw, "is_tmpfs", side_effect=lambda q: str(Path(q)).startswith((shm, ram))), self.readings(4.001), \
                patch.object(gw.os, "link", side_effect=OSError(18, "Invalid cross-device link")):
            self.assertEqual(gw.restore_slot_locked(), "restart_empty")
        self.assertTrue(moved.is_file(), "the session on the SSD is not at fault")

    def test_the_reclaim_asks_for_twice_the_shortfall_within_the_reclaimable_cache(self):
        gw = self.gw
        asked = []
        mem = {"MemFree": 500.0, "Active(file)": 0.0, "Inactive(file)": 9000.0, "Dirty": 0.0}
        with patch.object(gw.cache_reclaim, "reclaim_page_cache", side_effect=lambda want, *_a, **_k: (asked.append(want), (want, ""))[1]), \
                patch.dict(gw.RAM_RECLAIM, {}, clear=True):
            self.assertTrue(gw.ram_try_reclaim(1500, mem, "test"))
            self.assertFalse(gw.ram_try_reclaim(1500, mem, "test"), "10 s per caller")
            self.assertTrue(gw.ram_try_reclaim(4400, mem, "test", key="other"))
        self.assertEqual(asked, [2 * 1500 + 512, 9000])

    def test_a_reclaim_write_that_hangs_is_abandoned_after_its_time(self):
        gw = self.gw
        target = self.case.root / "memory.reclaim"
        target.write_text("")
        release = __import__("threading").Event()
        self.addCleanup(release.set)
        mem = {"MemFree": 500.0, "Active(file)": 3000.0, "Inactive(file)": 6000.0, "Dirty": 0.0}
        started = time.monotonic()
        with patch.object(gw.cache_reclaim, "find_reclaim_file", return_value=target), \
                patch.object(gw.cache_reclaim.os, "write", side_effect=lambda fd, data: release.wait(5)), \
                patch.object(gw.cache_reclaim, "meminfo_mib", return_value=mem), \
                patch.object(gw.cache_reclaim, "MAX_WRITE_SECONDS", 0.3):
            freed, error = gw.cache_reclaim.reclaim_page_cache(1000)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(freed, 0)
        self.assertIn("did not return within", error)

    # --- found by the live check: MemAvailable was not parsed, so the desktop-floor branch of the load never ran
    def test_the_load_keeps_the_desktop_floor_with_a_real_meminfo(self):
        gw = self.gw
        path, _saved = self.saved_session(self.slots)
        mc = MODEL
        resident = ram_budget.resident_fixed_estimate_mib(mc["nCpuMoe"], 0)

        def text(*_a):
            held = 2048 if path.exists() else 0
            free = resident + 2048 + 3000 - held            # the honest budget fits even with the session in RAM
            kb = lambda mib: int(mib * 1024)  # noqa: E731
            return ram_budget.parse_meminfo_mib(
                f"MemTotal: {kb(32768)} kB\nMemFree: {kb(free)} kB\nMemAvailable: {kb(free + 1000)} kB\nInactive(file): 0 kB\n"
                f"SwapTotal: {kb(16384)} kB\nSwapFree: {kb(16384)} kB\n")
        self.assertIn("MemAvailable", text())
        self.assertEqual(self.admit(text), "")
        # with the session in RAM MemAvailable after the load would be 3000 - 2048 + 1000 + 2048 = below 4 GiB: it goes first
        self.assertFalse(path.exists())
        self.assertTrue(any(r == "ram_spill" for r, _d in self.events), self.events)

    # --- a laptop without a writable user@ cgroup (cgroup v1, another init): an event, and the gateway works as before
    def test_without_a_reclaim_file_the_load_goes_on_as_before(self):
        gw = self.gw
        mem = {"MemFree": 500.0, "Active(file)": 3000.0, "Inactive(file)": 6000.0, "Dirty": 0.0}
        with patch.object(gw.cache_reclaim, "find_reclaim_file", return_value=None), \
                patch.object(gw.cache_reclaim, "meminfo_mib", return_value=mem), \
                patch.dict(gw.RAM_RECLAIM, {}, clear=True):
            self.assertFalse(gw.ram_try_reclaim(2000, mem, "shortfall 2000 MiB for the model"))
            read = self.meminfo_without(FIXED + 1300)                         # short: the ordinary ladder (wait, then guarded)
            self.assertEqual(gw.refusal_code(self.admit(read)), "ram_wait")
            gw.RAM_LADDER["until"] = time.time() - 1
            self.assertEqual(self.admit(read), "")
        reasons = [(e["reason"], e["detail"]) for e in gw.st.events]
        self.assertTrue(any(r == "ram_cache_reclaim" and "no writable memory.reclaim" in d for r, d in reasons), reasons)


if __name__ == "__main__":
    unittest.main()
