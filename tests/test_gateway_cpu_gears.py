"""Gateway integrations use mocked backends and private tmpfs checkpoints."""
import time
import unittest
from unittest.mock import Mock, patch
import test_gateway as fixture


class GatewayCpuGears(unittest.TestCase):
    setUpClass = classmethod(fixture.GatewayGuard.setUpClass.__func__)
    tearDownClass = classmethod(fixture.GatewayGuard.tearDownClass.__func__)
    def setUp(self):
        fixture.GatewayGuard.setUp(self)
        self.patch_attr("wait_fresh_vram", Mock(return_value=True))

    patch_attr = fixture.GatewayGuard.patch_attr
    fake_llama = fixture.GatewayGuard.fake_llama

    def test_no_first_sample_no_admission_then_fresh_recovers(self):
        gw = self.gw
        gw.G["vramGuard"] = False
        gw.st.vram_sample_at = None
        self.assertIn("Fresh NVML", gw.preload_check(gw.model_config(gw.st.model_id, False)))
        self.assertIsNotNone(gw.st.monitor_error)
        gw.monitor_tick(1, 12000, 24)
        self.assertEqual(gw.st.monitor_error, "")
        self.assertEqual(gw.preload_check(gw.model_config(gw.st.model_id, False)), "")

    def test_fake_monitor_wakes_a_waiting_first_request(self):
        gw = self.gw
        gw.st.proc = None
        gw.st.vram_sample_at = None
        def fresh(_delay):
            gw.monitor_tick(1, 12000, 24)
        with patch.object(gw.time, "sleep", side_effect=fresh), patch.object(gw, "start_llama", return_value="") as start:
            self.assertEqual(gw.acquire_model(gw.st.model_id, False), "")
        start.assert_called_once()
        self.assertEqual(gw.st.busy, 1)

    def test_busy_gap_no_restart(self):
        self.gw.st.busy = 1
        with patch.object(self.gw, "GearBackend") as backend:
            self.assertFalse(self.gw.apply_slots(24))
        backend.assert_not_called()

    def test_vision_is_not_changed_by_text_slot_policy(self):
        self.gw.st.vision = True
        with patch.object(self.gw, "GearBackend") as backend:
            self.assertFalse(self.gw.apply_slots(24))
        backend.assert_not_called()

    def test_model_overrides_are_separate(self):
        gw = self.gw
        models = {"tiel": {**gw.M["models"][gw.M["defaultModel"]], "moeCacheSlots": 26},
                  "occamy": {**gw.M["models"][gw.M["defaultModel"]], "moeCacheSlots": 26}}
        with patch.object(gw, "M", {**gw.M, "models": models}), patch.object(gw, "RUNTIME_SLOTS", {"tiel": 20}):
            self.assertEqual(gw.model_config("tiel", False)["moeCacheSlots"], 20)
            self.assertEqual(gw.model_config("occamy", False)["moeCacheSlots"], 26)
            self.assertEqual(models["tiel"]["moeCacheSlots"], 26)

    def test_snapshot_is_hardlink_and_survives_active_cleanup(self):
        gw = self.gw
        gw.G["slotSaveCheckpoints"] = 1
        with patch.object(gw, "running_footprint", return_value=(1, 10)):
            backend = gw.GearBackend()
        snapshot = backend.save()
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["path"].stat().st_ino, gw.st.ram_checkpoint["path"].stat().st_ino)
        gw.clear_ram_checkpoint()
        self.assertEqual(snapshot["path"].read_bytes(), b"test checkpoint")
        backend.release(snapshot)
        self.assertFalse(snapshot["path"].exists())

    def test_snapshot_failure_keeps_server_running(self):
        gw = self.gw
        gw.G["slotSaveCheckpoints"] = 1
        with patch.object(gw, "save_slot_locked", return_value=False), patch.object(gw, "running_footprint", return_value=(1, 10)):
            backend = gw.GearBackend()
            self.assertIsNone(backend.save())
        self.proc.terminate.assert_not_called()

    def test_transaction_refuses_changed_binary_config(self):
        gw = self.gw
        gw.G["slotSaveCheckpoints"] = 1
        with patch.object(gw, "RUNTIME_SLOTS", {}), patch.object(gw, "running_footprint", return_value=(1, 10)):
            backend = gw.GearBackend()
            snapshot = backend.save()
            with patch.object(gw, "checkpoint_signature", return_value="changed"), patch.object(gw, "stop_llama_locked") as stop:
                with self.assertRaises(RuntimeError):
                    backend.restart(20, snapshot)
                stop.assert_not_called()
            backend.release(snapshot)

    def test_gears_do_not_run_on_stale_monitor(self):
        gw = self.gw
        with patch.object(gw, "GEARS", {}), patch.object(gw, "apply_slots") as apply:
            gw.G["slotGears"] = True
            gw.st.vram_sample_at = time.time()-60
            gw.reliability_tick(0)
            gw.reliability_tick(61)
            apply.assert_not_called()

    def test_status_exposes_both_features_disabled(self):
        gw = self.gw
        gw.G["slotGears"] = False
        self.assertFalse(gw.status_snapshot()["slot_gears"]["enabled"])
        self.assertFalse(gw.status_snapshot()["polite_model"]["enabled"])

    def test_failed_new_load_restores_original_slots_and_context(self):
        gw = self.gw
        gw.G["slotSaveCheckpoints"] = 1
        model = gw.st.model_id
        original = gw.model_config(model, False)["moeCacheSlots"]
        seen = []
        def start(mid, vision, fresh=False, reclaim=(0, 0)):
            seen.append(gw.model_config(mid, vision)["moeCacheSlots"])
            gw.st.model_id, gw.st.vision = mid, vision
            if len(seen) == 1:
                return "injected new-load failure"
            gw.st.status = "READY"
            gw.st.proc = Mock(pid=4321)
            gw.st.proc.poll.return_value = None
            return gw.restore_slot_locked()
        with patch.object(gw, "RUNTIME_SLOTS", {}), patch.object(gw, "running_footprint", return_value=(1, 10)), \
                patch.object(gw, "start_llama", side_effect=start):
            self.assertFalse(gw.apply_slots(20))
            self.assertEqual(gw.model_config(model, False)["moeCacheSlots"], original)
            self.assertEqual(gw.st.status, "READY")
            self.assertIsNone(gw.st.ram_checkpoint)
        self.assertEqual(seen, [20, original])
        self.assertFalse((self.ram / "gear-rollback.bin").exists())

    def test_double_load_failure_keeps_a2_snapshot_and_original_config(self):
        gw = self.gw
        gw.G["slotSaveCheckpoints"] = 1
        model = gw.st.model_id
        original = gw.model_config(model, False)["moeCacheSlots"]
        with patch.object(gw, "RUNTIME_SLOTS", {}), patch.object(gw, "running_footprint", return_value=(1, 10)), \
                patch.object(gw, "start_llama", return_value="injected total failure"):
            with self.assertRaisesRegex(RuntimeError, "rollback both failed"):
                gw.apply_slots(20)
            self.assertEqual(gw.model_config(model, False)["moeCacheSlots"], original)
        self.assertEqual((self.ram / "gear-rollback.bin").read_bytes(), b"test checkpoint")

    def test_corrupt_budget_disables_controller(self):
        gw = self.gw
        gw.G["slotGears"] = True
        (self.ram / "gear-budget.json").write_text("not JSON")
        with patch.object(gw, "GEAR_BUDGET_LOADED", False), patch.object(gw, "apply_slots") as apply:
            gw.reliability_tick(0)
        self.assertFalse(gw.G["slotGears"])
        apply.assert_not_called()
        self.assertEqual(gw.st.last_event["reason"], "gears_disabled")

    def test_sensor_publication_does_not_wait_for_lifecycle_lock(self):
        gw = self.gw
        with gw.st.lock:
            gw.publish_vram_sample(12000, self.proc, 10600)
        self.assertEqual(gw.st.vram_free_mib, 12000)
        self.assertEqual(gw.st.proc_vram_pid, self.proc.pid)
        self.assertEqual(gw.st.proc_vram_mib, 10600)
        self.assertLess(time.time() - gw.st.vram_sample_at, 1)
