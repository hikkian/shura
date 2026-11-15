"""LAYOUT1: the gateway switches the model's layout by request size (short profile, then the longContextProfile) on a fake
server. Covers: switch to long and back, no ping-pong inside a conversation, hold time, a switch that disables itself at zero,
config validation, and a failed long start that puts the short layout back."""
import json
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gateway_harness import FAKE_BIN, GatewayHarness  # noqa: E402

DUMMY = Path("/dev/shm") / "layout-test-model.gguf"

BASE_MODEL = {
    "modelPath": str(DUMMY), "ctxSize": 32768, "parallelSlots": 1, "threads": 6, "threadsBatch": 6,
    "batchSize": 2048, "ubatchSize": 1024, "nGpuLayers": 99, "nCpuMoe": 10, "cacheTypeK": "q8_0", "cacheTypeV": "turbo4",
    "flashAttn": "on", "loadMode": "none", "moeCacheSlots": 20, "temperature": 0.6, "topK": 20, "topP": 0.95,
    "specType": "draft-mtp", "specDraftNMax": 2, "specDraftPMin": 0.2, "reasoning": "auto", "vramFreeMinGBToLoad": 0,
    "extraArgs": ["--no-mmproj-offload"],
}
LONG_PROFILE = {"ctxSize": 200000, "nCpuMoe": 14, "ubatchSize": 2048, "specType": "none", "moeCacheSlots": 20}
GUARD = {"vramGuard": True, "slotSaveCheckpoints": 1, "slotSaveMinTokens": 1}


def launch(profile=True, threshold=1000, hold=None, broken=False):
    model = dict(BASE_MODEL)
    if profile:
        p = dict(LONG_PROFILE)
        if broken:
            p["extraArgs"] = ["--fake-exit-at-start"]
        model["longContextProfile"] = p
        model["longContextSwitchTokens"] = threshold
        if hold is not None:
            model["longContextHoldSeconds"] = hold
    return {"exePath": str(FAKE_BIN), "defaultModel": "tiel-coder", "cacheRamMB": 0, "models": {"tiel-coder": model}}


def chat(h, text):
    import http.client
    body = json.dumps({"model": "tiel-coder", "messages": [{"role": "user", "content": text}], "max_tokens": 16}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", h.gw_port, timeout=60)
    conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def starts(h):
    """(ctx, mtp) of every llama-server start the gateway made, in order."""
    out = []
    for line in h.full_log().splitlines():
        if "Starting llama-server" in line:
            m = re.search(r" -c (\d+)", line)
            out.append((int(m.group(1)) if m else None, "draft-mtp" in line))
    return out


SHORT = "Коротко: привет."
LONG = "Текст: " + "слово " * 800   # about 4000 characters: over the 1000-token switch, under the 32k window
LONGER = LONG + " ещё " * 600        # the same conversation, grown


class Layout(unittest.TestCase):
    def run_gateway(self, **kw):
        h = GatewayHarness(guardian=dict(GUARD, **kw.pop("guard", {})), launch=launch(**kw))
        h.start(timeout=60)
        self.addCleanup(h.stop)
        return h

    def test_short_request_stays_on_the_short_layout(self):
        h = self.run_gateway()
        self.assertEqual(chat(h, SHORT)[0], 200)
        self.assertEqual(starts(h), [(32768, True)])

    def test_long_request_switches_to_the_long_profile_and_back_only_for_a_new_conversation(self):
        h = self.run_gateway(hold=0)
        chat(h, SHORT)
        self.assertEqual(chat(h, LONG)[0], 200)
        self.assertEqual(starts(h), [(32768, True), (200000, False)])
        # the same conversation, longer: stays long (no ping-pong)
        self.assertEqual(chat(h, LONGER)[0], 200)
        self.assertEqual(len(starts(h)), 2)
        # a new, short conversation with no hold: back to the short layout (MTP on again)
        self.assertEqual(chat(h, SHORT)[0], 200)
        self.assertEqual(starts(h), [(32768, True), (200000, False), (32768, True)])

    def test_size_counts_characters_not_escaped_json_bytes(self):
        # 2000 Cyrillic characters are about 800 tokens: short, although the escaped JSON body is about six times larger
        h = self.run_gateway(hold=0)
        chat(h, SHORT)
        text = "в" * 2000
        self.assertEqual(chat(h, text)[0], 200)
        self.assertEqual(starts(h), [(32768, True)])

    def test_status_publishes_the_switch_state(self):
        import urllib.request
        h = self.run_gateway(hold=0)
        chat(h, SHORT)
        chat(h, LONG)
        status = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{h.gw_port}/guardian/status", timeout=10).read())
        self.assertEqual(status["layout"], "long")
        self.assertEqual(status["layout_switch"]["state"], "done")
        self.assertEqual(status["layout_switch"]["to"], "long")

    def test_hold_time_keeps_the_long_layout_after_a_new_short_conversation(self):
        h = self.run_gateway(hold=3600)
        chat(h, SHORT)
        chat(h, LONG)
        chat(h, SHORT)
        self.assertEqual(starts(h), [(32768, True), (200000, False)])

    def test_snapshot_is_saved_before_the_switch_and_restored_after(self):
        h = self.run_gateway(hold=0)
        h.set_behaviour(prompt_tokens=3000, restore_tokens=3000)
        chat(h, SHORT)
        chat(h, LONG)
        lines = [l for l in h.log().splitlines() if "checkpoint_saved" in l or "checkpoint_restored" in l
                 or "Starting llama-server" in l or "switching to model" in l]
        saved = next(i for i, l in enumerate(lines) if "checkpoint_saved" in l)
        switch = next(i for i, l in enumerate(lines) if "switching to model" in l)
        long_start = next(i for i, l in enumerate(lines) if "-c 200000" in l)
        restored = next(i for i, l in enumerate(lines) if "checkpoint_restored" in l)
        self.assertLess(saved, switch)
        self.assertLess(switch, long_start)
        self.assertLess(long_start, restored)
        self.assertIn("layout_switched", h.log())

    def test_a32_to_b_switch_saves_16000_tokens_in_ram_below_disk_threshold(self):
        h = self.run_gateway(hold=0, threshold=20000, guard={"slotSaveMinTokens": 20000})
        h.set_behaviour(prompt_tokens=16000, restore_tokens=16000)
        self.assertEqual(chat(h, "seed prompt")[0], 200)
        self.assertEqual(chat(h, "x" * 51000)[0], 200)
        log = h.log()
        self.assertIn("Saved 16000 tokens to RAM", log)
        self.assertIn("Restored 16000 RAM tokens", log)
        self.assertNotIn("Saved slot (16000 tokens) to disk", log)
        slots_dir = h.tmp / "slots"
        self.assertFalse(slots_dir.exists() and any(slots_dir.iterdir()))
        self.assertEqual(starts(h), [(32768, True), (200000, False)])

    def test_switch_off_keeps_the_old_behaviour(self):
        h = self.run_gateway(threshold=0)
        chat(h, SHORT)
        chat(h, LONG)
        self.assertEqual(starts(h), [(32768, True)])

    def test_failed_long_start_puts_the_short_layout_back(self):
        h = self.run_gateway(broken=True, hold=0)
        chat(h, SHORT)
        status, data = chat(h, LONG)
        self.assertEqual(status, 503)
        self.assertIn("Could not switch to the long context", data.decode("utf-8", "replace"))
        self.assertIn("The short layout is loaded again", data.decode("utf-8", "replace"))
        # the short layout is back and serves the next request
        self.assertEqual(chat(h, SHORT)[0], 200)
        self.assertEqual(starts(h)[-1], (32768, True))

    def switch_failure_gateway(self, retry=2):
        """A server that does not report the A2 slot-size estimate (the state A2PORT1 found in the candidate build)."""
        h = self.run_gateway(hold=0, guard={"layoutSwitchRetrySeconds": retry})
        h.set_behaviour(prompt_tokens=3000, restore_tokens=3000, no_slot_estimate=True)
        self.assertEqual(chat(h, SHORT)[0], 200)
        return h

    def error_of(self, data):
        return json.loads(data)["error"]

    def test_a_failed_switch_is_answered_with_its_own_code_and_a_retry_after(self):
        import http.client
        h = self.switch_failure_gateway()
        body = json.dumps({"model": "tiel-coder", "messages": [{"role": "user", "content": LONG}], "max_tokens": 16}).encode()
        conn = http.client.HTTPConnection("127.0.0.1", h.gw_port, timeout=60)
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        self.assertEqual(resp.status, 503)
        self.assertEqual(self.error_of(data)["code"], "SHURA_LAYOUT_SWITCH_FAILED")
        self.assertIn("A2 slot-size estimate", self.error_of(data)["message"])
        self.assertIn("short layout is loaded again", self.error_of(data)["message"])
        self.assertEqual(resp.getheader("Retry-After"), "2")
        conn.close()
        self.assertEqual(starts(h)[0], (32768, True))
        self.assertEqual(chat(h, SHORT)[0], 200)      # the short layout still serves
        import urllib.request
        status = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{h.gw_port}/guardian/status", timeout=10).read())
        self.assertEqual(status["layout"], "short")
        self.assertEqual(status["layout_switch"]["state"], "failed")
        self.assertIn("A2 slot-size estimate", status["layout_switch"]["error"])

    def test_a_failed_switch_is_not_repeated_inside_the_retry_window(self):
        h = self.switch_failure_gateway(retry=3)
        first = chat(h, LONG)
        self.assertEqual(first[0], 503)
        before = len(starts(h))
        second = chat(h, LONG)                         # inside the window: refused at once, nothing is stopped or started
        self.assertEqual(second[0], 503)
        self.assertEqual(self.error_of(second[1])["code"], "SHURA_LAYOUT_SWITCH_FAILED")
        self.assertIn("not tried again", self.error_of(second[1])["message"])
        self.assertEqual(len(starts(h)), before)
        self.assertEqual(chat(h, SHORT)[0], 200)
        self.assertEqual(len(starts(h)), before)

    def test_the_switch_is_tried_again_after_the_window_and_works_once_the_server_reports_the_estimate(self):
        h = self.switch_failure_gateway(retry=1)
        self.assertEqual(chat(h, LONG)[0], 503)
        h.set_behaviour(no_slot_estimate=False)
        time.sleep(1.3)
        status, data = chat(h, LONG)
        self.assertEqual(status, 200, data)
        self.assertEqual(starts(h)[-1], (200000, False))

    def test_the_retry_window_is_validated(self):
        for bad in (0, 3601, "x", 1.5):
            h = GatewayHarness(guardian=dict(GUARD, layoutSwitchRetrySeconds=bad), launch=launch())
            with self.assertRaises(Exception):
                h.start(timeout=20)
            h.stop()

    def test_profile_must_be_larger_than_the_short_window(self):
        bad = launch()
        bad["models"]["tiel-coder"]["longContextProfile"]["ctxSize"] = 16000
        h = GatewayHarness(guardian=dict(GUARD), launch=bad)
        with self.assertRaises(RuntimeError) as cm:
            h.start(timeout=60)
        self.assertIn("longContextProfile.ctxSize must be larger than ctxSize", str(cm.exception))

    def test_switch_threshold_must_be_below_the_short_window(self):
        bad = launch(threshold=40000)
        h = GatewayHarness(guardian=dict(GUARD), launch=bad)
        with self.assertRaises(RuntimeError) as cm:
            h.start(timeout=60)
        self.assertIn("longContextSwitchTokens must be below ctxSize", str(cm.exception))


class SettleAndNotice(unittest.TestCase):
    """LAYOUT1b, in process: the VRAM settle wait and the desktop notice before a layout switch (import as in test_gateway)."""

    @classmethod
    def setUpClass(cls):
        import test_gateway as fixture
        cls.created_import = not hasattr(fixture.GatewayGuard, "gw")
        if cls.created_import:
            fixture.GatewayGuard.setUpClass()
        cls.gw = fixture.GatewayGuard.gw

    @classmethod
    def tearDownClass(cls):
        import test_gateway as fixture
        if cls.created_import:
            fixture.GatewayGuard.tearDownClass()

    def samples(self, values):
        """A free-VRAM reader that returns the given MiB values in turn, then the last one."""
        it = iter(values)
        last = [values[-1]]

        def read():
            try:
                last[0] = next(it)
            except StopIteration:
                pass
            return last[0] / 1024.0
        return read

    def test_settle_waits_until_the_card_stops_giving_memory_back(self):
        # the memory comes back over a few samples, then stays: settled at the plateau, not at the first sample
        reader = self.samples([2000, 2500, 3900, 4100, 4100, 4100, 4100, 4100, 4100, 4100])
        with patch.object(self.gw, "vram_free_gb", side_effect=reader):
            got = self.gw.wait_for_vram_settle(timeout=5.0, quiet=0.3, interval=0.01)
        self.assertEqual(got, 4100)

    def test_settle_caps_at_the_timeout_when_memory_keeps_rising(self):
        counter = [0]

        def rising():
            counter[0] += 50
            return (2000 + counter[0]) / 1024.0
        start = time.monotonic()
        with patch.object(self.gw, "vram_free_gb", side_effect=rising):
            got = self.gw.wait_for_vram_settle(timeout=0.3, quiet=5.0, interval=0.01)
        self.assertIsNone(got)
        self.assertLess(time.monotonic() - start, 2.0)

    def test_settle_returns_at_once_when_the_card_cannot_be_read(self):
        with patch.object(self.gw, "vram_free_gb", return_value=-1.0):
            self.assertIsNone(self.gw.wait_for_vram_settle(timeout=5.0, quiet=0.5, interval=0.01))


class CheckpointValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gateway as fixture
        cls.created_import = not hasattr(fixture.GatewayGuard, "gw")
        if cls.created_import:
            fixture.GatewayGuard.setUpClass()
        cls.gw = fixture.GatewayGuard.gw

    @classmethod
    def tearDownClass(cls):
        import test_gateway as fixture
        if cls.created_import:
            fixture.GatewayGuard.tearDownClass()

    def checkpoint(self, path, model_id):
        model = self.gw.M["models"][model_id]
        path.write_bytes(b"verified fake checkpoint")
        saved = {"model": model_id, "vision": False, "tokens": 10, "bytes": path.stat().st_size,
                 "sha256": self.gw.file_hash(path), "reload_min_gb": 0.0,
                 "signature": self.gw.checkpoint_signature(model_id, False), "saved_at": time.time()}
        self.gw.write_saved_slot_metadata(path, saved)
        return saved

    def test_integrity_rejection_identifies_size_and_hash_separately(self):
        with tempfile.TemporaryDirectory(prefix="layout-checkpoint-", dir="/dev/shm") as tmp:
            path = Path(tmp) / "slot.bin"
            saved = self.checkpoint(path, self.gw.M["defaultModel"])
            path.write_bytes(b"different size")
            with self.assertRaisesRegex(ValueError, "checkpoint size mismatch"):
                self.gw.validate_checkpoint(path, saved)
            path.write_bytes(b"X" * saved["bytes"])
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256 mismatch"):
                self.gw.validate_checkpoint(path, saved)
            path.write_bytes(b"verified fake checkpoint")
            saved["tokens"] = 0
            with self.assertRaisesRegex(ValueError, "checkpoint token count is invalid"):
                self.gw.validate_checkpoint(path, saved)

    def test_layout_switch_save_minimum_defaults_to_2000_and_is_validated(self):
        self.assertEqual(self.gw.G["layoutSwitchSaveMinTokens"], 2000)
        invalid = {**self.gw.G, "layoutSwitchSaveMinTokens": -1}
        with self.assertRaisesRegex(ValueError, "layoutSwitchSaveMinTokens"):
            self.gw.validate_load_pressure_config(invalid)

    def test_changed_sampling_or_model_identity_still_rejects_a_slot(self):
        with tempfile.TemporaryDirectory(prefix="layout-checkpoint-", dir="/dev/shm") as tmp:
            model_id = self.gw.M["defaultModel"]
            path = Path(tmp) / "slot.bin"
            self.checkpoint(path, model_id)
            model = self.gw.M["models"][model_id]
            original = dict(model)
            try:
                model["temperature"] = float(model.get("temperature", 0.6)) + 0.1
                with self.assertRaisesRegex(ValueError, "slot metadata does not match"):
                    self.gw.validate_saved_slot(path, model_id, False)
            finally:
                model.clear()
                model.update(original)

            alternate = model_id + "-different-model"
            self.gw.M["models"][alternate] = dict(model, modelPath=str(Path(tmp) / "other.gguf"))
            try:
                with self.assertRaisesRegex(ValueError, "slot metadata does not match"):
                    self.gw.validate_saved_slot(path, alternate, False)
            finally:
                del self.gw.M["models"][alternate]


if __name__ == "__main__":
    unittest.main()
