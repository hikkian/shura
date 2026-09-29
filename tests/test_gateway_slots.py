import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class AtomicSlotSaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_root = tempfile.TemporaryDirectory(prefix="gateway-config-")
        root = Path(cls.config_root.name)
        config = root / "config"
        config.mkdir()
        (config / "guardian.json").write_text(
            json.dumps({"slotSaveDir": str(root / "slots"), "slotSaveMinTokens": 1})
        )
        (config / "model-launch.json").write_text(json.dumps({"models": {}}))
        cls.old_config = os.environ.get("AI_GATEWAY_CONFIG")
        cls.old_state = os.environ.get("AI_GATEWAY_STATE")
        os.environ["AI_GATEWAY_CONFIG"] = str(config)
        os.environ["AI_GATEWAY_STATE"] = str(root / "state")
        path = Path(__file__).resolve().parents[1] / "gateway" / "ai_gateway.py"
        spec = importlib.util.spec_from_file_location("gateway_atomic_test", path)
        cls.gateway = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.gateway
        spec.loader.exec_module(cls.gateway)
        if cls.old_config is None:
            os.environ.pop("AI_GATEWAY_CONFIG", None)
        else:
            os.environ["AI_GATEWAY_CONFIG"] = cls.old_config
        if cls.old_state is None:
            os.environ.pop("AI_GATEWAY_STATE", None)
        else:
            os.environ["AI_GATEWAY_STATE"] = cls.old_state

    @classmethod
    def tearDownClass(cls):
        cls.config_root.cleanup()
        sys.modules.pop("gateway_atomic_test", None)

    def setUp(self):
        self.files = tempfile.TemporaryDirectory(prefix="gateway-slot-")
        self.gateway.SLOT_DIR = Path(self.files.name)
        self.gateway.G["slotSaveMinTokens"] = 1
        self.gateway.st.model_id = "test-model"
        self.gateway.st.vision = False

    def tearDown(self):
        self.files.cleanup()

    def test_saves_pending_then_atomically_replaces_final(self):
        final = self.gateway.SLOT_DIR / "test-model-text.bin"
        pending = self.gateway.SLOT_DIR / "test-model-text.bin.pending"
        final.write_bytes(b"previous-good-slot")
        calls = []

        def fake_call(method, path, body=None, timeout=10):
            calls.append((method, path, body))
            if method == "GET":
                return 200, [{"n_prompt_tokens": 24000}]
            self.assertEqual(body["filename"], pending.name)
            pending.write_bytes(b"new-complete-slot")
            return 200, {"n_saved": 24000}

        with patch.object(self.gateway, "llama_call", side_effect=fake_call):
            self.gateway.save_slot_locked()

        self.assertEqual(final.read_bytes(), b"new-complete-slot")
        self.assertFalse(pending.exists())
        self.assertEqual(calls[1][2]["filename"], pending.name)

    def test_failed_partial_save_keeps_previous_final_and_removes_pending(self):
        final = self.gateway.SLOT_DIR / "test-model-text.bin"
        pending = self.gateway.SLOT_DIR / "test-model-text.bin.pending"
        final.write_bytes(b"previous-good-slot")

        def fake_call(method, path, body=None, timeout=10):
            if method == "GET":
                return 200, [{"n_prompt_tokens": 24000}]
            pending.write_bytes(b"partial")
            return 500, {"error": "server stopped during save"}

        with patch.object(self.gateway, "llama_call", side_effect=fake_call):
            self.gateway.save_slot_locked()

        self.assertEqual(final.read_bytes(), b"previous-good-slot")
        self.assertFalse(pending.exists())


if __name__ == "__main__":
    unittest.main()
