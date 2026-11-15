"""DEPLOY3: a saved session is tied to the KV cache types of its layout, and a restart after a failed restore keeps the layout.

Found on the live smoke of shura.cpp 1.0: the long layout keeps K/V in q5_0 (set through its extraArgs), the short one q8_0/q4_0. The
checkpoint signature was computed from the short layout only, so the long server was asked to read the short session
("mismatched key type"), and the restart that followed came up in the short layout. Offline: no server starts."""
import unittest.mock
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["AI_GATEWAY_KERNEL_LOG"] = "off"
REPO = Path(__file__).resolve().parent.parent

from test_gateway_gwv1 import load_gateway  # noqa: E402  (same loader: example config in a temporary directory)


class KvIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gw = load_gateway(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_extra_args_override_the_config_fields(self):
        mc = {"cacheTypeK": "q8_0", "cacheTypeV": "q4_0",
              "extraArgs": ["--no-warmup", "-ctkd", "q8_0", "-ctvd", "q8_0", "-ctk", "q5_0", "--cache-type-v", "q5_0"]}
        self.assertEqual(self.gw.kv_identity(mc), {"k": "q5_0", "v": "q5_0", "kd": "q8_0", "vd": "q8_0"})

    def test_last_flag_wins_and_a_trailing_flag_is_ignored(self):
        mc = {"cacheTypeK": "f16", "cacheTypeV": "f16", "extraArgs": ["-ctk", "q4_0", "-ctk", "q8_0", "-ctv"]}
        self.assertEqual(self.gw.kv_identity(mc), {"k": "q8_0", "v": "f16", "kd": None, "vd": None})

    def _with_long(self, long_extra):
        mid = next(iter(self.gw.M["models"]))
        model = self.gw.M["models"][mid]
        base = {"cacheTypeK": "q8_0", "cacheTypeV": "q4_0", "extraArgs": ["--no-warmup"]}
        long_profile = {"ctxSize": 262144, "nCpuMoe": 18, "ubatchSize": 768}
        if long_extra is not None:
            long_profile["extraArgs"] = long_extra
        return mid, patch.dict(model, {**base, "longContextProfile": long_profile})

    def test_layouts_with_the_same_kv_share_a_session(self):
        mid, ctx = self._with_long(None)
        with ctx:
            self.assertEqual(self.gw.checkpoint_signature(mid, False, "short"), self.gw.checkpoint_signature(mid, False, "long"))

    def test_a_layout_with_other_kv_types_declines_the_session(self):
        mid, ctx = self._with_long(["--no-warmup", "-ctk", "q5_0", "-ctv", "q5_0"])
        with ctx:
            self.assertNotEqual(self.gw.checkpoint_signature(mid, False, "short"), self.gw.checkpoint_signature(mid, False, "long"))

    def test_default_layout_is_the_running_one_for_the_running_model(self):
        mid, ctx = self._with_long(["-ctk", "q5_0", "-ctv", "q5_0"])
        with ctx, patch.object(self.gw.st, "model_id", mid), patch.object(self.gw.st, "vision", False), \
                patch.object(self.gw.st, "layout", "long"):
            self.assertEqual(self.gw.checkpoint_signature(mid, False), self.gw.checkpoint_signature(mid, False, "long"))
        with ctx, patch.object(self.gw.st, "model_id", ""):
            self.assertEqual(self.gw.checkpoint_signature(mid, False), self.gw.checkpoint_signature(mid, False, "short"))


class RestartKeepsLayout(unittest.TestCase):
    """load_server_locked: a failed restore stops the server (which resets st.layout to "short") and starts a fresh one; the fresh
    one must come up in the layout that was being started (the live smoke got a 32k server for a 101k prompt)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gw = load_gateway(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_restart_after_a_failed_restore_keeps_the_long_layout(self):
        gw = self.gw
        mid = next(iter(gw.M["models"]))
        proc = unittest.mock.Mock(pid=4321)
        proc.poll.return_value = None
        restarts = []

        def fake_stop(reason, save=True, **kw):
            gw.st.proc, gw.st.model_id, gw.st.vision, gw.st.layout = None, "", False, "short"
            return True

        def fake_start(model_id, vision, **kw):
            restarts.append(kw.get("layout"))
            return ""

        gw.st.layout, gw.st.model_id = "long", mid
        with patch.object(gw, "build_args", return_value=["llama-server"]), \
                patch.object(gw, "model_extra_env", return_value=None), \
                patch.object(gw.subprocess, "Popen", return_value=proc), \
                patch.object(gw, "llama_call", return_value=(200, {})), \
                patch.object(gw, "governor_on", return_value=False), \
                patch.object(gw, "write_server_lease"), patch.object(gw.time, "sleep"), \
                patch.object(gw, "restore_slot_locked", return_value="restart_empty"), \
                patch.object(gw, "stop_llama_locked", side_effect=fake_stop), \
                patch.object(gw, "start_llama", side_effect=fake_start), \
                patch.object(gw, "claim_port", return_value=True), patch.object(gw.st, "override", "AUTO"), \
                patch.object(gw.st, "backend_blocked", ""), patch.object(gw.st, "multimedia_lock", False):
            self.result = gw.load_server_locked(mid, False, dict(gw.M["models"][mid]), False, None, 120)
        self.assertEqual(restarts, ["long"], self.result)


if __name__ == "__main__":
    unittest.main()
