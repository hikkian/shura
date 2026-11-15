"""GWV1: per-layout exePath and the shura.cpp 1.0 ("v1") governor flags. Offline: only argument lists are built, no server starts."""
import importlib.util
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["AI_GATEWAY_KERNEL_LOG"] = "off"
REPO = Path(__file__).resolve().parent.parent
SNAPSHOT = REPO / "tests" / "fixtures" / "gwv1_legacy_args.json"


def load_gateway(root):
    for name in ("guardian", "model-launch"):
        (root / f"{name}.json").write_text((REPO / "config" / f"{name}.example.json").read_text())
    spec = importlib.util.spec_from_file_location("gwv1_gateway_module", REPO / "gateway" / "ai_gateway.py")
    gw = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, AI_GATEWAY_CONFIG=str(root), AI_GATEWAY_STATE=str(root / "state")):
        spec.loader.exec_module(gw)
    return gw


# (case name, model id, vision, guardian overrides, model overrides)
CASES = [
    ("plain", "tiel-coder", False, {}, {}),
    ("plain-vision", "tiel-coder", True, {}, {}),
    ("apex", "tiel-coder-apex", False, {}, {}),
    ("slots", "tiel-coder", False, {}, {"moeCacheSlots": 26}),
    ("spec-nice", "tiel-coder", False, {}, {"specType": "mtp", "specDraftNMax": 3, "specDraftPMin": 0.5, "niceLevel": 5}),
    ("sampling", "tiel-coder", False, {}, {"minP": 0.05, "presencePenalty": 1.5, "repeatPenalty": 1.1, "poll": 0,
                                           "imageMaxTokens": 1024, "chatTemplateKwargs": {"terse": False},
                                           "extraArgs": ["-ot", "blk\\.40\\.ffn_(up|down|gate)_exps\\.weight=CPU"]}),
    ("valve", "tiel-coder", False, {"vramValve": True}, {"moeCacheSlots": 20}),
    ("governor", "tiel-coder", False, {"vramPolicy": "governor"}, {"moeCacheSlots": 26}),
    ("governor-chunk", "tiel-coder", False, {"vramPolicy": "governor", "vramChunkSlots": 6, "vramPollMs": 100}, {}),
]


def build_case(gw, model_id, vision, guardian, model):
    """The argument list the gateway builds for one fixed case, with everything machine-specific pinned."""
    mc = {**gw.M["models"][model_id], **model}
    saved_g = dict(gw.G)
    gw.G.update(guardian)
    gw.GOV["budget"] = None
    try:
        with patch.object(gw.common, "server_supports", lambda exe, flag: True), \
                patch.object(gw, "remember_server_slot_dir", lambda: "/slots"), \
                patch.object(gw, "api_key_file", lambda: None), \
                patch.object(gw, "_ram_cache_budget", lambda mc, ladder=True: (1024, 2)), \
                patch.object(gw.common, "cors_args", lambda exe, port: ["--cors-origins", f"http://127.0.0.1:{port}"]):
            return gw.build_args(mc, vision)
    finally:
        gw.G.clear()
        gw.G.update(saved_g)
        gw.GOV["budget"] = None


def snapshot_all(gw):
    cfg = str(gw.CFG_DIR) + "/"
    return {name: [x.replace(cfg, "<CFG>/").replace(os.path.expanduser("~"), "<HOME>") for x in build_case(gw, mid, vision, g, m)] for name, mid, vision, g, m in CASES}


class GatewayV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="shura-gwv1-")
        cls.gw = load_gateway(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def args(self, mc, vision=False, guardian=None):
        gw = self.gw
        saved = dict(gw.G)
        gw.G.update(guardian or {})
        gw.GOV["budget"] = None
        try:
            with patch.object(gw.common, "server_supports", lambda exe, flag: True), \
                    patch.object(gw, "remember_server_slot_dir", lambda: "/slots"), \
                    patch.object(gw, "api_key_file", lambda: None), \
                    patch.object(gw, "_ram_cache_budget", lambda mc, ladder=True: (1024, 2)):
                return gw.build_args(mc, vision)
        finally:
            gw.G.clear()
            gw.G.update(saved)
            gw.GOV["budget"] = None

    @staticmethod
    def exe(argv):
        return argv[argv.index("--host") - 1]

    def base(self, **extra):
        return {**self.gw.M["models"]["tiel-coder"], "moeCacheSlots": 26, **extra}

    def test_legacy_arguments_match_the_snapshot_byte_for_byte(self):
        expected = json.loads(SNAPSHOT.read_text())
        self.assertEqual(snapshot_all(self.gw), expected)
        self.assertEqual(set(expected), {c[0] for c in CASES})

    def test_v1_engine_uses_vram_chunk_mib_and_no_old_cache_flags(self):
        gov = {"vramPolicy": "governor"}
        legacy = self.args(self.base(), guardian=gov)
        self.assertIn("--moe-cache-chunk-slots", legacy)
        v1 = self.args(self.base(engine="v1"), guardian=gov)
        self.assertEqual(v1[v1.index("--vram-chunk-mib") + 1], "52")
        for flag in ("--moe-cache-chunk-slots", "--moe-cache-slots", "--moe-cache-profile"):
            self.assertNotIn(flag, v1)
        self.assertIn("--vram-reserve-mib", v1)
        self.assertEqual(self.args(self.base(engine="v1", vramChunkMiB=64), guardian=gov).count("--vram-chunk-mib"), 1)
        v1g = self.args(self.base(engine="v1"), guardian={**gov, "vramChunkMiB": 40})
        self.assertEqual(v1g[v1g.index("--vram-chunk-mib") + 1], "40")
        v1m = self.args(self.base(engine="v1", vramChunkMiB=64), guardian={**gov, "vramChunkMiB": 40})
        self.assertEqual(v1m[v1m.index("--vram-chunk-mib") + 1], "64")
        for bad in (52.5, "52", True, 0, -4):
            with self.assertRaises(ValueError):
                self.args(self.base(engine="v1", vramChunkMiB=bad), guardian=gov)
        with self.assertRaises(ValueError):
            self.args(self.base(engine="v2"))

    def test_each_layout_launches_its_own_binary(self):
        with tempfile.TemporaryDirectory(prefix="shura-gwv1-exe-") as d:
            old, new = Path(d) / "old-server", Path(d) / "new-server"
            for p in (old, new):
                p.write_text("#!/bin/sh\n")
                p.chmod(p.stat().st_mode | stat.S_IXUSR)
            short = self.base(exePath=str(old))
            long_ = self.base(exePath=str(new), engine="v1")
            self.assertEqual(self.exe(self.args(short)), str(old))
            self.assertEqual(self.exe(self.args(long_)), str(new))
            self.assertEqual(self.exe(self.args(self.base())), self.gw.M["exePath"])
            self.assertEqual(self.gw.exe_for(long_), str(new))

    def test_missing_binary_is_a_clear_error_and_the_old_layout_still_builds(self):
        gw = self.gw
        good = self.base()
        before = self.args(good)
        broken = self.base(exePath="/nonexistent/gwv1/llama-server")
        with self.assertRaises(ValueError) as caught:
            gw.exe_for(broken, "tiel-coder/long")
        self.assertIn("tiel-coder/long", str(caught.exception))
        self.assertIn("/nonexistent/gwv1/llama-server", str(caught.exception))
        with self.assertRaises(ValueError):
            self.args(broken)
        self.assertEqual(self.args(good), before)

    def test_start_llama_refuses_a_missing_binary_without_stopping_the_running_server(self):
        gw = self.gw
        mid = gw.M["defaultModel"]
        saved_models = gw.M["models"]
        gw.M["models"] = {**saved_models, mid: {**saved_models[mid], "exePath": "/nonexistent/gwv1/llama-server"}}
        try:
            with patch.object(gw, "stop_llama_locked", side_effect=AssertionError("must not stop the running server")), \
                    patch.object(gw, "event"), patch.object(gw, "st", gw.State()):
                reason = gw.start_llama(mid, False)
        finally:
            gw.M["models"] = saved_models
        self.assertIn("/nonexistent/gwv1/llama-server", reason)
        self.assertIn("Invalid model configuration", reason)


if __name__ == "__main__":
    unittest.main()
