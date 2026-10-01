import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
from game_mode import GamePolicy, GameSignals
import test_gateway as baseline


class Policy(unittest.TestCase):
    def policy(self):
        return GamePolicy({"gameMode": True})

    def test_default_off(self):
        p = GamePolicy({})
        p.update(0, ["game"])
        p.update(100, ["game"])
        self.assertFalse(p.active)

    def test_three_seconds_false_signal(self):
        p = self.policy()
        p.update(0, ["game"])
        p.update(3, [])
        self.assertFalse(p.active)

    def test_on_and_off_hysteresis(self):
        p = self.policy()
        p.update(0, ["game"])
        p.update(5, ["game"])
        self.assertTrue(p.active)
        p.update(10, [])
        p.update(69, [])
        self.assertTrue(p.active)
        p.update(70, [])
        self.assertFalse(p.active)

    def test_twenty_second_flaps_one_transition(self):
        p = self.policy()
        p.update(0, ["game"])
        p.update(5, ["game"])
        for t in (20, 60, 100):
            p.update(t, [])
            p.update(t + 20, ["game"])
            self.assertTrue(p.active)

    def test_manual_resume_not_bypass_real_game(self):
        p = self.policy()
        p.manual_switch(True, 0)
        p.update(1, ["game"])
        p.manual_switch(False, 2)
        self.assertTrue(p.active)

    def test_manual_pause_resume(self):
        p = self.policy()
        p.manual_switch(True, 0)
        self.assertTrue(p.active)
        p.manual_switch(False, 1)
        self.assertFalse(p.active)

    def test_errors_do_not_release_parked(self):
        p = self.policy()
        p.manual_switch(True, 0)
        p.manual = False
        p.update(1, [], ["NVML failed"])
        p.update(100, [], ["NVML failed"])
        self.assertTrue(p.active)

    def test_steam_marker_own_proc_fixture(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            d = root / "12345"
            d.mkdir()
            (d / "comm").write_text("sleep")
            (d / "environ").write_bytes(b"SteamAppId=1\0SECRET=do-not-log\0")
            reasons, errors = GameSignals({"gameDetectSteam": True}, root).collect(None)
            self.assertEqual(reasons, ["SteamAppId PID12345"])
            self.assertEqual(errors, [])

    def test_steam_zero_and_desktop_ignored(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            d = root / "12345"
            d.mkdir()
            (d / "comm").write_text("sleep")
            (d / "environ").write_bytes(b"SteamAppId=0\0")
            self.assertEqual(
                GameSignals({"gameDetectSteam": True}, root).collect(None)[0], []
            )
            (d / "comm").write_text("brave")
            (d / "environ").write_bytes(b"SteamAppId=1\0")
            self.assertEqual(
                GameSignals({"gameDetectSteam": True}, root).collect(None)[0], []
            )

    def test_nvml_ignores_model_and_desktop(self):
        n = Mock()
        n.processes.return_value = {1: 9999, 2: 3000, 3: 6000}
        n.process_utilization.return_value = ({2: 99}, 123)
        sensor = GameSignals({"gameDetectNvml": True})
        with patch.object(
            sensor, "process", side_effect=lambda pid: {2: "brave", 3: "game"}[pid]
        ):
            self.assertEqual(
                sensor.collect(1, n)[0], ["foreign VRAM PID3 game 6000MiB"]
            )

    def test_gpu_util_only_signal(self):
        n = Mock()
        n.processes.return_value = {3: 100}
        n.process_utilization.return_value = ({3: 95}, 123)
        sensor = GameSignals({"gameDetectNvml": True})
        with patch.object(sensor, "process", return_value="game"):
            self.assertEqual(sensor.collect(None, n)[0], ["foreign GPU PID3 game 95%"])

    def test_missing_gamemode_reported(self):
        with patch("game_mode.shutil.which", return_value=None):
            self.assertTrue(GameSignals({"gameDetectGameMode": True}).collect(None)[1])

    def test_gamemode_client_count(self):
        with (
            patch("game_mode.shutil.which", return_value="/bin/tool"),
            patch(
                "game_mode.subprocess.run",
                return_value=Mock(returncode=0, stdout="(<1>,)"),
            ),
        ):
            self.assertIn(
                "gamemoded",
                GameSignals({"gameDetectGameMode": True}).collect(None)[0][0],
            )


class Gateway(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        baseline.GatewayGuard.setUpClass()
        cls.gw = baseline.GatewayGuard.gw

    @classmethod
    def tearDownClass(cls):
        baseline.GatewayGuard.tearDownClass()

    def setUp(self):
        p = patch.object(self.gw, "st", self.gw.State())
        p.start()
        self.addCleanup(p.stop)
        cfg = {**self.gw.G, "gameMode": True}
        p = patch.object(self.gw, "G", cfg)
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(self.gw, "GAME", GamePolicy(cfg))
        p.start()
        self.addCleanup(p.stop)
        for key, val in {
            "log": Mock(),
            "event": Mock(),
            "save_slot_locked": Mock(return_value=True),
            "stop_llama_locked": Mock(return_value=True),
            "llama_call": Mock(return_value=(200, [{"is_processing": False}])),
        }.items():
            p = patch.object(self.gw, key, val)
            p.start()
            self.addCleanup(p.stop)
        self.gw.st.proc = Mock(pid=123)
        self.gw.st.proc.poll.return_value = None
        self.gw.st.status = "READY"

    def active(self):
        self.gw.GAME.manual_switch(True, time.monotonic())

    def test_save_failure_never_unloads(self):
        self.active()
        self.gw.save_slot_locked.return_value = False
        self.gw.game_tick(time.monotonic(), [])
        self.gw.stop_llama_locked.assert_not_called()
        self.assertEqual(self.gw.st.status, "READY")

    def test_processing_slot_not_saved(self):
        self.active()
        self.gw.llama_call.return_value = (200, [{"is_processing": True}])
        self.gw.game_tick(time.monotonic(), [])
        self.gw.save_slot_locked.assert_not_called()

    def test_park_uses_ram_save_then_stop(self):
        self.active()
        self.gw.game_tick(time.monotonic(), [])
        self.gw.save_slot_locked.assert_called_once_with(ram_only=True)
        self.gw.stop_llama_locked.assert_called_once_with("game mode", save=False)
        self.assertEqual(self.gw.st.status, "PARKED_BY_GAME")

    def test_active_response_waits(self):
        self.active()
        self.gw.st.busy = 1
        self.gw.game_tick(time.monotonic(), [])
        self.gw.stop_llama_locked.assert_not_called()

    def test_timeout_cancels_upstream_explicitly(self):
        self.active()
        self.gw.st.busy = 1
        lease = {"cancel": __import__("threading").Event(), "conn": Mock()}
        with patch.object(self.gw, "GAME_REQUESTS", {1: lease}):
            self.gw.game_tick(0, [])
            self.gw.game_tick(61, [])
        self.assertTrue(lease["cancel"].is_set())
        lease["conn"].sock.shutdown.assert_called_once()

    def test_default_off_routes_legacy(self):
        self.gw.GAME.enabled = False
        self.gw.G["vramGuard"] = False
        self.gw.G["slotSaveCheckpoints"] = 0
        with patch.object(self.gw, "legacy_main") as old:
            self.gw.main()
            old.assert_called_once()


class ProxyCancellation(unittest.TestCase):
    def test_stream_timeout_is_explicit(self):
        import http.client
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        baseline.GatewayGuard.setUpClass()
        gw = baseline.GatewayGuard.gw
        ready = threading.Event()
        release = threading.Event()

        class Backend(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                data = b'data: {"choices":[{"delta":{"content":"start"}}]}\n\n'
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
                ready.set()
                release.wait(5)
                try:
                    self.wfile.write(b"0\r\n\r\n")
                except OSError:
                    pass

        backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        frontend = ThreadingHTTPServer(("127.0.0.1", 0), gw.Handler)
        backend.daemon_threads = frontend.daemon_threads = True
        for server in (backend, frontend):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        config = {
            **gw.G,
            "gameMode": True,
            "gameDrainSeconds": 0,
            "llamaPort": backend.server_port,
        }
        state = gw.State()
        state.status = "READY"
        state.proc = Mock(pid=123)
        state.proc.poll.return_value = None
        state.model_id = gw.M["defaultModel"]
        policy = GamePolicy(config)
        policy.ready = True
        out = {}

        def client():
            c = http.client.HTTPConnection("127.0.0.1", frontend.server_port, timeout=5)
            c.request("POST", "/v1/chat/completions", json.dumps({"messages": []}))
            response = c.getresponse()
            out["status"] = response.status
            out["body"] = response.read()
            c.close()

        try:
            with (
                patch.object(gw, "G", config),
                patch.object(gw, "st", state),
                patch.object(gw, "GAME", policy),
                patch.object(gw, "GAME_REQUESTS", {}),
                patch.object(gw, "event"),
            ):
                t = threading.Thread(target=client)
                t.start()
                self.assertTrue(ready.wait(3))
                policy.manual_switch(True, time.monotonic())
                gw.game_tick(time.monotonic(), [])
                t.join(5)
                self.assertFalse(t.is_alive())
                self.assertEqual(out["status"], 200)
                self.assertIn(b"game_pause_timeout", out["body"])
                self.assertIn(b"[DONE]", out["body"])
                self.assertEqual(state.busy, 0)
        finally:
            release.set()
            frontend.shutdown()
            backend.shutdown()
            frontend.server_close()
            backend.server_close()
            baseline.GatewayGuard.tearDownClass()
