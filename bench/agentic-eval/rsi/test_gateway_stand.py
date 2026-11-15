"""RSI1: the gateway stand (gateway_stand.py) and the client criterion «200 и корректно» (executor.client_verdict). No GPU: the pin on a
throwaway git repository, the config from the candidate files (read only), and one stand end to end on the shura-qa fake llama-server.

    cd bench/agentic-eval && python3 -B -m unittest rsi.test_gateway_stand
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from . import executor, gateway_stand

REPO = Path(__file__).resolve().parents[3]
FAKE_BIN = Path(os.environ.get("SHURA_QA", "/home/alisher/Projects/shura-qa")) / "fake_llama" / "fake-llama-bin"


def tmpdir(test, prefix):
    d = Path(tempfile.mkdtemp(prefix=prefix, dir="/dev/shm" if Path("/dev/shm").is_dir() else None))
    test.addCleanup(shutil.rmtree, d, True)
    return d


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


class Pin(unittest.TestCase):
    def setUp(self):
        self.repo = tmpdir(self, "rsi1-pin-")
        git(self.repo, "init", "-q")
        (self.repo / "gateway").mkdir()
        (self.repo / "gateway" / "ai_gateway.py").write_text("# v1\n")
        git(self.repo, "add", "-A")
        git(self.repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "v1")
        self.v1 = git(self.repo, "rev-parse", "HEAD")

    def test_the_pinned_clean_worktree_passes(self):
        self.assertEqual(gateway_stand.check_pin(self.repo, self.v1[:9]), self.v1)

    def test_a_moved_worktree_stops_the_run(self):
        (self.repo / "gateway" / "ai_gateway.py").write_text("# v2\n")
        git(self.repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "v2")
        with self.assertRaises(SystemExit) as cm:
            gateway_stand.check_pin(self.repo, self.v1)
        self.assertIn("moved", str(cm.exception))

    def test_a_changed_file_stops_the_run(self):
        (self.repo / "gateway" / "ai_gateway.py").write_text("# edited\n")
        with self.assertRaises(SystemExit) as cm:
            gateway_stand.check_pin(self.repo, self.v1)
        self.assertIn("changes", str(cm.exception))

    def test_a_folder_without_the_gateway_is_refused(self):
        with self.assertRaises(SystemExit):
            gateway_stand.check_pin(tmpdir(self, "rsi1-empty-"), self.v1)


class Config(unittest.TestCase):
    def test_the_candidate_with_the_stands_ports_and_folders(self):
        work = tmpdir(self, "rsi1-cfg-")
        cfg = gateway_stand.build_config(work / "cfg", work / "state", "sage", REPO, 18091, 18092)
        g = json.loads((cfg / "guardian.json").read_text())
        m = json.loads((cfg / "model-launch.json").read_text())
        self.assertEqual((g["gatewayPort"], g["llamaPort"]), (18091, 18092))
        self.assertEqual(g["vramPolicy"], "governor")              # the candidate's, not changed by the stand
        self.assertFalse(g["ramGuard"])
        self.assertEqual(g["ramDeskReserveMiB"], gateway_stand.DESK_RESERVE_MIB)
        for key in ("slotSaveDir", "sessionCacheDir", "sessionPersistDir", "vramPressureSlotDir"):
            self.assertTrue(g[key].startswith(str(work / "state")), key)
        self.assertEqual(m["defaultModel"], "sage")
        self.assertIn("tiel-coder", m["models"])

    def test_a_state_folder_on_disk_gets_its_ram_folders_in_tmpfs(self):
        # RSI1 night 1: the run folder is on the SSD, and the gateway refuses RAM sessions and slot dumps anywhere but tmpfs
        import shutil
        import tempfile
        from unittest import mock
        disk = Path(tempfile.mkdtemp(prefix="rsi1-disk-", dir=str(Path.home() / ".cache")))
        self.addCleanup(shutil.rmtree, disk, True)
        ram = tmpdir(self, "rsi1-ram-")
        with mock.patch.object(gateway_stand, "RAM_STATE", ram):
            cfg = gateway_stand.build_config(disk / "myrun" / "gw" / "cfg", disk / "myrun" / "gw" / "state", "tiel-coder", REPO)
        g = json.loads((cfg / "guardian.json").read_text())
        for key in ("slotSaveDir", "sessionCacheDir", "sessionPersistDir", "vramPressureSlotDir"):
            self.assertTrue(g[key].startswith(str(ram / "myrun_gw")), (key, g[key]))
            self.assertTrue(Path(g[key]).is_dir())
        self.assertFalse((disk / "myrun" / "gw" / "state" / "slots").exists())

    def test_a_model_outside_the_candidate_is_refused(self):
        work = tmpdir(self, "rsi1-cfg-")
        with self.assertRaises(SystemExit):
            gateway_stand.build_config(work / "cfg", work / "state", "nobody", REPO)

    def test_each_model_sends_its_own_sampling(self):
        self.assertEqual(gateway_stand.model_sampling("tiel-coder"), {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
        sage = gateway_stand.model_sampling("sage")
        self.assertEqual((sage["temperature"], sage["presence_penalty"], sage["min_p"]), (1.0, 1.5, 0.0))

    def test_the_sampling_replaces_the_engines_options(self):
        cfg = tmpdir(self, "rsi1-wire-")
        (cfg / "shuracode.json").write_text(json.dumps({"provider": {"eval": {"models": {"m": {"options": {"temperature": 0.6}}}}},
                                                        "agent": {"build": {"temperature": 0.6, "top_p": 0.95}}, "plugin": []}))
        executor.wire_harness(cfg, cfg / "h", "sage", {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5})
        conf = json.loads((cfg / "shuracode.json").read_text())
        self.assertEqual(conf["provider"]["eval"]["models"]["sage"]["options"]["presence_penalty"], 1.5)
        self.assertEqual(conf["agent"]["build"], {"temperature": 1.0, "top_p": 0.95})
        self.assertEqual(conf["model"], "eval/sage")


class Pins(unittest.TestCase):
    def test_the_candidate_is_copied_once_and_the_agent_side_is_fingerprinted(self):
        work = tmpdir(self, "rsi1-pins-")
        stand = gateway_stand.GatewayStand(work, "sage", REPO, "HEAD", pin=False, guard=False, wrapper=False)
        self.assertEqual(stand.launch_file.read_bytes(), gateway_stand.LAUNCH_CANDIDATE.read_bytes())
        copy = json.loads(stand.launch_file.read_text())
        copy["models"]["tiel-coder"]["temperature"] = 0.123             # what the stand uses is its own copy, never re-read
        stand.launch_file.write_text(json.dumps(copy))
        again = gateway_stand.GatewayStand(work, "tiel-coder", REPO, "HEAD", pin=False, guard=False, wrapper=False)
        self.assertEqual(again.sampling["temperature"], 0.123)
        self.assertEqual(set(stand.pins), {"guardian", "launch", "engine", "plugins", "reliability", "model_harness"})


class ClientVerdict(unittest.TestCase):
    def log(self, *rows):
        p = tmpdir(self, "rsi1-log-") / "proxy.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return p

    def test_all_200_and_finished(self):
        v = executor.client_verdict(self.log({"status": 200, "done": True}, {"status": 200, "finish_reason": "stop"}))
        self.assertEqual((v["ok"], v["requests"]), (True, 2))

    def test_an_error_status_fails(self):
        v = executor.client_verdict(self.log({"status": 200, "done": True}, {"status": 503}))
        self.assertFalse(v["ok"])
        self.assertEqual(v["bad"], [{"status": 503, "why": "status"}])

    def test_an_unfinished_200_fails(self):
        self.assertFalse(executor.client_verdict(self.log({"status": 200}))["ok"])

    def test_the_agent_hanging_up_is_the_models_run(self):
        self.assertTrue(executor.client_verdict(self.log({"status": 200, "client_closed": True}))["ok"])

    def test_no_request_is_not_ok(self):
        self.assertFalse(executor.client_verdict(self.log())["ok"])


class StrictRunner(unittest.TestCase):
    def test_a_run_with_a_client_error_is_discarded_and_repeated(self):
        class Stand:
            fake, client_strict, sampling = False, True, {"temperature": 1.0}
            proxy_port, proxy_log, current_run = 0, "/nonexistent", ""

            def __init__(self):
                self.restarts = 0

            def alive(self):
                return True

            def memory_over_limit(self):
                return False

            def restart(self, reason=""):
                self.restarts += 1

        recs = [{"requests": 3, "client": {"ok": False, "bad_total": 1}}, {"requests": 3, "client": {"ok": True, "bad_total": 0}}]
        seen = []

        def fake_run_task(*a, sampling=None, **kw):
            seen.append(sampling)
            return recs.pop(0)
        old = executor.run_task
        executor.run_task = fake_run_task
        self.addCleanup(setattr, executor, "run_task", old)
        stand = Stand()
        runner = executor.Runner(stand, tmpdir(self, "rsi1-run-"))
        rec = runner.run({"id": "t"}, "/x", "sage", "base", 1, "/x")
        self.assertTrue(rec["client"]["ok"])
        self.assertEqual((stand.restarts, len(runner.discarded)), (1, 1))
        self.assertEqual(seen, [{"temperature": 1.0}] * 2)


@unittest.skipUnless(FAKE_BIN.exists(), "the shura-qa fake llama-server is not installed")
class EndToEnd(unittest.TestCase):
    """The real gateway of this checkout against the fake server, with the candidate layout (fake binary and model): the stand starts,
    sets the effort, answers through its proxy, knows its own server, restarts, and stops everything it started."""

    def test_the_stand_answers_restarts_and_leaves_nothing(self):
        work = tmpdir(self, "rsi1-gw-")
        dummy = work / "model.gguf"
        dummy.write_bytes(b"GGUF")
        launch = json.loads(gateway_stand.LAUNCH_CANDIDATE.read_text())
        launch["exePath"] = str(FAKE_BIN)
        for mc in launch["models"].values():
            mc.update({"modelPath": str(dummy), "vramFreeMinGBToLoad": 0})
            for key in ("mmprojPath", "visionOverrides", "draftModelPath"):
                mc.pop(key, None)
        (work / "launch.json").write_text(json.dumps(launch))
        gw_port, llama_port = executor.free_port(), executor.free_port()
        # the fake has no expert cache and no /vram endpoint: the governor and both valves need the real server (checked on the GPU)
        guardian = {"ramPolicy": "off", "vramPolicy": "legacy", "vramValve": False, "prefillCacheYield": False, "deskNeed": False,
                    "ramFreeMinGBToLoad": 0, "ramFreeMinGB": 0.1, "healthCheckTimeoutSeconds": 30, "pollIntervalSeconds": 1}
        os.environ["FAKE_LLAMA_CONFIG"] = str(work / "fake.json")
        (work / "fake.json").write_text(json.dumps({"n_tokens": 5, "token_delay_ms": 5, "load_seconds": 0.1}))
        stand = gateway_stand.GatewayStand(work / "stand", "sage", REPO, "HEAD", effort="high", gw_port=gw_port, llama_port=llama_port,
                                           wrapper=False, guard=False, pin=False, guardian=guardian, launch=work / "launch.json")
        with stand:
            self.assertTrue(stand.alive())
            self.assertEqual(stand.think.get("level"), "high")
            body = json.dumps({"model": "sage", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{stand.proxy_port}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    self.assertEqual(r.status, 200)
                    r.read()
            except urllib.error.HTTPError as err:
                self.fail(f"{err.code} {err.read()[:400]!r}\n{(stand.work / 'gateway.log').read_text()[-2500:]}")
            deadline = time.time() + 10                    # the proxy writes its line after the last chunk went out
            while not (stand.proxy_log.exists() and stand.proxy_log.read_text().endswith("\n")) and time.time() < deadline:
                time.sleep(0.1)
            self.assertTrue(executor.client_verdict(stand.proxy_log)["ok"], stand.proxy_log.read_text())
            first = stand.own_server_pids()
            self.assertEqual(len(first), 1)
            stand.restart("test")
            self.assertTrue(stand.alive())
            self.assertFalse(any(executor.pid_running(p) for p in first))
        self.assertEqual(executor.listener_pids(gw_port) | executor.listener_pids(llama_port), set())


if __name__ == "__main__":
    unittest.main()
