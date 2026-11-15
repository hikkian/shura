import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench/agentic-eval"))
import harness as H

HAVE = bool(shutil.which("bwrap") and shutil.which("socat"))


def sandbox_ok():
    try:
        return subprocess.run(["bwrap", "--unshare-net", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
                               "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "true"],
                              capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class BridgePlanTests(unittest.TestCase):
    def test_plan_ports_and_paths(self):
        plan = H.bridge_plan("/s dir", 9001, 9002, 9003, ["/opt/e ng/bin", "serve", "--port", "9002"])
        self.assertEqual(plan["host_model"], ["socat", "UNIX-LISTEN:/s dir/model.sock,fork,reuseaddr", "TCP:127.0.0.1:9001"])
        self.assertEqual(plan["host_engine"], ["socat", "TCP-LISTEN:9003,bind=127.0.0.1,fork,reuseaddr", "UNIX-CONNECT:/s dir/engine.sock"])
        script = plan["inner"][2]
        self.assertEqual(plan["inner"][:2], ["sh", "-c"])
        self.assertIn("TCP-LISTEN:9001,bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:'/s dir/model.sock' &", script)
        self.assertIn("UNIX-LISTEN:'/s dir/engine.sock',fork,reuseaddr TCP:127.0.0.1:9002 &", script)
        self.assertTrue(script.endswith("exec '/opt/e ng/bin' serve --port 9002"))

    def test_unshare_net_flag_position(self):
        self.assertEqual(H.sandboxed(["x"], "/s", "/s/w", unshare_net=True)[:2], ["bwrap", "--unshare-net"])
        self.assertNotIn("--unshare-net", H.sandboxed(["x"], "/s", "/s/w"))

    def test_model_port_from_config(self):
        with tempfile.TemporaryDirectory() as d:
            H.write_config(Path(d), 51587, plugins=False)
            self.assertEqual(H.model_port_from_config(d), 51587)


@unittest.skipUnless(HAVE and sandbox_ok(), "needs bwrap (working) and socat")
class SandboxNetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
        self.scratch = Path(self.tmp.name)
        self.work = self.scratch / "work"
        self.work.mkdir()
        (self.scratch / "home").mkdir()
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            H._stop_group(p)
        self.tmp.cleanup()

    def run_in(self, argv, **kw):
        cmd = H.sandboxed(argv, self.scratch, self.work, env={}, unshare_net=True)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, **kw)

    def test_no_external_network_but_loopback(self):
        script = self.work / "net.py"
        script.write_text(
            "import socket\n"
            "def tcp():\n"
            "    s = socket.socket(); s.settimeout(3)\n"
            "    try:\n        s.connect(('1.1.1.1', 443)); return 'OPEN'\n    except OSError as e:\n        return 'fail'\n"
            "def dns():\n"
            "    try:\n        socket.getaddrinfo('example.com', 443); return 'OPEN'\n    except OSError:\n        return 'fail'\n"
            "def loop():\n"
            "    s = socket.socket()\n"
            "    try:\n        s.bind(('127.0.0.1', 0)); return 'ok'\n    except OSError:\n        return 'fail'\n"
            "print(tcp(), dns(), loop())\n")
        r = self.run_in(["python3", str(script)])
        self.assertEqual(r.stdout.split(), ["fail", "fail", "ok"], r.stdout + r.stderr)

    def test_bridge_to_host_server(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(5)
        model_port = srv.getsockname()[1]

        def serve():
            while True:
                try:
                    c, _ = srv.accept()
                except OSError:
                    return
                c.sendall(b"hello-from-host")
                c.close()
        threading.Thread(target=serve, daemon=True).start()
        self.addCleanup(srv.close)
        free = socket.socket()
        free.bind(("127.0.0.1", 0))
        inner_port = free.getsockname()[1]      # the port the sandbox listens on (same number, own network)
        free.close()
        plan = H.bridge_plan(self.scratch, inner_port, 1, 2, ["true"])
        # the host side forwards to the real server, the inner side listens on inner_port
        host = subprocess.Popen(["socat", f"UNIX-LISTEN:{plan['model_sock']},fork,reuseaddr", f"TCP:127.0.0.1:{model_port}"],
                                start_new_session=True)
        self.procs.append(host)
        for _ in range(50):
            if os.path.exists(plan["model_sock"]):
                break
            time.sleep(0.1)
        script = self.work / "client.py"
        script.write_text(
            "import socket, time\n"
            "for _ in range(30):\n"
            "    try:\n"
            "        s = socket.create_connection(('127.0.0.1', %d), timeout=3); print(s.recv(100).decode()); break\n"
            "    except OSError:\n        time.sleep(0.2)\n" % inner_port)
        sh = (f"socat TCP-LISTEN:{inner_port},bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:{plan['model_sock']} & "
              f"python3 {script}; kill %1")
        r = self.run_in(["sh", "-c", sh])
        self.assertEqual(r.stdout.strip(), "hello-from-host", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
