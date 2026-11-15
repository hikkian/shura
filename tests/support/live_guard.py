"""Keep a test away from the live machine: a temporary HOME, and a PATH whose tools log their calls and do nothing.

The tools that could reach the live gateway, the desktop or the GPU (shura, systemctl, pkill, nvidia-smi, the KDE
helpers, docker) are replaced by stand-ins. Each stand-in appends its arguments to a log and exits without acting.
`hermetic()` also refuses to continue when `shutil.which("shura")` would not resolve to the stand-in, so a test
cannot silently reach the real command.

    from live_guard import hermetic
    class MyTest(unittest.TestCase):
        def setUp(self):
            self.live = hermetic(self)            # self.live.home, self.live.calls()
"""
import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock

# name -> exit code of the stand-in: 0 = "it worked" (shura, systemctl, docker), 1 = "not available" (GPU and desktop tools)
STANDINS = {
    "shura": 0, "shuracode": 0, "systemctl": 0, "docker": 0,
    "pkill": 1, "killall": 1, "nvidia-smi": 1, "busctl": 1, "kwriteconfig6": 1, "kscreen-doctor": 1, "qdbus6": 1,
}
STANDIN_BODY = '#!/bin/sh\necho "{name} $*" >> "$LIVE_GUARD_LOG"\nexit {code}\n'


class LiveGuard:
    def __init__(self, root):
        self.root = Path(root)
        self.home = self.root / "home"
        self.bin = self.root / "bin"
        self.log = self.root / "calls.log"

    def calls(self):
        """Every stand-in call made so far, one string per call: "<tool> <arguments>"."""
        return self.log.read_text().splitlines() if self.log.exists() else []


def hermetic(testcase):
    """Point HOME and PATH at a temporary folder for the rest of the test. Cleanup is registered on the test case."""
    root = Path(tempfile.mkdtemp(dir="/dev/shm"))
    testcase.addCleanup(shutil.rmtree, root, True)
    guard = LiveGuard(root)
    guard.home.mkdir()
    guard.bin.mkdir()
    for name, code in STANDINS.items():
        stand_in = guard.bin / name
        stand_in.write_text(STANDIN_BODY.format(name=name, code=code))
        stand_in.chmod(0o755)
    patcher = mock.patch.dict(os.environ, {
        "HOME": str(guard.home),
        "PATH": f"{guard.bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "LIVE_GUARD_LOG": str(guard.log),
    })
    patcher.start()
    testcase.addCleanup(patcher.stop)
    if shutil.which("shura") != str(guard.bin / "shura") or Path.home() != guard.home:
        raise AssertionError("live guard: PATH or HOME did not take over; refusing to run against the real tools")
    return guard
