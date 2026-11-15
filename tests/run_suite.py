"""Run the test suite with a watchdog: a test that runs longer than SHURA_TEST_HANG_SECONDS (600) prints every thread's stack
(faulthandler) and the run goes on; a hung suite then says where it hangs. Usage: python tests/run_suite.py [unittest args].

The suite also runs with a silent `notify-send` first on PATH: the gateway (and the gateways the harness starts) call the real one
for desktop notifications, and a test run must never pop a notification on the user's desktop."""
import faulthandler
import os
import stat
import sys
import tempfile
import unittest

HANG_SECONDS = float(os.environ.get("SHURA_TEST_HANG_SECONDS", "600"))
SILENCED = ("notify-send",)


def quiet_path(env=os.environ):
    """Put a folder with no-op stand-ins for the desktop commands in SILENCED first on PATH (of this process and every child)."""
    folder = tempfile.mkdtemp(prefix="shura-tests-quiet-")
    for name in SILENCED:
        path = os.path.join(folder, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(path, stat.S_IRWXU)
    env["PATH"] = folder + os.pathsep + env.get("PATH", "")
    return folder


class WatchedResult(unittest.TextTestResult):
    def startTest(self, test):
        faulthandler.dump_traceback_later(HANG_SECONDS, repeat=True, file=sys.__stderr__)
        super().startTest(test)

    def stopTest(self, test):
        faulthandler.cancel_dump_traceback_later()
        super().stopTest(test)


if __name__ == "__main__":
    quiet_path()
    args = sys.argv[1:] or ["discover", "-s", "tests"]
    unittest.main(module=None, argv=["run_suite"] + args, testRunner=unittest.TextTestRunner(resultclass=WatchedResult))
