"""Shared test helper: a deterministic amount of free disk space, so the installer tests depend on logic, not on the machine.

The installer checks the free space before any download (install.check_disk_plan, modelstore.check_disk). Both call
shutil.disk_usage, so one patch covers them. SHURA_TEST_REAL_DISK=1 switches the patch off (the real disk is measured).
"""
import os
import shutil
from unittest import mock

GiB = 1024 ** 3
FAKE_FREE = 500 * GiB


def fake_usage(free=FAKE_FREE, total=1000 * GiB):
    return lambda path: shutil._ntuple_diskusage(total, total - free, free)


def patch_free_space(test, free=FAKE_FREE):
    """Patch shutil.disk_usage for the length of one test (addCleanup). No-op with SHURA_TEST_REAL_DISK=1."""
    if os.environ.get("SHURA_TEST_REAL_DISK") == "1":
        return
    p = mock.patch.object(shutil, "disk_usage", fake_usage(free))
    p.start()
    test.addCleanup(p.stop)
