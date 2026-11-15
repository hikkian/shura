"""Which VRAM policy the gateway suite runs under (VRAMLOGIC1i).

SHURA_SUITE_VRAM_POLICY=governor runs every gateway test on the governor path. Tests that check a mechanism the governor
does not run (the expert-cache valve, prefill yield, slot gears, the lite-pressure check) are skipped there, each with its
reason, so the skip list says exactly which behaviour is legacy-only.
"""
import os
import unittest

GOVERNOR_SUITE = os.environ.get("SHURA_SUITE_VRAM_POLICY") == "governor"


def legacy_only(reason):
    return unittest.skipIf(GOVERNOR_SUITE, f"legacy-only under the governor: {reason}")
