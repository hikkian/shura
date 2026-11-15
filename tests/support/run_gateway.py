"""Start the gateway with a fake NVML, so tests need no GPU and no driver. Usage: run_gateway.py PATH_TO_ai_gateway.py"""
import importlib.util
import os
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location("ai_gateway", path)
gw = importlib.util.module_from_spec(spec)
sys.modules["ai_gateway"] = gw
spec.loader.exec_module(gw)


class FakeNvml:
    """Plenty of free VRAM; the server process 'uses' 8 GiB."""
    free_mib = int(os.environ.get("FAKE_NVML_FREE_MIB", "12000"))

    def card_free_mib(self):
        return self.free_mib

    def proc_mib(self, pid):
        return 8000

    def monitor_info(self, pid):
        return {"name": "fake", "total_mib": 12282, "free_mib": self.free_mib, "used_mib": 12282 - self.free_mib, "process_mib": 8000}

    def close(self):
        pass

    def __getattr__(self, name):
        raise AttributeError(f"FakeNvml has no {name}")


gw._NVML_DEVICE = FakeNvml()
# RAMLOGIC2: FAKE_PROC_ROOT points the RAM budget at a fake /proc (meminfo, pressure/memory) that a test rewrites while it runs
if os.environ.get("FAKE_PROC_ROOT"):
    import functools
    gw.ram_budget.read_meminfo = functools.partial(gw.ram_budget.read_meminfo, os.environ["FAKE_PROC_ROOT"])
    gw.ram_budget.read_pressure = functools.partial(gw.ram_budget.read_pressure, os.environ["FAKE_PROC_ROOT"])
gw.main()
