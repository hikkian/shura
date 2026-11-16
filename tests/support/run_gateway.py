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
    # a caller that names its own root (the valve's ramGuardProcRoot) is pointed at the fake one as well
    _meminfo, _pressure = gw.ram_budget.read_meminfo, gw.ram_budget.read_pressure
    gw.ram_budget.read_meminfo = lambda *_root: _meminfo(os.environ["FAKE_PROC_ROOT"])  # noqa: E731
    gw.ram_budget.read_pressure = lambda *_root: _pressure(os.environ["FAKE_PROC_ROOT"])  # noqa: E731
# RAMFIX1: FAKE_CGROUP_ROOT points the page cache reclaim at a fake cgroup tree (memory.reclaim a test watches) and a fake /proc
if os.environ.get("FAKE_CGROUP_ROOT"):
    _reclaim = gw.cache_reclaim.reclaim_page_cache
    gw.cache_reclaim.reclaim_page_cache = lambda want, _proc="/proc", **kw: _reclaim(  # noqa: E731
        want, os.environ.get("FAKE_PROC_ROOT", "/proc"), cgroup_root=os.environ["FAKE_CGROUP_ROOT"], **kw)
gw.main()
