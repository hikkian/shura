"""GPU telemetry via NVML (libnvidia-ml.so.1) - no nvidia-smi process per sample.

nvidia-smi re-initialises NVML on every call and can wait several seconds while the driver is busy with a
model load. Here NVML is initialised once; each query is a direct library call (~0.2 ms when idle)."""
import ctypes

NVML_ERROR_INSUFFICIENT_SIZE = 7


class _Mem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _Proc(ctypes.Structure):  # nvmlProcessInfo_t (v2/v3 API)
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


class NVML:
    def __init__(self, index=0):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")
        self.handle = ctypes.c_void_p()
        self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)), "GetHandle")

    @staticmethod
    def _check(rc, what):
        if rc != 0:
            raise RuntimeError(f"NVML {what} failed rc={rc}")

    def card_free_mib(self):
        m = _Mem()
        self._check(self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(m)), "GetMemoryInfo")
        return m.free >> 20

    def proc_mib(self, pid):
        for _ in range(3):
            count = ctypes.c_uint(64)
            buf = (_Proc * 64)()
            rc = self.lib.nvmlDeviceGetComputeRunningProcesses_v3(self.handle, ctypes.byref(count), buf)
            if rc == NVML_ERROR_INSUFFICIENT_SIZE:
                continue
            self._check(rc, "GetComputeRunningProcesses_v3")
            for i in range(count.value):
                if buf[i].pid == pid:
                    used = buf[i].usedGpuMemory
                    return None if used >= (1 << 63) else used >> 20  # NVML_VALUE_NOT_AVAILABLE
            return None
        raise RuntimeError("NVML process list too large")

    def close(self):
        self.lib.nvmlShutdown()
