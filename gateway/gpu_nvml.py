"""GPU telemetry via NVML (libnvidia-ml.so.1) - no nvidia-smi process per sample.

nvidia-smi re-initialises NVML on every call and can wait several seconds while the driver is busy with a
model load. Here NVML is initialised once; each query is a direct library call (~0.2 ms when idle)."""

import ctypes

NVML_ERROR_INSUFFICIENT_SIZE = 7


class _Mem(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _Proc(ctypes.Structure):  # nvmlProcessInfo_t (v2/v3 API)
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


class NVML:
    def __init__(self, index=0):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")
        self.handle = ctypes.c_void_p()
        self._check(
            self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)),
            "GetHandle",
        )

    @staticmethod
    def _check(rc, what):
        if rc != 0:
            raise RuntimeError(f"NVML {what} failed rc={rc}")

    def card_free_mib(self):
        m = _Mem()
        self._check(
            self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(m)),
            "GetMemoryInfo",
        )
        return m.free >> 20

    def proc_mib(self, pid):
        for _ in range(3):
            count = ctypes.c_uint(64)
            buf = (_Proc * 64)()
            rc = self.lib.nvmlDeviceGetComputeRunningProcesses_v3(
                self.handle, ctypes.byref(count), buf
            )
            if rc == NVML_ERROR_INSUFFICIENT_SIZE:
                continue
            self._check(rc, "GetComputeRunningProcesses_v3")
            for i in range(count.value):
                if buf[i].pid == pid:
                    used = buf[i].usedGpuMemory
                    return (
                        None if used >= (1 << 63) else used >> 20
                    )  # NVML_VALUE_NOT_AVAILABLE
            return None
        raise RuntimeError("NVML process list too large")

    def close(self):
        self.lib.nvmlShutdown()


class _Util(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("timeStamp", ctypes.c_ulonglong),
        ("smUtil", ctypes.c_uint),
        ("memUtil", ctypes.c_uint),
        ("encUtil", ctypes.c_uint),
        ("decUtil", ctypes.c_uint),
    ]


def _processes(self):
    result = {}
    for kind in ("Compute", "Graphics"):
        for size in (64, 256, 1024):
            count = ctypes.c_uint(size)
            buf = (_Proc * size)()
            rc = getattr(self.lib, "nvmlDeviceGet" + kind + "RunningProcesses_v3")(
                self.handle, ctypes.byref(count), buf
            )
            if rc == NVML_ERROR_INSUFFICIENT_SIZE:
                continue
            self._check(rc, "Get" + kind + "RunningProcesses_v3")
            for i in range(count.value):
                used = buf[i].usedGpuMemory
                mib = None if used >= (1 << 63) else used >> 20
                previous = result.get(buf[i].pid)
                result[buf[i].pid] = (
                    max(previous, mib)
                    if previous is not None and mib is not None
                    else mib
                )
            break
        else:
            raise RuntimeError("NVML process list exceeds1024")
    return result


def _process_utilization(self, last_seen):
    fn = self.lib.nvmlDeviceGetProcessUtilization
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Util),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_ulonglong,
    ]
    for size in (256, 1024, 4096):
        count = ctypes.c_uint(size)
        buf = (_Util * size)()
        rc = fn(self.handle, buf, ctypes.byref(count), last_seen)
        if rc == NVML_ERROR_INSUFFICIENT_SIZE:
            continue
        if rc == 6:
            return {}, last_seen
        self._check(rc, "GetProcessUtilization")
        values = {}
        latest = last_seen
        for i in range(count.value):
            if buf[i].timeStamp >= latest:
                latest = max(latest, buf[i].timeStamp)
            values[buf[i].pid] = buf[i].smUtil
        return values, latest
    raise RuntimeError("NVML utilization buffer exceeds4096")


NVML.processes = _processes
NVML.process_utilization = _process_utilization
