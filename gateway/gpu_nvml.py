"""GPU telemetry via NVML (libnvidia-ml.so.1) - no nvidia-smi process per sample.

nvidia-smi re-initialises NVML on every call and can wait several seconds while the driver is busy with a
model load. Here NVML is initialised once; each query is a direct library call (~0.2 ms when idle)."""

import ctypes

NVML_ERROR_INSUFFICIENT_SIZE = 7
# Uninitialized, driver not loaded, GPU lost, reset required, library/driver version mismatch:
# the handle is dead and only a fresh nvmlInit can recover it.
NVML_REINIT_ERRORS = frozenset({1, 9, 15, 16, 18})


class _Mem(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _UtilizationRates(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Proc(ctypes.Structure):  # nvmlProcessInfo_t (v2/v3 API)
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


class NVML:
    def __init__(self, index=0):
        self.needs_reinit = False
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")
        self.handle = ctypes.c_void_p()
        try:
            self._check(
                self.lib.nvmlDeviceGetHandleByIndex_v2(
                    index, ctypes.byref(self.handle)
                ),
                "GetHandle",
            )
        except BaseException:
            self.lib.nvmlShutdown()  # do not leak an init count per failed retry
            raise

    def device_name(self):
        value = ctypes.create_string_buffer(96)
        fn = self.lib.nvmlDeviceGetName
        fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        self._check(fn(self.handle, value, len(value)), "GetName")
        return value.value.decode("utf-8", errors="replace")

    def driver_version(self):
        value = ctypes.create_string_buffer(96)
        fn = self.lib.nvmlSystemGetDriverVersion
        fn.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        self._check(fn(value, len(value)), "GetDriverVersion")
        return value.value.decode("utf-8", errors="replace")

    def _check(self, rc, what):
        if rc != 0:
            if rc in NVML_REINIT_ERRORS:
                self.needs_reinit = True  # the owner replaces this handle on its next request
            raise RuntimeError(f"NVML {what} failed rc={rc}")

    def card_free_mib(self):
        m = _Mem()
        self._check(
            self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(m)),
            "GetMemoryInfo",
        )
        return m.free >> 20

    def memory_info(self):
        m = _Mem()
        fn = self.lib.nvmlDeviceGetMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Mem)]
        fn.restype = ctypes.c_int
        self._check(fn(self.handle, ctypes.byref(m)), "GetMemoryInfo")
        return {"total_mib": m.total >> 20, "free_mib": m.free >> 20, "used_mib": m.used >> 20}

    def utilization(self):
        rates = _UtilizationRates()
        fn = self.lib.nvmlDeviceGetUtilizationRates
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_UtilizationRates)]
        fn.restype = ctypes.c_int
        self._check(fn(self.handle, ctypes.byref(rates)), "GetUtilizationRates")
        return {"gpu_percent": int(rates.gpu), "memory_percent": int(rates.memory)}

    def temperature_c(self):
        value = ctypes.c_uint()
        fn = self.lib.nvmlDeviceGetTemperature
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
        fn.restype = ctypes.c_int
        self._check(fn(self.handle, 0, ctypes.byref(value)), "GetTemperature")
        return int(value.value)

    def power_usage_w(self):
        value = ctypes.c_uint()
        fn = self.lib.nvmlDeviceGetPowerUsage
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        fn.restype = ctypes.c_int
        self._check(fn(self.handle, ctypes.byref(value)), "GetPowerUsage")
        return value.value / 1000.0

    def pcie_link(self):
        values = {}
        for key, symbol in (("generation", "nvmlDeviceGetCurrPcieLinkGeneration"),
                            ("width", "nvmlDeviceGetCurrPcieLinkWidth")):
            value = ctypes.c_uint()
            fn = getattr(self.lib, symbol)
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
            fn.restype = ctypes.c_int
            self._check(fn(self.handle, ctypes.byref(value)), symbol)
            values[key] = int(value.value)
        return values

    def monitor_info(self, pid=None):
        """One on-demand sample. Every caller can degrade fields to n/a independently."""
        result = {}
        try:
            result.update(self.memory_info())
        except (OSError, RuntimeError, ValueError):
            result.update({"total_mib": None, "free_mib": None, "used_mib": None})
        try:
            result["utilization_percent"] = self.utilization()["gpu_percent"]
        except (AttributeError, OSError, RuntimeError, ValueError):
            result["utilization_percent"] = None
        for key, method in (("temperature_c", self.temperature_c), ("power_w", self.power_usage_w)):
            try:
                result[key] = method()
            except (AttributeError, OSError, RuntimeError, ValueError):
                result[key] = None
        try:
            result.update({f"pcie_{k}": v for k, v in self.pcie_link().items()})
        except (AttributeError, OSError, RuntimeError, ValueError):
            result.update({"pcie_generation": None, "pcie_width": None})
        try:
            processes = self.processes()
            result["model_process_mib"] = processes.get(pid, 0 if pid is None else None)
        except (AttributeError, OSError, RuntimeError, ValueError):
            result["model_process_mib"] = 0 if pid is None else None
        return result

    def proc_mib(self, pid):
        try:
            fn = self.lib.nvmlDeviceGetComputeRunningProcesses_v3
        except AttributeError as error:  # old driver without the v3 symbol
            raise RuntimeError("NVML process list API unavailable") from error
        for size in (64, 256, 1024):
            count = ctypes.c_uint(size)
            buf = (_Proc * size)()
            rc = fn(self.handle, ctypes.byref(count), buf)
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
            try:
                fn = getattr(self.lib, "nvmlDeviceGet" + kind + "RunningProcesses_v3")
            except AttributeError as error:  # old driver without the v3 symbol
                raise RuntimeError("NVML process list API unavailable") from error
            rc = fn(self.handle, ctypes.byref(count), buf)
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
            # Samples are not guaranteed to be chronological: keep the peak per pid.
            values[buf[i].pid] = max(values.get(buf[i].pid, 0), buf[i].smUtil)
        return values, latest
    raise RuntimeError("NVML utilization buffer exceeds4096")


NVML.processes = _processes
NVML.process_utilization = _process_utilization
