"""Which GPU telemetry the gateway reads: NVIDIA through NVML when its library is installed, otherwise the AMD card (sysfs, fdinfo).

Both devices have the same calls (see gpu_nvml.NVML and gpu_amd.AMDDevice), so callers do not change."""

from gpu_amd import AMDDevice, AMDError
from gpu_nvml import NVML


class GPUError(RuntimeError):
    """Neither backend can read the GPU; the message says why for each."""

    backend = "GPU"


def open_device(nvml=NVML, index=0):
    try:
        return nvml(index)
    except (OSError, RuntimeError) as nvidia_error:  # no libnvidia-ml.so.1, or no NVIDIA device answers
        try:
            return AMDDevice()
        except AMDError as amd_error:
            raise GPUError(f"no GPU telemetry: NVIDIA ({nvidia_error}); AMD ({amd_error})") from amd_error
