"""Hardware detection: turns what the operating system reports into the hardware profile the planner uses.

Every function here PARSES text (lscpu, /proc/meminfo, nvidia-smi, lspci, sysctl, PowerShell JSON) so that it can be
tested with recorded output from machines we do not own. The only code that touches the machine is `SystemEnv`,
which runs a command or reads a file and returns text or None. Nothing needs root and nothing is installed.
Whatever cannot be determined is left out and named in `detection_notes`, never guessed silently.
"""
import json
import os
import platform
import re
import shutil
import subprocess

MiB = 1024 * 1024
GiB = 1024 * MiB

# Published memory bandwidth (GB/s) of common GPUs. Unknown GPUs are not guessed: the planner falls back to a
# conservative default and the installer can measure the GPU instead.
GPU_BANDWIDTH_GBS = {
    "rtx 3060": 360, "rtx 3070": 448, "rtx 3080": 760, "rtx 3090": 936, "rtx 4060 ti": 288, "rtx 4060": 272,
    "rtx 4070 ti super": 672, "rtx 4070 ti": 504, "rtx 4070 super": 504, "rtx 4070": 504, "rtx 4080 super": 736,
    "rtx 4080": 717, "rtx 4090": 1008, "rtx 5090": 1792,
    "rx 7900 xtx": 960, "rx 7900 xt": 800, "rx 7800 xt": 624, "rx 9070 xt": 640,
    "arc a770": 560,
    "m2 ultra": 800, "m2 max": 400, "m3 ultra": 800, "m4 max": 546,
}
VENDOR_IDS = {"10de": "nvidia", "1002": "amd", "8086": "intel"}


class SystemEnv:
    """The real machine. Tests substitute a fake with the same four methods."""

    def run(self, cmd, timeout=15):
        if not shutil.which(cmd[0]):
            return None
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout if r.returncode == 0 else None

    def read(self, path):
        try:
            with open(path, errors="replace") as f:
                return f.read()
        except OSError:
            return None

    def exists(self, path):
        return os.path.exists(path)

    def listdir(self, path):
        try:
            return sorted(os.listdir(path))
        except OSError:
            return []


def gpu_bandwidth(name):
    n = name.lower()
    for key in sorted(GPU_BANDWIDTH_GBS, key=len, reverse=True):      # longest match first: "4070 ti super" < "4070"
        if key in n:
            return GPU_BANDWIDTH_GBS[key]
    return None


# ------------------------------------------------------------------------------------------------ parsers
def parse_lscpu(text):
    kv = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            kv[k.strip()] = v.strip()
    sockets = int(kv.get("Socket(s)", 1) or 1)
    per_socket = int(kv.get("Core(s) per socket", 0) or 0)
    logical = int(kv.get("CPU(s)", 0) or 0)
    physical = per_socket * sockets if per_socket else logical
    return {"model": kv.get("Model name", "unknown"), "physical_cores": max(1, physical), "logical_cores": max(1, logical),
            "numa_nodes": max(1, int(kv.get("NUMA node(s)", 1) or 1)),
            "flags": sorted(f for f in kv.get("Flags", "").split() if f.startswith(("avx", "amx", "fma", "sse4")))}


def parse_meminfo(text):
    mem = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in text.splitlines() if ":" in line}
    return {"total": mem["MemTotal"], "available": mem.get("MemAvailable", mem.get("MemFree", 0))}


def parse_nvidia_smi(text):
    """`nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader,nounits`."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            name, total, used, driver = parts[:4]
            gpus.append({"vendor": "nvidia", "name": name, "vram_total": int(float(total)) * MiB,
                         "vram_used": int(float(used)) * MiB, "driver": driver})
    return gpus


def parse_lspci(text):
    """`lspci -mm` lines for display controllers -> [{vendor, name}]."""
    out = []
    for line in text.splitlines():
        if not re.search(r'"(VGA compatible controller|3D controller|Display controller)"', line):
            continue
        fields = re.findall(r'"([^"]*)"', line)
        vid = re.search(r"\[([0-9a-f]{4})\]", line)
        vendor = VENDOR_IDS.get(vid.group(1)) if vid else None
        if vendor is None:
            low = fields[1].lower() if len(fields) > 1 else ""
            vendor = "nvidia" if "nvidia" in low else "amd" if ("amd" in low or "advanced micro" in low) \
                else "intel" if "intel" in low else None
        if vendor:
            out.append({"vendor": vendor, "name": (fields[2] if len(fields) > 2 else "GPU")})
    return out


def parse_sysctl(text):
    kv = dict(line.split(": ", 1) for line in text.strip().splitlines() if ": " in line)
    return {"memsize": int(kv.get("hw.memsize", 0)), "physical": int(kv.get("hw.physicalcpu", 0)),
            "logical": int(kv.get("hw.logicalcpu", 0)), "brand": kv.get("machdep.cpu.brand_string", "Apple Silicon")}


def parse_windows_video(text):
    """`Get-CimInstance Win32_VideoController | Select Name,AdapterRAM,DriverVersion | ConvertTo-Json`.
    AdapterRAM is a 32-bit field and cannot exceed 4 GiB, so it is not trusted for VRAM (see detection_notes)."""
    data = json.loads(text)
    data = data if isinstance(data, list) else [data]
    out = []
    for d in data:
        name = d.get("Name", "")
        low = name.lower()
        vendor = "nvidia" if "nvidia" in low else "amd" if ("amd" in low or "radeon" in low) else "intel" if "intel" in low else None
        if vendor:
            out.append({"vendor": vendor, "name": name, "driver": d.get("DriverVersion")})
    return out


# ------------------------------------------------------------------------------------------ detection flow
def _amd_vram(env, names):
    """VRAM of AMD GPUs from sysfs (amdgpu): (total, used) per card with mem_info_vram_total."""
    cards = []
    for card in env.listdir("/sys/class/drm"):
        if not re.fullmatch(r"card\d+", card):
            continue
        base = f"/sys/class/drm/{card}/device"
        if (env.read(f"{base}/vendor") or "").strip() != "0x1002":
            continue
        total, used = env.read(f"{base}/mem_info_vram_total"), env.read(f"{base}/mem_info_vram_used")
        if total:
            cards.append((int(total), int(used or 0)))
    return cards


def detect_linux(env):
    notes, gpus = [], []
    lscpu = env.run(["lscpu"])
    cpu = parse_lscpu(lscpu) if lscpu else None
    if not cpu:
        n = os.cpu_count() or 1
        cpu = {"model": "unknown", "physical_cores": n, "logical_cores": n, "numa_nodes": 1, "flags": []}
        notes.append("lscpu is not available: core counts are approximate")
    mem = parse_meminfo(env.read("/proc/meminfo") or "MemTotal: 0 kB")
    smi = env.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                   "--format=csv,noheader,nounits"])
    for g in parse_nvidia_smi(smi or ""):
        g["backends"] = ["cuda", "vulkan"] if env.run(["vulkaninfo", "--summary"]) else ["cuda"]
        gpus.append(g)
    other = [g for g in parse_lspci(env.run(["lspci", "-mm"]) or "") if g["vendor"] != "nvidia"]
    amd_mem = iter(_amd_vram(env, other))
    for g in other:
        if g["vendor"] == "amd":
            vram = next(amd_mem, None)
            if not vram:
                notes.append(f"{g['name']}: VRAM not readable from sysfs, GPU skipped (llama-server --list-devices will tell)")
                continue
            g["vram_total"], g["vram_used"] = vram
            g["backends"] = (["rocm"] if env.exists("/dev/kfd") else []) + ["vulkan"]
        else:
            notes.append(f"{g['name']}: VRAM is not read without vendor tools, GPU skipped (llama-server --list-devices will tell)")
            continue
        gpus.append(g)
    return cpu, mem, gpus, notes


def detect_macos(env):
    s = parse_sysctl(env.run(["sysctl", "hw.memsize", "hw.physicalcpu", "hw.logicalcpu", "machdep.cpu.brand_string"]) or "")
    cpu = {"model": s["brand"], "physical_cores": max(1, s["physical"]), "logical_cores": max(1, s["logical"]),
           "numa_nodes": 1, "flags": []}
    mem = {"total": s["memsize"], "available": int(s["memsize"] * 0.5)}     # not reported simply; conservative half
    gpus, notes = [], ["available memory on macOS is estimated as half of RAM"]
    if "apple" in s["brand"].lower():
        gpus.append({"vendor": "apple", "name": s["brand"], "vram_total": s["memsize"], "vram_used": 0, "unified": True,
                     "display": True, "driver": None, "backends": ["metal"]})
    return cpu, mem, gpus, notes


def _windows_memory():
    """(total, available) bytes through GlobalMemoryStatusEx; (0, 0) when not on Windows or on failure."""
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("tpage", ctypes.c_ulonglong), ("apage", ctypes.c_ulonglong),
                        ("tvirt", ctypes.c_ulonglong), ("avirt", ctypes.c_ulonglong), ("aext", ctypes.c_ulonglong)]
        st = Status()
        st.length = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return int(st.total), int(st.avail)
    except (AttributeError, OSError, ImportError):
        return 0, 0


def detect_windows(env):
    n = os.cpu_count() or 1
    cpu = {"model": platform.processor() or "unknown", "physical_cores": max(1, n // 2), "logical_cores": n,
           "numa_nodes": 1, "flags": []}
    notes = ["Windows detection is limited: CPU cores are estimated, VRAM only comes from nvidia-smi"]
    smi = env.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                   "--format=csv,noheader,nounits"])
    gpus = parse_nvidia_smi(smi or "")
    for g in gpus:
        g["backends"] = ["cuda", "vulkan"]
    total, avail = _windows_memory()
    mem = {"total": total, "available": avail}
    if not total:
        notes.append("RAM size could not be read")
    return cpu, mem, gpus, notes


def build_profile(system, cpu, mem, gpus, notes, *, ram_bandwidth_gbs=None, disk_free=0, arch="x86_64", distro=None):
    for g in gpus:
        g.setdefault("unified", False)
        g.setdefault("display", g["vram_used"] > 256 * MiB)
        g.setdefault("driver", None)
        bw = gpu_bandwidth(g["name"])
        if bw:
            g["bandwidth_gbs"] = bw
    memory = dict(mem)
    if ram_bandwidth_gbs:
        memory["bandwidth_gbs"] = ram_bandwidth_gbs
    os_ = {"family": system, "arch": arch}
    if distro:
        os_["distro"] = distro
    return {"schema": 1, "os": os_, "cpu": cpu, "memory": memory, "gpus": gpus, "disk": {"free": disk_free},
            "detection_notes": notes}


def detect(env=None, *, ram_bandwidth_gbs=None):
    """Hardware profile of this machine. `ram_bandwidth_gbs` comes from the bandwidth measurement (see HARDWARE.md)."""
    env = env or SystemEnv()
    system = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(platform.system(), "linux")
    cpu, mem, gpus, notes = {"linux": detect_linux, "macos": detect_macos, "windows": detect_windows}[system](env)
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64"
    return build_profile(system, cpu, mem, gpus, notes, ram_bandwidth_gbs=ram_bandwidth_gbs, arch=arch,
                         disk_free=shutil.disk_usage(os.path.expanduser("~")).free)
