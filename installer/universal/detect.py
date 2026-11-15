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
import time
from pathlib import Path

MiB = 1024 * 1024
GiB = 1024 * MiB

# Published memory bandwidth (GB/s) of common GPUs. Unknown GPUs are not guessed: the planner falls back to a
# conservative default and the installer can measure the GPU instead.
GPU_BANDWIDTH_GBS = {      # memory bandwidth in GB/s from the vendors' specifications (longest key wins: "4070 ti super" before "4070")
    "rtx 3050": 224, "rtx 3060 ti": 448, "rtx 3060": 360, "rtx 3070 ti": 608, "rtx 3070": 448, "rtx 3080 ti": 912,
    "rtx 3080": 760, "rtx 3090 ti": 1008, "rtx 3090": 936,
    "rtx 4060 ti": 288, "rtx 4060": 272, "rtx 4070 ti super": 672, "rtx 4070 ti": 504, "rtx 4070 super": 504,
    "rtx 4070": 504, "rtx 4080 super": 736, "rtx 4080": 717, "rtx 4090": 1008,
    "rtx 5050": 320, "rtx 5060 ti": 448, "rtx 5060": 448, "rtx 5070 ti": 896, "rtx 5070": 672, "rtx 5080": 960,
    "rtx 5090": 1792,
    "rx 6600 xt": 256, "rx 6600": 224, "rx 6650 xt": 280, "rx 6700 xt": 384, "rx 6750 xt": 432, "rx 6800 xt": 512,
    "rx 6800": 512, "rx 6900 xt": 512, "rx 6950 xt": 576,
    "rx 7600 xt": 288, "rx 7600": 288, "rx 7700 xt": 432, "rx 7800 xt": 624, "rx 7900 gre": 576, "rx 7900 xtx": 960,
    "rx 7900 xt": 800, "rx 9060 xt": 322, "rx 9070 xt": 640, "rx 9070 gre": 576, "rx 9070": 640,
    "arc a380": 186, "arc a580": 512, "arc a750": 512, "arc a770": 560, "arc b570": 380, "arc b580": 456,
    "m1 max": 400, "m1 ultra": 800, "m2 ultra": 800, "m2 max": 400, "m3 max": 400, "m3 ultra": 800, "m4 max": 546,
    "m4 pro": 273, "m3 pro": 150, "m2 pro": 200,
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
            try:
                vram_total, vram_used = int(float(total)) * MiB, int(float(used)) * MiB
            except ValueError:                  # "[N/A]" on some drivers and devices: no memory size, so the GPU is not used
                continue
            gpus.append({"vendor": "nvidia", "name": name, "vram_total": vram_total,
                         "vram_used": vram_used, "driver": driver})
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
            out.append({"vendor": vendor, "name": (fields[2] if len(fields) > 2 else "GPU"), "bus": line.split()[0]})
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


def parse_windows_cpu(text):
    """`Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json` ->
    (name, physical cores, logical cores) summed over sockets, or None."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    data = data if isinstance(data, list) else [data]
    try:
        cores = sum(int(d["NumberOfCores"]) for d in data)
        logical = sum(int(d["NumberOfLogicalProcessors"]) for d in data)
    except (KeyError, TypeError, ValueError):
        return None
    return (data[0].get("Name") or "unknown").strip(), max(1, cores), max(1, logical), len(data)


def parse_windows_vram(text):
    """Registry `HardwareInformation.qwMemorySize` per display adapter -> {adapter name: bytes}. The 32-bit `AdapterRAM` of
    Win32_VideoController stops at 4 GiB, this 64-bit value does not. Some drivers store it as 8 raw bytes (a JSON list)."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return {}
    data = data if isinstance(data, list) else [data]
    out = {}
    for d in data:
        raw = d.get("HardwareInformation.qwMemorySize")
        if isinstance(raw, list) and raw and all(isinstance(x, int) for x in raw):
            raw = int.from_bytes(bytes(x & 255 for x in raw[:8]), "little")
        if isinstance(raw, int) and raw > 0 and d.get("DriverDesc"):
            out[d["DriverDesc"].strip()] = max(raw, out.get(d["DriverDesc"].strip(), 0))
    return out


# ------------------------------------------------------------------------------------------ detection flow
def _amd_vram(env, names):
    """VRAM of AMD GPUs from sysfs (amdgpu): (PCI slot or None, total, used) per card with mem_info_vram_total."""
    cards = []
    for card in env.listdir("/sys/class/drm"):
        if not re.fullmatch(r"card\d+", card):
            continue
        base = f"/sys/class/drm/{card}/device"
        if (env.read(f"{base}/vendor") or "").strip() != "0x1002":
            continue
        total, used = env.read(f"{base}/mem_info_vram_total"), env.read(f"{base}/mem_info_vram_used")
        try:
            total_b, used_b = int(total or 0), int(used or 0)
        except ValueError:                  # text such as "N/A" in the counters: this card's memory is not known
            continue
        if total_b > 0:                     # a card that reports no memory at all is not a GPU the planner can use
            cards.append((_pci_slot(env.read(f"{base}/uevent")), total_b, used_b))
    return cards


def _pci_slot(uevent):
    """'2d:00.0' from the uevent file of a PCI device (PCI_SLOT_NAME=0000:2d:00.0), or None."""
    m = re.search(r"PCI_SLOT_NAME=(?:[0-9a-f]{4}:)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])", uevent or "")
    return m.group(1) if m else None


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
    listed = parse_lspci(env.run(["lspci", "-mm"]) or "")
    described = parse_nvidia_smi(smi or "")
    for g in described:
        g["backends"] = ["cuda", "vulkan"] if env.run(["vulkaninfo", "--summary"]) else ["cuda"]
        gpus.append(g)
    unseen = sum(1 for g in listed if g["vendor"] == "nvidia") - len(described)
    if unseen > 0:                       # lspci shows a card that nvidia-smi does not describe: say so, never drop it silently
        notes.append(f"{unseen} NVIDIA GPU(s) listed by lspci have no entry in nvidia-smi (driver not installed, or no readable "
                     f"memory size): not used")
    other = [g for g in listed if g["vendor"] != "nvidia"]
    amd_mem = _amd_vram(env, other)
    by_slot = {slot: (total, used) for slot, total, used in amd_mem if slot}   # the card that lspci shows at this bus address
    unslotted = iter([(total, used) for slot, total, used in amd_mem if not slot])   # no uevent: the old order is all we have
    for g in other:
        if g["vendor"] == "amd":
            vram = by_slot.get(g.get("bus")) or next(unslotted, None)
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


PS_CPU = "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"
PS_VIDEO = "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress"
PS_VRAM = (r"Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0*' "
           r"-ErrorAction SilentlyContinue | Select-Object DriverDesc,'HardwareInformation.qwMemorySize' | ConvertTo-Json -Compress")
WINDOWS_DESKTOP_VRAM = 1024 ** 3        # what the Windows desktop and a browser typically hold in VRAM (not measurable cheaply)
UNKNOWN_VRAM = 4 * 1024 ** 3            # placeholder when the driver does not tell; the llama.cpp build corrects it at install


def _powershell(env, script):
    for exe in ("powershell", "pwsh"):
        out = env.run([exe, "-NoProfile", "-NonInteractive", "-Command", script], timeout=40)
        if out:
            return out
    return None


def detect_windows(env):
    n = os.cpu_count() or 1
    cpu = {"model": platform.processor() or "unknown", "physical_cores": max(1, n // 2), "logical_cores": n,
           "numa_nodes": 1, "flags": []}
    notes = []
    parsed = parse_windows_cpu(_powershell(env, PS_CPU) or "")
    if parsed:
        name, cores, logical, sockets = parsed
        cpu.update(model=name, physical_cores=cores, logical_cores=logical, numa_nodes=max(1, sockets))
    else:
        notes.append("CPU cores are estimated (half of the logical processors)")
    smi = env.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                   "--format=csv,noheader,nounits"])
    gpus = parse_nvidia_smi(smi or "")
    for g in gpus:
        g["backends"] = ["cuda", "vulkan"]
    video = _powershell(env, PS_VIDEO)
    if video:
        vram = parse_windows_vram(_powershell(env, PS_VRAM) or "")
        have = {g["name"].lower() for g in gpus}
        for v in parse_windows_video(video):
            if v["vendor"] == "nvidia" and gpus:
                continue                                       # nvidia-smi already described it, with the exact numbers
            if v["name"].lower() in have:
                continue
            total = vram.get(v["name"])
            if not total:
                total = UNKNOWN_VRAM
                notes.append(f"{v['name']}: video memory size is unknown, assuming {UNKNOWN_VRAM // 1024 ** 3} GiB until the "
                             f"llama.cpp build reports it")
            backends = {"amd": ["rocm", "vulkan"], "intel": ["sycl", "vulkan", "openvino"],
                        "nvidia": ["cuda", "vulkan"]}[v["vendor"]]
            gpus.append({"vendor": v["vendor"], "name": v["name"], "vram_total": total,
                         "vram_used": min(WINDOWS_DESKTOP_VRAM, total // 2), "display": True,
                         "driver": v.get("driver"), "backends": backends})
            have.add(v["name"].lower())
        if any(g["vendor"] != "nvidia" for g in gpus):
            notes.append("VRAM in use by the desktop is assumed (1 GiB) on non-NVIDIA GPUs under Windows")
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


# ------------------------------------------------------------------------------------- extended detection (INSTALL1 step 1)
# Every parser below takes the TEXT of a command or file (or a path to a recorded fixture) and returns plain data.
# `collect_extra` is the only function here that talks to the machine, and only through `env`.
def _text(src):
    """The text itself, or the contents of the file `src` points to. A file is read only when `src` is a pathlib.Path
    (explicit); a string is always text, even if it happens to spell an existing path (command output is untrusted)."""
    if isinstance(src, Path):
        with open(src, errors="replace") as f:
            return f.read()
    return src or ""


PCI_DISPLAY_CLASSES = {"0300": "vga", "0302": "3d", "0380": "display"}
_LSPCI_NN = re.compile(r"^(\S+)\s+(.+?) \[([0-9a-f]{4})\]: (.+?) \[([0-9a-f]{4}):([0-9a-f]{4})\](?: \(rev [0-9a-f]+\))?\s*$")


def _slot(s):
    """'0000:2d:00.0' or '2d:00.0' -> '2d:00.0'."""
    m = re.search(r"(?:[0-9a-f]{4}:)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])", (s or "").lower())
    return m.group(1) if m else None


def parse_lspci_nn(src):
    """`lspci -nn` -> every display device (class VGA 0300, 3D 0302, Display 0380):
    [{slot, kind, vendor, vendor_id, device_id, name, integrated}]. `integrated` is True only for an Intel device on PCI bus 00
    (where Intel iGPUs sit); for other vendors it is None: the bus address alone does not tell."""
    out = []
    for line in _text(src).splitlines():
        m = _LSPCI_NN.match(line.strip())
        if not m or m.group(3) not in PCI_DISPLAY_CLASSES:
            continue
        slot, vid, did = _slot(m.group(1)), m.group(5), m.group(6)
        vendor = VENDOR_IDS.get(vid, "other")
        integrated = True if (vendor == "intel" and slot and slot.startswith("00:")) else (False if vendor == "intel" else None)
        out.append({"slot": slot, "kind": PCI_DISPLAY_CLASSES[m.group(3)], "vendor": vendor, "vendor_id": vid,
                    "device_id": did, "name": m.group(4), "integrated": integrated})
    return out


def parse_drm_cards(entries):
    """Which GPU drives the screen. `entries` = [{card, slot, vendor_id, boot_vga, connected}] (boot_vga: text of
    drm/cardN/device/boot_vga, connected: number of connectors with status 'connected'). Returns
    {display_slot, method, boot_vga_slot, cards}. A single card with a connected connector wins ('connector'); otherwise the
    single boot_vga=1 card ('boot_vga'); otherwise display_slot is None (the caller must say so, not guess)."""
    cards = []
    for e in entries:
        bv = (e.get("boot_vga") or "").strip()
        cards.append({"card": e.get("card"), "slot": _slot(e.get("slot")), "vendor_id": (e.get("vendor_id") or "").strip().replace("0x", ""),
                      "boot_vga": True if bv == "1" else False if bv == "0" else None, "connected": int(e.get("connected") or 0)})
    boot = [c for c in cards if c["boot_vga"]]
    wired = [c for c in cards if c["connected"] > 0]
    res = {"display_slot": None, "method": None, "boot_vga_slot": boot[0]["slot"] if len(boot) == 1 else None, "cards": cards}
    if len(wired) == 1 and wired[0]["slot"]:
        res.update(display_slot=wired[0]["slot"], method="connector")
    elif len(boot) == 1 and boot[0]["slot"]:
        res.update(display_slot=boot[0]["slot"], method="boot_vga")
    return res


def parse_nvidia_smi_header(src):
    """Plain `nvidia-smi` banner -> {driver, cuda}. cuda is None when the banner has no 'CUDA Version'."""
    t = _text(src)
    d = re.search(r"Driver Version:\s*([\d.]+)", t)
    c = re.search(r"CUDA Version:\s*([\d.]+)", t)
    return {"driver": d.group(1) if d else None, "cuda": c.group(1) if c else None}


def parse_rocm_version(src):
    """/opt/rocm/.info/version ('7.2.4-123') -> '7.2.4-123', or None."""
    m = re.search(r"\d+(?:\.\d+)+(?:-[\w.]+)?", _text(src))
    return m.group(0) if m else None


def parse_kfd_gfx(src):
    """A KFD topology node `properties` file -> 'gfx1101' (from gfx_target_version=110001), or None (CPU node / not present)."""
    m = re.search(r"^gfx_target_version\s+(\d+)", _text(src), re.M)
    if not m or int(m.group(1)) == 0:
        return None
    v = int(m.group(1))
    return f"gfx{v // 10000}{(v // 100) % 100:x}{v % 100:x}"


def parse_vulkaninfo_summary(src):
    """`vulkaninfo --summary` -> {instance, devices: [{name, type, api, driver}]}."""
    t = _text(src)
    inst = re.search(r"Vulkan Instance Version:\s*([\d.]+)", t)
    devices = []
    for block in re.split(r"^GPU\d+:\s*$", t, flags=re.M)[1:]:
        kv = dict((k.strip(), v.strip()) for k, v in re.findall(r"^\s*(\w+)\s*=\s*(.+)$", block, re.M))
        if "deviceName" in kv:
            devices.append({"name": kv["deviceName"], "type": kv.get("deviceType", "").replace("PHYSICAL_DEVICE_TYPE_", "").lower(),
                            "api": kv.get("apiVersion"), "driver": kv.get("driverInfo") or kv.get("driverName")})
    return {"instance": inst.group(1) if inst else None, "devices": devices}


def parse_vulkan_icd_names(names):
    """File names of /usr/share/vulkan/icd.d (or /etc/vulkan/icd.d) -> sorted list of ICD .json names (a loader exists, but
    this does not prove the driver works: vulkaninfo does)."""
    return sorted({n for n in names if n.endswith(".json")})


def parse_sycl_ls(src):
    """`sycl-ls` (oneAPI) -> [{backend, type, name}], e.g. '[level_zero:gpu][level_zero:0] Intel(R) Level-Zero, ...'."""
    out = []
    for line in _text(src).splitlines():
        m = re.match(r"\s*\[([\w]+):(\w+)\]\[[^\]]*\]\s*(.+)$", line)
        if m:
            out.append({"backend": m.group(1), "type": m.group(2), "name": m.group(3).strip()})
    return out


def parse_proc_swaps(src):
    """/proc/swaps -> [{kind, size, used, priority}] in bytes. kind: 'zram' for /dev/zram*, else the Type column
    ('partition', 'file'). File names are dropped on purpose (they can hold user names)."""
    out = []
    for line in _text(src).splitlines()[1:]:
        p = line.split()
        if len(p) >= 5:
            try:
                out.append({"kind": "zram" if p[0].startswith("/dev/zram") else p[1], "size": int(p[2]) * 1024,
                            "used": int(p[3]) * 1024, "priority": int(p[4])})
            except ValueError:
                continue
    return out


def parse_zram_device(disksize, comp_algorithm, mm_stat):
    """One /sys/block/zramN: disksize (bytes), comp_algorithm ('lzo lzo-rle [zstd]'), mm_stat -> dict.
    Fields that are missing or unreadable are None."""
    def num(x, i=0):
        try:
            return int(_text(x).split()[i])
        except (ValueError, IndexError):
            return None
    sel = re.search(r"\[([\w-]+)\]", _text(comp_algorithm))
    return {"disksize": num(disksize), "algorithm": sel.group(1) if sel else None,
            "orig_bytes": num(mm_stat, 0), "compressed_bytes": num(mm_stat, 1)}


def parse_cgroup_controllers(src):
    """/sys/fs/cgroup/cgroup.controllers ('cpuset cpu io memory pids') -> sorted list."""
    return sorted(_text(src).split())


def parse_power_supply(entries):
    """`entries` = [{type, online, status}] from /sys/class/power_supply/*. -> {has_battery, on_battery, mains_online}.
    on_battery is None when it cannot be told (a battery with no mains 'online' file and no clear charge status)."""
    ents = [{"type": (e.get("type") or "").strip().lower(), "online": (e.get("online") or "").strip(),
             "status": (e.get("status") or "").strip().lower()} for e in entries]
    bats = [e for e in ents if e["type"] == "battery"]
    mains = [e for e in ents if e["type"] in ("mains", "usb", "ac") and e["online"] in ("0", "1")]
    mains_online = any(e["online"] == "1" for e in mains) if mains else None
    if not bats:
        return {"has_battery": False, "on_battery": False, "mains_online": mains_online}
    if mains:
        on_batt = not mains_online
    elif any(b["status"] == "discharging" for b in bats):
        on_batt = True
    elif any(b["status"] in ("charging", "full") for b in bats):
        on_batt = False
    else:
        on_batt = None
    return {"has_battery": True, "on_battery": on_batt, "mains_online": mains_online}


def measure_disk_read(path, *, size=64 * MiB, budget_s=8.0, opener=open, clock=time.monotonic):
    """Short sequential read of up to `size` bytes from the sample file `path` -> (MiB/s or None, note or None).
    Never writes. Stops at `budget_s`. The kernel's cached pages of that file are dropped first (posix_fadvise DONTNEED,
    allowed without root), so a repeat run is not just a RAM read. Needs at least 16 MiB read to give a number."""
    try:
        f = opener(path, "rb")
    except OSError as e:
        return None, f"disk read speed: sample file cannot be opened ({type(e).__name__})"
    got = 0
    try:
        try:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError, ValueError):
            pass
        t0 = clock()
        while got < size:
            chunk = f.read(min(4 * MiB, size - got))
            if not chunk:
                break
            got += len(chunk)
            if clock() - t0 > budget_s:
                break
        dt = clock() - t0
    except OSError as e:
        return None, f"disk read speed: read failed ({type(e).__name__})"
    finally:
        f.close()
    if got < 16 * MiB or dt <= 0:
        return None, "disk read speed: sample file is shorter than 16 MiB or the read was too fast to time"
    return round(got / MiB / dt, 1), None


def _drm_entries(env):
    entries = []
    names = env.listdir("/sys/class/drm")
    for card in names:
        if not re.fullmatch(r"card\d+", card):
            continue
        base = f"/sys/class/drm/{card}/device"
        connected = sum(1 for n in names if n.startswith(card + "-")
                        and (env.read(f"/sys/class/drm/{n}/status") or "").strip() == "connected")
        entries.append({"card": card, "slot": _pci_slot(env.read(f"{base}/uevent")), "vendor_id": env.read(f"{base}/vendor"),
                        "boot_vga": env.read(f"{base}/boot_vga"), "connected": connected})
    return entries


def collect_extra(env, gpus, *, disk_sample=None, disk_benchmark=False):
    """Everything INSTALL1 step 1 adds, read through `env`. Returns (system dict, disk dict, notes). `gpus` (the list from
    detect_*) gets its `display` flag corrected where the display GPU is known. Linux only; elsewhere says so in the notes."""
    notes = []
    sysd = {"gpus_all": [], "display": {"display_slot": None, "method": None, "boot_vga_slot": None},
            "drivers": {"rocm": None, "rocm_gfx": [], "nvidia_driver": None, "cuda": None, "vulkan": None, "oneapi": None},
            "swap": [], "zram": [], "cgroup_v2": None, "cgroup_controllers": [], "power": None}
    disk = {"read_mibs": None}
    # 2. all video cards
    lspci = env.run(["lspci", "-nn"])
    if lspci is None:
        notes.append("lspci -nn is not available: the list of video cards is limited to what other tools report")
    else:
        sysd["gpus_all"] = parse_lspci_nn(lspci)
    vendors = {g["vendor"] for g in sysd["gpus_all"]}
    # 1. integrated + display
    if env.exists("/sys/class/drm"):
        disp = parse_drm_cards(_drm_entries(env))
        sysd["display"] = {k: disp[k] for k in ("display_slot", "method", "boot_vga_slot")}
        if disp["display_slot"] is None:
            notes.append("which GPU draws the screen could not be determined (no connected connector, no single boot_vga)")
        elif disp["method"] == "boot_vga":
            notes.append("display GPU is taken from boot_vga (no connected connector was visible): on a laptop with a MUX switch it may differ")
    else:
        notes.append("/sys/class/drm is not readable: the display GPU is unknown")
    ds = sysd["display"]["display_slot"]
    for g in sysd["gpus_all"]:
        g["display"] = (g["slot"] == ds) if ds else None
    if sum(1 for g in sysd["gpus_all"] if g["integrated"] is None) and len(sysd["gpus_all"]) > 1:
        notes.append("integrated or discrete is not told by the PCI listing for a non-Intel GPU next to another GPU: left unknown")
    if ds:
        for g in gpus:
            slot = _slot(g.get("bus"))
            if slot:
                g["display"] = slot == ds
            elif sum(1 for a in sysd["gpus_all"] if a["vendor"] == g["vendor"]) == 1:
                match = [a for a in sysd["gpus_all"] if a["vendor"] == g["vendor"]]
                g["display"] = match[0]["slot"] == ds
    # 3. drivers and runtimes
    d = sysd["drivers"]
    if "nvidia" in vendors or lspci is None:
        banner = env.run(["nvidia-smi"])
        if banner is None:
            if "nvidia" in vendors:
                notes.append("nvidia-smi is not available: NVIDIA driver and CUDA versions unknown")
        else:
            h = parse_nvidia_smi_header(banner)
            d["nvidia_driver"], d["cuda"] = h["driver"], h["cuda"]
            if not h["cuda"]:
                notes.append("nvidia-smi does not report a CUDA version")
    if "amd" in vendors:
        d["rocm"] = parse_rocm_version(env.read("/opt/rocm/.info/version") or "")
        if not d["rocm"]:
            notes.append("ROCm version not found (/opt/rocm/.info/version): ROCm is not installed system-wide, or lives elsewhere")
        gfx = []
        for n in env.listdir("/sys/class/kfd/kfd/topology/nodes"):
            t = parse_kfd_gfx(env.read(f"/sys/class/kfd/kfd/topology/nodes/{n}/properties") or "")
            if t:
                gfx.append(t)
        d["rocm_gfx"] = gfx
        if not gfx:
            notes.append("AMD GPU architecture (gfx target) not readable from KFD")
    if sysd["gpus_all"]:
        vk = env.run(["vulkaninfo", "--summary"])
        if vk is not None:
            d["vulkan"] = parse_vulkaninfo_summary(vk)
        else:
            icds = parse_vulkan_icd_names(env.listdir("/usr/share/vulkan/icd.d") + env.listdir("/etc/vulkan/icd.d"))
            d["vulkan"] = {"instance": None, "devices": [], "icd_files": icds}
            notes.append("vulkaninfo is not available: " + (f"{len(icds)} Vulkan ICD file(s) found, not tested"
                                                             if icds else "no Vulkan loader or driver files found"))
    if "intel" in vendors:
        sy = env.run(["sycl-ls"])
        if sy is None:
            notes.append("sycl-ls is not available: oneAPI/SYCL not detected")
        else:
            d["oneapi"] = parse_sycl_ls(sy)
    # 5. swap and zram
    sw = env.read("/proc/swaps")
    if sw is None:
        notes.append("/proc/swaps is not readable: swap unknown")
    else:
        sysd["swap"] = parse_proc_swaps(sw)
    for n in env.listdir("/sys/block"):
        if re.fullmatch(r"zram\d+", n):
            b = f"/sys/block/{n}"
            sysd["zram"].append(parse_zram_device(env.read(f"{b}/disksize") or "", env.read(f"{b}/comp_algorithm") or "",
                                                  env.read(f"{b}/mm_stat") or ""))
    # 6. cgroup v2
    cg = env.read("/sys/fs/cgroup/cgroup.controllers")
    sysd["cgroup_v2"] = cg is not None
    if cg is not None:
        sysd["cgroup_controllers"] = parse_cgroup_controllers(cg)
    else:
        notes.append("cgroup v2 not found (/sys/fs/cgroup/cgroup.controllers): resource limits through cgroups are not available")
    # 7. power
    ps = []
    for n in env.listdir("/sys/class/power_supply"):
        b = f"/sys/class/power_supply/{n}"
        ps.append({"type": env.read(f"{b}/type"), "online": env.read(f"{b}/online"), "status": env.read(f"{b}/status")})
    sysd["power"] = parse_power_supply(ps)
    if sysd["power"]["has_battery"] and sysd["power"]["on_battery"] is None:
        notes.append("battery present but mains/battery state could not be determined")
    # 4. disk read speed (free space is measured by detect() itself)
    if not disk_benchmark or not disk_sample:
        notes.append("disk read speed: not measured" + ("" if disk_sample or disk_benchmark else " (no sample file given)"))
    else:
        disk["read_mibs"], note = measure_disk_read(disk_sample)
        if note:
            notes.append(note)
    return sysd, disk, notes


def detect(env=None, *, ram_bandwidth_gbs=None, disk_sample=None, disk_benchmark=False):
    """Hardware profile of this machine. `ram_bandwidth_gbs` comes from the bandwidth measurement (see HARDWARE.md).
    `disk_sample` is a file of 64 MiB or more to read for the disk speed (only read when `disk_benchmark` is True)."""
    env = env or SystemEnv()
    system = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(platform.system(), "linux")
    cpu, mem, gpus, notes = {"linux": detect_linux, "macos": detect_macos, "windows": detect_windows}[system](env)
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64"
    profile = build_profile(system, cpu, mem, gpus, notes, ram_bandwidth_gbs=ram_bandwidth_gbs, arch=arch,
                            disk_free=shutil.disk_usage(os.path.expanduser("~")).free)
    if system == "linux":
        profile["system"], extra_disk, extra_notes = collect_extra(env, gpus, disk_sample=disk_sample, disk_benchmark=disk_benchmark)
        profile["disk"].update(extra_disk)
        profile["detection_notes"].extend(extra_notes)
    else:
        profile["detection_notes"].append("drivers, swap, cgroup, power and disk speed are detected on Linux only so far")
    # what could build our fork here: the planner offers the ROCm build of it only where hipconfig and a build chain exist
    profile["toolchain"] = {t: bool(shutil.which(t)) for t in ("hipconfig", "cmake", "ninja", "git")}
    return profile
