"""Choosing a prebuilt shura.cpp 1.0 build from `release-manifest.json` (the file itself is prepared by card RELBIN1).

The contract (names, fields, rules) is docs/RELEASE-CONTRACT.md. Shape: {"schema": 1, "releases": [{"id", "engine": "shura.cpp", "version", "family": HIP|CUDA|Vulkan|SYCL|CPU,
"gpu_targets": [gfx codes / generations], "runtime": {"rocm_min": "6.2", "rocm_max": ..., "nvidia_driver_min": ...,
"vulkan_min": ..., "oneapi_min": ...}, "glibc_min": "2.31", "arch": "x86_64" (default), "parts": ["vram_governor", ...], "built_with": {"commit", "cmake_flags", ...},
"cpu_isa": [...] (CPU only, informational), "assets": [{"name": "shura-cpp-<version>-<family>-linux-<arch>.tar.gz", "url", "sha256", "size"}]}]}

Pure functions, no network, no guessing: a field the machine does not report, or a release does not carry, means "not suitable".
"""
import json
import os
import platform
import re
import subprocess

from . import backends

SCHEMA = 1
FAMILIES = {"HIP": "rocm", "CUDA": "cuda", "Vulkan": "vulkan", "SYCL": "sycl", "CPU": "cpu"}
RUNTIME_OF = {"HIP": "rocm", "CUDA": "nvidia_driver", "Vulkan": "vulkan", "SYCL": "oneapi"}   # which runtime a family needs
RUNTIME_KEYS = {"rocm", "nvidia_driver", "vulkan", "oneapi"}
SHA = re.compile(r"^[0-9a-fA-F]{64}$")


class ManifestError(Exception):
    pass


def vkey(text):
    """'2.31' -> (2, 31); None when it is not a dotted number."""
    m = re.match(r"^\s*(\d+(?:\.\d+)*)", str(text) if text is not None else "")
    return tuple(int(x) for x in m[1].split(".")) if m else None


def check_asset_name(name):
    """The asset name must be a plain file name: it is used as a path on disk."""
    if (not isinstance(name, str) or not name or "/" in name or "\\" in name or ".." in name
            or name.startswith(".") or "\0" in name):
        raise ManifestError(f"asset name {name!r} is not a plain file name (no '/', '\\', '..', leading '.' or NUL)")


def rocm_same_major(runtime, installed_version):
    """(ok, reason): the installed ROCm must have the same major version as both `rocm_min` and `rocm_max` of the release."""
    if not isinstance(runtime, dict) or not runtime.get("rocm_min") or not runtime.get("rocm_max"):
        return False, "the release does not give both rocm_min and rocm_max"
    lo, hi, have = vkey(runtime["rocm_min"]), vkey(runtime["rocm_max"]), vkey(installed_version)
    if lo is None or hi is None:
        return False, "the release's rocm_min / rocm_max are not version numbers"
    if have is None:
        return False, "ROCm version unknown"
    if have[0] != lo[0] or have[0] != hi[0]:
        need = str(lo[0]) if lo[0] == hi[0] else f"{lo[0]}..{hi[0]}"
        return False, f"ROCm major {have[0]}, release needs {need}"
    return True, "ROCm major version matches"


def glibc_host_ok(glibc_min, host_glibc):
    """(ok, reason): the host's glibc must be at least `glibc_min` (the release's ceiling for the oldest supported system)."""
    host, need = vkey(host_glibc), vkey(glibc_min)
    if host is None:
        return False, "host glibc unknown"
    if need is None:
        return False, f"bad glibc_min {glibc_min!r}"
    if host < need:
        return False, f"host glibc {host_glibc}, release needs {glibc_min}"
    return True, "host glibc is new enough"


def check_release_asset(rel, asset):
    """The archive name must follow the contract and agree with the release's own family, version and architecture."""
    check_asset_name(asset["name"])
    p = backends.parse_release_asset(asset["name"])
    if not p:
        raise ManifestError(f"asset name {asset['name']!r} does not follow shura-cpp-<version>-<family>-<os>-<arch>.tar.gz")
    want = (rel["family"], str(rel["version"]), rel.get("arch", "x86_64"))
    if (p["family"], p["version"], p["arch"]) != want:
        raise ManifestError(f"asset name {asset['name']!r} says {p['family']} {p['version']} {p['arch']}, "
                            f"the release says {want[0]} {want[1]} {want[2]}")


def validate(manifest):
    """The manifest itself, or ManifestError with a plain-language reason."""
    if not isinstance(manifest, dict):
        raise ManifestError("the manifest is not a JSON object")
    if manifest.get("schema") != SCHEMA:
        raise ManifestError(f"unknown manifest schema {manifest.get('schema')!r} (this installer reads {SCHEMA})")
    rels = manifest.get("releases")
    if not isinstance(rels, list):
        raise ManifestError("the manifest has no 'releases' list")
    for i, r in enumerate(rels):
        where = f"release #{i + 1}" + (f" ({r.get('id')})" if isinstance(r, dict) and r.get("id") else "")
        if not isinstance(r, dict):
            raise ManifestError(f"{where} is not an object")
        for k in ("id", "engine", "version", "family", "gpu_targets", "runtime", "glibc_min", "assets"):
            if k not in r:
                raise ManifestError(f"{where} has no field '{k}'")
        if r["family"] not in FAMILIES:
            raise ManifestError(f"{where}: unknown family {r['family']!r} (expected {', '.join(FAMILIES)})")
        if not isinstance(r["gpu_targets"], list) or not all(isinstance(t, str) for t in r["gpu_targets"]):
            raise ManifestError(f"{where}: 'gpu_targets' must be a list of strings")
        if r["family"] not in ("HIP", "CUDA") and r["gpu_targets"]:
            raise ManifestError(f"{where}: 'gpu_targets' must be empty for {r['family']} (only HIP and CUDA name targets)")
        if r["family"] in ("HIP", "CUDA") and not r["gpu_targets"]:
            raise ManifestError(f"{where}: 'gpu_targets' is empty for {r['family']}")
        if r.get("arch", "x86_64") not in ("x86_64", "arm64"):
            raise ManifestError(f"{where}: bad arch {r.get('arch')!r}")
        if not isinstance(r["version"], str) or not r["version"]:
            raise ManifestError(f"{where}: bad version {r['version']!r}")
        if not isinstance(r.get("parts"), list) or not all(isinstance(t, str) and t for t in r["parts"]):
            raise ManifestError(f"{where} has no 'parts' list of strings (our parts that work in this build)")
        bw = r.get("built_with")
        if not isinstance(bw, dict) or not isinstance(bw.get("commit"), str) or not bw["commit"]:
            raise ManifestError(f"{where} has no 'built_with' object with a 'commit'")
        if "cpu_isa" in r and not (isinstance(r["cpu_isa"], list) and all(isinstance(t, str) for t in r["cpu_isa"])):
            raise ManifestError(f"{where}: 'cpu_isa' must be a list of strings")
        rt = r["runtime"]
        if not isinstance(rt, dict):
            raise ManifestError(f"{where}: 'runtime' must be an object")
        for k, v in rt.items():
            base = k[:-4] if k.endswith(("_min", "_max")) else None
            if base not in RUNTIME_KEYS or vkey(v) is None:
                raise ManifestError(f"{where}: bad runtime entry {k!r}: {v!r}")
        if vkey(r["glibc_min"]) is None:
            raise ManifestError(f"{where}: bad glibc_min {r['glibc_min']!r}")
        if not isinstance(r["assets"], list) or not r["assets"]:
            raise ManifestError(f"{where} has no assets")
        for a in r["assets"]:
            if not isinstance(a, dict):
                raise ManifestError(f"{where}: an asset is not an object")
            for k in ("name", "url", "sha256"):
                if not isinstance(a.get(k), str) or not a[k]:
                    raise ManifestError(f"{where}: an asset has no '{k}'")
            check_release_asset(r, a)
            if not SHA.match(a["sha256"]):
                raise ManifestError(f"{where}: asset {a['name']} has a bad sha256")
            if not a["url"].startswith("https://"):
                raise ManifestError(f"{where}: asset {a['name']} has a non-HTTPS url")
            if "size" in a and not (isinstance(a["size"], int) and a["size"] > 0):
                raise ManifestError(f"{where}: asset {a['name']} has a bad size")
    return manifest


def parse(text):
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as e:
        raise ManifestError(f"the manifest is not valid JSON ({e})") from None
    return validate(data)


def load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return parse(f.read())
    except OSError as e:
        raise ManifestError(f"the manifest cannot be read ({e})") from None


def machine_from_profile(hw, candidates, *, run=None, read=None, glibc=None):
    """What the manifest is matched against: {backends, gpu_targets, rocm, nvidia_driver, vulkan, oneapi, glibc, os, arch}.
    `hw` is the detect.py profile; the extra probes (rocminfo, /opt/rocm version, vulkaninfo, libc) can be replaced in tests.
    A value that cannot be read stays None."""
    run = run or _run
    read = read or _read
    osf, arch = hw["os"]["family"], hw["os"]["arch"]
    m = {"os": osf, "arch": arch, "backends": list(candidates), "gpu_targets": [], "rocm": None, "nvidia_driver": None,
         "vulkan": None, "oneapi": None, "glibc": glibc if glibc is not None else _glibc()}
    for g in hw.get("gpus", []):
        for k in ("gfx", "arch", "sm"):
            if g.get(k):
                m["gpu_targets"].append(str(g[k]).lower())
        if g.get("generation"):
            m["gpu_targets"].append(str(g["generation"]).lower())
        if g.get("vendor") == "nvidia" and g.get("driver") and not m["nvidia_driver"]:
            m["nvidia_driver"] = str(g["driver"])
    if "rocm" in candidates:
        m["rocm"] = (read("/opt/rocm/.info/version") or "").strip() or None
        if not any(t.startswith("gfx") for t in m["gpu_targets"]):
            m["gpu_targets"] += sorted(set(re.findall(r"\bgfx[0-9a-f]+\b", (run(["rocminfo"]) or "").lower())))
    if "vulkan" in candidates:
        mm = re.search(r"Vulkan Instance Version:\s*(\d+\.\d+(?:\.\d+)?)", run(["vulkaninfo", "--summary"]) or "")
        m["vulkan"] = mm[1] if mm else None
    m["gpu_targets"] = sorted(set(m["gpu_targets"]))
    return m


def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _glibc():
    try:
        name, ver = os.confstr("CS_GNU_LIBC_VERSION").split()
        return ver if name == "glibc" else None
    except (OSError, ValueError, AttributeError):
        lib, ver = platform.libc_ver()
        return ver if lib == "glibc" and ver else None


def why_not(rel, machine):
    """None when `rel` fits `machine`, else the plain reason it does not."""
    fam, rid = rel["family"], rel["id"]
    if rel["engine"] != "shura.cpp":
        return f"{rid}: engine {rel['engine']!r} is not shura.cpp"
    if machine.get("os") != "linux":
        return f"{rid}: prebuilt shura.cpp builds are for Linux, this is {machine.get('os')}"
    if rel.get("arch", "x86_64") != machine.get("arch"):
        return f"{rid}: built for {rel.get('arch', 'x86_64')}, this machine is {machine.get('arch')}"
    if FAMILIES[fam] not in machine.get("backends", []):
        return f"{rid}: {fam} is not a backend this machine can use"
    if fam != "CPU":
        have = {t.lower() for t in machine.get("gpu_targets", [])}
        if fam in ("HIP", "CUDA") and not have:
            return f"{rid}: the GPU target (gfx code / generation) of this machine is unknown"
        if fam in ("HIP", "CUDA") and not have & {t.lower() for t in rel["gpu_targets"]}:
            return f"{rid}: built for {', '.join(rel['gpu_targets']) or 'no GPU'}, this machine has {', '.join(sorted(have))}"
        base = RUNTIME_OF[fam]
        if not any(k in (base + "_min", base + "_max") for k in rel["runtime"]):
            return f"{rid}: the manifest does not say which {base} versions it needs"
    for k, bound in rel["runtime"].items():
        base, kind = k[:-4], k[-3:]
        have = vkey(machine.get(base))
        if have is None:
            return f"{rid}: needs {base} {kind} {bound}, but this machine's {base} version is unknown"
        want = vkey(bound)
        if (kind == "min" and have < want) or (kind == "max" and have > want):
            return f"{rid}: needs {base} {kind} {bound}, this machine has {machine.get(base)}"
    have_c = vkey(machine.get("glibc"))
    if have_c is None:
        return f"{rid}: this machine's glibc version is unknown"
    if have_c < vkey(rel["glibc_min"]):
        return f"{rid}: needs glibc {rel['glibc_min']}, this machine has {machine.get('glibc')}"
    return None


def best_release(manifest, machine):
    """(release, reason) for the best fitting build, or (None, reason). Preference: the machine's own backend order
    (candidates come best first), then the newest version."""
    order = {b: i for i, b in enumerate(machine.get("backends", []))}
    fits, misses = [], []
    for rel in manifest["releases"]:
        why = why_not(rel, machine)
        (misses if why else fits).append(why or rel)
    if not fits:
        return None, ("no prebuilt shura.cpp build fits this machine: " + "; ".join(misses)) if misses else \
            "the manifest lists no builds"
    fits.sort(key=lambda r: (order[FAMILIES[r["family"]]], tuple(-x for x in (vkey(r["version"]) or (0,)))))
    r = fits[0]
    return r, f"prebuilt shura.cpp {r['version']} ({r['family']}) fits this machine"
