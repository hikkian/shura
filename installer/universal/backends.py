"""Choosing which prebuilt llama.cpp build to download for a backend.

llama.cpp publishes ready-made builds for many backends on every release, named like
`llama-b11301-bin-ubuntu-vulkan-x64.tar.gz`. Asset names change between releases, so the names are parsed with a
tolerant pattern and the selection works on whatever the release actually lists. Pure functions: the release
listing (GitHub API JSON) goes in, an asset comes out. Downloading and self-testing are done by the installer.
"""
import hashlib
import re

_NAME = re.compile(
    r"^llama-(?P<tag>b\d+)-bin-(?P<os>ubuntu|macos|win)-(?P<rest>.+?)\.(?:tar\.gz|zip)$")
_OS = {"ubuntu": "linux", "macos": "macos", "win": "windows"}
_ARCH = {"x64": "x86_64", "arm64": "arm64"}


def parse_asset(name):
    """`llama-b11301-bin-ubuntu-cuda-13.4-x64.tar.gz` -> {os, arch, backend, version, variant} or None."""
    m = _NAME.match(name)
    if not m:
        return None
    parts = m["rest"].split("-")
    if parts[-1] not in _ARCH:
        return None
    arch, words = _ARCH[parts.pop()], parts
    backend, version, variant = "cpu", None, None
    for w in words:
        if w in ("cuda", "rocm", "vulkan", "sycl", "openvino", "opencl", "hip"):
            backend = "rocm" if w == "hip" else w
        elif re.fullmatch(r"\d+(\.\d+)*", w):
            version = w
        elif w in ("fp16", "fp32", "cpu", "adreno", "snapdragon"):
            variant = w
    if "snapdragon" in words or "adreno" in words:
        return None                                   # phone/ARM-laptop builds are out of scope
    if backend == "cpu" and not words and m["os"] == "macos":
        backend = "metal"                             # the Apple Silicon build uses Metal by default
    return {"os": _OS[m["os"]], "arch": arch, "backend": backend, "version": version, "variant": variant,
            "tag": m["tag"], "name": name}


_TQP = re.compile(r"^turboquant-plus-(?P<tag>tqp-v[\d.]+)-(?P<os>linux|macos|windows)-(?P<arch>x64|arm64)-"
                  r"(?P<backend>cpu|vulkan|metal|cuda)(?P<version>[\d.]*)\.(?:tar\.gz|zip)$")


def parse_tqp_asset(name):
    """`turboquant-plus-tqp-v0.4.0-linux-x64-vulkan.tar.gz` (prebuilts of the TurboQuant+ llama.cpp fork: turbo KV cache,
    expert cache, MTP) -> {os, arch, backend, version, tag, name} or None."""
    m = _TQP.match(name)
    if not m:
        return None
    return {"os": m["os"], "arch": _ARCH[m["arch"]], "backend": m["backend"], "version": m["version"] or None,
            "tag": m["tag"], "name": name}


def pick_tqp_asset(assets, os_family, arch, backend):
    for a in assets:
        p = parse_tqp_asset(a["name"])
        if p and (p["os"], p["arch"], p["backend"]) == (os_family, arch, backend):
            return a
    return None


def _vkey(version):
    return tuple(int(x) for x in version.split(".")) if version else ()


def cuda_versions_allowed(driver):
    """Highest CUDA major a driver can run (the self-test still has the last word)."""
    try:
        major = int(str(driver).split(".")[0])
    except (TypeError, ValueError):
        return None
    return 13 if major >= 580 else 12 if major >= 525 else 0


def pick_asset(assets, os_family, arch, backend, driver=None, prefer_variant="fp16"):
    """Best matching asset (dict from the release listing) for a backend, or None.

    CUDA: the newest build the driver supports. Several versions of ROCm/OpenVINO: the newest. SYCL: fp16 by default.
    """
    want = "metal" if backend == "metal" else backend
    found = []
    for a in assets:
        p = parse_asset(a["name"])
        if p and (p["os"], p["arch"], p["backend"]) == (os_family, arch, want):
            found.append((p, a))
    if backend == "cuda":
        allowed = cuda_versions_allowed(driver)
        if allowed is not None:
            found = [(p, a) for p, a in found if p["version"] and int(p["version"].split(".")[0]) <= allowed]
    if not found:
        return None
    found.sort(key=lambda pa: (pa[0]["variant"] == prefer_variant, _vkey(pa[0]["version"])), reverse=True)
    return found[0][1]


def builds_for(profile, backend_candidates, assets):
    """Ordered [(backend, asset)] for this machine: every candidate backend that has a build on this release."""
    osf, arch = profile["os"]["family"], profile["os"]["arch"]
    driver = next((g.get("driver") for g in profile["gpus"] if g["vendor"] == "nvidia"), None)
    out = []
    for b in backend_candidates:
        a = pick_asset(assets, osf, arch, b, driver)
        if a:
            out.append((b, a))
    return out


def verify_sha256(path, expected):
    """True if the file's SHA-256 equals `expected` (GitHub gives it as `sha256:<hex>`; a bare hex is accepted too)."""
    want = expected.split(":", 1)[-1].lower()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == want
