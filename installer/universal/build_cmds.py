"""Build commands for shura.cpp from source for CUDA, Vulkan and CPU (HIP has its own script). This module only PRODUCES
the command lists and checks that the tools exist; it never runs a build and never calls sudo."""
import os
import shutil
from pathlib import Path

from . import install

BACKEND_FLAGS = {"cuda": ["-DGGML_CUDA=ON"], "vulkan": ["-DGGML_VULKAN=ON"], "cpu": []}
VULKAN_HEADER = "/usr/include/vulkan/vulkan.h"

_EXTRA_PKGS = {  # tool -> package per distro family (text only)
    "nvcc": {"fedora": "cuda-toolkit (from the NVIDIA repository)", "debian": "nvidia-cuda-toolkit", "arch": "cuda"},
    "vulkan-headers": {"fedora": "vulkan-headers vulkan-loader-devel", "debian": "libvulkan-dev", "arch": "vulkan-headers vulkan-icd-loader"},
    "glslc": {"fedora": "glslc", "debian": "glslc", "arch": "shaderc"},
}


def build_commands(backend, src, build_dir=None, jobs=None):
    """List of argv lists (cmake configure, then cmake --build) for building llama-server from `src` for `backend`."""
    if backend not in BACKEND_FLAGS:
        raise ValueError(f"no source build for backend {backend!r} (supported: {', '.join(BACKEND_FLAGS)})")
    src = str(Path(src).expanduser())
    build = str(Path(build_dir).expanduser()) if build_dir else str(Path(src) / f"build-{backend}")
    configure = ["cmake", "-S", src, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", *BACKEND_FLAGS[backend]]
    compile_ = ["cmake", "--build", build, "--target", "llama-server"] + (["-j", str(int(jobs))] if jobs else [])
    return [configure, compile_]


def missing_tools(backend, which=shutil.which, exists=os.path.exists):
    """Short names of what the build needs and is not found: cmake, ninja, compiler; nvcc (CUDA); vulkan-headers, glslc (Vulkan)."""
    miss = [t for t in ("cmake", "ninja") if not which(t)]
    if not any(which(c) for c in ("c++", "g++", "clang++")):
        miss.append("compiler")
    if backend == "cuda" and not which("nvcc"):
        miss.append("nvcc")
    if backend == "vulkan":
        if not exists(VULKAN_HEADER):
            miss.append("vulkan-headers")
        if not which("glslc"):
            miss.append("glslc")
    return miss


def check_tools(backend, which=shutil.which, exists=os.path.exists, family=None):
    """[] when all is there; otherwise plain lines: what is missing and the install command (for you to run with administrator rights)."""
    miss = missing_tools(backend, which, exists)
    if not miss:
        return []
    names = {"compiler": "a C++ compiler", "nvcc": "nvcc (CUDA toolkit)", "vulkan-headers": "Vulkan headers", "glslc": "glslc (shader compiler)"}
    lines = [f"Build from source for {backend} needs " + ", ".join(names.get(t, t) for t in miss) + ", and they are not installed.",
             "Install them yourself with administrator rights, then run the same command again:"]
    fam = family or install._distro_family()
    for f in ([fam] if fam else ["fedora", "debian", "arch"]):
        pk = " ".join(dict.fromkeys(" ".join((install._PKGS.get(t) or _EXTRA_PKGS[t])[f] for t in miss).split()))
        lines.append(("  " + (f + ": " if not fam else "")) + install._FAMILY_CMD[f].format(pk))
    return lines
