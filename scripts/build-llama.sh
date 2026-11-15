#!/usr/bin/env bash
# Build the llama.cpp fork of this project (base: a thecodacus/llama.cpp `perf` snapshot) at the exact benchmarked commit,
# with the TurboQuant fix applied. The fork is published as hikkian/llama.cpp-shura (FORKPUB1: prepared, not pushed yet).
#
#   scripts/build-llama.sh [--attention-decode] [destination]        (default: ~/ai/llama.cpp-perf)
#   CUDA_ARCH=86 scripts/build-llama.sh          (RTX 30xx; default 89 = RTX 40xx)
#   SHURA_CUDA_BIN=/opt/cuda-12/bin scripts/build-llama.sh      the folder of nvcc when it is not in PATH (default search: /usr/local/cuda/bin, /opt/cuda/bin)
#   SHURA_BUILD_TARGETS="llama-server llama-perplexity test-backend-ops" SHURA_BUILD_EXAMPLES=OFF limits HIP build outputs.
#   scripts/build-llama.sh --hip [gfx1101] [destination]     AMD with ROCm (HIP): the same fork and its expert cache, built for
#                                                            your GPU (default: the one `rocminfo` reports). UNVERIFIED on AMD:
#                                                            the author has no AMD card; it takes tens of minutes.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HOME/ai/llama.cpp-perf"
ATTENTION_DECODE=0
HIP=0
HIP_ARCH=""
DEST_SET=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --attention-decode) ATTENTION_DECODE=1 ;;
    --hip) HIP=1
      if [ "$#" -gt 1 ] && [[ "$2" =~ ^gfx[0-9a-f]+$ ]]; then HIP_ARCH="$2"; shift; fi ;;
    --*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)
      if [ "$DEST_SET" -eq 1 ]; then echo "Only one destination is allowed" >&2; exit 1; fi
      DEST="$1"
      DEST_SET=1
      ;;
  esac
  shift
done
# The measured fork commit. It is taken from SHURA_LLAMA_REPO when set (a local checkout of the fork works too); the default
# is the public fork. Until the fork is published, the default clone fails with a message that says so.
COMMIT="95d26d19e332eea80caf9b84e30366aba4a21659"
LLAMA_REPO="${SHURA_LLAMA_REPO:-https://github.com/hikkian/llama.cpp-shura.git}"
PATCH="$REPO_DIR/patches/turboquant-pr12-fix.patch"
SLOT_PATCH="$REPO_DIR/patches/slot-checkpoints.patch"
ATTENTION_PATCH="$REPO_DIR/patches/attention-decode.patch"
CUDA_ARCH="${CUDA_ARCH:-89}"
BUILD_JOBS="${SHURA_BUILD_JOBS:-6}"
read -r -a BUILD_TARGETS <<< "${SHURA_BUILD_TARGETS:-}"
for target in "${BUILD_TARGETS[@]}"; do
  [[ "$target" =~ ^[A-Za-z0-9_][A-Za-z0-9_-]*$ ]] || { echo "Invalid build target: $target" >&2; exit 1; }
done
if ! [[ "$BUILD_JOBS" =~ ^[1-6]$ ]]; then
  echo "SHURA_BUILD_JOBS must be between 1 and 6" >&2
  exit 1
fi

if [ "$HIP" -eq 1 ]; then
  [ "$ATTENTION_DECODE" -eq 0 ] || { echo "--attention-decode is an NVIDIA (CUDA) patch and cannot be combined with --hip" >&2; exit 1; }
  command -v hipconfig >/dev/null || { echo "hipconfig not found - install the ROCm SDK first (Fedora: sudo dnf install rocm-hip-devel rocblas-devel hipblas-devel cmake gcc-c++ ninja-build; Arch: sudo pacman -S rocm-hip-sdk cmake ninja; Ubuntu/Debian: sudo apt install rocm-hip-sdk cmake ninja-build)"; exit 1; }
  if [ -z "$HIP_ARCH" ]; then
    HIP_ARCH="$(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]\{3,5\}' | head -1 || true)"
    [ -n "$HIP_ARCH" ] || { echo "could not tell your GPU's gfx target: pass it, e.g. --hip gfx1101 (rocminfo | grep gfx)"; exit 1; }
  fi
  echo "Building for HIP, GPU target $HIP_ARCH (this takes tens of minutes)"
else
  cuda_dirs=(/usr/local/cuda/bin /opt/cuda/bin)                   # NVIDIA's packages, and Arch
  if [ -n "${SHURA_CUDA_BIN:-}" ]; then cuda_dirs=("$SHURA_CUDA_BIN"); fi
  for cuda_bin in "${cuda_dirs[@]}"; do
    command -v nvcc >/dev/null || { [ -x "$cuda_bin/nvcc" ] && export PATH="$cuda_bin:$PATH"; } || true
  done
  command -v nvcc >/dev/null || { echo "nvcc not found (looked in PATH and ${cuda_dirs[*]}) - install the CUDA toolkit first (Fedora/Ubuntu: NVIDIA's cuda-toolkit; Arch: sudo pacman -S cuda; see docs/INSTALL.md), or set SHURA_CUDA_BIN to the folder that holds nvcc"; exit 1; }
fi

if [ ! -d "$DEST/.git" ]; then
  git clone --quiet "$LLAMA_REPO" "$DEST" || {
    echo "could not clone $LLAMA_REPO - the public fork is not published yet (FORKPUB1); set SHURA_LLAMA_REPO to a local checkout of the fork" >&2
    exit 1
  }
fi
git -C "$DEST" fetch --quiet origin || true
if ! git -C "$DEST" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
  echo "commit $COMMIT not found in $LLAMA_REPO - see FORKPUB1 (the fork is not published yet); set SHURA_LLAMA_REPO to a local checkout of the fork" >&2
  exit 1
fi
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$COMMIT"

if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "TurboQuant fix already applied"
else
  git -C "$DEST" apply "$PATCH"
  echo "Applied $(basename "$PATCH")"
fi

if git -C "$DEST" apply --reverse --check "$SLOT_PATCH" 2>/dev/null; then
  echo "Slot checkpoint patch already applied"
else
  git -C "$DEST" apply --check "$SLOT_PATCH"
  git -C "$DEST" apply "$SLOT_PATCH"
  echo "Applied $(basename "$SLOT_PATCH")"
fi

ATTENTION_CMAKE=OFF
if [ "$ATTENTION_DECODE" -eq 0 ] && git -C "$DEST" apply --reverse --check "$ATTENTION_PATCH" 2>/dev/null; then
  # An earlier run with --attention-decode left the patch in the tree: this run did not ask for it, so take it out again.
  git -C "$DEST" apply --reverse "$ATTENTION_PATCH"
  echo "Removed $(basename "$ATTENTION_PATCH") (left by an earlier --attention-decode build; rerun with the flag to use it)"
fi
if [ "$ATTENTION_DECODE" -eq 1 ]; then
  if git -C "$DEST" apply --reverse --check "$ATTENTION_PATCH" 2>/dev/null; then
    echo "Attention decode patch already applied"
  else
    git -C "$DEST" apply --check "$ATTENTION_PATCH"
    git -C "$DEST" apply "$ATTENTION_PATCH"
    echo "Applied $(basename "$ATTENTION_PATCH")"
  fi
  ATTENTION_CMAKE=ON
fi

# The measured HIP stack (REPORT-FATTN2 / REPORT-PREFILL1c): applied in this order, each patch checked strictly.
# The Vulkan shaders patch (vulkan-fa-shaders.patch) is for the Vulkan build only and is not applied here.
apply_forward() {
  local name="$1" path="$2"
  git -C "$DEST" apply --check "$path"
  git -C "$DEST" apply "$path"
  echo "Applied $name"
}
HIP_STACK="upstream-picks forktune1 gqavec2 op-profile fuse1 fuse3 path-a-prefillprof4 fattn2-upstream-28102 upstream-fattn-fixes gemma-fa512 cachepf1-prefill-off cachex1c-pinned-source cachex1c-stage3-two-channel cachex1d-gemma-architecture cachex1e-live-resize"
if [ "$HIP" -eq 1 ]; then
  # A reverse check cannot tell whether an early patch is in place once a later patch has changed the same lines,
  # so the stack is recognised by a stamp file in the tree that lists the sha256 of every patch applied.
  wanted="$(for n in $HIP_STACK; do sha256sum "$REPO_DIR/patches/$n.patch" | cut -c1-16; done | tr '\n' ' ')"
  stamp="$DEST/.shura-hip-stack"
  if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$wanted" ]; then
    echo "HIP patch stack already applied (same patches)"
  else
    for n in $HIP_STACK; do apply_forward "$n.patch" "$REPO_DIR/patches/$n.patch"; done
    echo "$wanted" > "$stamp"
  fi
fi
if [ "${SHURA_STOP_AFTER_PATCH:-0}" = "1" ]; then
  echo "Patches applied to $DEST (SHURA_STOP_AFTER_PATCH=1: not building)"
  exit 0
fi

configure() { cmake -S "$DEST" -B "$DEST/build" -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_TURBO3_DECODE_GROUP8="$ATTENTION_CMAKE" "$@"; }
if [ "$HIP" -eq 1 ]; then
  HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S "$DEST" -B "$DEST/build" -G Ninja -DGGML_HIP=ON -DGPU_TARGETS="$HIP_ARCH" -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_EXAMPLES="${SHURA_BUILD_EXAMPLES:-ON}"
elif ! configure; then
  # Rolling distros (e.g. Arch) can ship a GCC newer than nvcc officially supports; it usually works anyway.
  echo "configure failed - retrying with -allow-unsupported-compiler"
  rm -rf "$DEST/build/CMakeCache.txt" "$DEST/build/CMakeFiles"       # forget the failed configure, keep the objects already built
  configure -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler
fi
if [ "$HIP" -eq 1 ]; then
  configured_arch="$(sed -n 's/^GPU_TARGETS:[^=]*=//p' "$DEST/build/CMakeCache.txt")"
  [ "$configured_arch" = "$HIP_ARCH" ] || { echo "HIP target mismatch: requested $HIP_ARCH, configured $configured_arch" >&2; exit 1; }
  echo "Configured HIP GPU target: $configured_arch"
fi
if [ "${#BUILD_TARGETS[@]}" -gt 0 ]; then
  cmake --build "$DEST/build" --config Release -j"$BUILD_JOBS" --target "${BUILD_TARGETS[@]}"
else
  cmake --build "$DEST/build" --config Release -j"$BUILD_JOBS"
fi

echo
echo "Built: $DEST/build/bin/llama-server"
echo "Set \"exePath\" in config/model-launch.json to that path."
