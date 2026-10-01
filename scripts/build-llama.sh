#!/usr/bin/env bash
# Build the thecodacus/llama.cpp `perf` fork at the exact benchmarked commit, with the TurboQuant fix applied.
#
#   scripts/build-llama.sh [--attention-decode] [destination]        (default: ~/ai/llama.cpp-perf)
#   CUDA_ARCH=86 scripts/build-llama.sh          (RTX 30xx; default 89 = RTX 40xx)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HOME/ai/llama.cpp-perf"
ATTENTION_DECODE=0
DEST_SET=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --attention-decode) ATTENTION_DECODE=1 ;;
    --*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)
      if [ "$DEST_SET" -eq 1 ]; then echo "Only one destination is allowed" >&2; exit 1; fi
      DEST="$1"
      DEST_SET=1
      ;;
  esac
  shift
done
COMMIT="27c54b4bbcefadedcec6397477cc2e866c1db716"
PATCH="$REPO_DIR/patches/turboquant-pr12-fix.patch"
SLOT_PATCH="$REPO_DIR/patches/slot-checkpoints.patch"
ATTENTION_PATCH="$REPO_DIR/patches/attention-decode.patch"
CUDA_ARCH="${CUDA_ARCH:-89}"
BUILD_JOBS="${SHURA_BUILD_JOBS:-6}"
if ! [[ "$BUILD_JOBS" =~ ^[1-6]$ ]]; then
  echo "SHURA_BUILD_JOBS must be between 1 and 6" >&2
  exit 1
fi

command -v nvcc >/dev/null || export PATH="/usr/local/cuda/bin:$PATH"
command -v nvcc >/dev/null || { echo "nvcc not found - install the CUDA toolkit first (see docs/INSTALL.md)"; exit 1; }

if [ ! -d "$DEST/.git" ]; then
  git clone --branch perf --single-branch https://github.com/thecodacus/llama.cpp.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin perf || true
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

configure() { cmake -S "$DEST" -B "$DEST/build" -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_TURBO3_DECODE_GROUP8="$ATTENTION_CMAKE" "$@"; }
if ! configure; then
  # Rolling distros (e.g. Arch) can ship a GCC newer than nvcc officially supports; it usually works anyway.
  echo "configure failed - retrying with -allow-unsupported-compiler"
  rm -rf "$DEST/build"
  configure -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler
fi
cmake --build "$DEST/build" --config Release -j"$BUILD_JOBS"

echo
echo "Built: $DEST/build/bin/llama-server"
echo "Set \"exePath\" in config/model-launch.json to that path."
