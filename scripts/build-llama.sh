#!/usr/bin/env bash
# Build the thecodacus/llama.cpp `perf` fork at the exact benchmarked commit, with the TurboQuant fix applied.
#
#   scripts/build-llama.sh [destination]        (default: ~/ai/llama.cpp-perf)
#   CUDA_ARCH=86 scripts/build-llama.sh          (RTX 30xx; default 89 = RTX 40xx)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$HOME/ai/llama.cpp-perf}"
COMMIT="27c54b4bbcefadedcec6397477cc2e866c1db716"
PATCH="$REPO_DIR/patches/turboquant-pr12-fix.patch"
CUDA_ARCH="${CUDA_ARCH:-89}"

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

cmake -S "$DEST" -B "$DEST/build" -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release
cmake --build "$DEST/build" --config Release -j"$(nproc)"

echo
echo "Built: $DEST/build/bin/llama-server"
echo "Set \"exePath\" in config/model-launch.json to that path."
