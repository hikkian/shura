#!/usr/bin/env bash
# Runs inside a distro container: detects the family, installs the build packages setup.sh would install,
# checks the tools exist and that NVIDIA's CUDA repository URL for this release is reachable.
set -euo pipefail
if ! command -v sudo >/dev/null; then  # containers run as root without sudo
  printf '#!/bin/sh\nexec "$@"\n' > /usr/local/bin/sudo
  chmod +x /usr/local/bin/sudo
fi
# shellcheck disable=SC1091  # setup.sh is mounted at /repo inside the container
SHURA_SOURCE_ONLY=1 . /repo/setup.sh
detect_family
echo "family=$FAMILY os=$OS_NAME"
read -ra PKGS <<< "$(build_packages)"
case "$FAMILY" in arch) pacman -Sy --noconfirm >/dev/null ;; esac
pkg_install "${PKGS[@]}" >/tmp/pkg.log 2>&1 || { tail -20 /tmp/pkg.log; exit 1; }
for t in git curl cmake node npm python3; do command -v "$t" >/dev/null || { echo "MISSING $t"; exit 1; }; done
command -v ninja >/dev/null || command -v ninja-build >/dev/null || { echo "MISSING ninja"; exit 1; }
echo "tools: cmake $(cmake --version | head -1 | awk '{print $3}'), node $(node --version), python $(python3 --version | awk '{print $2}')"
url="$(cuda_repo_url)"
if [ -n "$url" ]; then
  code="$(curl -s -o /dev/null -w '%{http_code}' -I "$url")"; echo "cuda repo: $code $url"; [ "$code" = 200 ]
else
  echo "cuda: from distro repos (pacman -S cuda)"; pacman -Si cuda >/dev/null && echo "cuda package available"
fi
echo "OK"
