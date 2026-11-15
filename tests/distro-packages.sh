#!/usr/bin/env bash
# Runs inside a distro container (or anywhere with bash): detects the family and checks that pkg_install only PRINTS the
# install command for each family (dnf / apt-get / pacman) and runs nothing. No network, no sudo, nothing installed.
set -euo pipefail
# shellcheck disable=SC1091  # setup.sh is mounted at /repo inside the container
SHURA_SOURCE_ONLY=1 . "${SETUP_SH:-/repo/setup.sh}"
detect_family
echo "family=$FAMILY os=$OS_NAME"
orig="$FAMILY"
read -ra PKGS <<< "$(build_packages)"
[ "${#PKGS[@]}" -gt 0 ] || { echo "build_packages is empty for $FAMILY"; exit 1; }
bin="$(mktemp -d)"; trap 'rm -rf "$bin"' EXIT
for tool in sudo dnf apt-get pacman; do  # tripwires: if pkg_install ran any of them, the marker appears
  printf '#!/bin/sh\necho "$0 $*" >> "%s/ran"\nexit 0\n' "$bin" > "$bin/$tool"; chmod +x "$bin/$tool"
done
for fam in fedora debian arch; do
  FAMILY=$fam
  case "$fam" in fedora) want="dnf install" ;; debian) want="apt-get install" ;; arch) want="pacman -S" ;; esac
  if PATH="$bin:$PATH" pkg_install git curl >"$bin/out" 2>&1; then echo "pkg_install returned 0 for $fam (it must not install)"; exit 1; fi
  line="$(grep -F "sudo " "$bin/out" || true)"
  case "$line" in *"$want"*"git curl"*) ;; *) echo "pkg_install printed no $fam command: $line"; exit 1 ;; esac
  echo "$fam: $line"
done
[ ! -e "$bin/ran" ] || { echo "pkg_install ran a command: $(cat "$bin/ran")"; exit 1; }
FAMILY="$orig"
url="$(cuda_repo_url)"
echo "cuda repo url for $FAMILY: ${url:-"(distro repos)"}"
echo "OK"
