#!/usr/bin/env bash
# The opposite of scripts/install.sh: removes the systemd user service, the `shura` command and the app-menu launcher
# that install.sh set up. It never touches config/*.json, models, the llama.cpp build, ShuraCode or ~/.local/share/shura
# (that folder may hold your 20 GB model; `shura uninstall` only cleans what `shura install` downloaded).
#
#   scripts/uninstall.sh [--dry-run] [--yes] [--searxng]
#
#   --dry-run   only print what would be removed
#   --yes       do not ask
#   --searxng   also remove the 'searxng' container (SHURA_DOCKER sets the command) and ~/.config/searxng
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY=0 YES=0 SEARXNG=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --yes|-y) YES=1 ;;
    --searxng) SEARXNG=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
unit="$HOME/.config/systemd/user/ai-gateway.service"
link="$HOME/.local/bin/shura"
apps="$HOME/.local/share/applications" icons="$HOME/.local/share/icons/hicolor"
entry="$apps/io.github.hikkian.Shura.desktop"
read -ra DOCKER <<< "${SHURA_DOCKER:-docker}"

todo=()
if [ -f "$unit" ]; then
  if grep -qF "$REPO_DIR/gateway/ai_gateway.py" "$unit"; then todo+=("stop and remove the service $unit")
  else echo "  $unit belongs to another checkout: left alone"; fi
fi
if [ -L "$link" ]; then
  if [ "$(readlink "$link")" = "$REPO_DIR/scripts/shura" ]; then todo+=("remove the command $link")
  else echo "  $link points to $(readlink "$link"): left alone"; fi
fi
[ -f "$entry" ] && todo+=("remove the app-menu entry $entry")
for size in 16 24 32 48 64 128 256 512; do
  [ -f "$icons/${size}x${size}/apps/shura.png" ] && todo+=("remove the icon $icons/${size}x${size}/apps/shura.png")
done
if [ "$SEARXNG" = 1 ]; then
  todo+=("remove the container 'searxng' and the folder $HOME/.config/searxng")
fi

if [ "${#todo[@]}" -eq 0 ]; then echo "Nothing to remove."; exit 0; fi
say "This removes:"
for t in "${todo[@]}"; do echo "  - $t"; done
kept="config/*.json, models, llama.cpp build, ShuraCode, ~/.local/share/shura"
[ "$SEARXNG" = 1 ] || kept="$kept, SearXNG (add --searxng)"
echo "  Kept: $kept."
[ "$DRY" = 1 ] && exit 0
if [ "$YES" != 1 ]; then
  [ -t 0 ] || { echo "No terminal to ask on: pass --yes" >&2; exit 1; }
  read -r -p "Remove them? [y/N] " reply
  [[ "$reply" =~ ^[Yy] ]] || { echo "Nothing was removed."; exit 0; }
fi

if [ -f "$unit" ] && grep -qF "$REPO_DIR/gateway/ai_gateway.py" "$unit"; then
  say "systemd user service"
  systemctl --user disable --now ai-gateway.service || echo "  (could not stop it: continue anyway)"
  rm -f "$unit"
  systemctl --user daemon-reload
fi
if [ -L "$link" ] && [ "$(readlink "$link")" = "$REPO_DIR/scripts/shura" ]; then rm -f "$link"; fi
rm -f "$entry"
for size in 16 24 32 48 64 128 256 512; do rm -f "$icons/${size}x${size}/apps/shura.png"; done
if command -v update-desktop-database >/dev/null; then update-desktop-database -q "$apps" 2>/dev/null || true; fi
if command -v gtk-update-icon-cache >/dev/null && [ -d "$icons" ]; then gtk-update-icon-cache -q -t "$icons" 2>/dev/null || true; fi
if [ "$SEARXNG" = 1 ]; then
  say "SearXNG"
  "${DOCKER[@]}" rm -f searxng >/dev/null 2>&1 || echo "  (no container 'searxng' removed)"
  rm -rf "$HOME/.config/searxng"
fi
say "Removed."
