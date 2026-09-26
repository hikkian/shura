#!/usr/bin/env bash
# Idempotent installer: local configs, systemd user service, OpenCode (offline) config, private SearXNG,
# the `shura` command and an app-menu launcher.
# Never overwrites an existing config silently - existing files are kept or backed up first.
#
#   scripts/install.sh [--no-service] [--no-opencode] [--no-searxng] [--no-desktop]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DO_SERVICE=1 DO_OPENCODE=1 DO_SEARXNG=1 DO_DESKTOP=1
for arg in "$@"; do
  case "$arg" in
    --no-service) DO_SERVICE=0 ;;
    --no-opencode) DO_OPENCODE=0 ;;
    --no-searxng) DO_SEARXNG=0 ;;
    --no-desktop) DO_DESKTOP=0 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
render() { sed "s|@REPO_DIR@|$REPO_DIR|g" "$1"; }
backup_if_different() {  # $1 = target, $2 = new content file
  if [ -f "$1" ] && ! cmp -s "$1" "$2"; then
    cp "$1" "$1.bak-$(date +%Y%m%d-%H%M%S)"
    say "Backed up existing $1"
  fi
}

say "Checking prerequisites"
for cmd in python3 node npx nvidia-smi; do
  command -v "$cmd" >/dev/null && echo "  ok  $cmd" || echo "  MISSING  $cmd"
done

say "Local gateway configs (gitignored)"
for name in guardian model-launch; do
  if [ -f "$REPO_DIR/config/$name.json" ]; then
    echo "  keeping existing config/$name.json"
  else
    cp "$REPO_DIR/config/$name.example.json" "$REPO_DIR/config/$name.json"
    echo "  created config/$name.json from example - review the paths in it"
  fi
done

if [ "$DO_SERVICE" = 1 ]; then
  say "systemd user service ai-gateway"
  mkdir -p "$HOME/.config/systemd/user"
  render "$REPO_DIR/systemd/ai-gateway.service.in" > "$HOME/.config/systemd/user/ai-gateway.service"
  systemctl --user daemon-reload
  systemctl --user enable ai-gateway.service
  systemctl --user restart ai-gateway.service
  echo "  gateway: http://127.0.0.1:8080  (status: curl -s localhost:8080/guardian/status)"
fi

if [ "$DO_OPENCODE" = 1 ]; then
  say "OpenCode config (~/.config/opencode)"
  mkdir -p "$HOME/.config/opencode"
  tmp="$(mktemp)"
  render "$REPO_DIR/opencode/opencode.json.in" > "$tmp"
  backup_if_different "$HOME/.config/opencode/opencode.json" "$tmp"
  mv "$tmp" "$HOME/.config/opencode/opencode.json"
  backup_if_different "$HOME/.config/opencode/AGENTS.md" "$REPO_DIR/opencode/AGENTS.md"
  cp "$REPO_DIR/opencode/AGENTS.md" "$HOME/.config/opencode/AGENTS.md"

  # Without these OpenCode contacts api.opencode.ai at startup and can hang offline.
  mkdir -p "$HOME/.config/environment.d"
  printf '%s\n' OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_SHARE=1 \
    OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DISABLE_DEFAULT_PLUGINS=1 \
    > "$HOME/.config/environment.d/opencode-offline.conf"
  echo "  offline env written to ~/.config/environment.d/opencode-offline.conf (applies after re-login)"
fi

if [ "$DO_SEARXNG" = 1 ]; then
  say "Private SearXNG (Docker, 127.0.0.1:8888)"
  if ! command -v docker >/dev/null; then
    echo "  docker not found - skipping (web search MCP will not work)"
  elif docker ps -a --format '{{.Names}}' | grep -qx searxng; then
    echo "  container 'searxng' already exists - leaving it alone"
  else
    mkdir -p "$HOME/.config/searxng"
    if [ ! -f "$HOME/.config/searxng/settings.yml" ]; then
      sed "s|@SEARXNG_SECRET@|$(openssl rand -hex 32)|" "$REPO_DIR/searxng/settings.yml" > "$HOME/.config/searxng/settings.yml"
    fi
    docker run -d --name searxng --restart unless-stopped -p 127.0.0.1:8888:8080 \
      -v "$HOME/.config/searxng:/etc/searxng:Z" searxng/searxng:latest >/dev/null
    echo "  started searxng container"
  fi
fi

say "shura command (~/.local/bin/shura)"
mkdir -p "$HOME/.local/bin"
ln -sf "$REPO_DIR/scripts/shura" "$HOME/.local/bin/shura"
echo "  try: shura status"

if [ "$DO_DESKTOP" = 1 ]; then
  say "App-menu launcher 'Shura'"
  mkdir -p "$HOME/.local/share/applications" "$HOME/.local/share/icons/hicolor/scalable/apps"
  cp "$REPO_DIR/assets/shura.svg" "$HOME/.local/share/icons/hicolor/scalable/apps/shura.svg"
  rm -f "$HOME/.local/share/applications/shura.desktop"  # pre-app-ID launcher name
  sed "s|@BIN@|$HOME/.local/bin/shura|g" "$REPO_DIR/desktop/io.github.hikkian.Shura.desktop.in" \
    > "$HOME/.local/share/applications/io.github.hikkian.Shura.desktop"
  if command -v update-desktop-database >/dev/null; then update-desktop-database -q "$HOME/.local/share/applications" || true; fi
  if command -v gtk-update-icon-cache >/dev/null; then gtk-update-icon-cache -q -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true; fi
  echo "  search for 'Shura' in the app menu (right-click: load / unload model)"
fi

say "Done. Next: see docs/INSTALL.md for building llama.cpp and the Playwright browser."
