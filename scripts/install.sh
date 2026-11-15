#!/usr/bin/env bash
# Idempotent installer: local configs, systemd user service, ShuraCode (the coding agent), private SearXNG,
# the `shura` command and an app-menu launcher.
# Never overwrites an existing config silently: config/*.json are kept as they are; the systemd unit and a regular
# file in the place of the `shura` command are backed up (*.bak-<time>) before they are replaced.
# A rerun does not restart a running gateway (that would drop the loaded model and the active session): pass --restart,
# or run `shura restart`, to apply a changed unit.
#
#   scripts/install.sh [--no-service] [--no-shuracode] [--no-searxng] [--no-desktop] [--restart]
#
# What it downloads and runs: ShuraCode is cloned from https://github.com/hikkian/shuracode (its install.sh then runs),
# SearXNG is the container image searxng/searxng:latest. Pin them with SHURACODE_REF=<tag or commit> and
# SHURA_SEARXNG_IMAGE=<image>@sha256:<digest>. SHURA_DOCKER sets the container command (docker, podman, "sudo docker").
# ShuraCode and SearXNG are optional: if one of them fails, the rest is still installed and the problem is listed at the end.
# The opposite of this script is scripts/uninstall.sh.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DO_SERVICE=1 DO_SHURACODE=1 DO_SEARXNG=1 DO_DESKTOP=1 FORCE_RESTART=0
for arg in "$@"; do
  case "$arg" in
    --no-service) DO_SERVICE=0 ;;
    --no-shuracode) DO_SHURACODE=0 ;;
    --no-searxng) DO_SEARXNG=0 ;;
    --no-desktop) DO_DESKTOP=0 ;;
    --restart) FORCE_RESTART=1 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

TMPFILES=()
trap 'rm -f ${TMPFILES[@]+"${TMPFILES[@]}"}' EXIT
WARNINGS=()
say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '  \033[1;33mWARNING\033[0m %s\n' "$*" >&2; WARNINGS+=("$*"); }
subst() {  # subst FILE PLACEHOLDER VALUE: literal replacement (sed would treat | & \ in the value as syntax)
  SUBST_VALUE="$3" awk -v ph="$2" '{ out = ""; while ((i = index($0, ph)) > 0) { out = out substr($0, 1, i - 1) ENVIRON["SUBST_VALUE"]; $0 = substr($0, i + length(ph)) } print out $0 }' "$1"
}
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
  case "$REPO_DIR" in
    *\"*|*\\*|*\$*|*%*|*$'\n'*)
      echo "  the repository path contains one of \" \\ \$ % or a newline, which systemd unit files cannot hold: $REPO_DIR" >&2
      echo "  move the repository to a plainer path, or use --no-service" >&2
      exit 1 ;;
  esac
  unit="$HOME/.config/systemd/user/ai-gateway.service"
  new_unit="$(mktemp)"; TMPFILES+=("$new_unit")
  subst "$REPO_DIR/systemd/ai-gateway.service.in" "@REPO_DIR@" "$REPO_DIR" > "$new_unit"
  mkdir -p "$HOME/.config/systemd/user"
  if [ -f "$unit" ] && cmp -s "$unit" "$new_unit"; then
    echo "  unit unchanged"
  else
    backup_if_different "$unit" "$new_unit"
    cp "$new_unit" "$unit.tmp.$$" && mv -f "$unit.tmp.$$" "$unit"
    echo "  wrote $unit"
  fi
  systemctl --user daemon-reload
  systemctl --user enable ai-gateway.service
  if systemctl --user is-active --quiet ai-gateway.service; then
    if [ "$FORCE_RESTART" = 1 ]; then
      systemctl --user restart ai-gateway.service
      echo "  gateway restarted (--restart)"
    else
      echo "  gateway is running: left alone, so the loaded model stays. To apply a changed unit or config: shura restart"
    fi
  else
    systemctl --user start ai-gateway.service
    echo "  gateway started"
  fi
  echo "  gateway: http://127.0.0.1:8080  (status: curl -s localhost:8080/guardian/status)"
fi

install_shuracode() {  # 0 = installed; a failure is reported by the caller as a warning and does not abort the whole install
  # A clone next to this repository wins (development); otherwise keep a copy under ~/.local/share/shura.
  local sc="${SHURACODE_DIR:-}" ref="${SHURACODE_REF:-}" url="https://github.com/hikkian/shuracode.git"
  if [ -z "$sc" ] && [ -x "$REPO_DIR/../shuracode/install.sh" ]; then sc="$(cd "$REPO_DIR/../shuracode" && pwd)"; fi
  if [ -z "$sc" ]; then
    sc="$HOME/.local/share/shura/shuracode"
    if [ -d "$sc/.git" ]; then
      if [ -n "$ref" ]; then
        git -C "$sc" fetch -q --depth 1 origin "$ref" && git -C "$sc" checkout -q FETCH_HEAD || return 1
      else
        git -C "$sc" pull --ff-only -q || echo "  could not update $sc - using the copy that is there"
      fi
    else
      git clone -q --depth 1 "$url" "$sc" || return 1
      if [ -n "$ref" ]; then
        git -C "$sc" fetch -q --depth 1 origin "$ref" && git -C "$sc" checkout -q FETCH_HEAD || return 1
      fi
    fi
    echo "  downloaded from $url at commit $(git -C "$sc" rev-parse --short HEAD 2>/dev/null || echo '?')${ref:+ (pinned: $ref)}; its install.sh runs now"
  fi
  echo "  from $sc"
  "$sc/install.sh" --base-url http://127.0.0.1:8080/v1 --status-url http://127.0.0.1:8080/guardian/status 2>&1 | sed 's/^/  /'
  return "${PIPESTATUS[0]}"
}

if [ "$DO_SHURACODE" = 1 ]; then
  say "ShuraCode (coding agent)"
  if ! install_shuracode; then warn "ShuraCode was not installed (see above). Rerun scripts/install.sh when the cause is fixed."; fi
fi

make_secret() {  # 64 hex characters, or a non-zero status: never an empty secret
  local s=""
  if command -v openssl >/dev/null; then s="$(openssl rand -hex 32 2>/dev/null)" || s=""; fi
  if [ "${#s}" -ne 64 ]; then s="$(python3 -c 'import secrets; print(secrets.token_hex(32))' 2>/dev/null)" || s=""; fi
  [ "${#s}" -eq 64 ] || return 1
  printf '%s' "$s"
}

install_searxng() {
  local conf="$HOME/.config/searxng/settings.yml" bin names secret tmp image="${SHURA_SEARXNG_IMAGE:-searxng/searxng:latest}"
  local -a DOCKER
  read -ra DOCKER <<< "${SHURA_DOCKER:-docker}"  # setup.sh passes "sudo docker" right after installing Docker
  bin="${DOCKER[0]}"; if [ "$bin" = sudo ]; then bin="${DOCKER[1]:-docker}"; fi
  if ! command -v "$bin" >/dev/null; then
    echo "  $bin not found - skipping (web search MCP will not work)"
    return 0
  fi
  # Capture first: `docker ps | grep -q` under pipefail can fail on SIGPIPE and miss an existing container.
  if ! names="$("${DOCKER[@]}" ps -a --format '{{.Names}}' 2>&1)"; then
    warn "$bin is installed but cannot be used (daemon stopped, or your user is not allowed to use it?): ${names%%$'\n'*}. SearXNG skipped; fix it and rerun scripts/install.sh."
    return 1
  fi
  if grep -qx searxng <<< "$names"; then
    echo "  container 'searxng' already exists - leaving it alone"
    return 0
  fi
  mkdir -p "$HOME/.config/searxng"
  # A settings.yml with an empty or unreplaced secret (an earlier run without openssl) is not kept: it would never be repaired.
  if [ -f "$conf" ] && grep -Eq 'secret_key:[[:space:]]*("")?[[:space:]]*$|@SEARXNG_SECRET@' "$conf"; then
    cp "$conf" "$conf.bak-$(date +%Y%m%d-%H%M%S)"
    rm -f "$conf"
    echo "  the existing settings.yml had no secret key: regenerating it (backup kept)"
  fi
  if [ ! -f "$conf" ]; then
    if ! secret="$(make_secret)"; then
      warn "could not generate a SearXNG secret (neither openssl nor python3 secrets worked). SearXNG skipped."
      return 1
    fi
    tmp="$conf.tmp.$$"; TMPFILES+=("$tmp")
    subst "$REPO_DIR/searxng/settings.yml" "@SEARXNG_SECRET@" "$secret" > "$tmp"
    mv -f "$tmp" "$conf"       # mode left readable: the container user (not you) must read it
  fi
  if "${DOCKER[@]}" run -d --name searxng --restart unless-stopped -p 127.0.0.1:8888:8080 \
      -v "$HOME/.config/searxng:/etc/searxng:Z" "$image" >/dev/null; then
    echo "  started searxng container ($image)"
  else
    warn "starting the searxng container failed (see the message above). Web search will not work until it is fixed."
    return 1
  fi
}

if [ "$DO_SEARXNG" = 1 ]; then
  say "Private SearXNG (container, 127.0.0.1:8888)"
  install_searxng || true
fi

say "shura command (~/.local/bin/shura)"
mkdir -p "$HOME/.local/bin"
link="$HOME/.local/bin/shura" target="$REPO_DIR/scripts/shura"
if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
  echo "  already links to $target"
else
  if [ -e "$link" ] && [ ! -L "$link" ]; then     # a real file of the user's: keep it
    mv "$link" "$link.bak-$(date +%Y%m%d-%H%M%S)"
    say "Backed up existing $link"
  elif [ -L "$link" ]; then
    echo "  replacing the link to $(readlink "$link")"
  fi
  ln -sfn "$target" "$link"
fi
echo "  try: shura status"

if [ "$DO_DESKTOP" = 1 ]; then
  say "App-menu launcher 'Shura'"
  apps="$HOME/.local/share/applications" icons="$HOME/.local/share/icons/hicolor" changed=0
  mkdir -p "$apps"
  # The desktop watches these folders live. Files are replaced atomically and only when their content
  # changed: GNOME's menu code (gnome-menus 3.38) crashes gnome-shell if it reads a half-written entry.
  put_file() {  # $1 = source, $2 = target; returns 0 only if the target was (re)written
    if [ -f "$2" ] && cmp -s "$1" "$2"; then return 1; fi
    local tmp; tmp="$(dirname "$2")/.$(basename "$2").tmp.$$"  # not *.desktop, so it is never indexed
    cp "$1" "$tmp" && chmod 644 "$tmp" && mv -f "$tmp" "$2"
  }
  for old in "$icons/scalable/apps/shura.svg" "$apps/shura.desktop"; do  # names from earlier versions
    if [ -e "$old" ]; then rm -f "$old"; changed=1; fi
  done
  for size in 16 24 32 48 64 128 256 512; do
    mkdir -p "$icons/${size}x${size}/apps"
    if put_file "$REPO_DIR/assets/icons/shura-$size.png" "$icons/${size}x${size}/apps/shura.png"; then changed=1; fi
  done
  entry="$(mktemp)"; TMPFILES+=("$entry")
  bin_exec="$HOME/.local/bin/shura"
  case "$bin_exec" in *" "*) bin_exec="\"$bin_exec\"" ;; esac     # a desktop Exec= line needs quotes around a path with spaces
  subst "$REPO_DIR/desktop/io.github.hikkian.Shura.desktop.in" "@BIN@" "$bin_exec" > "$entry"
  if put_file "$entry" "$apps/io.github.hikkian.Shura.desktop"; then changed=1; fi
  if [ "$changed" = 1 ]; then
    if command -v update-desktop-database >/dev/null; then update-desktop-database -q "$apps" || true; fi
    if command -v gtk-update-icon-cache >/dev/null; then gtk-update-icon-cache -q -t "$icons" 2>/dev/null || true; fi
    echo "  installed/updated - search for 'Shura' in the app menu (right-click: load / unload model)"
  else
    echo "  already up to date (nothing rewritten)"
  fi
fi

if [ "${#WARNINGS[@]}" -gt 0 ]; then
  say "Done, with problems in optional parts:"
  for w in "${WARNINGS[@]}"; do echo "  - $w"; done
else
  say "Done. Next: see docs/INSTALL.md for building llama.cpp."
fi
