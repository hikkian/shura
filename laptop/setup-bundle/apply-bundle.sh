#!/usr/bin/env bash
# Copies the files of this bundle to the places where Shura reads them. Run AFTER: ~/shura (branch laptop) is cloned and ~/.local/share/shura/shuracode is
# cloned and installed (SETUP-ZHOMART.md steps 5-7). Replaces the author's home path /home/hikkian by $HOME in text configs. Backs up every file it overwrites to
# $HOME/.shura-setup-backup/<timestamp>/ . Use HOME=/some/test/dir to try it without touching the real home.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); TS=$(date +%Y%m%d-%H%M%S); BK="$HOME/.shura-setup-backup/$TS"; mkdir -p "$BK"
put() { # put <src> <dst> [sed]
  local s=$1 d=$2 sedit=${3:-1}
  mkdir -p "$(dirname "$d")"
  [ -e "$d" ] && { mkdir -p "$BK/$(dirname "${d#$HOME/}")"; cp -a "$d" "$BK/${d#$HOME/}"; }
  if [ "$sedit" = 1 ]; then sed "s#/home/hikkian#$HOME#g" "$s" > "$d"; else cp -a "$s" "$d"; fi
  echo "  -> $d"
}
echo "gateway (python files, overwrite the repo copies):"
for f in ai_gateway.py lang_guard.py typo_fix.py game_mode.py gpu_nvml.py; do put "$B/gateway/$f" "$HOME/shura/gateway/$f" 0; done
echo "gateway config:"
put "$B/config/model-launch.json" "$HOME/shura/config/model-launch.json"
put "$B/config/guardian.json" "$HOME/shura/config/guardian.json"
for f in "$B"/config/templates/*.jinja; do put "$f" "$HOME/shura/config/templates/$(basename "$f")" 0; done
for f in "$B"/config/moe-trace/*.csv; do put "$f" "$HOME/shura/config/moe-trace/$(basename "$f")" 0; done
echo "ShuraCode config:"
for f in shuracode.json package.json; do put "$B/config/$f" "$HOME/.config/shuracode/$f"; done
put "$B/config/AGENTS.md" "$HOME/.config/shuracode/AGENTS.md" 0
put "$B/config/prompt.txt" "$HOME/.config/shuracode/prompt.txt" 0
for f in "$B"/config/mcp/*; do put "$f" "$HOME/.config/shuracode/mcp/$(basename "$f")"; done
for f in "$B"/config/commands/*; do put "$f" "$HOME/.config/shuracode/commands/$(basename "$f")" 0; done
echo "ShuraCode program files (on top of the clone of hikkian/shuracode):"
put "$B/shuracode/shuracode-wrapper" "$HOME/.local/share/shura/shuracode/bin/shuracode" 0; chmod +x "$HOME/.local/share/shura/shuracode/bin/shuracode"
put "$B/shuracode/server.ts" "$HOME/.local/share/shura/shuracode/plugin/server.ts" 0
mkdir -p "$HOME/.local/share/shura/shuracode/safe-bin"; for f in "$B"/shuracode/safe-bin/*; do put "$f" "$HOME/.local/share/shura/shuracode/safe-bin/$(basename "$f")" 0; chmod +x "$HOME/.local/share/shura/shuracode/safe-bin/$(basename "$f")"; done
put "$B/shuracode/shura-launcher" "$HOME/.local/bin/shura" 0; chmod +x "$HOME/.local/bin/shura"
echo "systemd user units:"
put "$B/systemd/ai-gateway.service" "$HOME/.config/systemd/user/ai-gateway.service"
mkdir -p "$HOME/.config/systemd/user/ai-gateway.service.d"
for f in "$B"/systemd/ai-gateway.service.d/*; do put "$f" "$HOME/.config/systemd/user/ai-gateway.service.d/$(basename "$f")"; done
put "$B/systemd/searxng.service" "$HOME/.config/systemd/user/searxng.service" 0
echo "done. Backups of replaced files: $BK"
