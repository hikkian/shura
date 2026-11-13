#!/usr/bin/env bash
set -euo pipefail
ENG="$HOME/.local/share/shuracode/engine/1.18.32"
ln -sfn "$ENG/node_modules/opencode-ai/bin/opencode.exe" "$HOME/.local/share/shuracode/libexec/shuracode"
echo "rolled back: $(readlink "$HOME/.local/share/shuracode/libexec/shuracode")"
