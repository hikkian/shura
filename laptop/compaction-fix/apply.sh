#!/usr/bin/env bash
# Switch ShuraCode to the UI-patched engine (see patch_ui.py). Rollback: ./rollback.sh
set -euo pipefail
ENG="$HOME/.local/share/shuracode/engine/1.18.32"
LINK="$HOME/.local/share/shuracode/libexec/shuracode"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ENG/patched"
python3 "$HERE/patch_ui.py" "$ENG/node_modules/opencode-linux-x64/bin/opencode" "$ENG/patched/shuracode"
"$ENG/patched/shuracode" --version
ln -sfn "$ENG/patched/shuracode" "$LINK"
echo "ok: $LINK -> $(readlink "$LINK") (already running sessions are not affected; restart shuracode)"
