#!/bin/bash
# Restore everything apply.sh changed.
D=$(cd "$(dirname "$0")" && pwd); B=$D/backup
cp -p $B/model-launch.json $HOME/shura/config/model-launch.json; cp -p $B/shuracode.json $HOME/.config/shuracode/shuracode.json
cp -p $B/prompt.txt $HOME/.config/shuracode/prompt.txt; cp -p $B/server.ts $HOME/.local/share/shura/shuracode/plugin/server.ts
systemctl --user restart ai-gateway; echo "rolled back"
