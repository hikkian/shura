#!/bin/bash
# Install the safe pkill/pgrep wrappers into ShuraCode. Run ONLY after stage15 (held-out comparison) has finished:
# changing the plugin mid-run would put some repetitions under different tool conditions.
set -e
D=$(cd "$(dirname "$0")" && pwd); T=$HOME/.local/share/shura/shuracode
[ -f ~/Desktop/AI/bench/ruspell/STAGE15_DONE ] || { echo "stage15 not finished - refusing"; exit 1; }
mkdir -p $T/safe-bin; cp $D/bin/pkill $D/bin/pgrep $T/safe-bin/
cp -n $T/plugin/server.ts $D/server.ts.installed-backup
cp $D/server.ts.new $T/plugin/server.ts; echo installed
