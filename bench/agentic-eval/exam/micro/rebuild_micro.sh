#!/usr/bin/env bash
# EXAM200e: rebuild the micro set of source A from the open benchmarks and check it.
# Needs python3 and git; downloads only the two named files (see SOURCES.md). Everything runs under a 4 GiB memory scope.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RAW=/dev/shm/EXAM200e/raw
CAND=/dev/shm/EXAM200e/cand
OUT=/dev/shm/EXAM200e/out
CAP="systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nice -n 19 ionice -c3"
mkdir -p "$RAW"
# 1. download (HumanEval) and clone (QuixBugs) and check the hash of HumanEval
curl -sL -A "exam-builder" -o "$RAW/HumanEval.jsonl.gz" \
  "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
echo "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef  $RAW/HumanEval.jsonl.gz" | sha256sum -c -
[ -d "$RAW/quixbugs" ] || git clone -q --depth 1 https://github.com/jkoppel/QuixBugs.git "$RAW/quixbugs"
# 2. candidates and validation (every run under the memory cap, timeout 10 s per hidden script)
$CAP python3 "$HERE/build_std_tasks.py" build --raw "$RAW" --cand "$CAND"
rm -rf "$OUT"
$CAP python3 "$HERE/build_std_tasks.py" validate --cand "$CAND" --out "$OUT"
# 3. manifest v2 (statements and reader notes are made by hand and by the blind reader; they are not rebuilt here)
$CAP python3 "$HERE/make_micro_manifest_v2.py"
