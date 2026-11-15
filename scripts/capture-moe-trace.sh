#!/usr/bin/env bash
# Build a MoE routing profile for --moe-cache-profile from prompts that look like YOUR workload.
# config/moe-trace/tiel-coder-agentic.csv is already provided for Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.
#
#   scripts/capture-moe-trace.sh <model.gguf> [prompt-dir] [output.csv] [llama.cpp-perf dir]
#
# prompt-dir: *.txt files, each a full chat-formatted prompt (e.g. a real agent request + a task).
# Only GENERATED tokens are kept: experts used while reading a long prompt differ from the ones used
# while writing, and decode speed depends on the latter (see docs/BENCHMARKS.md).
set -euo pipefail

MODEL="${1:?usage: capture-moe-trace.sh <model.gguf> [prompt-dir] [output.csv] [llama.cpp-perf dir]}"
PROMPTS="${2:-}"
OUT="${3:-moe-trace.csv}"
LLAMA_DIR="${4:-$HOME/ai/llama.cpp-perf}"
TRACE="$LLAMA_DIR/build/bin/llama-moe-trace"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
[ -x "$TRACE" ] || { echo "tracer not found or not executable: $TRACE (build llama.cpp-perf first, or pass its folder as the 4th argument)" >&2; exit 1; }
[ -d "$(dirname "$OUT")" ] || { echo "the folder of the output file does not exist: $(dirname "$OUT")" >&2; exit 1; }

if [ -z "$PROMPTS" ]; then
  PROMPTS="$TMP/prompts"; mkdir -p "$PROMPTS"
  printf '%s' "Write a Python function that implements a binary search tree with insert, delete, and in-order traversal. Include type hints and tests." > "$PROMPTS/code.txt"
  printf '%s' "Explain in plain language how long context slows down LLM generation and how KV-cache quantization and expert caching help." > "$PROMPTS/explain.txt"
  printf '%s' "Here is a failing pytest run: AssertionError: assert is_palindrome('racecar') is True. The function returns s == s[::2]. Find the root cause, fix it, and show the corrected function." > "$PROMPTS/fix.txt"
fi

shopt -s nullglob
files=("$PROMPTS"/*.txt)
[ "${#files[@]}" -gt 0 ] || { echo "no *.txt prompt files in $PROMPTS" >&2; exit 1; }

# Everything goes to a temporary file; $OUT (it may be the shipped, hand-tuned config/moe-trace/*.csv) is replaced only
# after every prompt was traced successfully, so a failed run leaves it as it was.
RESULT="$TMP/result.csv"
: > "$RESULT"
for f in "${files[@]}"; do
  echo "Tracing $(basename "$f")..."
  rm -f "$TMP/trace.csv"                      # a failed prompt must not re-add the previous prompt's rows
  if ! MOE_TRACE_OUT="$TMP/trace.csv" "$TRACE" -m "$MODEL" -ngl 99 -ncmoe 99 -fa 1 -c 32768 -n 384 -f "$f" >"$TMP/tracer.log" 2>&1; then
    echo "the tracer failed on $(basename "$f"); last lines of its output:" >&2
    tail -n 15 "$TMP/tracer.log" >&2
    echo "$OUT was not changed." >&2
    exit 1
  fi
  [ -s "$TMP/trace.csv" ] || { echo "the tracer wrote no trace for $(basename "$f"); $OUT was not changed." >&2; exit 1; }
  awk -F, '$1 >= 0' "$TMP/trace.csv" >> "$RESULT"   # generated tokens only
done
[ -s "$RESULT" ] || { echo "no generated-token rows were captured; $OUT was not changed." >&2; exit 1; }
cp "$RESULT" "$OUT.tmp.$$" && mv -f "$OUT.tmp.$$" "$OUT"
echo "Wrote $OUT ($(wc -l < "$OUT") routing rows from generated tokens)"
