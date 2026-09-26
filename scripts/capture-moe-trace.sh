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

if [ -z "$PROMPTS" ]; then
  PROMPTS="$TMP/prompts"; mkdir -p "$PROMPTS"
  printf '%s' "Write a Python function that implements a binary search tree with insert, delete, and in-order traversal. Include type hints and tests." > "$PROMPTS/code.txt"
  printf '%s' "Explain in plain language how long context slows down LLM generation and how KV-cache quantization and expert caching help." > "$PROMPTS/explain.txt"
  printf '%s' "Here is a failing pytest run: AssertionError: assert is_palindrome('racecar') is True. The function returns s == s[::2]. Find the root cause, fix it, and show the corrected function." > "$PROMPTS/fix.txt"
fi

: > "$OUT"
for f in "$PROMPTS"/*.txt; do
  echo "Tracing $(basename "$f")..."
  MOE_TRACE_OUT="$TMP/trace.csv" "$TRACE" -m "$MODEL" -ngl 99 -ncmoe 99 -fa 1 -c 32768 -n 384 -f "$f" >/dev/null 2>&1
  awk -F, '$1 >= 0' "$TMP/trace.csv" >> "$OUT"   # generated tokens only
done
echo "Wrote $OUT ($(wc -l < "$OUT") routing rows from generated tokens)"
