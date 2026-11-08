#!/usr/bin/env bash
# bench.sh <agent> <task> <rep> -> appends one JSON line to results/<agent>.jsonl
# agent: shura (installed ShuraCode config, as the user runs it) | codex (OpenAI Codex CLI, if installed)
set -u
E=/home/hikkian/Desktop/AI/shuracode-prompt/eval
H=/home/hikkian/Desktop/AI/shura-report
A=$1; T=$2; R=$3
# deferral switch: lets a queued script skip an arm that will be re-run later in a better order
[ -f "$H/SKIP_$A" ] && exit 0
if [ -x "$H/HOOK_$A" ]; then mv "$H/HOOK_$A" "$H/HOOK_$A.ran"; "$H/HOOK_$A.ran" > "$H/HOOK_$A.log" 2>&1; fi  # one-shot setup before an arm
# work dirs on disk (not tmpfs = RAM); TMPDIR on disk so ShuraCode/Bun does not leak libopentui.so into RAM-backed /tmp
W=$H/work/$A-$T-$R
rm -rf /tmp/claude-1000/-home-hikkian-Desktop-AI/0b728dcb-3144-4e4f-a160-f3550c3e91a9/scratchpad/reportwork/$A-$T-$R
export TMPDIR=$H/tmp
export PLAYWRIGHT_MCP_HEADLESS=true  # bench: browser tool runs invisibly, no windows on the user screen
# small_model = the model under test: otherwise every run first switches the gateway to tiel-coder and back (~25 s reload each)
case $A in arm_*) SM=tiel-local/$(echo "$A" | cut -d_ -f2);; qwens) SM=tiel-local/qwen36s;; qwen) SM=tiel-local/qwen36;; *) SM="";; esac
NT=""; case $A in *_notask) NT=",\"tools\":{\"task\":false},\"permission\":{\"task\":\"deny\"}";; esac
[ -n "$SM" ] && export OPENCODE_CONFIG_CONTENT="{\"small_model\":\"$SM\"$NT}"
rm -rf "$W"; mkdir -p "$W/proj" "$W/mem" "$W/xdg"
cp -r "$E/tasks/$T/repo/." "$W/proj/"
cd "$W/proj" && git init -q && git add -A && git -c user.email=e@e -c user.name=e commit -qm init
PROMPT="$(cat "$E/tasks/$T/prompt.txt")"
ATT=(); [ -f "$E/tasks/$T/attach.png" ] && ATT=(-f "$E/tasks/$T/attach.png")
# /guardian/status, NOT /health: the gateway forwards /health to the DEFAULT model (tiel-coder) and reloads it
gw_ready() { # wait until the gateway would accept a request (not paused for memory, not disabled); max 30 min
  [ "$A" = codex ] && return 0
  for i in $(seq 360); do curl -s -m 5 127.0.0.1:8080/guardian/status | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if not d['memory_pressure'] and not d['multimedia_lock'] and d['override']!='OFF' and not d['game_mode']['active'] else 1)" && return 0; sleep 5; done; }
for attempt in 1 2 3; do
gw_ready
rm -rf "$W/proj" "$W/xdg" "$W/mem"; mkdir -p "$W/proj" "$W/mem" "$W/xdg"; cp -r "$E/tasks/$T/repo/." "$W/proj/"
cd "$W/proj" && git init -q && git add -A && git -c user.email=e@e -c user.name=e commit -qm init
t0=$(date +%s)
case $A in
  arm_*) AM=$(echo "$A" | cut -d_ -f2); AV=$(echo "$A" | cut -d_ -f3)
         TO=1500; [ "$AV" = max ] && TO=3000  # max = full verification + subagent review, ~20 min typical
         XDG_DATA_HOME="$W/xdg" SHURACODE_MEMORY_DIR="$W/mem" timeout $TO \
           /home/hikkian/.local/bin/shuracode run --auto -m tiel-local/$AM --variant $AV --format json "$PROMPT" "${ATT[@]}" > "$W/events.jsonl" 2> "$W/stderr.txt" ;;
  qwens) XDG_DATA_HOME="$W/xdg" SHURACODE_MEMORY_DIR="$W/mem" timeout 1500 \
           /home/hikkian/.local/bin/shuracode run --auto -m tiel-local/qwen36s --format json "$PROMPT" "${ATT[@]}" > "$W/events.jsonl" 2> "$W/stderr.txt" ;;
  qwen) XDG_DATA_HOME="$W/xdg" SHURACODE_MEMORY_DIR="$W/mem" timeout 1500 \
           /home/hikkian/.local/bin/shuracode run --auto -m tiel-local/qwen36 --format json "$PROMPT" "${ATT[@]}" > "$W/events.jsonl" 2> "$W/stderr.txt" ;;
  final|shura) XDG_DATA_HOME="$W/xdg" SHURACODE_MEMORY_DIR="$W/mem" timeout 1500 \
           /home/hikkian/.local/bin/shuracode run --auto --format json "$PROMPT" "${ATT[@]}" > "$W/events.jsonl" 2> "$W/stderr.txt" ;;
  codex) CI=1 timeout 1500 codex exec --json --skip-git-repo-check --ephemeral -C "$W/proj" -s workspace-write \
           -c sandbox_workspace_write.network_access=true -m gpt-6-luna -c model_reasoning_effort=xhigh -o "$W/last.txt" \
           $([ -f "$E/tasks/$T/attach.png" ] && echo "-i $E/tasks/$T/attach.png") -- "$PROMPT" \
           > "$W/events.jsonl" 2> "$W/stderr.txt" < /dev/null ;;
esac
rc=$?; t1=$(date +%s)
# infrastructure failure (gateway refused / connection reset) and the agent did nothing: retry instead of scoring it
if [ $rc -ne 0 ] && ! grep -q '"type":"tool_use"' "$W/events.jsonl" 2>/dev/null && grep -q -E 'APIError|ECONNRESET|memory pressure|paused' "$W/events.jsonl" "$W/stderr.txt" 2>/dev/null; then
  echo "infra failure on attempt $attempt ($A $T $R), retrying" >&2; sleep 30; continue; fi
break; done
SHOT_DIR="$H/shots/$A-$R" ; mkdir -p "$SHOT_DIR"
SHOT_DIR="$SHOT_DIR" python3 "$H/score2.py" "$E" "$A" "$T" "$R" "$W" $((t1-t0)) $rc >> "$H/results/$A.jsonl"
mkdir -p "$H/runs/$A-$T-$R"; cp "$W/events.jsonl" "$W/answer.txt" "$H/runs/$A-$T-$R/" 2>/dev/null
tail -1 "$H/results/$A.jsonl"
