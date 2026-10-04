#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 /dev/shm/run-root --stub | --server-argv-json /dev/shm/llama-argv.json [--single-arm]" >&2
  exit 2
fi
SINGLE_ARM=0
if [[ "${!#}" == "--single-arm" ]]; then
  SINGLE_ARM=1
  run_args=("$@")
  unset 'run_args[-1]'
  set -- "${run_args[@]}"
fi

ROOT=$(realpath "$1")
MODE=$2
SERVER_ARGV=""
if [[ "$MODE" == "--server-argv-json" ]]; then
  [[ $# -ge 3 ]] || { echo "missing server argv JSON path" >&2; exit 2; }
  SERVER_ARGV=$(realpath "$3")
elif [[ "$MODE" != "--stub" ]]; then
  echo "expected --stub or --server-argv-json" >&2
  exit 2
fi
case "$ROOT/" in /dev/shm/*) ;; *) echo "run root must be under /dev/shm" >&2; exit 2 ;; esac
if [[ -n "$SERVER_ARGV" ]]; then
  case "$SERVER_ARGV" in /dev/shm/*) ;; *) echo "server argv JSON must be under /dev/shm" >&2; exit 2 ;; esac
  [[ -r "$SERVER_ARGV" ]] || { echo "server argv JSON is not readable" >&2; exit 2; }
fi

SHURA_REPO=/home/alisher/Projects/shura
SHURACODE_REPO=/home/alisher/Projects/shuracode
ENGINE=/home/alisher/.local/share/shuracode/libexec/shuracode
SDK=/home/alisher/.config/shuracode/node_modules

run_arm() {
  local arm_name=$1
  local plugins=$2
  local args=(
    --tasks "$ROOT/tasks-polyglot"
    --task-ids-file "$ROOT/validated-ids.txt"
    --out "$ROOT/run"
    --scratch "$ROOT/scratch"
    --config-snapshot "$ROOT/config.snapshot.json"
    --model tiel
    --languages "python,go,cpp"
    --arm-name "$arm_name"
    --context 131072
    --cpus "0,1"
  )
  [[ -z "$plugins" ]] || args+=(--plugins "$plugins")
  if [[ "$MODE" == "--stub" ]]; then
    args+=(--stub)
  else
    args+=(--server-argv-json "$SERVER_ARGV")
  fi

  local devices=()
  if [[ "$MODE" == "--server-argv-json" ]]; then
    for device in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
      [[ -e "$device" ]] && devices+=(--dev-bind "$device" "$device")
    done
    devices+=(--ro-bind /home/alisher/Desktop/AI/llama.cpp-perf-a2/build /home/alisher/Desktop/AI/llama.cpp-perf-a2/build)
    devices+=(--ro-bind /home/alisher/Desktop/AI/models /home/alisher/Desktop/AI/models)
    devices+=(--ro-bind /usr/local /usr/local)
  fi

  taskset -c 0,1 nice -n 19 bwrap --die-with-parent --unshare-net \
    --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib \
    --symlink usr/lib64 /lib64 --ro-bind /etc /etc --ro-bind /sys /sys \
    --proc /proc --dev /dev "${devices[@]}" --tmpfs /tmp \
    --ro-bind "$SHURA_REPO" "$SHURA_REPO" \
    --ro-bind "$SHURACODE_REPO" "$SHURACODE_REPO" \
    --ro-bind /home/alisher/.local/share/shuracode /home/alisher/.local/share/shuracode \
    --ro-bind "$SDK" "$SDK" \
    --ro-bind /home/alisher/.config/shuracode/package.json /home/alisher/.config/shuracode/package.json \
    --ro-bind /home/alisher/.config/shuracode/package-lock.json /home/alisher/.config/shuracode/package-lock.json \
    --ro-bind /home/alisher/.config/shuracode/AGENTS.md /home/alisher/.config/shuracode/AGENTS.md \
    --bind "$ROOT" "$ROOT" --chdir "$SHURA_REPO" \
    --setenv SHURA_POLYGLOT_NETNS 1 --setenv SHURACODE_ENGINE "$ENGINE" \
    --setenv PYTHONDONTWRITEBYTECODE 1 --setenv POLYGLOT_TMPDIR "$ROOT/tmp" \
    /usr/bin/python3 -B "$SHURA_REPO/bench/agentic-eval/external/polyglot_harness.py" "${args[@]}"
}

run_arm tiel-no-plugins ""
if [[ "$SINGLE_ARM" == 0 ]]; then
  run_arm tiel-all-plugins "completion-gate,soft-edit,loop-detector,task-card"
fi
