#!/usr/bin/env bash
# Shura setup: from a Linux machine with an NVIDIA GPU to a tuned, fully local coding agent.
#
#   curl -fsSL https://raw.githubusercontent.com/hikkian/shura/main/setup.sh | bash
#   git clone https://github.com/hikkian/shura && cd shura && ./setup.sh
#
# Options:
#   --quant Q            force a Tiel-Coder quant (default: chosen for your RAM/VRAM)
#   --model FILE         reuse an already downloaded .gguf (verified by SHA-256)
#   --quick              shorter auto-tune (skips the full-context check)
#   --desktop-ram-gb N   RAM to keep free for your browser/IDE/messengers (default 6)
#   --no-search          skip the private SearXNG web search (no Docker needed)
#   --yes                do not ask for confirmation
#
# Everything big lives in $SHURA_HOME (default ~/.local/share/shura): models, llama.cpp build.
set -euo pipefail

REPO_URL="https://github.com/hikkian/shura.git"
SHURA_HOME="${SHURA_HOME:-$HOME/.local/share/shura}"
QUANT="" MODEL="" QUICK="" DESKTOP_RAM=6 SEARCH=1 YES=""

say()  { printf '\n\033[1;36m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }
GATEWAY_STOPPED_BY_SETUP=""
restart_gateway_we_stopped() {  # setup stops the gateway to measure VRAM: any exit before the install step must give it back
  if [ -n "$GATEWAY_STOPPED_BY_SETUP" ]; then
    GATEWAY_STOPPED_BY_SETUP=""
    if systemctl --user start ai-gateway 2>/dev/null; then info "restarted the gateway that setup had stopped"; fi
  fi
}
trap restart_gateway_we_stopped EXIT
ask()  {  # ask "question" -> 0 for yes
  [ -n "$YES" ] && return 0
  local reply; read -r -p "    $1 [Y/n] " reply </dev/tty || reply=y
  [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}

# ---------------------------------------------------------------------------------------- distro
detect_family() {
  local release="${SHURA_OS_RELEASE:-/etc/os-release}"
  [ -r "$release" ] || die "cannot read $release"
  # shellcheck disable=SC1091
  . "$release"
  OS_ID="$ID" OS_VER="${VERSION_ID:-}" OS_NAME="${PRETTY_NAME:-$ID}"
  case " $ID ${ID_LIKE:-} " in
    *" fedora "*|*" rhel "*) FAMILY=fedora ;;
    *" ubuntu "*|*" debian "*) FAMILY=debian ;;
    *" arch "*) FAMILY=arch ;;
    *) FAMILY=unsupported ;;
  esac
}

preflight_checks() {  # read-only checks; called before clone, mkdir, package changes, or any sudo
  say "Read-only preflight checks"
  [ "$FAMILY" != unsupported ] || die "unsupported distribution ($OS_NAME). Supported: Fedora, Ubuntu/Debian, Arch."
  [ "$(uname -s 2>/dev/null)" = Linux ] || die "setup.sh requires Linux"
  [ "$(uname -m 2>/dev/null)" = x86_64 ] || die "setup.sh requires x86_64"
  info "OS: $OS_NAME ($FAMILY), architecture x86_64"

  local ram_kb disk_probe meminfo="${SHURA_MEMINFO:-/proc/meminfo}"
  ram_kb="$(awk '/MemTotal/ {print $2; exit}' "$meminfo" 2>/dev/null || true)"
  [[ "$ram_kb" =~ ^[0-9]+$ ]] && [ "$ram_kb" -gt 0 ] || die "could not read system RAM from /proc/meminfo"
  [ "$ram_kb" -ge $((16 * 1024 * 1024)) ] || die "setup.sh needs at least 16 GiB RAM (detected $((ram_kb / 1048576)) GiB)"
  info "RAM: $((ram_kb / 1048576)) GiB"

  disk_probe="$SHURA_HOME"
  while [ ! -e "$disk_probe" ] && [ "$disk_probe" != / ]; do disk_probe="$(dirname "$disk_probe")"; done
  local free_gb
  free_gb="$(df -BG --output=avail "$disk_probe" 2>/dev/null | tail -1 | tr -dc 0-9)"
  [[ "$free_gb" =~ ^[0-9]+$ ]] && [ "$free_gb" -ge 30 ] || die "need at least 30 GB free near $SHURA_HOME (detected ${free_gb:-unknown} GB)"
  info "Disk: ${free_gb} GB free near $SHURA_HOME"

  command -v nvidia-smi >/dev/null || die "nvidia-smi is missing; install the NVIDIA driver, reboot and rerun setup.sh"
  GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)"
  [ -n "$GPU" ] || die "no working NVIDIA GPU/driver found; install the driver, reboot and rerun setup.sh"
  DRIVER_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA (UMD )?Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)"
  info "GPU: $GPU (driver CUDA ${DRIVER_CUDA:-version unknown})"

  if command -v curl >/dev/null; then
    curl -fsSI --max-time 8 https://github.com/ -o /dev/null || die "cannot reach GitHub; check the network before rerunning setup.sh"
    curl -fsSI --max-time 8 https://huggingface.co/ -o /dev/null || die "cannot reach Hugging Face; check the network before rerunning setup.sh"
  elif command -v python3 >/dev/null; then
    python3 -c 'import urllib.request; urllib.request.urlopen("https://github.com/", timeout=8).close(); urllib.request.urlopen("https://huggingface.co/", timeout=8).close()' \
      || die "cannot reach GitHub and Hugging Face; check the network before rerunning setup.sh"
  else
    die "need curl or Python 3 to check network access before making system changes"
  fi
  info "Network: GitHub and Hugging Face reachable"
}

build_packages() {  # build tools per distro family (Docker and CUDA are handled separately)
  case "$FAMILY" in
    fedora) echo git curl cmake ninja-build gcc-c++ python3 nodejs npm zenity libnotify ;;
    debian) echo git curl cmake ninja-build build-essential python3 nodejs npm zenity libnotify-bin ;;
    arch)   echo git curl cmake ninja base-devel python nodejs npm zenity libnotify ;;
  esac
}

cuda_repo_url() {  # NVIDIA CUDA repository (Fedora .repo / Debian+Ubuntu keyring); empty for Arch
  case "$FAMILY" in
    fedora) echo "https://developer.download.nvidia.com/compute/cuda/repos/fedora${OS_VER}/x86_64/cuda-fedora${OS_VER}.repo" ;;
    debian)
      if [ "$OS_ID" = ubuntu ]; then dist="ubuntu${OS_VER//./}"; else dist="debian${OS_VER%%.*}"; fi
      echo "https://developer.download.nvidia.com/compute/cuda/repos/$dist/x86_64/cuda-keyring_1.1-1_all.deb" ;;
  esac
}

pkg_cmd() {  # pkg_cmd pkg... -> the exact install command for this distribution (text only, never executed)
  case "$FAMILY" in
    fedora) echo "sudo dnf install -y $*" ;;
    debian) echo "sudo apt-get update && sudo apt-get install -y $*" ;;
    arch)   echo "sudo pacman -S --needed $*" ;;
  esac
}

pkg_install() {  # pkg_install pkg...  Never installs: prints the command for the user and returns 1.
  info "Run this yourself with administrator rights:"
  info "  $(pkg_cmd "$@")"
  return 1
}

# --------------------------------------------------------------------------------------- bootstrap
[ "${SHURA_SOURCE_ONLY:-}" = 1 ] && return 0
detect_family
preflight_checks
SELF="${BASH_SOURCE[0]:-}"
if [ -z "$SELF" ] || [ ! -f "$(dirname "$SELF")/installer/shura_setup.py" ]; then
  # Piped from curl: fetch the repository, then run the real script from it.
  [ "$FAMILY" = unsupported ] && die "unsupported distribution ($OS_NAME). Supported: Fedora, Ubuntu/Debian, Arch."
  command -v git >/dev/null || { pkg_install git; die "git is required to fetch Shura. Install it with the command above, then rerun this script."; }
  if [ -d "$SHURA_HOME/app/.git" ]; then git -C "$SHURA_HOME/app" pull --ff-only -q
  else git clone -q --depth 1 "$REPO_URL" "$SHURA_HOME/app"; fi
  exec bash "$SHURA_HOME/app/setup.sh" "$@" </dev/tty
fi
REPO_DIR="$(cd "$(dirname "$SELF")" && pwd)"
PY="$REPO_DIR/installer/shura_setup.py"

while [ $# -gt 0 ]; do
  case "$1" in
    --quant) QUANT="$2"; shift ;;
    --model) MODEL="$(realpath "$2")"; shift ;;
    --quick) QUICK=1 ;;
    --desktop-ram-gb) DESKTOP_RAM="$2"; shift ;;
    --no-search) SEARCH="" ;;
    --yes|-y) YES=1 ;;
    -h|--help) sed -n '2,19p' "$REPO_DIR/setup.sh" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
  shift
done
mkdir -p "$SHURA_HOME" "$HOME/.cache/shura"
exec > >(tee -a "$HOME/.cache/shura/setup.log") 2>&1

printf '\n\033[1;36m  Shura · شورى\033[0m  local AI coding workstation setup\n'

# ---------------------------------------------------------------------------------------- checks
say "Preflight passed; proceeding with setup"

# ------------------------------------------------------------------------------------- packages
say "System packages"
read -ra PKGS <<< "$(build_packages)"
missing=()
for p in cmake git curl node npm python3; do command -v "$p" >/dev/null || missing+=("$p"); done
command -v ninja >/dev/null || command -v ninja-build >/dev/null || missing+=(ninja)
if [ ${#missing[@]} -gt 0 ]; then
  info "missing: ${missing[*]}"
  info "setup never runs sudo itself; install the packages, then rerun setup.sh"
  pkg_install "${PKGS[@]}"
  die "build tools are missing (${missing[*]}). Run the command printed above, then rerun setup.sh"
else
  info "build tools already installed"
fi

DOCKER=""
if [ -n "$SEARCH" ]; then
  if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
    DOCKER="docker"
  else
    SEARCH=""
    info "skipping web search: Docker is not usable without administrator rights (setup never calls sudo)"
    if ! command -v docker >/dev/null; then
      case "$FAMILY" in fedora) pkg_name=moby-engine ;; debian) pkg_name=docker.io ;; arch) pkg_name=docker ;; esac
      pkg_install "$pkg_name" || true
      info "then enable it:  sudo systemctl enable --now docker"
    fi
    info "and allow your user to use it:  sudo usermod -aG docker \$USER   (log out and in), then rerun setup.sh"
  fi
fi

# ----------------------------------------------------------------------------------------- CUDA
say "CUDA toolkit"
for d in /usr/local/cuda/bin /opt/cuda/bin; do [ -x "$d/nvcc" ] && export PATH="$d:$PATH"; done
if ! command -v nvcc >/dev/null; then
  info "nvcc not found: the CUDA toolkit is not installed (the driver is left untouched)"
  info "setup never runs sudo itself. To install the toolkit, run these yourself with administrator rights:"
  case "$FAMILY" in
    fedora)
      info "  sudo dnf config-manager addrepo --from-repofile=$(cuda_repo_url)"
      info "  sudo dnf install -y cuda-toolkit" ;;
    debian)
      info "  wget $(cuda_repo_url) -O /tmp/cuda-keyring.deb"
      info "  sudo dpkg -i /tmp/cuda-keyring.deb"
      info "  sudo apt-get update && sudo apt-get install -y cuda-toolkit" ;;
    arch) info "  $(pkg_cmd cuda)" ;;
  esac
  die "this setup builds llama.cpp with CUDA, so it cannot continue without nvcc. Install the toolkit, then rerun; or run ./scripts/shura install for the Vulkan/CPU path that needs no CUDA"
fi
NVCC_VER="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9.]+$')"
info "nvcc $NVCC_VER"
if [ -n "$DRIVER_CUDA" ] && [ "$(printf '%s\n%s\n' "$NVCC_VER" "$DRIVER_CUDA" | sort -V | tail -1)" != "$DRIVER_CUDA" ]; then
  die "CUDA toolkit $NVCC_VER is newer than your driver supports ($DRIVER_CUDA). Update the NVIDIA driver, then rerun."
fi

# ---------------------------------------------------------------------------------------- build
say "Building llama.cpp for your GPU"
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
LLAMA_DIR="${SHURA_LLAMA_DIR:-$SHURA_HOME/llama.cpp-perf}"
info "CUDA architecture sm_$ARCH (first build takes 10-20 minutes)"
CUDA_ARCH="$ARCH" "$REPO_DIR/scripts/build-llama.sh" "$LLAMA_DIR" > "$HOME/.cache/shura/build.log" 2>&1 \
  || die "build failed - see ~/.cache/shura/build.log"
LLAMA_BIN="$LLAMA_DIR/build/bin/llama-server"
[ -x "$LLAMA_BIN" ] || die "llama-server was not built - see ~/.cache/shura/build.log"
info "built $LLAMA_BIN"

# ----------------------------------------------------------------------------------------- plan
say "Choosing the model size for your hardware"
if systemctl --user is-active --quiet ai-gateway 2>/dev/null; then
  info "stopping the running gateway so VRAM can be measured"
  systemctl --user stop ai-gateway
  GATEWAY_STOPPED_BY_SETUP=1
fi
PLAN="$(python3 "$PY" plan ${QUANT:+--quant "$QUANT"} --desktop-ram-gb "$DESKTOP_RAM")" || die "planning failed"
jget() {  # jget JSON key [key...]
  python3 -c 'import json,sys
d = json.loads(sys.argv[1])
for k in sys.argv[2:]: d = d[k]
print("" if d is None else d)' "$@"
}
QUANT="$(jget "$PLAN" quant)"
[ -n "$QUANT" ] || die "$(jget "$PLAN" reason)"
info "Tiel-Coder $QUANT - $(jget "$PLAN" reason)"
info "download: $(awk -v b="$(jget "$PLAN" size)" 'BEGIN {printf "%.1f GB", b/1e9}') + 0.9 GB vision projector"
info "starting layout: experts of the first $(jget "$PLAN" placement ncmoe) layers in RAM, $(jget "$PLAN" placement slots) hot-expert cache slots (auto-tune refines this)"
case "$QUANT" in
  IQ3_XXS|Q2_K_XL) info "note: this smaller quant writes noticeably worse code than IQ4_XS; more RAM or VRAM allows IQ4_XS" ;;
esac

# ---------------------------------------------------------------------------------------- model
say "Model"
FILE="$(jget "$PLAN" file)"
if [ -z "$MODEL" ]; then
  found="$(find "$SHURA_HOME/models" "$HOME" -maxdepth 4 -name "$FILE" -not -path '*/.cache/*' 2>/dev/null | head -1 || true)"
  if [ -n "$found" ] && ask "Found $found - reuse it instead of downloading?"; then MODEL="$found"; fi
fi
if [ -n "$MODEL" ]; then
  PATHS="$(python3 "$PY" fetch --quant "$QUANT" --dest "$SHURA_HOME/models" --existing "$MODEL")" || die "model check failed"
else
  ask "Download it now?" || die "a model is required"
  PATHS="$(python3 "$PY" fetch --quant "$QUANT" --dest "$SHURA_HOME/models")" || die "download failed"
fi
MODEL="$(jget "$PATHS" model)"
MMPROJ="$(jget "$PATHS" mmproj)"

# -------------------------------------------------------------------------------------- autotune
say "Auto-tuning for your hardware"
info "measures a few GPU/CPU layouts on your GPU${QUICK:+ (quick mode)}${QUICK:- and checks the best one at full 200k context}"
python3 "$PY" autotune --model "$MODEL" --mmproj "$MMPROJ" --llama "$LLAMA_BIN" --llama-src "$LLAMA_DIR" \
  --desktop-ram-gb "$DESKTOP_RAM" ${QUICK:+--quick} | tee "$HOME/.cache/shura/autotune.json" || die "auto-tune failed"
RESULT="$(tail -1 "$HOME/.cache/shura/autotune.json")"

# ----------------------------------------------------------------------------------- services
say "Installing Shura"
SHURA_DOCKER="$DOCKER" "$REPO_DIR/scripts/install.sh" ${SEARCH:---no-searxng}
GATEWAY_STOPPED_BY_SETUP=""       # install.sh started it again

say "Done"
speed="$(jget "$RESULT" decode_compare) tok/s at ${QUICK:+12k}${QUICK:-100k} context"
[ -z "$QUICK" ] && speed="$speed, $(jget "$RESULT" decode_full) tok/s at ~185k"
info "Tiel-Coder $QUANT: $speed"
info "Open a project:  cd your-project && shura      or: app menu -> Shura      (agent: shuracode)"
info "Status: shura status     Help: shura help     Log: ~/.cache/shura/setup.log"
