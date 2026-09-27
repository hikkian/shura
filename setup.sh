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
ask()  {  # ask "question" -> 0 for yes
  [ -n "$YES" ] && return 0
  local reply; read -r -p "    $1 [Y/n] " reply </dev/tty || reply=y
  [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}

# ---------------------------------------------------------------------------------------- distro
detect_family() {
  [ -r /etc/os-release ] || die "cannot read /etc/os-release"
  # shellcheck disable=SC1091
  . /etc/os-release
  OS_ID="$ID" OS_VER="${VERSION_ID:-}" OS_NAME="${PRETTY_NAME:-$ID}"
  case " $ID ${ID_LIKE:-} " in
    *" fedora "*|*" rhel "*) FAMILY=fedora ;;
    *" ubuntu "*|*" debian "*) FAMILY=debian ;;
    *" arch "*) FAMILY=arch ;;
    *) FAMILY=unsupported ;;
  esac
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

pkg_install() {  # pkg_install pkg...
  case "$FAMILY" in
    fedora) sudo dnf install -y "$@" ;;
    debian) sudo apt-get update -qq && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y "$@" ;;
    arch)   sudo pacman -S --needed --noconfirm "$@" ;;
  esac
}

# --------------------------------------------------------------------------------------- bootstrap
[ "${SHURA_SOURCE_ONLY:-}" = 1 ] && return 0
detect_family
SELF="${BASH_SOURCE[0]:-}"
if [ -z "$SELF" ] || [ ! -f "$(dirname "$SELF")/installer/shura_setup.py" ]; then
  # Piped from curl: fetch the repository, then run the real script from it.
  [ "$FAMILY" = unsupported ] && die "unsupported distribution ($OS_NAME). Supported: Fedora, Ubuntu/Debian, Arch."
  command -v git >/dev/null || { echo "Installing git..."; pkg_install git; }
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
say "Checking this machine"
[ "$FAMILY" = unsupported ] && die "unsupported distribution ($OS_NAME). Supported: Fedora, Ubuntu/Debian, Arch."
[ "$(uname -m)" = x86_64 ] || die "only x86_64 is supported"
info "system: $OS_NAME ($FAMILY family)"
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  info "No working NVIDIA driver found. Install it first, reboot, then rerun this script:"
  case "$FAMILY" in
    fedora) info "  RPM Fusion: sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda" ;;
    debian) info "  Ubuntu: sudo ubuntu-drivers install    Debian: see wiki.debian.org/NvidiaGraphicsDrivers" ;;
    arch)   info "  sudo pacman -S nvidia-open (or nvidia-open-dkms)" ;;
  esac
  die "Shura needs an NVIDIA GPU with the proprietary driver (the MoE expert cache is CUDA-only)."
fi
GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
DRIVER_CUDA="$(nvidia-smi | grep -oE 'CUDA (UMD )?Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
info "GPU: $GPU (driver supports CUDA ${DRIVER_CUDA:-?})"
info "RAM: $(awk '/MemTotal/ {printf "%.0f GB", $2/1048576}' /proc/meminfo), CPU: $(lscpu -p=Core,Socket 2>/dev/null | grep -vc '^#') physical cores"
free_gb=$(df -BG --output=avail "$SHURA_HOME" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 30 ] || die "need ~30-45 GB free in $SHURA_HOME (have ${free_gb} GB)"

# ------------------------------------------------------------------------------------- packages
say "System packages"
read -ra PKGS <<< "$(build_packages)"
missing=()
for p in cmake git curl node npm python3; do command -v "$p" >/dev/null || missing+=("$p"); done
command -v ninja >/dev/null || command -v ninja-build >/dev/null || missing+=(ninja)
if [ ${#missing[@]} -gt 0 ]; then
  info "missing: ${missing[*]}"
  info "will run: sudo package install ${PKGS[*]}"
  ask "Install them now?" || die "cannot continue without build tools"
  pkg_install "${PKGS[@]}"
else
  info "build tools already installed"
fi

DOCKER=""
if [ -n "$SEARCH" ]; then
  if command -v docker >/dev/null; then
    if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi
  elif ask "Web search needs Docker (for a private SearXNG). Install Docker?"; then
    case "$FAMILY" in fedora) pkg_install moby-engine ;; debian) pkg_install docker.io ;; arch) pkg_install docker ;; esac
    sudo systemctl enable --now docker
    DOCKER="sudo docker"
  else
    SEARCH=""
    info "skipping web search (rerun with Docker installed to add it)"
  fi
fi

# ----------------------------------------------------------------------------------------- CUDA
say "CUDA toolkit"
for d in /usr/local/cuda/bin /opt/cuda/bin; do [ -x "$d/nvcc" ] && export PATH="$d:$PATH"; done
if ! command -v nvcc >/dev/null; then
  info "nvcc not found - installing the CUDA toolkit (the driver is left untouched)"
  ask "Add NVIDIA's CUDA repository and install cuda-toolkit?" || die "the CUDA toolkit is required to build llama.cpp"
  case "$FAMILY" in
    fedora)
      repo="$(cuda_repo_url)"
      curl -fsI "$repo" >/dev/null || die "NVIDIA has no CUDA repository for Fedora $OS_VER yet ($repo)"
      sudo dnf config-manager addrepo --from-repofile="$repo" 2>/dev/null || sudo dnf config-manager --add-repo "$repo"
      sudo dnf install -y cuda-toolkit ;;
    debian)
      keyring="$(cuda_repo_url)"
      curl -fsSL "$keyring" -o /tmp/cuda-keyring.deb || die "NVIDIA has no CUDA repository for $OS_NAME ($keyring)"
      sudo dpkg -i /tmp/cuda-keyring.deb && sudo apt-get update -qq && sudo apt-get install -y cuda-toolkit ;;
    arch) pkg_install cuda ;;
  esac
  for d in /usr/local/cuda/bin /opt/cuda/bin; do [ -x "$d/nvcc" ] && export PATH="$d:$PATH"; done
  command -v nvcc >/dev/null || die "nvcc still not found after installing the CUDA toolkit"
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

# -------------------------------------------------------------------------------------- opencode
say "OpenCode"
if [ -w "$(npm config get prefix)/lib" ] 2>/dev/null; then NPM_BIN="$(npm config get prefix)/bin"
else npm config set prefix "$HOME/.npm-global"; NPM_BIN="$HOME/.npm-global/bin"; fi
export PATH="$NPM_BIN:$PATH"
if ! command -v opencode >/dev/null; then
  npm install -g opencode-ai >/dev/null
  pkg="$(npm root -g)/opencode-ai"
  if ! opencode --version >/dev/null 2>&1; then (cd "$pkg" && node ./postinstall.mjs >/dev/null); fi
fi
info "opencode $(opencode --version)"
info "installing the browser for the Playwright MCP tool"
npx -y @playwright/mcp@latest --version >/dev/null 2>&1 || true
pw="$(find "$HOME/.npm/_npx" -path '*node_modules/playwright/cli.js' 2>/dev/null | head -1)"
if [ -n "$pw" ]; then (cd "$(dirname "$pw")" && node cli.js install chromium >/dev/null 2>&1) || info "(browser install failed - web browsing tool may not work)"; fi

# -------------------------------------------------------------------------------------- autotune
say "Auto-tuning for your hardware"
info "measures a few GPU/CPU layouts on your GPU${QUICK:+ (quick mode)}${QUICK:- and checks the best one at full 200k context}"
python3 "$PY" autotune --model "$MODEL" --mmproj "$MMPROJ" --llama "$LLAMA_BIN" --llama-src "$LLAMA_DIR" \
  --desktop-ram-gb "$DESKTOP_RAM" ${QUICK:+--quick} | tee "$HOME/.cache/shura/autotune.json" || die "auto-tune failed"
RESULT="$(tail -1 "$HOME/.cache/shura/autotune.json")"

# ----------------------------------------------------------------------------------- services
say "Installing Shura"
SHURA_DOCKER="$DOCKER" "$REPO_DIR/scripts/install.sh" ${SEARCH:---no-searxng}

say "Done"
speed="$(jget "$RESULT" decode_compare) tok/s at ${QUICK:+12k}${QUICK:-100k} context"
[ -z "$QUICK" ] && speed="$speed, $(jget "$RESULT" decode_full) tok/s at ~185k"
info "Tiel-Coder $QUANT: $speed"
info "Open a project:  cd your-project && shura      or: app menu -> Shura"
info "Status: shura status     Help: shura help     Log: ~/.cache/shura/setup.log"
info "Log out and back in once so the OpenCode offline settings apply everywhere."
