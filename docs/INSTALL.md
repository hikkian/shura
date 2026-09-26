# Installation

Tested on Fedora 44 with an RTX 4070 SUPER. Other distros and NVIDIA cards with ≥12 GB VRAM should work
with the obvious package-manager substitutions (set `CUDA_ARCH` for your GPU).

## 1. NVIDIA driver and CUDA toolkit

Driver from RPM Fusion (`akmod-nvidia`). The CUDA *toolkit* (nvcc, headers, cuBLAS) comes from NVIDIA's
repository. Install only `cuda-toolkit`, not `cuda`/`cuda-drivers`, so it does not replace the RPM Fusion
driver:

```bash
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/cuda-fedora44.repo
sudo dnf install -y cuda-toolkit
echo 'export PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
nvcc --version   # CUDA 13.4 was used here
```

Build tools: `sudo dnf install -y cmake ninja-build gcc-c++ git nodejs python3 docker`.

## 2. This repository

```bash
git clone https://github.com/hikkian/shura.git ~/Projects/shura
cd ~/Projects/shura
```

## 3. Build llama.cpp (perf fork + TurboQuant fix)

```bash
scripts/build-llama.sh              # -> ~/ai/llama.cpp-perf/build/bin/llama-server
CUDA_ARCH=86 scripts/build-llama.sh # RTX 30xx
```

The script pins the exact benchmarked commit and applies `patches/turboquant-pr12-fix.patch`.

## 4. Model

Download `Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf` and `mmproj-BF16.gguf` from
[peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP)
into `~/ai/models/`. Keep them on a normal Linux filesystem (ext4/xfs/btrfs), not on an NTFS or FUSE
mount: the model is read at load time, and a FUSE mount makes loading slow.

## 5. Install services and configs

```bash
scripts/install.sh
```

This:

- creates `config/guardian.json` and `config/model-launch.json` from the examples (gitignored; edit
  the paths if you did not use `~/ai/...`);
- installs and starts the `ai-gateway` systemd **user** service on `127.0.0.1:8080`;
- writes the OpenCode config (backing up any existing one) and the offline environment variables;
- starts a private SearXNG container on `127.0.0.1:8888` with a freshly generated secret.

Flags: `--no-service`, `--no-opencode`, `--no-searxng`.

## 6. OpenCode and the MCP browser

```bash
npm config set prefix ~/.npm-global && export PATH=~/.npm-global/bin:$PATH
npm install -g opencode-ai
# Install Chromium with the SAME Playwright version the MCP server bundles (avoids version mismatch):
npx -y @playwright/mcp@latest --version
cd "$(dirname "$(find ~/.npm/_npx -path '*node_modules/playwright/cli.js' | head -1)")" && node cli.js install chromium
```

Log out and back in once so `~/.config/environment.d/opencode-offline.conf` takes effect.

## 7. Verify

```bash
curl -s localhost:8080/guardian/status          # "status": "UNLOADED" until the first request
opencode mcp list                               # playwright + searxng: connected
python3 bench/correctness_probes.py             # loads the model (~12 s), expects 9/9
```

Optional full benchmark:

```bash
python3 bench/make_context.py --source ~/ai/llama.cpp-perf --tokens 187000 -o /tmp/ctx_187k.txt
python3 bench/deep_context_bench.py --context /tmp/ctx_187k.txt
```

## Tuning for other hardware

- **`nCpuMoe`**: lower values put more expert layers on the GPU. Use the lowest value that loads and
  survives a full-context generation.
- **`moeCacheSlots`**: raise it until decode stops improving. Then back off, because oversubscribing
  VRAM does not error, it just slows down run over run. Always test at full context: a config that
  loads can still OOM mid-generation.
- **`cacheRamMB`**: RAM for the cross-session prompt cache (~0.8 GB per 187k session).
- **`threads` / `threadsBatch`**: set them to your physical core count, not the thread count. The
  CPU-side expert compute is memory-bandwidth bound, so extra threads add no speed and only take CPU
  away from the desktop.
- **`niceLevel`** (default 10): runs llama-server at lower CPU priority. It costs nothing on an idle
  desktop, and interactive apps win whenever they compete.
- **`specDraftNMax`**: 1-3 perform about the same. 1 has the highest acceptance and uses the least VRAM.
- To build a routing profile for a different model or your own workload: `scripts/capture-moe-trace.sh <model.gguf> [prompt-dir]`. Put a few realistic chat-formatted prompts in `prompt-dir`; only generated tokens are kept.
