# Installation

Tested only on Fedora 44 + GNOME (Wayland) + Ghostty with an RTX 4070 SUPER (full list: [README → Tested environment](../README.md#tested-environment)). Other distros and NVIDIA cards with ≥12 GB VRAM should work
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

Optional, for the app-menu launcher: a folder dialog (`zenity`, `kdialog` or `yad`) and `notify-send`.
Any desktop environment and terminal emulator works.

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
- installs [ShuraCode](https://github.com/hikkian/shuracode), the coding agent: its engine, config, memory,
  the `shuracode` command and the browser for its web tool;
- starts a private SearXNG container on `127.0.0.1:8888` with a freshly generated secret;
- links the `shura` command into `~/.local/bin` and adds a **Shura** launcher to the app menu.

Flags: `--no-service`, `--no-shuracode`, `--no-searxng`, `--no-desktop`.

## 6. Verify

```bash
shura status                                    # model: UNLOADED until the first request
shuracode doctor                                # engine, config, memory, gateway: ok
python3 bench/correctness_probes.py             # loads the model (~12 s), expects 9/9
```

Then open a project with **Shura** from the app menu, or run `shura` inside the project folder.

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
- **`chatTemplateKwargs`** (optional, e.g. `{"terse": false}`): extra variables for the model's chat template, passed as `--chat-template-kwargs`. Tiel-Coder's template adds a "be concise" system prompt by default (`terse`); on 44 short agentic tasks it made no measurable difference to quality or tokens, so switching it off is a matter of taste.
- **`specDraftNMax`**: 1-3 perform about the same. 1 has the highest acceptance and uses the least VRAM.
- To build a routing profile for a different model or your own workload: `scripts/capture-moe-trace.sh <model.gguf> [prompt-dir]`. Put a few realistic chat-formatted prompts in `prompt-dir`; only generated tokens are kept.

## A second model: Occamy 1.0 with MTP

Accio-Lab ships Occamy 1.0 GGUFs without a multi-token-prediction head, and the official head (`Accio-Lab/occamy-1.0-MTP`) only as BF16
safetensors for SGLang. `scripts/graft_mtp_head.py` puts that head into the GGUF, so `--spec-type draft-mtp` works from one file the way
it does for Tiel-Coder (a separate draft model would cost another gigabyte of VRAM):

1. `graft_mtp_head.py head  occamy-1.0-IQ4_XS.gguf mtp-trained.safetensors head-bf16.gguf`
2. `llama-quantize --tensor-type 'ffn_.*_exps\.weight=q3_k' head-bf16.gguf head-q.gguf q8_0 6` (1612 MiB -> 371 MiB, the size of Tiel-Coder's head)
3. `graft_mtp_head.py merge occamy-1.0-IQ4_XS.gguf head-q.gguf occamy-1.0-IQ4_XS-MTP.gguf`

Checked on a CPU run with temperature 0: the model loads, `common_speculative_init_result` creates the MTP context, 57 of 57 drafted tokens
were accepted on a short coding prompt (an easy prompt: expect less on real work). MTP never changes the answer, only the speed. Not yet
measured on the GPU: speed against Tiel-Coder, how many expert-cache slots fit, and a routing profile of its own (the config borrows
Tiel-Coder's until one is captured with `scripts/capture-moe-trace.sh`).
