# Installer (`setup.sh`)

One script goes from a Linux machine with an NVIDIA GPU to a tuned, fully local coding agent. It picks
the Tiel-Coder quant that fits your RAM and VRAM, builds llama.cpp for your GPU, measures a few layouts on
your card and writes the fastest safe one.

```bash
curl -fsSL https://raw.githubusercontent.com/hikkian/shura/main/setup.sh | bash
# or
git clone https://github.com/hikkian/shura.git && cd shura && ./setup.sh
```

| Option | |
|---|---|
| `--quant Q` | force a quant (`Q5_K_XL`, `Q4_K_XL`, `IQ4_XS`, `Q3_K_XL`, `IQ3_XXS`, `Q2_K_XL`) |
| `--model FILE` | reuse a `.gguf` you already have (checked against the Hugging Face SHA-256) |
| `--quick` | one candidate layout and no full-context check |
| `--desktop-ram-gb N` | RAM kept free for the browser, IDE and messengers (default 6) |
| `--no-search` | skip the private SearXNG web search, so Docker is not needed |
| `--yes` | do not ask before each step |

Requirements: x86_64, Fedora / Ubuntu / Debian / Arch (or derivatives), an NVIDIA GPU with the
proprietary driver already installed, at least 16 GB RAM, and 30–45 GB free disk space.

## What it does

1. **Checks** the distro, the NVIDIA driver (and prints the install command if it is missing), and free
   disk space.
2. **System packages**: build tools, Node, Python, a folder dialog. It asks before running `sudo`.
3. **Docker**, optional, for the private SearXNG.
4. **CUDA toolkit** from NVIDIA's repository (Arch: `pacman -S cuda`). Only `cuda-toolkit` is
   installed; the driver is never touched. The script stops if the toolkit is newer than your driver
   supports.
5. **Builds llama.cpp**: the pinned perf fork plus the TurboQuant fix, compiled for your GPU's compute
   capability. The log goes to `~/.cache/shura/build.log`.
6. **Plans**: picks the quant and a starting GPU/CPU layout (see below).
7. **Model**: reuses a matching file found under your home folder, or downloads it. The download
   resumes if interrupted and is checked against its SHA-256.
8. **Auto-tunes** on your GPU (see below).
9. **Installs** the gateway service, [ShuraCode](https://github.com/hikkian/shuracode) (the coding agent,
   with its own engine, config and memory), the `shura` command and the app-menu entry, using
   `scripts/install.sh`.

Large files go in `$SHURA_HOME` (default `~/.local/share/shura`): the models, the llama.cpp build and,
in curl mode, the repository itself. The full log is `~/.cache/shura/setup.log`. Rerunning the script
is safe: existing configs are backed up before being replaced, and an existing llama.cpp build or model
is reused (the build is incremental).

## How the quant is chosen

The planner reads the real tensor sizes from each GGUF header on Hugging Face (a 16 MB range request,
not the whole file). It then works out, per layer, how many bytes of routed experts can stay on the GPU.
The rest of each layer's experts go to RAM, with a hot-expert cache on the GPU in front of them.

- **Upgrade** to Q5_K_XL or Q4_K_XL only if the CPU-side experts stay small (≤ 2 GB). A higher quant
  that makes decoding RAM-bound is not worth it.
- Otherwise, **step down** from IQ4_XS through Q3_K_XL, IQ3_XXS and Q2_K_XL until the model fits in
  RAM with `--desktop-ram-gb` still free.

Examples from the planner (desktop using ~0.7 GB VRAM):

| RAM | GPU VRAM | Quant | Experts in RAM |
|---|---|---|---|
| 16 GB | 8 GB | Q2_K_XL | 30 of 40 layers |
| 16 GB | 12 GB | IQ3_XXS | 16 layers |
| 16 GB | 16 GB | IQ4_XS | — |
| 16 GB | 24 GB | Q4_K_XL | 1 layer |
| 32 GB | 8 GB | IQ4_XS | 35 layers |
| 32 GB | 12 GB | IQ4_XS | 24 layers (the reference machine) |

IQ3_XXS and Q2_K_XL write noticeably worse code than IQ4_XS. The installer says so when it picks them.

Constants in [`installer/shura_setup.py`](../installer/shura_setup.py), calibrated on the reference
machine (RTX 4070 SUPER, 200k context, turbo3 KV cache, MTP):

| Constant | Value | Covers |
|---|---|---|
| `GPU_OVERHEAD` | 2040 MB | KV cache, MTP draft context, compute buffers, CUDA context |
| `VRAM_RESERVE` | 600 MB | desktop spikes and CUDA pool growth at full context |
| `RAM_BASE` | 1500 MB | llama-server without its prompt cache |
| `VISION_EXTRA` | 1000 MB | vision projector; vision mode moves a layer or trims the cache to make room |
| `MAX_SLOTS` | 64 | expert-cache slots per layer |

## Auto-tune

The planner's layout is the starting point. On your GPU, auto-tune then:

1. Tries the planner's layout and one or two variants that move more experts to RAM, freeing VRAM for
   cache slots. `--quick` tries only the first.
2. Runs the correctness probes on each layout. If TurboQuant's turbo3 KV cache gives wrong answers on
   your GPU, it falls back to q4_0.
3. Measures decode speed at ~100k tokens of context (~12k with `--quick`): one warm-up run is
   discarded, then the median is taken. Short contexts mislead: at 32k, one layout here was 9% faster
   and then tied at 187k.
4. Treats layouts within 5% of the fastest as a tie and keeps the one that leaves the most VRAM free,
   so the desktop stays smooth.
5. Checks the winner at ~185k tokens (skipped with `--quick`).
6. Writes `config/model-launch.json` and `config/guardian.json`, backing up any previous versions.

## What it changes on your system

| Where | What |
|---|---|
| System packages (`sudo`, asks first) | build tools, Node, Python, zenity, libnotify; optionally Docker; NVIDIA's CUDA repository and `cuda-toolkit` |
| `~/.local/share/shura/` | models, llama.cpp build, the repository (curl mode) |
| `~/.config/systemd/user/ai-gateway.service` | the gateway service (enabled) |
| `~/.config/shuracode/`, `~/.local/share/shuracode/`, `~/.local/bin/shuracode` | ShuraCode: config, engine, memory, command |
| `~/.local/bin/shura`, `~/.local/share/applications/`, `~/.local/share/icons/` | command and app-menu entry |
| Docker container `searxng` | private web search on 127.0.0.1:8888 |

### Uninstall

```bash
systemctl --user disable --now ai-gateway
rm ~/.config/systemd/user/ai-gateway.service ~/.local/bin/shura \
   ~/.local/share/applications/io.github.hikkian.Shura.desktop \
   ~/.local/share/icons/hicolor/*/apps/shura.png ~/.local/bin/shuracode
rm -rf ~/.local/share/shura ~/.cache/shura      # models and build: frees 25-40 GB
rm -rf ~/.config/shuracode ~/.local/share/shuracode   # ShuraCode, including its memory
docker rm -f searxng                            # if installed
```

To keep ShuraCode's memory, copy `~/.local/share/shuracode/memory/` somewhere first. System packages and the CUDA
toolkit stay installed. Remove them with your package manager if you want.

## Test status

| Part | How it was tested |
|---|---|
| Distro detection, package names, CUDA repository URLs | containers: `ubuntu:24.04`, `debian:12`, `archlinux`, `fedora:44` ([`tests/distro-packages.sh`](../tests/distro-packages.sh), run in CI) |
| Quant choice and layout invariants | unit tests on real GGUF header sizes ([`tests/test_installer.py`](../tests/test_installer.py)) |
| Auto-tune | real runs on the reference machine |
| A full fresh install on a clean machine, or any GPU other than an RTX 4070 SUPER | **not tested yet** |

Run the tests locally:

```bash
python3 -m unittest discover -s tests
docker run --rm -v "$PWD:/repo:ro,Z" ubuntu:24.04 bash /repo/tests/distro-packages.sh
```


---

# `shura install` (every other machine)

`setup.sh` above is the path for NVIDIA on Linux (it builds our CUDA fork). Everything else (AMD, Intel, Apple, Windows,
CPU-only servers, NVIDIA on Windows or macOS) uses the universal installer, which needs only Python 3.9+ and uses prebuilt
llama.cpp, so nothing is compiled:

```bash
./scripts/shura install --dry-run   # the plan, the download sizes and the disk needed; changes nothing
./scripts/shura install             # asks, then: engine -> model -> test launch -> ready
./scripts/shura start | status | stop
./scripts/shura uninstall           # deletes everything under the Shura home folder
```

| Option | |
|---|---|
| `--dry-run` | print the plan and what would be downloaded; touch nothing |
| `--yes` | do not ask (required when there is no terminal; without a terminal it never downloads silently) |
| `--quant Q`, `--model-id ID` | force a quant or a model from the catalog |
| `--context N` | never use a window larger than N tokens |
| `--dir DIR` | Shura home (default `$SHURA_HOME`, else `~/.local/share/shura`, `~/Library/Application Support/shura`, `%LOCALAPPDATA%\shura`) |
| `--port N` | server port (default 8080; the server listens on 127.0.0.1 only) |
| `--no-turbo` | never use the third-party TurboQuant+ llama.cpp build (by default it is tried first and compared with upstream on your machine) |
| `--ram-gbs N` | RAM read speed in GB/s when it cannot be measured (no C compiler, as on most Windows PCs): dual-channel DDR4-3200 is about 40, DDR5-6000 about 80 |

What it does, in order, and what protects you at each step:

1. **Plan.** The same planner as `shura check` ([how it decides](HARDWARE.md)). If nothing fits it says why and stops.
2. **Engine.** Downloads the prebuilt llama.cpp for each backend that could run your GPU (AMD and Intel have more than one),
   checks it against the release's SHA-256, runs a 10-second self-test on a 19 MB model and keeps the fastest that works.
   If no GPU build works, it plans again for the CPU and tells you.
3. **Model.** Size and SHA-256 come from the Hugging Face file list. The free disk space is checked first, a partial file
   is resumed (`.part`), and a file that fails the hash is moved aside as `.corrupt` and never used.
4. **Test launch.** Starts the real server with the planned settings and generates tokens. If it runs out of memory, the plan
   is made again with 1 GiB more left free (then 2 GiB and a window of at most 131k), up to three attempts. You end with a
   configuration that was seen to work, or with a report and the downloads kept.
5. **Ready.** Writes `state.json` and `shura-hardware-report.md` (no names, paths or IDs) into the Shura home and prints the
   OpenAI-compatible endpoint. Nothing is ever sent anywhere by the installer; `shura report --issue` prints a pre-filled
   GitHub link that you open yourself.

Files: `engines/` (llama.cpp builds), `models/`, `logs/`, `state.json`, `server.pid`. All of it is under the Shura home and
`shura uninstall` removes it. SSD wear: the only large write is the model download itself (once; it is reused by reruns).
