<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-wordmark-dark.png">
  <img alt="Shura · شورى" src="assets/logo-wordmark-light.png" width="520">
</picture>

**A fully local AI coding workstation: a 35B MoE model at 200k context on a single 12 GB GPU,
~40 tok/s decode — without slowing down the desktop you work on.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Linux-informational)
![GPU](https://img.shields.io/badge/GPU-12%20GB%20VRAM-76B900)
![Python](https://img.shields.io/badge/gateway-stdlib%20Python-3776AB)
[![CI](https://github.com/hikkian/shura/actions/workflows/ci.yml/badge.svg)](https://github.com/hikkian/shura/actions/workflows/ci.yml)

English · [Русский](README.ru.md)

</div>

---

> *Shura* (شورى) is Arabic for "consultation, council". In a Mixture-of-Experts model every token is
> decided by a council: a router consults 8 of 256 experts. This project is about seating that council
> on modest hardware.

This repository documents and packages a working setup for running
[Tiel-Coder-35B-A3B-MTP](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP)
(`qwen35moe`, 256 experts / 8 active) as a local coding agent. The model is 18 GB; the GPU has 12 GB.
Everything runs offline: inference, web search, browser automation. The machine stays usable for a
browser, an IDE and messengers while the model works.

It contains:

- a **tuned llama.cpp build recipe** that took decode at real 187k context from 26 to ~40 tok/s;
- a **fix for TurboQuant KV quantization** on head_dim-256 models, which otherwise crashes or
  silently returns wrong answers;
- an **agentic MoE routing profile** that decides which experts live in VRAM, built from generated
  tokens of realistic agent sessions;
- a small **AI gateway** that loads the model on demand, switches text/vision automatically,
  survives crashes, and keeps session state in RAM so the SSD is barely written;
- a hands-on **comparison of four agent harnesses**, and an **OpenCode** config hardened for offline use;
- **reproducible benchmark scripts** for every number below, including the ideas that did not work.

## Results

Decode speed at **~187k tokens of real context**, RTX 4070 SUPER 12 GB + Ryzen 5 5600 + 32 GB DDR4:

| Configuration | Mean tok/s |
|---|---:|
| Stock llama.cpp, default settings | 26.1 |
| + `--load-mode none` (no mmap) | 29.7 |
| + perf fork with MoE expert cache (16 slots) | 35.9 |
| + patched TurboQuant `turbo3` KV → room for 24 expert slots | 38.3 |
| + MTP draft length 1, 6 threads at `nice 10`, agentic routing profile | **~41.6**¹ |

¹ Measured with the faster slot-restore method (the same 187k session restored before every request).
On that method the previous row measures ~40.4, so the last step is worth about +3%. Both methods are
described in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Session resume at 187k | Time |
|---|---:|
| Cold prefill | 264 s |
| Switch back after other sessions (RAM prompt cache) | **1.9 s** |
| After a full model unload/reload (slot restored from disk) | **13 s** |

| Desktop impact during generation | |
|---|---:|
| UI scheduler latency, p99 (idle desktop: 0.1-0.5 ms) | 0.1-0.3 ms |
| Free RAM with a browser, IDE and messenger open | ~12.5 GB of 32 GB |
| CPU threads used by the model | 6 of 12, at lower priority |

## How it works

```mermaid
flowchart LR
    OC["OpenCode<br/>(offline)"] -- OpenAI API --> GW["AI Gateway :8080"]
    GW -- spawns / proxies --> LS["llama-server :8090"]
    LS --- GPU["GPU: attention, MTP head,<br/>24 hot experts per layer, turbo3 KV"]
    LS --- CPU["RAM: cold experts, prompt cache"]
    OC -- MCP --> PW["Playwright"]
    OC -- MCP --> SX["SearXNG (Docker)"]
```

The cold experts of the first 26 layers stay in system RAM, and the CPU computes them on 6 threads at
reduced priority. The most-routed experts of those layers stay in VRAM; which ones is decided by a
routing profile. The model's built-in MTP head drafts one token ahead for self-speculative decoding
(~85% accepted, +32% vs no speculation). TurboQuant shrinks the KV cache enough to fit 24 hot-expert
slots instead of 16. The gateway's design is described in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Key findings

- **The expert cache is the biggest lever, and too many slots fail silently.** Oversubscribed VRAM does
  not error, it slows down run over run. Configs can load fine and still OOM mid-generation at full
  context.
- **Prompt experts ≠ generation experts.** A routing profile built from long prompts (tool schemas,
  file contents) is *worse* for decode than a tiny synthetic one. Profiles for decode speed must be
  traced from generated tokens.
- **TurboQuant is not a speedup by itself.** Its value is turning saved KV-cache VRAM into more expert
  slots. It also needs [a fix](docs/TURBOQUANT-FIX.md) on this model family.
- **CPU threads beyond the physical cores add nothing.** The CPU side is memory-bandwidth bound: 6, 8
  and 10 threads decode at the same speed, and fewer threads leave the desktop responsive.
- **Many popular tricks did not transfer** to a 12 GB card at 200k context: all experts on CPU with a
  large cache, prefetch/host-register env vars, larger ubatch, quantized MTP draft cache. Each is
  measured in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
- **Saving KV slots to disk after every response** would write up to ~640 GB/day in agentic use. The
  gateway keeps sessions in RAM and writes one file only when the model is unloaded.

## Quick start

```bash
git clone https://github.com/hikkian/shura.git && cd shura
scripts/build-llama.sh          # perf fork @ pinned commit + TurboQuant fix
# download the model into ~/ai/models/ (see docs/INSTALL.md)
scripts/install.sh              # gateway service, OpenCode config, private SearXNG
python3 bench/correctness_probes.py
opencode
```

Full walkthrough, including the CUDA toolkit, the Playwright browser and tuning for other GPUs:
**[docs/INSTALL.md](docs/INSTALL.md)**.

## Daily use

Everything starts with the desktop session; nothing needs to be launched by hand.

- **App menu → Shura.** Pick a project folder and OpenCode opens there in its own window, under the
  Shura icon. Right-click the icon for *Load model now* / *Unload model*. Works on any desktop (GNOME,
  KDE, XFCE, tiling WMs) with any common terminal; set `SHURA_TERMINAL` to choose one.
- **Terminal:** `cd` into a project and type `shura`. The model starts loading in the background while
  OpenCode opens, so it is usually ready by the time you type.

```
shura status    # model state, free RAM/VRAM, idle timer
shura warm      # load the model now          shura unload   # free RAM/VRAM now
shura off / on  # disable / enable the AI     shura logs     # follow logs
shura window    # new Shura window for the current folder
```

The model unloads itself after 30 idle minutes. The last long session is saved and restored on the
next start.

## Tested environment

> [!NOTE]
> Everything in this repository was developed and tested on **one machine**. Anything outside this list
> is supported by design but **untested** — reports and fixes are welcome.

| | Tested |
|---|---|
| OS | Fedora 44, kernel 7.2, x86_64 |
| Desktop | GNOME Shell 50 on Wayland |
| Terminal | Ghostty 1.3 |
| GPU / driver | NVIDIA RTX 4070 SUPER 12 GB, driver 615.71 (RPM Fusion), CUDA 13.4 |
| CPU / RAM | AMD Ryzen 5 5600, 32 GB DDR4-3200 |
| Software | OpenCode 1.18, Docker 29, Python 3.14, Node 24 |

**Not tested on real systems:**
- other distributions: `install.sh` and the docs use Fedora package names;
- KDE, XFCE and tiling WMs;
- terminals other than Ghostty: launch commands for 11 other terminals were checked in dry-run only;
- the `kdialog`/`yad` folder dialogs;
- other GPUs: all numbers above are specific to a 12 GB card, and other cards will need re-tuning (see
  [docs/INSTALL.md](docs/INSTALL.md#tuning-for-other-hardware)).

AMD and Intel GPUs are not supported: the expert-cache fork is CUDA-only.

## Repository layout

```
gateway/     ai_gateway.py - on-demand llama-server manager + OpenAI-compatible proxy (stdlib only)
config/      example configs + agentic MoE routing profile for Tiel-Coder
patches/     TurboQuant head_dim-256 fix for the perf fork
scripts/     shura (daily CLI), build-llama.sh, install.sh, capture-moe-trace.sh
desktop/     app-menu launcher template
assets/      logo (light/dark), app icons 16-512 px, social preview, source artwork
bench/       deep-context benchmark, depth sweep, session-cache test, correctness probes
opencode/    OpenCode config template + AGENTS.md (verification + relative-path rules)
mcp/         stdio proxy that trims MCP tool schemas, with allowlists
systemd/     user service unit
searxng/     private SearXNG settings template
docs/        benchmarks, architecture, harness comparison, TurboQuant fix, install, troubleshooting
```

## Documentation

| | |
|---|---|
| [BENCHMARKS.md](docs/BENCHMARKS.md) | Every measurement: build tuning, MTP, threads, routing profiles, prefill, desktop impact, dead ends |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Gateway state machine, endpoints, KV cache and SSD-wear design |
| [TURBOQUANT-FIX.md](docs/TURBOQUANT-FIX.md) | Why TurboQuant crashes / corrupts output on this model family, and the fix |
| [HARNESS-COMPARISON.md](docs/HARNESS-COMPARISON.md) | OpenCode vs Pi vs Pithagoras vs DeepSeek Harness, tested hands-on |
| [INSTALL.md](docs/INSTALL.md) | Step-by-step setup and tuning for other GPUs |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Known failure modes and their fixes |
| [SECURITY.md](SECURITY.md) | Threat model: localhost-only services, no auth, what not to expose |

## Acknowledgements

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and the
  [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp) `perf` fork (MoE expert cache, TurboQuant)
- **wilky2005** for the TurboQuant fix ([thecodacus/llama.cpp#12](https://github.com/thecodacus/llama.cpp/pull/12))
- [Tiel-Coder-35B-A3B-MTP](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP) by peculiar-ragdoll
- [OpenCode](https://github.com/anomalyco/opencode), [SearXNG](https://github.com/searxng/searxng),
  [Playwright MCP](https://github.com/microsoft/playwright-mcp)

## License

[MIT](LICENSE) for everything in this repository. The patch in `patches/` is derived from llama.cpp (MIT).
Model weights are not included and are covered by their own license.
