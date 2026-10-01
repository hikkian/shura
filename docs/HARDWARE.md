# How Shura chooses a model and settings for your hardware

The universal installer does four things: it **describes** your machine (a hardware profile), **measures** what the
profile cannot tell (memory bandwidth), **plans** (a model, a quant and settings from the model catalog) and then
**calibrates** on your machine. This page documents the data formats and the planner's logic, so that you can check
a decision, change it, or add your own hardware and models.

> [!NOTE]
> **Status.** The planner and both formats below are implemented and unit-tested on described machines
> (`tests/fixtures/hardware/`). Only one machine has been verified for real: NVIDIA on Linux. Everything else is
> "supported by design, waiting for reports". See [the status table](#status-of-each-platform).

## The pipeline

```
detect hardware ─► hardware profile ─┐
measure bandwidth ───────────────────┤
model catalog (data) ────────────────┴─► planner ─► plan ─► download ─► calibrate ─► run
backend probe (which build works) ──────────────────────────────────────┘
```

- **Detect** (per OS, no root): CPU cores and NUMA nodes, RAM, GPUs with VRAM, drivers and the backends that can drive
  them. `llama-server --list-devices` of the downloaded build is the final word on what GPUs exist.
- **Measure**: RAM bandwidth is measured, never read from a spec (you cannot see XMP, channel count or ranks without
  root, and the number is what decides the speed). The helper reads one big buffer from all cores, the way an inference
  engine reads weights; giving each thread its own allocation measured about a third lower on the reference machine.
- **Plan** (this page): pure function, no I/O, deterministic.
- **Probe**: for every backend that could run your GPU, a 10-second self-test on a tiny model; the best working one wins.
  AMD and Intel usually have two candidates (ROCm or Vulkan; SYCL, Vulkan or OpenVINO), and community benchmarks show that
  neither always wins, so it is measured rather than guessed.
- **Calibrate**: a short real benchmark on the chosen model corrects the prediction and picks threads and NUMA mode.

## Hardware profile (JSON, `schema: 1`)

| Field | Meaning |
|---|---|
| `os.family`, `os.arch` | `linux`/`macos`/`windows`, `x86_64`/`arm64` |
| `cpu.physical_cores`, `logical_cores`, `numa_nodes` | used for thread count and NUMA mode |
| `memory.total`, `available` | bytes |
| `memory.bandwidth_gbs` | **measured** RAM read bandwidth in GB/s (omit if unknown: a conservative 20 GB/s is assumed and confidence drops) |
| `gpus[]` | empty for CPU-only machines. Each: `vendor` (`nvidia`/`amd`/`intel`/`apple`), `name`, `vram_total`, `vram_used` (what the desktop uses right now), `bandwidth_gbs`, `backends`, `unified` (Apple and APU-style shared memory), `display` (drives a screen: more VRAM is kept free) |
| `disk.free` | bytes |

`installer/universal/schema.py` validates a profile and lists every problem at once.

## Model catalog (data, `catalog/models.json`)

Adding a model is a data change, not a code change. Per model: `id`, `name`, `arch`, `kind` (`dense`/`moe`),
`params_total_b`, `params_active_b`, `n_layers`, `context_max`, `kv_bytes_per_token_f16` (KV cache size per token at f16),
`capability` (orders models: higher is more capable), `default_quant`, `source` (Hugging Face repo and file pattern),
`quants[]` (`id`, `file_bytes`, `nonexpert_bytes`, `quality`: unique, higher is better), `features`, `tested[]`.
MoE models also need `n_experts`, `experts_used` and `expert_active_fraction` (the share of the bytes read per token that
belongs to routed experts; attention, shared experts and the output head are the rest).

`tested[]` records who ran it on what and what they measured. It is how a model earns trust.

## The speed model

Decoding reads the *active* weights once per token, so speed is limited by memory bandwidth:

```
active bytes   = file size x active params / total params            (dense: the whole file)
time per token = bytes on fast memory / its bandwidth  +  bytes in RAM / RAM bandwidth
tok/s          = 1 / ( time per token / efficiency  +  KV cache in use / (KV memory bandwidth x attention efficiency) )
```

- **The KV cache** is read for every generated token, so a fuller window is slower. The planner judges speed with the
  **window half full** (a working session, not an empty chat) and also reports the speed for an empty and for a full
  window (`speed_by_fill`). The KV cache sits in VRAM (or unified memory) when a GPU is used, otherwise in RAM.
- **MoE on a discrete GPU** keeps attention, shared experts and the KV cache in VRAM and fills what is left with whole
  expert layers (`--n-cpu-moe` = how many layers' experts stay in RAM). The routed experts read per token are split by
  the share of layers on the GPU.
- **Dense on a discrete GPU** splits layers the same way the bandwidth times add up.
- **Unified memory** (Apple, APUs) is one pool, limited to 70% of RAM for the GPU.
- **CPU-only** reads everything from RAM. CPU-only does **not** mean small models: a dual-socket server with
  150-250 GB/s runs models that no 12 GB card can hold; the planner only asks what fits and how fast it would be.

**Efficiency constants** (`installer/universal/planner.py`, `DEFAULTS["efficiency"]`): hybrid GPU+CPU MoE 0.30, GPU only 0.50,
CPU only 0.40, unified 0.45, attention (KV reads) 0.25. They absorb everything the pure bandwidth sum ignores: kernel launches, per-layer
synchronisation, dequantisation, attention over a long context. **They were calibrated on one machine** (RTX 4070
SUPER, DDR4-3200, about 40 GB/s as measured by the shipped helper): the raw bandwidth sum predicts roughly three
times the speed stock llama.cpp really reaches there (about 26-30 tok/s at 187k context), and 0.30 brings the
prediction to about 30. They are
conservative, and they are meant for *choosing and ranking*. A prediction is shown as a range (0.6x to 1.35x of the
mid value) and as `low`/`medium` confidence (`low` when bandwidth was not measured, on multi-NUMA machines and on
unified memory). Calibration on the user's machine is what makes the final decision; if it disagrees strongly, a more
modest choice is made and the user is told why.

## The fork tier (NVIDIA + Linux)

Our CUDA fork adds three things that change both what fits and how fast it runs: the `turbo3` KV cache (about 4 KB per
token instead of 10 KB for `q8_0`), an **expert cache** (the hottest experts of the layers that stay in RAM are kept in
VRAM) and MTP (speculative decoding, about x1.32). The planner therefore searches two numbers *together*: how many expert
layers stay in RAM (`--n-cpu-moe`) and how many cache slots the remaining VRAM pays for. They compete for the same bytes,
which is why "layers first, cache with what is left" is worse than the joint search.

```
seconds/token = ( scale x (GPU bytes / GPU bandwidth + RAM bytes / RAM bandwidth) + sync x CPU layers ) / MTP
                + KV read at depth (turbo3, measured 7.2 ms per token at 187k)
cache hit rate = slots / (slots + 12)
```

The constants are fitted to **one** machine and **anchored** on its measurements instead of trusting raw bandwidth:
ncmoe 26 + 24 slots reaches 54 tok/s near empty and 38.6-39.2 at 187k (`docs/BENCHMARKS.md`), 16 slots cost about 8%, and
"all experts in RAM + 80-110 slots" was measured *slower* (per-layer synchronisation), which the `sync` term reproduces.
VRAM is budgeted from the same log: 885 MiB of compute buffers and state, 400 MiB of CUDA context, 6 KB per token of
window (turbo3 KV plus the f16 KV of the MTP draft) and about 36 MiB per cache slot.
The search stays near what was measured: at most 24 slots, and a layout that keeps more layers in VRAM than the measured
machine (fewer than 20 in RAM) is marked *extrapolated* with lower confidence. The prediction only picks candidates;
calibration on your machine has the last word.

## Selection policy

Everything that fits is a *candidate*: every allowed quant, every window from 16k to the model's maximum, every KV cache type
that fits (and, on the fork tier, the best split of layers and cache slots). Each candidate gets **one score**; the best score
wins. There are no thresholds that flip a plan for one token per second: an earlier version had a ladder of speed rungs, and
a prediction of 34.0 tok/s against a target of 35 cost a machine 70,000 tokens of window.

```
score = quality(quant)^1.0  x  quality(KV type)  x  (window / model maximum)^0.4  x  speed credit^0.3 (shape below)
```

- **Reserves** (never touched): RAM `max(6 GiB, 15%)` for the system and desktop; VRAM what the desktop already uses plus
  0.8 GiB (0.3 GiB on a headless GPU).
- **Quality of the quant** is a prior from its bits per weight (it falls off quickly below about 4 bits), plus 3% for the
  catalog's tested default. **Quality of the KV cache:** `q8_0` 1.0, `turbo3` 0.985, `q4_0` 0.975. The quant is never below the
  catalog's `min_quant` (Q3_K_XL for Tiel-Coder) unless you force one.
- **The window** is worth a diminishing amount: 65k scores 0.61, 131k 0.76, 200k 0.90, 262k 1.0 (before speed).
- **Speed** is judged with the window half full. It counts fully up to 1.3x the target (35 tok/s), because predictions are only
  good to +-40% and a margin is worth having; far above that it is a windfall and counts for almost nothing. Below the
  target the slope is soft, below 20 tok/s steeper, and a candidate under 15 tok/s is only taken when nothing faster exists.
- **A window nobody has run** (beyond the one a model was verified at: 200k for Tiel-Coder on the fork) is taken only with
  25% of the VRAM budget free and 15% more speed than the target. A 12 GB card stays at 200k; a faster 16 GB card gets 262k.
- **KV cache type** is just another candidate, and where TurboQuant is not available the best substitutes are `q8_0`
  (near lossless, 0.53 of f16), **`q5_0`** (5.5 bits, 0.34: about 1% quality for a third of the memory, usually the best
  trade) and `q4_0` (0.28, the smallest the flash-attention kernels everywhere support). The CUDA and ROCm prebuilts only
  run flash attention with `q8_0` or `q4_0` pairs, so `q5_0` may not start there: the installer then retries with `q8_0` /
  `q4_0` by itself. The quality figures are priors from general experience, not our measurements.
- **Model**: among the models whose best candidate reaches the comfortable speed (20 tok/s) the most capable wins; if none
  does, among those that reach the minimum; if none does, the fastest that fits, with a warning. If nothing fits, the plan is
  a refusal with the reason.
- **Backends.** NVIDIA: CUDA, then Vulkan. AMD: ROCm and Vulkan. Intel: SYCL, Vulkan and OpenVINO. Apple: Metal. No GPU:
  CPU. With more than one candidate the installer measures them.

**Profiles.** The weights are exponents, so a different priority is a different weight, not different code:
`--optimize balanced` (default), `fast` (short window, speed counts more), `long` (the biggest window), `quality` (the quant
counts more). `shura check` shows what each would choose on your machine.

**Closed loop.** The prediction is only a prediction. After the test launch the installer compares the measured speed with
the predicted one; if they differ by more than 15%, it plans again with all predictions scaled by the measured/predicted ratio
(same model file, nothing is downloaded), starts the new settings and keeps them only if they really work.

What the planner gives for described machines (Tiel-Coder, predictions, not promises; `tests/fixtures/hardware/`):

| Machine | Plan |
|---|---|
| RTX 4070 SUPER 12 GB + 32 GB | fork: IQ4_XS, 200k, 27 layers in RAM + 24 slots; 44 tok/s at 100k filled, 37 full |
| 12 GB + 16 GB RAM | fork: Q3_K_XL (IQ4_XS does not fit the RAM after the reserve), 200k; about 41 at 100k filled |
| RTX 4070 Ti SUPER 16 GB + 32 GB | fork: IQ4_XS, 262k; about 47 at 131k filled |
| RTX 4060 8 GB + 32 GB | fork: IQ4_XS, 200k, 27 tok/s: 35 is out of reach on this bandwidth, the window is kept |
| RTX 4090 24 GB + 64 GB | fork: Q5_K_XL, 262k; about 60 |
| Radeon RX 9070 16 GB + 48 GB DDR4 | standard (upstream llama.cpp): IQ4_XS, 262k with `q4_0` KV (`q8_0` would cost speed), MTP draft on; about 41 at 131k filled |
| Dual EPYC 256 GB, CPU only | standard: IQ4_XS, full 262k window with `q4_0` KV, about 25 at 131k filled |

The thresholds are in `DEFAULTS` and can be overridden (`config=`), so a hardware report can argue for a different
value with data.

## What the plan contains

`ok`, `model`, `quant`, `tier` (`fork`/`standard`), `speed_rung` (`target`/`comfort`/`minimum`/`none`), `backend_candidates`,
`needs_probe`, `mode` (`cpu`/`hybrid`/`gpu`/`unified`), `settings` (context, KV type, `n_cpu_moe`, `moe_cache_slots`, GPU layers, threads, `numa`), `predicted_tok_s` (`low`/`mid`/`high`), `confidence`,
`memory` (needed and available RAM and VRAM), `reasons`, `warnings` and `alternatives`. Settings are *intents*
(what we want); the launcher turns them into flags for the chosen llama.cpp build after checking what that build supports.

## Limits, stated plainly

- NVIDIA-specific speed-ups (the MoE expert cache, `turbo3` KV, MTP, the VRAM guard) live in our CUDA fork (Linux only);
  other backends use upstream llama.cpp and the standard tier.
- Several GPUs: the largest one is planned for; a multi-GPU split is not planned yet.
- The efficiency constants come from one machine. Expect the prediction to be wrong by a factor on other hardware,
  more on dual-socket servers. That is why calibration exists and why reports matter.

## Status of each platform

Same table as the [README](../README.md#what-is-verified-and-what-is-not). Only the first row has ever run for real.

| Platform | Installed by | State |
|---|---|---|
| NVIDIA, Linux, 12 GB VRAM + 32 GB RAM (our CUDA fork) | `setup.sh` | **Verified** on one machine (RTX 4070 SUPER, Fedora 44) |
| NVIDIA, Linux, other sizes | `setup.sh` | Unverified: the fork-tier speed model is fitted to that one machine |
| AMD (ROCm, Vulkan), Intel (SYCL, Vulkan, OpenVINO), Apple (Metal), CPU-only servers | `shura install` | Unverified on real hardware; planner and install flow unit-tested on described machines and a fake server; the CPU build of llama.cpp is downloaded and run in CI |
| Windows | `shura install` | Unverified; unit tests and CI runners only (the gateway and ShuraCode setup still target Linux) |

A class becomes *verified* when a hardware report with a real measurement arrives: the gap between the predicted and the
measured speed is what corrects the constants above.

## Adding your hardware or a model

1. Run `shura report` (coming with the installer) and open a *hardware report* issue, or
2. add a profile to `tests/fixtures/hardware/` (copy one and change the numbers) and, if the planner's choice for it is
   wrong, say what you measured; or
3. add a model to `catalog/models.json` (`python3 -m unittest tests.test_universal` validates the schema).
