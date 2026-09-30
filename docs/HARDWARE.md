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
tok/s          = efficiency / time per token
```

- **MoE on a discrete GPU** keeps attention, shared experts and the KV cache in VRAM and fills what is left with whole
  expert layers (`--n-cpu-moe` = how many layers' experts stay in RAM). The routed experts read per token are split by
  the share of layers on the GPU.
- **Dense on a discrete GPU** splits layers the same way the bandwidth times add up.
- **Unified memory** (Apple, APUs) is one pool, limited to 70% of RAM for the GPU.
- **CPU-only** reads everything from RAM. CPU-only does **not** mean small models: a dual-socket server with
  150-250 GB/s runs models that no 12 GB card can hold; the planner only asks what fits and how fast it would be.

**Efficiency constants** (`installer/universal/planner.py`, `DEFAULTS["efficiency"]`): hybrid GPU+CPU MoE 0.30, GPU only 0.50,
CPU only 0.40, unified 0.45. They absorb everything the pure bandwidth sum ignores: kernel launches, per-layer
synchronisation, dequantisation, attention over a long context. **They were calibrated on one machine** (RTX 4070
SUPER, DDR4-3200, about 40 GB/s as measured by the shipped helper): the raw bandwidth sum predicts roughly three
times the speed stock llama.cpp really reaches there (about 26-30 tok/s at 187k context), and 0.30 brings the
prediction to about 30. They are
conservative, and they are meant for *choosing and ranking*. A prediction is shown as a range (0.6x to 1.35x of the
mid value) and as `low`/`medium` confidence (`low` when bandwidth was not measured, on multi-NUMA machines and on
unified memory). Calibration on the user's machine is what makes the final decision; if it disagrees strongly, a more
modest choice is made and the user is told why.

## Selection policy

1. **Reserves.** RAM: `max(4 GiB, 15%)` stays free for the system and desktop. VRAM: 1.5 GiB stays free on a GPU that
   drives a display (0.5 GiB on a headless one), on top of what the desktop already uses.
2. **Context.** Among the contexts that fit (16k to 256k, KV cache `q8_0`, then `q4_0` if tight) the planner takes the
   largest one that keeps at least 92% of the best predicted speed, but never above a default cap per mode (CPU-only
   32k, GPU+RAM 64k, GPU only and unified memory 128k): long contexts slow generation in ways the speed model cannot
   see, so a bigger window is something you ask for, not something you get by default. (Our tuned CUDA fork runs 200k.)
3. **Quant.** Each model has a tested `default_quant`. If it fits and reaches the comfortable speed (20 tok/s) it is taken,
   and a higher-quality quant is taken only if it still reaches 1.5x that speed. If the default is too slow or does not
   fit, the next smaller quants are tried, first for comfortable speed and then for the 15 tok/s minimum.
4. **Model.** Among the models that reach the comfortable speed, the most capable one wins; if none does, among those
   that reach the minimum; if none does, the fastest that fits, with a warning. If nothing fits, the plan is a refusal
   with the reason (for example, how much RAM is missing).
5. **Backends.** NVIDIA: CUDA, then Vulkan. AMD: ROCm and Vulkan. Intel: SYCL, Vulkan and OpenVINO. Apple: Metal. No GPU:
   CPU. When there is more than one candidate the installer measures them.

The thresholds are in `DEFAULTS` and can be overridden (`config=`), so a hardware report can argue for a different
value with data.

## What the plan contains

`ok`, `model`, `quant`, `backend_candidates`, `needs_probe`, `mode` (`cpu`/`hybrid`/`gpu`/`unified`), `settings`
(context, KV type, `n_cpu_moe` or GPU layers, threads, `numa`), `predicted_tok_s` (`low`/`mid`/`high`), `confidence`,
`memory` (needed and available RAM and VRAM), `reasons`, `warnings` and `alternatives`. Settings are *intents*
(what we want); the launcher turns them into flags for the chosen llama.cpp build after checking what that build supports.

## Limits, stated plainly

- NVIDIA-specific speed-ups (the MoE expert cache, `turbo3` KV, the VRAM guard) live in our CUDA fork and are an
  optional upgrade on top of this plan; other backends use upstream llama.cpp.
- Several GPUs: the largest one is planned for; a multi-GPU split is not planned yet.
- The efficiency constants come from one machine. Expect the prediction to be wrong by a factor on other hardware,
  more on dual-socket servers. That is why calibration exists and why reports matter.

## Status of each platform

| Platform | State |
|---|---|
| NVIDIA, Linux (our CUDA fork) | **Verified** on one machine (RTX 4070 SUPER, Fedora 44) |
| NVIDIA, Linux, upstream build | Planner tested on described machines; awaiting reports |
| AMD (ROCm, Vulkan), Intel (SYCL, Vulkan, OpenVINO), Apple (Metal), CPU-only servers | Planner tested on described machines; awaiting reports |
| Windows, macOS | Detection code is tested on fixtures only; the gateway still targets Linux (WSL2 is the path on Windows) |

## Adding your hardware or a model

1. Run `shura report` (coming with the installer) and open a *hardware report* issue, or
2. add a profile to `tests/fixtures/hardware/` (copy one and change the numbers) and, if the planner's choice for it is
   wrong, say what you measured; or
3. add a model to `catalog/models.json` (`python3 -m unittest tests.test_universal` validates the schema).
