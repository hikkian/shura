# Benchmarks

All numbers were measured on one machine, in one continuous session, with the scripts in
[`bench/`](../bench). Nothing here is copied from other people's reports.

## Test system

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 5 5600 (6C/12T) |
| RAM | 32 GB DDR4-3200 (8 GB zram swap) |
| GPU | NVIDIA RTX 4070 SUPER, 12 GB VRAM (Ada, sm_89) |
| SSD | Kingston KC3000 2 TB NVMe |
| OS | Fedora 44, kernel 7.2, NVIDIA driver 615.71 (RPM Fusion), CUDA 13.4 |
| Model | [Tiel-Coder-35B-A3B-MTP](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP) `UD-IQ4_XS` (18.1 GB), `qwen35moe` arch, 256 experts / 8 active |

The model (18 GB) does not fit in 12 GB of VRAM, so the MoE expert weights of the first 26 layers
live in system RAM (`--n-cpu-moe 26`). Everything below is about making that split fast.

## Methodology

- **Context:** ~187,000 tokens of real C/C++/CUDA source (llama.cpp's own code), built with
  [`make_context.py`](../bench/make_context.py). Server context size is 200,000.
- **Request:** OpenAI `/v1/chat/completions`, "write a short function to reverse a linked list",
  `max_tokens=200` with `ignore_eos` (every run decodes exactly 200 tokens), temperature 0.6.
- **Runs:** 5 consecutive identical requests ([`deep_context_bench.py`](../bench/deep_context_bench.py)).
  Request 1 prefills the whole context; requests 2-5 reuse the cached prefix and measure pure decode.
  Reported "decode" is llama-server's own `predicted_per_second`.
- **Speculative decoding:** the model's built-in MTP head (`--spec-type draft-mtp`, `n_max 2`,
  `p_min 0.2`) is on in every run; draft acceptance was 75-84%.

## Results at real ~187k context

| # | Configuration | Decode tok/s (5 requests) | Mean |
|---|---|---|---:|
| 1 | Stock llama.cpp `2145525`, default mmap, q4_0 KV | 24.27 · 26.72 · 23.77 · 28.32 · 27.59 | 26.1 |
| 2 | Stock llama.cpp, `--load-mode none` (no mmap) | 31.16 · 32.28 · 28.45 · 28.51 · 27.87 | 29.7 |
| 3 | perf fork + MoE expert cache, 10 slots, q4_0 KV | 30.13 · 36.47 · 32.16 · 35.14 · 36.22 | 34.0 |
| 4 | perf fork + MoE expert cache, 16 slots, q4_0 KV | 36.97 · 35.41 · 35.87 · 34.57 · 36.73 | 35.9 |
| 5 | #4 + `GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1` | 21.53 · 25.56 · 24.17 · 23.04 · 28.79 | 24.6 |
| 6 | perf fork + TurboQuant `turbo3` KV (patched), 16 slots | 35.68 · 34.12 · 39.51 · 35.68 · 32.98 | 35.6 |
| 7 | turbo3, 22 slots | 36.25 · 37.34 · 38.49 · 39.79 · 35.17 | 37.4 |
| **8** | **turbo3, 24 slots — final config** | **40.57 · 40.13 · 37.16 · 36.96 · 36.90** | **38.3** |
| 9 | turbo3, 28 slots | 39.21 · 38.25 · 36.88 · 30.89 · 30.60 | 35.2 |
| 10 | [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) `cdf232c`, no mmap, q4_0 KV | 34.72 · 35.00 · 34.53 · 35.30 · 35.26 | 35.0 |

"perf fork" = [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp) branch `perf` at `27c54b4`.
All rows except #1 use `--load-mode none`.

### What the table says

- **No-mmap wins on Linux too** (+14% decode, and prefill 212 s vs 322 s). It also used *less* RAM:
  12.5 GB RSS vs 15.1 GB with mmap.
- **The MoE expert cache is the single biggest lever.** Moving to the perf fork with its expert cache
  (the most-routed experts of the CPU-resident layers stay in VRAM) took decode from 29.7 to 35.9 tok/s
  and removed most of the run-to-run variance. Rows 3 → 4 isolate the cache itself: going from 10 to
  16 slots on the same build gave +5.6%.
- **Slot count is capped by VRAM, and too many slots fail silently.** With q4_0 KV, 25 slots failed at
  load and 20 slots loaded fine but crashed mid-generation (CUDA OOM in a transient compute buffer at
  full context). With turbo3, 28 slots loaded but decode degraded run over run — the classic sign of
  VRAM oversubscription. 24 is the sweet spot right before that cliff.
- **TurboQuant does not speed up decode by itself** (#6 ≈ #4). Its value is VRAM: the smaller KV cache
  freed enough memory to go from 16 to 24 expert-cache slots, and *that* produced the gain (#8).
  It only works after [the fix in `patches/`](TURBOQUANT-FIX.md).
- **Host-register + expert prefetch env vars hurt** when the expert cache is on (#5); do not combine them.
- **ik_llama.cpp** has excellent raw CPU MoE kernels (35.0 with no expert cache vs 29.7 for stock),
  but no expert-residency feature, and its prefill at this depth was 2.8x slower (587 s vs 212 s).
  Using mmap + `--prefetch-experts` made it worse (prefill > 600 s, near-OOM page cache).

### VRAM (12 GB card, ~0.7 GB taken by the desktop)

| Config | VRAM used |
|---|---:|
| q4_0 KV, 10 expert slots | 11.57 GB |
| q4_0 KV, 16 slots | 11.85 GB |
| turbo3 KV, 16 slots | 11.49 GB |
| turbo3 KV, 22 slots | 11.59 GB |
| turbo3 KV, 28 slots | 11.83 GB (degrades) |

## Decode speed vs. context depth

MoE cache 16 slots, q4_0 KV, no mmap ([`depth_sweep.py`](../bench/depth_sweep.py)):

| Prompt tokens | Decode tok/s |
|---:|---:|
| 4,203 | 53.35 |
| 16,762 | 45.38 |
| 67,439 | 41.14 |
| 123,064 | 34.76 |
| 160,631 | 31.81 |
| 187,066 | 34.82 |

A smooth decline with no cliff. This rules out the upstream qwen35-hybrid decode-collapse bug
([ggml-org/llama.cpp#27623](https://github.com/ggml-org/llama.cpp/issues/27623), reported for a dense
Qwen3.8-27B hybrid); the A3B MoE variant is not affected.

**Why published numbers differ so much.** At near-empty context the same stack does 52-60 tok/s
(stock llama.cpp: 42-51). Many "70+ tok/s" reports for this model family are measured at 32-64k
context or with an empty prompt. Attention over a 187k-token KV cache is real work that no
expert-cache trick removes, so always compare at the same depth.

### Final configuration, filled context (2026-09-27)

Live settings (ncmoe 26, 24 cache slots, turbo3 KV, MTP n_max 1, 200k context). The context is filled
for real and restored from a slot before every run. Each value is the median of 3–5 runs of 200 tokens.

| Filled context | Decode tok/s | Desktop lag p99 (fill / decode) |
|---:|---:|---:|
| 4k | 53.9 | — |
| 16k | 53.2 | — |
| 32k | 50.8 | 0.12 ms |
| 64k | 49.4–51.5 | 0.10 ms |
| 128k | 42.4 | 0.12 ms |
| 187k | **38.6–39.2** | 0.14 / 0.10 ms |

- Run-to-run spread between sessions is about ±5–8%, driven by background desktop activity. That is
  why comparisons are only made within one session and interleaved (A, B, A, B).
- A 10 ms sleep probe measured desktop latency during both the 187k fill and generation: it stays at
  idle level (idle p99 ≈ 0.1–0.2 ms).
- Loading the 18 GB model sometimes causes one ~0.2 s stall while the kernel reclaims memory. This
  happens only at load time and not on every load.

## VRAM budget: fitting a 1.5 GB desktop reserve

Goal: keep 1.5 GB of the 12 GB card for the desktop. The model's own peak (measured per process, not
from the card's free memory) must then stay ≤ 12282 − 1536 = 10,746 MiB. The live config peaks at
**10,970 MiB**.

Where it goes, from the verbose load log:
- weights and expert cache on the GPU: ~7.5 GB plus the cache;
- main KV cache (turbo3, 10 attention layers): 764 MiB;
- main compute buffer: 770 MiB;
- recurrent state: 126 MiB;
- **MTP draft KV cache (f16): 391 MiB**;
- draft compute buffer: 236 MiB.

Everything below was measured in one session (decode at filled context; quality checked with the 9
correctness probes):

| Change | Model peak | Decode | Verdict |
|---|---:|---|---|
| none (live) | 10,970 MiB | 39.2 @187k, 50.9 @64k | 224 MiB over budget |
| 100k context + 40 slots ("normal" mode) | 10,578 | 45.3 @99.8k vs 48.3 for the live config at the same depth | not faster, dropped |
| `-ub 384` / `-ub 256` | 10,838 / 10,714 | −13% / −16%, prefill −30% / −42% | rejected |
| draft KV `q8_0` (`-ctkd/-ctvd`) | OOM at load | — | quantized KV is expanded to a full f16 scratch buffer (MMA flash attention; see [llama.cpp#29371](https://github.com/ggml-org/llama.cpp/issues/29371)) |
| draft KV `turbo3` | 11,046 | same | no saving |
| `-cmoed` (draft MoE on CPU) | 10,970 | same | no effect: the draft uses target layer 40 |
| layer 40 (MTP) experts on CPU (`-ot`) + 24 slots | 10,638 | −7% @64k | fits |
| layer 40 experts on CPU + 26 slots | 10,712 | **−4%** @64k | fits, best full-budget option |
| 16 cache slots | 10,682 | −8% @187k | fits |

Takeaways:
- A smaller context does not make decode faster at the same depth.
- One cache slot costs about 36 MiB of VRAM.
- The cheapest fix was outside the model: turning off hardware acceleration in VS Code
  (`"disable-hardware-acceleration": true` in `argv.json`) removes its ~180–210 MiB GPU process. The
  desktop then idles at ~0.5–0.6 GB, and the live config keeps its full speed.

## Session resume (KV cache reuse)

Measured with [`session_cache_test.py`](../bench/session_cache_test.py) at ~187k tokens:

| Scenario | Tokens re-processed | Wall time |
|---|---:|---:|
| Cold prefill of the session | 187,038 | 264 s |
| Back to the session after two unrelated sessions (llama-server `--cache-ram`) | 19 | **1.9 s** |
| Model unloaded by the gateway (slot saved to disk) then reloaded + slot restored | 20 | **13 s** (10 s is model load) |

Slot file size: 64 MiB at an empty context, 781 MiB at 187k. Save 0.47 s, restore 0.22 s.
See [ARCHITECTURE.md](ARCHITECTURE.md#kv-cache-and-ssd-wear) for why the gateway only writes it
to disk on unload.

## Output correctness

After every kernel or KV-cache change, [`correctness_probes.py`](../bench/correctness_probes.py)
must pass 9/9 (simple arithmetic and facts). The final config (patched turbo3, 24 slots) passes 9/9,
and text + vision answers were spot-checked by hand.

## Speculative decoding and desktop-impact tuning

This workstation is also a desktop running a browser, an IDE and a messenger. So besides tok/s, each
config was checked for what it costs the desktop:

- **Card VRAM left free.** GPU-accelerated apps share the card; the desktop itself used ~760 MB.
- **Scheduler latency.** A probe thread measures how late a 10 ms sleep wakes up during generation.
  Its p99 is a proxy for UI jank; on an idle desktop it is 0.1-0.5 ms.

Method: a 187k-token session is saved once and restored before every request (~28 tokens
re-processed), so each config costs about a minute instead of a 4.5-minute prefill. MoE cache
24 slots, turbo3 KV, ncmoe 26 unless noted.

### MTP draft length

| `--spec-draft-n-max` / `p-min` | Decode tok/s | Draft acceptance | Card VRAM free |
|---|---:|---:|---:|
| no speculation | 28.5 | — | 1298 MB |
| 1 / 0.0 | 37.6 | 85% | 237 MB |
| **1 / 0.2** | **38.1** | 83% | 274 MB |
| 2 / 0.0 | 38.7 | 72% | 107 MB |
| 2 / 0.2 (previous default) | 37.6 | 70% | 121 MB |
| 3 / 0.0 | 36.8 | 58% | 52 MB |
| 3 / 0.2 | 37.9 | 60% | 103 MB |

- **MTP is worth +32%.** Draft lengths 1-3 are within run-to-run noise of each other. `n_max 1` has
  the highest acceptance and leaves the most VRAM, so it is the default now.
- The MTP draft context keeps its own **F16 KV cache for the full 200k window (~1.1 GB VRAM)**.
  Compressing it does not work here:
  - `--spec-draft-type-k/v q8_0` or `q4_0` → CUDA OOM (no flash-attention path for quantized draft KV
    at head_dim 256, so it falls back to dequantizing the whole cache);
  - `turbo3` runs, but used 78 MB *more* and crashed in one of two runs.

### CPU threads and priority

Interleaved, two passes each to cancel drift:

| Threads | Decode tok/s (pass 1 / 2) | Scheduler lag p99 |
|---|---|---:|
| 10 | 42.6 / 41.2 | 0.8-1.3 ms |
| 8 | 42.9 / 41.1 | 0.1-0.6 ms |
| **6 + `nice 10`** | 43.1 / 41.0 (6 without nice) | **0.1 ms** |

Decode is memory-bandwidth bound on the CPU side: beyond 6 threads (the physical core count) nothing
is gained, and 10 threads just compete with the desktop. `-tb 10` did not speed up prefill either
(761 t/s at 8k in both cases) but raised latency to 1.5 ms. `nice 10` costs nothing when the desktop is
idle and makes UI work win any contention.

### All experts on CPU + a large expert cache (`-ncmoe 99`)

This is the layout from the perf fork author's own videos: every layer's experts in RAM, the hottest
N per layer cached in VRAM. On routing traces it should serve more expert calls from VRAM at equal
VRAM (≈59% vs ≈47%). Measured:

| Layout | Decode tok/s | Prefill (8k) | RAM available | Lag p99 |
|---|---:|---:|---:|---:|
| **ncmoe 26 + 24 slots** | **39.9** | **761 t/s** | **12.6 GB** | **0.3 ms** |
| ncmoe 99 + 80 slots | 36.3 | 600 t/s | 7.3 GB | 1.2 ms |
| ncmoe 99 + 90 slots | 36.9 | 615 t/s | 7.4 GB | 1.2 ms |
| ncmoe 99 + 100 slots | 35.5 | 624 t/s | 7.3 GB | 1.8 ms |
| ncmoe 99 + 110 slots | 37.8 | 642 t/s | 7.4 GB | 1.7 ms |

It lost on every axis in this build: slower decode and prefill, 5 GB less free RAM, more desktop
latency. The likely reason is per-layer GPU↔CPU synchronisation. The fork author's peak numbers rely
on concurrent GPU/CPU execution chains, which do not appear to be in the public `perf` branch, and
were measured at a 64k window with a Q4_K_M quant.

### Final settings

`-t 6 -tb 6`, `nice 10`, `--spec-draft-n-max 1 --spec-draft-p-min 0.2`, ncmoe 26, 24 cache slots,
turbo3 KV, F16 draft KV, agentic routing profile (below). Decode ~40-43 tok/s at 187k, correctness
probes 9/9, and the desktop stays at idle-level latency during generation.

## MoE routing profile: which experts to cache

The expert cache keeps the most frequently routed experts of each CPU-resident layer in VRAM. The
choice comes from a routing profile. The original profile was traced from two short synthetic prompts,
about 1,000 generated tokens in total.

**Evaluation.** A realistic corpus of 10 agentic-coding scenarios was traced: the real OpenCode system
prompt with its 29 tool schemas, code files in Python/TS/Bash/JS/C++, a pytest traceback, a diff
review, a tool-call loop, and explanations in Russian. Profiles were scored leave-one-out: built on 9
scenarios, measured on the 10th. The score is the share of CPU-layer expert calls, during *generated*
tokens, that the top-24 cached experts would serve.

| Profile | Mean | Worst scenario |
|---|---:|---:|
| Original (2 synthetic prompts) | 24.7% | 15.6% |
| Realistic corpus, all tokens (prompt + generated) | 13.4% | 7.1% |
| Realistic corpus, generated tokens weighted ×5 | 21.6% | 12.4% |
| **Realistic corpus, generated tokens only + original traces** | **29.9%** | **22.7%** |

The key finding is that **the experts used while reading a prompt are not the experts used while
writing.** A profile dominated by long prompts (tool schemas, file contents) is worse for decode than
a tiny synthetic one. Profiles for decode speed must be built from generated tokens only.
[`scripts/capture-moe-trace.sh`](../scripts/capture-moe-trace.sh) does this and accepts a directory of
your own prompts.

**Decode A/B at 187k** (interleaved, restore method):

| Profile | Pass 1 | Pass 2 | Pass 3 | Mean |
|---|---:|---:|---:|---:|
| Original | 40.3 | 40.5 | — | 40.4 |
| Agentic (generated tokens) | 40.2 | 43.3 | 41.4 | **41.6** |

This is a small, noisy gain (+0-7%, about +3% on average), consistent with the ~7% reduction in CPU
expert calls. It costs no RAM or VRAM, was never slower, and is more robust on agentic scenarios. It
is the shipped profile: [`config/moe-trace/tiel-coder-agentic.csv`](../config/moe-trace/tiel-coder-agentic.csv)
(numbers only: token position, layer, expert IDs).

## Prefill experiments (none adopted)

Prefill of a fresh 8k prompt runs at ~760 t/s. Attempts to speed it up:

| Change | Prefill | Decode | Verdict |
|---|---:|---:|---|
| Baseline | 768 t/s | 41.4 | — |
| `GGML_CUDA_REGISTER_HOST=1` | 764 t/s | 41.5 | no effect |
| `GGML_SCHED_PREFETCH_EXPERTS=1` | 757 t/s | 40.3 | no effect |
| Both | 751 t/s | 39.6 | slightly worse |
| `-ub 1024` / `-ub 2048` | — | — | fails to load: needs +470 MB of compute buffer |

The published +64% prefill gains for these patches were measured with `-ub 2048`, on 2k-token prompts,
without a 200k window, MTP or an expert cache. On a 12 GB card that already holds 24 cache slots
there is no VRAM for the larger ubatch, and trading cache slots (decode, used constantly) for prefill
(once per new session) is a bad deal. Returning to an existing session is served from the RAM prompt
cache in seconds anyway.

## Previous Windows 11 baseline (same hardware)

The same model and flags on Windows 11 measured 37.4 → 29.7 → 24.7 → 25.1 → 24.6 tok/s over five
consecutive requests at ~180k context: it started fast and settled around 25. On Linux no run showed
that decay pattern.


## A2 slot checkpoint restore (2026-09-30)

At 187k tokens on the accepted A2 test layout (12 cache slots, 26 MoE layers), a slot with N=1 stored one 432,683,008-byte checkpoint appendix. The complete file was 1,273,555,436 bytes; save took 1.00 s and restore 0.89 s. A plain read of this tmpfs file took 0.23 s; FNV-1a 64-bit took 1.25 s, adding about 1.02 s for a 1.27 GB file. At one idle save per day this is about 465 GB/year; at three, about 1.39 TB/year.

The final deterministic, interleaved 187k N0/N1 block used the same prompt, seed 1, temperature 0 and 200 generated tokens. All six answers were token-identical, each run reused 187,187 cached tokens and evaluated 32 new tokens, and draft acceptance was 86/112 for both flags. `other_busy_cpus` medians were 1.477 (N0) and 1.300 (N1), so the block passed the 1.6 filter. Median speed was 34.856 tok/s (N0) and 34.844 tok/s (N1), −0.03%; process peak was 10,486 MiB in every run and card-free minimum was 616 MiB.

Three real ShuraCode turns passed with N=1 and three with N=2 in isolated temporary projects. Rewritten-turn restore evaluated 44 new prompt tokens for both, after restoring one or two checkpoints respectively; full request body is in the local, ignored result bundle. Corrupt hash, truncated appendix, and foreign-version files logged a safe legacy restore. Killing the test server during a gateway save left the prior final file byte-for-byte intact, removed the pending file, and the prior slot restored. Final `correctness_probes.py` passed 9/9. `--ctx-checkpoints 4` and `6` had the same 3 live checkpoints and 972,577,372 checkpoint bytes at 187k; neither setting showed a useful RAM reduction.

A corrected completed-dialog quality check produced semantically equivalent but non-identical wording between full prefill and restore (`cache_n=187216`, `prompt_n=27` after restore versus 187,243-token full prefill). The N0/N1 interleaved outputs themselves remained exactly identical. The phrasing difference is recorded as a likely floating-point path variation; no contradictory answer appeared.

Two safety stops are retained: one low-free-VRAM stop coincided with Brave PID 4003 using 589–637 MiB, and the first fault attempt tripped a zram baseline/phase issue during server startup; the corrected retries passed. Earlier speed blocks with `other_busy_cpus` above 1.6 were excluded. One incomplete assistant-turn quality probe was also discarded as a harness error. No user process was terminated.

The NVMe written-sector counter changed from 9,487,950,848 to 44,993,583,104 bytes over the full A2 work window (delta 35,505,632,256 bytes). Test builds, slots, corpora and logs were kept in tmpfs; the historical counter cannot identify which process wrote the difference, so attribution remains unknown and is not claimed as test output. Compact local evidence, exact launch argv and the captured ShuraCode request are in `bench/results/elastic-vram-a2-20260930/` (ignored, not committed).


## Elastic VRAM Part A guardian acceptance (2026-09-30)

The opt-in gateway guardian passed an isolated 187k pressure/save/restore test. The test gateway used ports8081/8091, 12 MoE slots, checkpoint N=1 and a pressure threshold overridden only in its temporary config. The 187177-token session was saved in RAM, then unloaded after the answer; restore reused187180 tokens and evaluated29 new tokens. The slot measured1273512736 bytes, including a432670720-byte checkpoint appendix. The test-only setting does not change the live config; shipped defaults are `vramGuard=false` and `slotSaveCheckpoints=0`.

The isolated server recovered after its PID was killed, and a restarted gateway recovered its lease and answered a request. Final correctness was9/9; 83 unit tests, compileall, ShellCheck and diff checks passed. Peak model process was10532MiB, minimum card free569MiB, p99 UI lag0.076ms, maximum lag3.36ms, GPU73°C and CPU Tctl52.5°C. Disk swap stayed0 and no zram stop condition occurred. A 1GiB reserve initially rejected the save safely; a512MiB extra reservation in the prototype/example passed while retaining the configured1.5GiB free-RAM floor.

A runner race and one continuation missing a middle exchange were classified as harness issues; the latter's `cache_n=0` was excluded from product results. Two transient NVML failures were recorded and recovered through the safe fallback. NVMe field7 increased359383040 bytes system-wide over the phase; attribution is unavailable and the amount is below2GB. No live gateway/config/binary was changed. Machine-readable evidence: [`summary.json`](../bench/results/elastic-vram-part-a-20260930/summary.json) (local ignored result).
