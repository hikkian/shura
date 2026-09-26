# TurboQuant on head_dim-256 models: crash + silent corruption, and the fix

[TurboQuant](https://arxiv.org/abs/2504.19874) KV-cache quantization (`-ctk/-ctv turbo2|turbo3|turbo4`)
ships in the [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp) `perf` branch. On
Tiel-Coder / other `qwen35moe` models it is unusable out of the box.

## Symptom

The server aborts during the warmup decode, before serving anything:

```
ggml/src/ggml-cuda/template-instances/../fattn-common.cuh:1450: GGML_ASSERT(n_kv_max > 0) failed
  ... ggml_cuda_flash_attn_ext_mma_turbo_case<256, 256, 2, 8, (ggml_type)44, (ggml_type)44>
  ... launch_fattn<256, 2, 8>
```

## Root cause

Two independent bugs in the fused tensor-core decode path added by PR #4 of the fork:

1. **Crash.** `ggml_cuda_flash_attn_ext_mma_turbo_case` in `fattn-mma-turbo.cuh` calls
   `launch_fattn(...)` without the `use_sparse` argument. The following `warp_size_host` (32) lands in
   the `use_sparse` slot, which corrupts the `n_kv_max` computation to 0 and trips the assert.
2. **Silent wrong answers.** The turbo tile loaders in `fattn-mma-f16.cuh` ignore the shared-memory
   XOR swizzle that the kernel enables for head_dim 256. Attention reads scrambled tiles, so even
   with the crash patched out the model produces fluent but wrong output. The fix author measured
   0/9 on arithmetic probes before the fix.

The fork author's only validation was a single "capital of France" check, which a partially broken
kernel can still pass. That is why [`bench/correctness_probes.py`](../bench/correctness_probes.py)
exists.

## Fix

[`patches/turboquant-pr12-fix.patch`](../patches/turboquant-pr12-fix.patch) is
[thecodacus/llama.cpp#12](https://github.com/thecodacus/llama.cpp/pull/12) by **wilky2005**, which was
still unmerged when this was written. It passes the missing argument and threads the swizzle flag
into the turbo2/3/4 tile loaders. [`scripts/build-llama.sh`](../scripts/build-llama.sh) applies it
automatically.

## Validation on Tiel-Coder

| Check | Result |
|---|---|
| Server starts with `turbo3` K/V | ✅ |
| `correctness_probes.py` | 9/9 |
| Vision (mmproj) answers, long-context answers | spot-checked correct |
| Decode at 187k, 16 expert slots | 35.6 tok/s (≈ q4_0's 35.9) |
| VRAM freed vs q4_0 | enough for 24 instead of 16 expert slots → **38.3 tok/s** |

TurboQuant is not a speedup by itself here. It turns KV-cache bytes into expert-cache slots.
