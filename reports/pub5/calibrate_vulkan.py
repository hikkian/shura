"""PUB5: fit the Vulkan speed prediction on open llama-bench data (tg128 = decode tokens/s).

Efficiency = measured tg128 x bytes read per token / peak memory bandwidth. Bytes per token are the whole file for a dense
model and the active experts (plus shared weights) for an MoE. Run: python3 calibrate_vulkan.py [out_dir]
Standard library only. Every row carries its source; rows with a known problem stay in the CSV with `use` = 0.
"""
import csv
import os
import statistics
import sys

DENSE_BYTES = 3.83e9        # llama-2-7b.Q4_0.gguf, about 3.83 GB (size of the same architecture's Q4_0 file, see report)
MOE_ACTIVE_BYTES = 2.0e9    # Qwen3-30B-A3B Q4_K_M: 3.3 B active params x about 4.9 bits/weight ~ 2.0 GB (estimate, +-10%)

SCOREBOARD = "https://github.com/ggml-org/llama.cpp/discussions/10879"       # Vulkan scoreboard, Llama 2 7B Q4_0
THREAD_15021 = "https://github.com/ggml-org/llama.cpp/discussions/15021"    # ROCm/HIP thread, comments carry Vulkan rows
RADXA = "https://forum.radxa.com/t/llama-cpp-benchmarks/27813/17"            # volyrique post, Qwen3-30B-A3B Vulkan

# (gpu, family, vram_gb, peak_gbs, peak_source, model, quant, fa, tg128, source, use, note)
# peak bandwidth: manufacturer specs. Only the RX 7800 XT value was checked against a search result in this session.
ROWS = [
    # scoreboard, Llama 2 7B Q4_0, no flash attention (commits differ per row; see the CSV)
    ("RX 7900 XTX", "RDNA3", 24, 960, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 182.63, SCOREBOARD, 1, ""),
    ("RX 7900 XT", "RDNA3", 20, 800, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 123.18, SCOREBOARD, 1, ""),
    ("RX 7800 XT", "RDNA3", 16, 624, "search: 256-bit x 19.5 Gbps", "llama2-7b", "Q4_0", "no", 118.27, SCOREBOARD, 1, ""),
    ("RX 7900 GRE", "RDNA3", 16, 576, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 116.11, SCOREBOARD, 1, ""),
    ("RX 7600", "RDNA3", 8, 288, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 58.03, SCOREBOARD, 1, ""),
    ("RX 7600 XT", "RDNA3", 16, 288, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 53.02, SCOREBOARD, 1, ""),
    ("RX 6900 XT", "RDNA2", 16, 512, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 108.00, SCOREBOARD, 1, "Infinity Cache"),
    ("RX 6800 XT", "RDNA2", 16, 512, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 100.32, SCOREBOARD, 1, "Infinity Cache"),
    ("RX 6800", "RDNA2", 16, 512, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 101.47, SCOREBOARD, 1, "Infinity Cache"),
    ("RX 6750 XT", "RDNA2", 12, 432, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 81.98, SCOREBOARD, 1, "Infinity Cache"),
    ("RX 6700 XT", "RDNA2", 12, 384, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 83.88, SCOREBOARD, 1, "Infinity Cache"),
    ("RX 6650 XT", "RDNA2", 8, 280, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 62.14, SCOREBOARD, 1, "Infinity Cache"),
    ("RX 6600 XT", "RDNA2", 8, 256, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 53.92, SCOREBOARD, 1, "Infinity Cache"),
    ("RX 6600", "RDNA2", 8, 224, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 50.57, SCOREBOARD, 1, "Infinity Cache"),
    ("RTX 4090", "NVIDIA", 24, 1008, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 187.97, SCOREBOARD, 1, "coopmat2"),
    ("RTX 4080 Super", "NVIDIA", 16, 736, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 147.13, SCOREBOARD, 1, "coopmat2"),
    ("RTX 4070 Ti Super", "NVIDIA", 16, 672, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 129.45, SCOREBOARD, 1, "coopmat2"),
    ("RTX 4070 Ti", "NVIDIA", 12, 504, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 110.53, SCOREBOARD, 1, "coopmat2"),
    ("RTX 4070 Super", "NVIDIA", 12, 504, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 108.74, SCOREBOARD, 1, "coopmat2"),
    ("RTX 4070", "NVIDIA", 12, 504, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 92.29, SCOREBOARD, 1, ""),
    ("RTX 3090", "NVIDIA", 24, 936, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 164.05, SCOREBOARD, 1, "coopmat2"),
    ("RTX 3080", "NVIDIA", 10, 760, "spec for the 10 GB card; the row does not say which", "llama2-7b", "Q4_0", "no", 139.15, SCOREBOARD, 0, "VRAM variant unknown"),
    ("RTX 3070", "NVIDIA", 8, 448, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 78.71, SCOREBOARD, 1, ""),
    ("RTX 3060 Ti", "NVIDIA", 8, 448, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 86.74, SCOREBOARD, 1, "coopmat2"),
    ("RTX 3060", "NVIDIA", 12, 360, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 75.94, SCOREBOARD, 1, "coopmat2"),
    ("Arc A770", "Intel", 16, 560, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 52.56, SCOREBOARD, 1, "Intel Arc: lower efficiency"),
    ("Arc A750", "Intel", 8, 512, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 42.66, SCOREBOARD, 1, "Intel Arc: lower efficiency"),
    ("Arc B580", "Intel", 12, 456, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 70.14, SCOREBOARD, 1, "Intel Arc: lower efficiency"),
    ("Arc B570", "Intel", 10, 380, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 49.64, SCOREBOARD, 1, "Intel Arc: lower efficiency"),
    # the same scoreboard with flash attention on (fa column), RDNA3 and NVIDIA only: a check on the flash-attention effect
    ("RX 7800 XT", "RDNA3", 16, 624, "search: 256-bit x 19.5 Gbps", "llama2-7b", "Q4_0", "yes", 124.86, SCOREBOARD, 1, ""),
    ("RX 7900 XTX", "RDNA3", 24, 960, "spec, not re-checked", "llama2-7b", "Q4_0", "yes", 190.92, SCOREBOARD, 1, ""),
    ("RTX 4090", "NVIDIA", 24, 1008, "spec, not re-checked", "llama2-7b", "Q4_0", "yes", 190.10, SCOREBOARD, 1, "coopmat2"),
    # comments in the ROCm/HIP thread: the same cards on Vulkan, other builds and dates (spread check)
    ("RX 7800 XT", "RDNA3", 16, 624, "search: 256-bit x 19.5 Gbps", "llama2-7b", "Q4_0", "no", 96.89, THREAD_15021, 1, "comment, other build"),
    ("RX 7600 XT", "RDNA3", 16, 288, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 52.84, THREAD_15021, 1, "comment, other build"),
    ("RX 6800 XT", "RDNA2", 16, 512, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 91.59, SCOREBOARD, 1, "comment, Jan 2025 build"),
    ("RX 7900 XTX", "RDNA3", 24, 960, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 136.24, SCOREBOARD, 1, "comment, Jan 2025 build, one device"),
    ("RTX 3090", "NVIDIA", 24, 936, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 123.72, SCOREBOARD, 1, "comment, Jan 2025 build"),
    # MoE, Qwen3-30B-A3B Q4_K_M (active ~2.0 GB per token)
    ("RX 7900 XTX", "RDNA3", 24, 960, "spec, not re-checked", "qwen3-30b-a3b", "Q4_K_M", "no", 57.64, RADXA, 1, "one card, one build; see report"),
    ("RX 7900 XTX", "RDNA3", 24, 960, "spec, not re-checked", "qwen3-30b-a3b", "Q4_K_M", "no", 61.92, RADXA, 1, "taskset variant, -t 8"),
    ("RX 7900 XTX", "RDNA3", 24, 960, "spec, not re-checked", "qwen3-30b-a3b", "Q4_K_M", "no", 62.28, RADXA, 1, "taskset variant, -t 1"),
    # excluded: the same post reports this card's dense Llama 2 7B at 99.8 (fa 0) and 47.5 (fa 1) tok/s, half of the scoreboard
    ("RX 7900 XTX", "RDNA3", 24, 960, "spec, not re-checked", "llama2-7b", "Q4_0", "no", 99.82, RADXA, 0,
     "same post: contradicts the scoreboard (182.6) on the same card; build or settings differ"),
]


def efficiency(tg, bytes_per_token, peak_gbs):
    return tg * bytes_per_token / (peak_gbs * 1e9)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for gpu, fam, vram, peak, peak_src, model, quant, fa, tg, src, use, note in ROWS:
        moe = model.startswith("qwen3")
        b = MOE_ACTIVE_BYTES if moe else DENSE_BYTES
        rows.append({"gpu": gpu, "family": fam, "vram_gb": vram, "peak_gbs": peak, "peak_source": peak_src,
                     "model": model, "quant": quant, "flash_attn": fa, "bytes_per_token_gb": round(b / 1e9, 2),
                     "tg128_tok_s": tg, "eff": round(efficiency(tg, b, peak), 3), "source": src, "use": use, "note": note})
    with open(os.path.join(out_dir, "vulkan_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    dense = [r for r in rows if r["use"] and r["model"] == "llama2-7b" and r["flash_attn"] == "no"]
    print("dense Llama 2 7B Q4_0, Vulkan, no FA (use=1): n =", len(dense))
    for fam in ("RDNA3", "RDNA2", "NVIDIA", "Intel"):
        e = [r["eff"] for r in dense if r["family"] == fam]
        if e:
            print(f"  {fam:7s} n={len(e):2d} median={statistics.median(e):.3f} min={min(e):.3f} max={max(e):.3f}")
    rdna3_nv = [r["eff"] for r in dense if r["family"] in ("RDNA3", "NVIDIA")]
    q = statistics.quantiles(rdna3_nv, n=4)
    print(f"  RDNA3+NVIDIA n={len(rdna3_nv)} median={statistics.median(rdna3_nv):.3f} quartiles={q[0]:.3f}/{q[2]:.3f}")
    rx7800 = [r["eff"] for r in rows if r["use"] and r["gpu"] == "RX 7800 XT" and r["model"] == "llama2-7b"]
    print("RX 7800 XT dense, all sources:", [round(x, 3) for x in rx7800])
    moe = [r for r in rows if r["model"].startswith("qwen3") and r["use"]]
    print("MoE Qwen3-30B-A3B, Vulkan (use=1):", [(r["tg128_tok_s"], r["eff"]) for r in moe])
    print("CSV:", os.path.join(out_dir, "vulkan_rows.csv"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
