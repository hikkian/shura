#!/usr/bin/env python3
"""Делает пару мини-GGUF (база Q2_K/Q4_K-эксперты, донор) для теста splice_gguf.py. Usage: make_mini_gguf.py DIR"""
import sys
sys.path.insert(0, "/home/hikkian/.unsloth/llama.cpp/gguf-py")
import numpy as np, gguf
from gguf import GGMLQuantizationType as QT
from gguf.quants import quant_shape_to_byte_shape
def quantize(x, t):
    # квантизации Q2_K/Q4_K в gguf-py нет: случайные байты нужной формы (для теста логики достаточно)
    return np.random.default_rng(int(x.sum()*1000) & 0xffff).integers(0, 256, quant_shape_to_byte_shape(x.shape, t), dtype=np.uint8)
from pathlib import Path
d = Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
def mk(path, down_t, gate_t, seed):
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(str(path), "qwen35moe")
    w.add_block_count(3); w.add_name("mini"); w.add_file_type(10)
    w.add_token_list(["a", "b", "c"]) if hasattr(w, "add_token_list") else None
    w.add_uint32("general.alignment", 32); w.data_alignment = 32
    w.add_tensor("token_embd.weight", rng.standard_normal((16, 256)).astype(np.float32))
    w.add_tensor("output_norm.weight", rng.standard_normal(256).astype(np.float32))
    for L in range(2):
        w.add_tensor(f"blk.{L}.attn_norm.weight", rng.standard_normal(256).astype(np.float16))
        for kind, t in (("ffn_down_exps", down_t), ("ffn_gate_exps", gate_t)):
            x = rng.standard_normal((4, 8, 512)).astype(np.float32)
            w.add_tensor(f"blk.{L}.{kind}.weight", quantize(x, t), raw_dtype=t)
    # MTP-слой
    w.add_tensor("blk.2.nextn.eh_proj.weight", rng.standard_normal((8, 256)).astype(np.float32))
    x = rng.standard_normal((4, 8, 512)).astype(np.float32)
    w.add_tensor("blk.2.ffn_down_exps.weight", quantize(x, down_t), raw_dtype=down_t)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
mk(d / "mini-base.gguf", QT.Q2_K, QT.Q2_K, 1)
mk(d / "mini-donor.gguf", QT.Q4_K, QT.Q4_K, 2)
