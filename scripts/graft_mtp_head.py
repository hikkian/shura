#!/usr/bin/env python3
"""Put a Qwen3.5/3.6-MoE multi-token-prediction (MTP) head into a GGUF that has none, so that `--spec-type draft-mtp` works
from one file (the way Tiel-Coder ships), without a second draft model that would cost another gigabyte of VRAM.

    graft_mtp_head.py head   BODY.gguf HEAD.safetensors HEAD-bf16.gguf     # step 1: the head as GGUF tensors (BF16 / F32)
    # step 2 (llama.cpp): llama-quantize --allow-requantize --tensor-type 'ffn_.*_exps=q3_k' ... HEAD-bf16.gguf HEAD-q.gguf q8_0
    graft_mtp_head.py merge  BODY.gguf HEAD-q.gguf OUT.gguf                # step 3: body tensors + head tensors, one file

The tensor mapping and the two numeric rules are the ones in llama.cpp's convert_hf_to_gguf (conversion/qwen.py):
every `*norm.weight` gets +1 (HF stores w-1) and is kept as F32, and the fused `experts.gate_up_proj` is split in half on
the row axis (first half = gate, second half = up). MTP is lossless (the target model verifies every drafted token), so a
wrong head can only lower the acceptance rate, never change the answer; check the acceptance rate after grafting.

Needs numpy and llama.cpp's gguf-py (GGUF_PY or ../llama.cpp-perf/gguf-py). Reads the body through a memory map; writes the output once.
"""
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("GGUF_PY", str(Path.home() / "Desktop/AI/llama.cpp-perf/gguf-py")))
import gguf  # noqa: E402

PER_LAYER = {
    "input_layernorm.weight": ("attn_norm.weight", True),
    "post_attention_layernorm.weight": ("post_attention_norm.weight", True),
    "self_attn.q_proj.weight": ("attn_q.weight", False),
    "self_attn.k_proj.weight": ("attn_k.weight", False),
    "self_attn.v_proj.weight": ("attn_v.weight", False),
    "self_attn.o_proj.weight": ("attn_output.weight", False),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", True),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", True),
    "mlp.gate.weight": ("ffn_gate_inp.weight", False),
    "mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp.weight", False),
    "mlp.shared_expert.up_proj.weight": ("ffn_up_shexp.weight", False),
    "mlp.shared_expert.down_proj.weight": ("ffn_down_shexp.weight", False),
    "mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp.weight", False),
    "mlp.experts.down_proj": ("ffn_down_exps.weight", False),
}
TOP = {
    "mtp.fc.weight": ("nextn.eh_proj.weight", False),
    "mtp.pre_fc_norm_embedding.weight": ("nextn.enorm.weight", True),
    "mtp.pre_fc_norm_hidden.weight": ("nextn.hnorm.weight", True),
    "mtp.norm.weight": ("nextn.shared_head_norm.weight", True),
}
F32_KEEP = ("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight")      # router and shared-expert gate stay F32 (as in Tiel-Coder)


def read_safetensors(path):
    """name -> (dtype string, shape, bytes view); the file stays memory-mapped."""
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    n = struct.unpack("<Q", bytes(mm[:8]))[0]
    header = json.loads(bytes(mm[8:8 + n]))
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        out[name] = (meta["dtype"], tuple(meta["shape"]), mm[8 + n + a:8 + n + b])
    return out


def to_f32(dtype, raw):
    if dtype == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == "F32":
        return raw.view(np.float32)
    if dtype == "F16":
        return raw.view(np.float16).astype(np.float32)
    raise SystemExit(f"unsupported dtype {dtype}")


def head_tensors(st, layer):
    """Yield (gguf name, numpy array, ggml type) for the MTP head, stored as block `layer`."""
    def emit(name, arr, as_norm=False, force_f32=False):
        if as_norm or force_f32 or arr.dtype == np.float32:
            return (name, np.ascontiguousarray(arr, dtype=np.float32), gguf.GGMLQuantizationType.F32)
        return (name, arr, gguf.GGMLQuantizationType.BF16)

    def get(name, norm=False, f32=False, squeeze=False):
        dtype, shape, raw = st[name]
        if dtype == "BF16" and not (norm or f32):
            arr = raw.view(np.uint16).reshape(shape)
        else:
            arr = to_f32(dtype, raw).reshape(shape)
            if norm:
                arr = arr + 1.0                         # HF stores w-1 for these norms
        if squeeze:
            arr = arr.reshape(-1)
        return arr

    for hf, (g, norm) in TOP.items():
        yield emit(f"blk.{layer}.{g}", get(hf, norm=norm), as_norm=norm)
    for hf, (g, norm) in PER_LAYER.items():
        name = f"mtp.layers.0.{hf}"
        f32 = g in F32_KEEP
        yield emit(f"blk.{layer}.{g}", get(name, norm=norm, f32=f32, squeeze=g == "ffn_gate_inp_shexp.weight"), as_norm=norm, force_f32=f32)
    dtype, shape, raw = st["mtp.layers.0.mlp.experts.gate_up_proj"]
    arr = raw.view(np.uint16).reshape(shape) if dtype == "BF16" else to_f32(dtype, raw).reshape(shape)
    half = shape[-2] // 2
    kind = gguf.GGMLQuantizationType.BF16 if dtype == "BF16" else gguf.GGMLQuantizationType.F32
    yield (f"blk.{layer}.ffn_gate_exps.weight", np.ascontiguousarray(arr[:, :half, :]), kind)
    yield (f"blk.{layer}.ffn_up_exps.weight", np.ascontiguousarray(arr[:, half:, :]), kind)


def body_metadata(reader):
    arch = reader.get_field("general.architecture").contents()
    return arch, [f for f in reader.fields.values() if f.name != "general.architecture" and not f.name.startswith("GGUF.")]


def copy_kv(writer, fields, overrides):
    for f in fields:
        val_type = f.types[0]
        sub = f.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = overrides.get(f.name, f.contents())
        writer.add_key_value(f.name, value, val_type, sub_type=sub)


def new_counts(reader, arch):
    blocks = reader.get_field(f"{arch}.block_count").contents()
    over = {f"{arch}.block_count": blocks + 1, f"{arch}.nextn_predict_layers": 1}
    rec = reader.get_field(f"{arch}.attention.recurrent_layers")
    if rec is not None:
        # newer conversions store one flag per layer; the MTP block has ordinary (full) attention, so its flag is False
        flags = [bool(x) for x in rec.contents()]
        if len(flags) == blocks:
            over[f"{arch}.attention.recurrent_layers"] = flags + [False]
    return blocks, over


def cmd_head(body, safetensors, out):
    reader = gguf.GGUFReader(body)
    arch, fields = body_metadata(reader)
    blocks, over = new_counts(reader, arch)
    if any(t.name.startswith(f"blk.{blocks}.nextn") for t in reader.tensors):
        raise SystemExit("the body already has an MTP head")
    tensors = list(head_tensors(read_safetensors(safetensors), blocks))
    w = gguf.GGUFWriter(out, arch)
    copy_kv(w, fields, over)
    if "nextn_predict_layers" not in "".join(f.name for f in fields):
        w.add_key_value(f"{arch}.nextn_predict_layers", 1, gguf.GGUFValueType.UINT32)
    for name, arr, kind in tensors:
        w.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, kind)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for _, arr, _ in tensors:
        w.write_tensor_data(arr)
    w.close()
    print(f"{len(tensors)} head tensors for block {blocks} -> {out}")


def cmd_merge(body, head, out):
    rb, rh = gguf.GGUFReader(body), gguf.GGUFReader(head)
    arch, fields = body_metadata(rb)
    blocks, over = new_counts(rb, arch)
    head_t = [t for t in rh.tensors if t.name.startswith(f"blk.{blocks}.")]
    if not head_t:
        raise SystemExit(f"{head} has no blk.{blocks}.* tensors")
    w = gguf.GGUFWriter(out, arch)
    names = {f.name for f in fields}
    copy_kv(w, fields, over)
    if f"{arch}.nextn_predict_layers" not in names:
        w.add_key_value(f"{arch}.nextn_predict_layers", 1, gguf.GGUFValueType.UINT32)
    tensors = list(rb.tensors) + head_t
    for t in tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in tensors:
        w.write_tensor_data(t.data)
    w.close()
    print(f"{len(rb.tensors)} body + {len(head_t)} head tensors -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] not in ("head", "merge"):
        raise SystemExit(__doc__)
    {"head": cmd_head, "merge": cmd_merge}[sys.argv[1]](*sys.argv[2:])
