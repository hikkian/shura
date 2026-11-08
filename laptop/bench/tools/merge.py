#!/usr/bin/env python3
"""merge.py <alpha_experts> <alpha_other> <out.gguf> [--limit N]
Task-arithmetic merge without training: W = Qwen3.6 + alpha * (Ornith - Qwen3.6), tensor by tensor, in Q8_0 GGUF space.
alpha 1 = Ornith, 0 = Qwen3.6. Separate alpha for routed experts (ffn_*_exps) and for everything else.
Tensors that exist only in Ornith (MTP layer blk.40) are copied from Ornith. Metadata = Ornith's (chat template etc.)."""
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q, GGUFValueType as VT
from gguf.quants import quantize, dequantize

O = "/home/hikkian/Desktop/AI/bench/rufix/src/Ornith-1.5-35B-Q8_0.gguf"
QW = "/home/hikkian/Desktop/AI/bench/rufix/src/Qwen3.6-35B-A3B-Q8_0.gguf"
ae, ao, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

ro, rq = GGUFReader(O), GGUFReader(QW)
TQ = {t.name: t for t in rq.tensors}
w = GGUFWriter(out, arch=ro.fields["general.architecture"].contents(), use_temp_file=False)
for k, f in ro.fields.items():
    if k.startswith("GGUF.") or k == "general.architecture":
        continue
    vt = f.types[0]
    w.add_key_value(k, f.contents(), vt, sub_type=f.types[-1]) if vt == VT.ARRAY else w.add_key_value(k, f.contents(), vt)


def mix(to, tq, a):
    if to.tensor_type == Q.F32:
        o = np.asarray(to.data, dtype=np.float32); q = np.asarray(tq.data, dtype=np.float32)
        return (q + a * (o - q)).astype(np.float32), None
    rows = to.data.shape[0] if to.data.ndim == 2 else None
    od, qd = np.asarray(to.data), np.asarray(tq.data)
    flat_o, flat_q = od.reshape(-1, od.shape[-1]), qd.reshape(-1, qd.shape[-1])
    res = np.empty_like(flat_o)
    CH = 4096
    for s in range(0, flat_o.shape[0], CH):
        x = dequantize(flat_o[s:s + CH], to.tensor_type); y = dequantize(flat_q[s:s + CH], to.tensor_type)
        res[s:s + CH] = quantize((y + a * (x - y)).astype(np.float32), to.tensor_type)
    return res.reshape(od.shape), to.tensor_type


# pass 1: tensor infos (same shapes/types as Ornith), pass 2: compute and stream each tensor to disk
for t in ro.tensors:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
n = 0
for t in ro.tensors:
    tq = TQ.get(t.name)
    a = ae if "_exps" in t.name else ao
    if tq is None or a == 1.0 or (limit is not None and n >= limit):
        w.write_tensor_data(np.asarray(t.data))
        continue
    data, _ = mix(t, tq, a)
    n += 1
    w.write_tensor_data(data)
    if n % 50 == 0:
        print("mixed", n, t.name, flush=True)
w.close()
print("wrote", out, "mixed tensors:", n)
