#!/usr/bin/env python3
"""Обёртка над ~/shura/scripts/graft_mtp_head.py для моделей, где эксперты MTP лежат по одному
(mtp.layers.0.mlp.experts.N.{gate,up,down}_proj.weight), а не склеенными, как ждёт скрипт.

  graft_unfused.py head BODY.gguf SHARD.safetensors HEAD-bf16.gguf

Складывает 256 экспертов в experts.gate_up_proj (E, 2I, H) и experts.down_proj (E, H, I),
остальное делает исходный скрипт без изменений."""
import importlib.util, re, sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("graft", Path.home() / "shura/scripts/graft_mtp_head.py")
graft = importlib.util.module_from_spec(spec)
sys.argv_saved = sys.argv
spec.loader.exec_module(graft)  # main() вызывается только под __name__ == "__main__"

orig_read = graft.read_safetensors


def as_u16(dtype, shape, raw):
    assert dtype == "BF16", dtype
    return raw.view(np.uint16).reshape(shape)


def read_fused(path):
    st = orig_read(path)
    rx = re.compile(r"^mtp\.layers\.0\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
    per = {}
    for k in list(st):
        m = rx.match(k)
        if m:
            per.setdefault(int(m[1]), {})[m[2]] = st.pop(k)
    if not per:
        return st
    n = max(per) + 1
    assert sorted(per) == list(range(n)), "не все эксперты на месте"
    gate_up = np.stack([np.concatenate([as_u16(*per[e]["gate"]), as_u16(*per[e]["up"])], axis=0) for e in range(n)])
    down = np.stack([as_u16(*per[e]["down"]) for e in range(n)])
    st["mtp.layers.0.mlp.experts.gate_up_proj"] = ("BF16", gate_up.shape, gate_up.view(np.uint8).reshape(-1))
    st["mtp.layers.0.mlp.experts.down_proj"] = ("BF16", down.shape, down.view(np.uint8).reshape(-1))
    print(f"склеено {n} экспертов: gate_up {gate_up.shape}, down {down.shape}", flush=True)
    return st


graft.read_safetensors = read_fused
cmd = sys.argv[1]
{"head": graft.cmd_head, "merge": graft.cmd_merge}[cmd](*sys.argv[2:])
