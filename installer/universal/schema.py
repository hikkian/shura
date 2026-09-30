"""Validation of the two data formats the planner consumes: the hardware profile and the model catalog.

Both are plain JSON. Validation returns a list of human-readable problems (empty list = valid) so that CI can
show every problem in a contributed file at once. The formats are documented in docs/HARDWARE.md.
"""

OS_FAMILIES = {"linux", "macos", "windows"}
ARCHES = {"x86_64", "arm64"}
VENDORS = {"nvidia", "amd", "intel", "apple"}
BACKENDS = {"cuda", "rocm", "vulkan", "sycl", "openvino", "metal", "cpu"}
KINDS = {"dense", "moe"}


def _num(d, key, errs, path, *, minimum=0, optional=False, kind=(int, float)):
    v = d.get(key)
    if v is None:
        if not optional:
            errs.append(f"{path}.{key}: missing")
        return
    if isinstance(v, bool) or not isinstance(v, kind) or v < minimum:
        errs.append(f"{path}.{key}: expected a number >= {minimum}, got {v!r}")


def _str(d, key, errs, path, *, choices=None, optional=False):
    v = d.get(key)
    if v is None:
        if not optional:
            errs.append(f"{path}.{key}: missing")
        return
    if not isinstance(v, str) or not v or (choices and v not in choices):
        errs.append(f"{path}.{key}: expected one of {sorted(choices)}, got {v!r}" if choices
                    else f"{path}.{key}: expected a non-empty string, got {v!r}")


def validate_hardware(hw):
    errs = []
    if not isinstance(hw, dict):
        return ["hardware profile must be an object"]
    if hw.get("schema") != 1:
        errs.append("schema: must be 1")
    osd, cpu, mem = hw.get("os", {}), hw.get("cpu", {}), hw.get("memory", {})
    _str(osd, "family", errs, "os", choices=OS_FAMILIES)
    _str(osd, "arch", errs, "os", choices=ARCHES)
    _str(cpu, "model", errs, "cpu")
    _num(cpu, "physical_cores", errs, "cpu", minimum=1, kind=int)
    _num(cpu, "logical_cores", errs, "cpu", minimum=1, kind=int)
    _num(cpu, "numa_nodes", errs, "cpu", minimum=1, kind=int)
    _num(mem, "total", errs, "memory", minimum=1, kind=int)
    _num(mem, "available", errs, "memory", minimum=0, kind=int)
    _num(mem, "bandwidth_gbs", errs, "memory", minimum=0.1, optional=True)
    if mem.get("total") and mem.get("available") and mem["available"] > mem["total"]:
        errs.append("memory.available: larger than memory.total")
    if not isinstance(hw.get("gpus"), list):
        errs.append("gpus: must be a list (empty for CPU-only machines)")
    else:
        for i, g in enumerate(hw["gpus"]):
            p = f"gpus[{i}]"
            _str(g, "vendor", errs, p, choices=VENDORS)
            _str(g, "name", errs, p)
            _num(g, "vram_total", errs, p, minimum=1, kind=int)
            _num(g, "vram_used", errs, p, minimum=0, kind=int)
            _num(g, "bandwidth_gbs", errs, p, minimum=0.1, optional=True)
            bs = g.get("backends", [])
            if not isinstance(bs, list) or any(b not in BACKENDS for b in bs):
                errs.append(f"{p}.backends: expected a list drawn from {sorted(BACKENDS)}, got {bs!r}")
    _num(hw.get("disk", {}), "free", errs, "disk", minimum=0, kind=int)
    return errs


def validate_catalog(cat):
    errs = []
    if not isinstance(cat, dict) or cat.get("schema") != 1 or not isinstance(cat.get("models"), list):
        return ["catalog: expected {'schema': 1, 'models': [...]}"]
    seen = set()
    for i, m in enumerate(cat["models"]):
        p = f"models[{i}]"
        _str(m, "id", errs, p)
        if m.get("id") in seen:
            errs.append(f"{p}.id: duplicate {m.get('id')!r}")
        seen.add(m.get("id"))
        _str(m, "name", errs, p)
        _str(m, "arch", errs, p)
        _str(m, "kind", errs, p, choices=KINDS)
        _num(m, "params_total_b", errs, p, minimum=0.01)
        _num(m, "params_active_b", errs, p, minimum=0.01)
        _num(m, "n_layers", errs, p, minimum=1, kind=int)
        _num(m, "context_max", errs, p, minimum=512, kind=int)
        _num(m, "kv_bytes_per_token_f16", errs, p, minimum=1, kind=int)
        _num(m, "capability", errs, p, minimum=0, kind=int)
        if m.get("params_active_b") and m.get("params_total_b") and m["params_active_b"] > m["params_total_b"]:
            errs.append(f"{p}.params_active_b: larger than params_total_b")
        if m.get("kind") == "moe":
            _num(m, "n_experts", errs, p, minimum=2, kind=int)
            _num(m, "experts_used", errs, p, minimum=1, kind=int)
            _num(m, "expert_active_fraction", errs, p, minimum=0.01)
            if isinstance(m.get("expert_active_fraction"), (int, float)) and m["expert_active_fraction"] > 1:
                errs.append(f"{p}.expert_active_fraction: must be <= 1")
        src = m.get("source", {})
        _str(src, "hf_repo", errs, f"{p}.source")
        _str(src, "file_pattern", errs, f"{p}.source")
        qs = m.get("quants")
        if not isinstance(qs, list) or not qs:
            errs.append(f"{p}.quants: expected a non-empty list")
            continue
        ids = [q.get("id") for q in qs]
        if m.get("default_quant") not in ids:
            errs.append(f"{p}.default_quant: {m.get('default_quant')!r} is not one of {ids}")
        for j, q in enumerate(qs):
            qp = f"{p}.quants[{j}]"
            _str(q, "id", errs, qp)
            _num(q, "file_bytes", errs, qp, minimum=1, kind=int)
            _num(q, "nonexpert_bytes", errs, qp, minimum=0, kind=int)
            _num(q, "quality", errs, qp, minimum=0, kind=int)
            if q.get("nonexpert_bytes") and q.get("file_bytes") and q["nonexpert_bytes"] > q["file_bytes"]:
                errs.append(f"{qp}.nonexpert_bytes: larger than file_bytes")
        if len(set(q.get("quality") for q in qs)) != len(qs):
            errs.append(f"{p}.quants: quality values must be unique (they order the quants)")
    return errs
