"""The universal planner: hardware profile + model catalog -> plan (model, quant, settings, prediction).

No I/O, stdlib only, deterministic. The decision rests on one idea: token generation is limited by memory
bandwidth, so speed ~ efficiency * bandwidth / bytes of active weights read per token, where the bytes are split
between fast memory (VRAM or unified memory) and system RAM. The efficiency constants below were calibrated on ONE
machine (RTX 4070 SUPER + DDR4-3200, upstream-style `--n-cpu-moe` layout) and are deliberately conservative: the
prediction is for choosing and ranking, never a promise. A short measurement on the user's machine (calibration)
corrects it. See docs/HARDWARE.md.
"""
import copy

from .schema import validate_catalog, validate_hardware

GiB = 1024 ** 3
GB = 10 ** 9

DEFAULTS = {
    "min_tok_s": 15.0,            # below this a model is not recommended at all
    "comfort_tok_s": 20.0,        # a pick should reach this; otherwise we take the best that reaches min_tok_s
    "upgrade_margin": 1.5,        # a quant above the catalog default needs comfort * margin predicted speed
    "efficiency": {"hybrid": 0.30, "gpu": 0.50, "cpu": 0.40, "unified": 0.45},
    "os_reserve_min": 4 * GiB,    # RAM kept for the OS, desktop and apps: max(min, frac * total)
    "os_reserve_frac": 0.15,
    "vram_reserve_display": int(1.5 * GiB),   # VRAM kept for the desktop on a GPU that drives a display
    "vram_reserve_headless": int(0.5 * GiB),
    "unified_fraction": 0.70,     # share of unified memory the GPU may use (macOS wired limit is about this)
    "compute_buffer": 1 * GiB,    # GPU compute buffers and runtime context
    "host_base": int(1.5 * GiB),  # the server process itself
    "min_context": 16384,
    "contexts": (262144, 200000, 131072, 65536, 32768, 16384),
    "kv_types": (("q8_0", 0.53), ("q4_0", 0.28)),   # (name, size relative to f16)
    "speed_keep": 0.92,           # the context chosen keeps at least this share of the best predicted speed
    "bandwidth_fallback_gbs": {"ram": 20.0, "gpu": 150.0},
    "uncertainty": (0.6, 1.35),   # prediction range relative to the mid estimate
}

BACKEND_ORDER = {
    "nvidia": ("cuda", "vulkan"),
    "amd": ("rocm", "vulkan"),
    "intel": ("sycl", "vulkan", "openvino"),
    "apple": ("metal",),
}


def _cfg(overrides):
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in (overrides or {}).items():
        if isinstance(cfg.get(k), dict) and isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def _resources(hw, cfg):
    mem = hw["memory"]
    total = mem["total"]
    ram_bw = mem.get("bandwidth_gbs")
    res = {
        "ram_budget": total - max(cfg["os_reserve_min"], cfg["os_reserve_frac"] * total),
        "ram_bw": (ram_bw or cfg["bandwidth_fallback_gbs"]["ram"]) * GB,
        "ram_bw_measured": ram_bw is not None,
        "gpu": None,
        "cores": hw["cpu"]["physical_cores"],
        "numa_nodes": hw["cpu"]["numa_nodes"],
        "total": total,
    }
    gpus = sorted(hw["gpus"], key=lambda g: g["vram_total"], reverse=True)
    if gpus:
        g = gpus[0]
        bw = g.get("bandwidth_gbs")
        res["gpu"] = g
        res["gpu_bw"] = (bw or cfg["bandwidth_fallback_gbs"]["gpu"]) * GB
        res["gpu_bw_measured"] = bw is not None
        if g.get("unified"):
            res["pool"] = cfg["unified_fraction"] * total
        else:
            display = g.get("display", g["vram_used"] > 256 * 1024 * 1024)
            reserve = cfg["vram_reserve_display"] if display else cfg["vram_reserve_headless"]
            res["vram_budget"] = g["vram_total"] - g["vram_used"] - reserve
    return res


def backend_candidates(hw):
    """Backends worth probing, most likely first. When there is more than one, the installer measures them."""
    if not hw["gpus"]:
        return ["cpu"], False
    g = max(hw["gpus"], key=lambda x: x["vram_total"])
    usable = [b for b in BACKEND_ORDER.get(g["vendor"], ()) if not g.get("backends") or b in g["backends"]]
    if not usable:
        usable = [b for b in g.get("backends", []) if b != "cpu"] or ["vulkan"]
    return usable + ["cpu"], len(usable) > 1


def _active_bytes(m, q):
    if m["kind"] == "moe":
        return q["file_bytes"] * m["params_active_b"] / m["params_total_b"]
    return float(q["file_bytes"])


def _fit(res, cfg, m, q, ctx, kv_name, kv_factor):
    """Placement and predicted speed of one (model, quant, context, KV type), or (None, reason)."""
    file, kv = q["file_bytes"], ctx * m["kv_bytes_per_token_f16"] * kv_factor
    active, eff, g = _active_bytes(m, q), cfg["efficiency"], res["gpu"]
    s = {"context": ctx, "kv_type": kv_name, "flash_attn": True, "parallel": 1, "n_cpu_moe": None}

    def finish(mode, time_s, ram_need, vram_need, ngl, eta):
        if ram_need > res["ram_budget"]:
            return None, (f"needs {ram_need / GiB:.1f} GiB of RAM, only {res['ram_budget'] / GiB:.1f} GiB is "
                          f"available after keeping memory for the system")
        s["ngl"] = ngl
        s["threads"] = min(res["cores"], 8) if mode in ("hybrid", "gpu", "unified") else res["cores"]
        s["numa"] = "distribute" if res["numa_nodes"] > 1 and mode in ("cpu", "hybrid") else None
        return {"mode": mode, "tok_s": eta / time_s if time_s else 0.0, "ram_need": ram_need, "vram_need": vram_need,
                "settings": s}, ""

    if g and g.get("unified"):                                    # Apple-style unified memory
        need = file + kv + cfg["compute_buffer"]
        if need > res["pool"]:
            return None, f"needs {need / GiB:.1f} GiB of unified memory, the GPU may use {res['pool'] / GiB:.1f} GiB"
        return finish("unified", active / res["gpu_bw"], cfg["host_base"], need, "all", eff["unified"])

    budget = res.get("vram_budget", 0) if g else 0
    if g and m["kind"] == "moe":
        expert_bytes = file - q["nonexpert_bytes"]
        fixed = q["nonexpert_bytes"] + kv + cfg["compute_buffer"]
        if fixed <= budget and expert_bytes > 0:
            per_layer = expert_bytes / m["n_layers"]
            on_gpu = min(m["n_layers"], int((budget - fixed) // per_layer))
            share = on_gpu / m["n_layers"]
            routed = active * m["expert_active_fraction"]
            gpu_b, cpu_b = active - routed + routed * share, routed * (1 - share)
            s["n_cpu_moe"] = m["n_layers"] - on_gpu
            resident_cpu = expert_bytes * (1 - share)
            time_s = gpu_b / res["gpu_bw"] + cpu_b / res["ram_bw"]
            mode = "gpu" if on_gpu == m["n_layers"] else "hybrid"
            return finish(mode, time_s, resident_cpu + cfg["host_base"], fixed + on_gpu * per_layer, "all",
                          eff[mode])
    elif g:                                                        # dense model, layers split by bandwidth time
        fixed = kv + cfg["compute_buffer"]
        if budget - fixed > 0:
            share = min(1.0, (budget - fixed) / file)
            time_s = share * file / res["gpu_bw"] + (1 - share) * file / res["ram_bw"]
            mode = "gpu" if share >= 1 else "hybrid"
            # a dense layer split adds the two bandwidths without per-layer expert routing, so its overhead is that of
            # CPU inference (the MoE "hybrid" constant would make offloading look slower than not offloading)
            return finish(mode, time_s, (1 - share) * file + cfg["host_base"], fixed + share * file,
                          max(1, round(share * m["n_layers"])) if share < 1 else "all",
                          eff["gpu"] if share >= 1 else eff["cpu"])
    s["ngl"] = 0                                                    # CPU only (no GPU, or none that is usable)
    return finish("cpu", active / res["ram_bw"], file + kv + cfg["host_base"], 0, 0, eff["cpu"])


def _best_fit(res, cfg, m, q):
    """Largest context (and cheapest KV type) that fits and keeps most of the best speed; else (None, reason)."""
    fits, reason = [], ""
    for ctx in cfg["contexts"]:
        if ctx > m["context_max"] or ctx < cfg["min_context"]:
            continue
        for name, factor in cfg["kv_types"]:
            fit, why = _fit(res, cfg, m, q, ctx, name, factor)
            if fit:
                fits.append(fit)
                break
            reason = why
    if not fits:
        return None, reason or "no context size fits"
    best = max(f["tok_s"] for f in fits)
    return next(f for f in fits if f["tok_s"] >= cfg["speed_keep"] * best), ""


def _pick_quant(res, cfg, m, forced=None):
    qs = sorted(m["quants"], key=lambda q: q["quality"])
    by_id = {q["id"]: q for q in qs}
    fits = {}
    for q in qs:
        fit, why = _best_fit(res, cfg, m, q)
        fits[q["id"]] = (fit, why)
    comfort, floor = cfg["comfort_tok_s"], cfg["min_tok_s"]
    ok = lambda q, thr: fits[q["id"]][0] and fits[q["id"]][0]["tok_s"] >= thr       # noqa: E731
    default = by_id[m["default_quant"]]
    if forced:
        q = by_id.get(forced)
        if not q:
            return None
        chosen, why = q, "forced by the user"
    else:
        above = [q for q in qs if q["quality"] > default["quality"]]
        below = [q for q in reversed(qs) if q["quality"] < default["quality"]]
        chosen = why = None
        if ok(default, comfort):
            up = [q for q in reversed(above) if ok(q, comfort * cfg["upgrade_margin"])]
            chosen = up[0] if up else default
            why = ("a higher-quality quant still keeps a wide speed margin" if up
                   else "the catalog's tested default fits with enough speed")
        else:
            for thr, label in ((comfort, "the tested default is too slow or does not fit; this smaller quant is "
                                         "the best one that still feels responsive"),
                               (floor, "nothing reaches the comfortable speed; this is the best that is still "
                                       "usable")):
                cand = [q for q in below if ok(q, thr)] or ([default] if ok(default, thr) else [])
                if cand:
                    chosen, why = cand[0], label
                    break
        if chosen is None:
            feasible = [q for q in qs if fits[q["id"]][0]]
            if not feasible:
                return {"model": m, "fit": None, "reason": fits[qs[0]["id"]][1], "quant": None, "alts": []}
            chosen = max(feasible, key=lambda q: fits[q["id"]][0]["tok_s"])
            why = "nothing reaches the minimum speed; this is the fastest that fits"
    fit = fits[chosen["id"]][0]
    alts = [{"model": m["id"], "quant": q["id"], "tok_s": round(fits[q["id"]][0]["tok_s"], 1)}
            for q in reversed(qs) if q is not chosen and ok(q, floor)]
    return {"model": m, "quant": chosen, "fit": fit, "why": why, "alts": alts, "reason": ""}


def plan(hw, catalog, *, config=None, model=None, quant=None):
    """Return a plan dict. Raises ValueError if the hardware profile or the catalog is malformed."""
    problems = validate_hardware(hw) + validate_catalog(catalog)
    if problems:
        raise ValueError("invalid input: " + "; ".join(problems))
    cfg = _cfg(config)
    res = _resources(hw, cfg)
    backends, probe = backend_candidates(hw)
    picks, refusals = [], []
    for m in sorted(catalog["models"], key=lambda x: -x["capability"]):
        if model and m["id"] != model:
            continue
        p = _pick_quant(res, cfg, m, quant if model else None)
        (picks if p and p["fit"] else refusals).append(p or {"model": m, "reason": f"unknown quant {quant!r}"})
    comfort, floor = cfg["comfort_tok_s"], cfg["min_tok_s"]
    pool = ([p for p in picks if p["fit"]["tok_s"] >= comfort] or [p for p in picks if p["fit"]["tok_s"] >= floor]
            or sorted(picks, key=lambda p: -p["fit"]["tok_s"])[:1])
    if not pool:
        why = "; ".join(f"{r['model']['name']}: {r['reason']}" for r in refusals) or "the catalog is empty"
        return {"ok": False, "reasons": [f"no model in the catalog fits this machine ({why})"],
                "backend_candidates": backends, "needs_probe": probe}
    best = max(pool, key=lambda p: (p["model"]["capability"], p["quant"]["quality"]))
    fit, mid = best["fit"], best["fit"]["tok_s"]
    lo, hi = cfg["uncertainty"]
    warnings = []
    if mid < floor:
        warnings.append(f"predicted {mid:.0f} tok/s is below the {floor:.0f} tok/s minimum; this is the best "
                        f"available on this machine")
    if not res["ram_bw_measured"] and fit["mode"] != "unified":
        warnings.append("RAM bandwidth was not measured, a conservative estimate was used")
    if res["numa_nodes"] > 1:
        warnings.append(f"{res['numa_nodes']} NUMA nodes: calibration should compare --numa modes")
    confidence = "low" if (res["numa_nodes"] > 1 or fit["mode"] == "unified" or not res["ram_bw_measured"]) else "medium"
    reasons = [best["why"], f"mode: {fit['mode']}, {fit['settings']['context']} tokens of context, "
                            f"{fit['settings']['kv_type']} KV cache"]
    return {
        "ok": True,
        "model": best["model"]["id"], "model_name": best["model"]["name"], "quant": best["quant"]["id"],
        "backend_candidates": backends, "needs_probe": probe,
        "settings": fit["settings"], "mode": fit["mode"],
        "predicted_tok_s": {"low": round(mid * lo, 1), "mid": round(mid, 1), "high": round(mid * hi, 1)},
        "confidence": confidence,
        "memory": {"ram_need": int(fit["ram_need"]), "ram_budget": int(res["ram_budget"]),
                   "vram_need": int(fit["vram_need"]),
                   "vram_budget": int(res.get("vram_budget", res.get("pool", 0)))},
        "reasons": reasons, "warnings": warnings,
        "alternatives": (best["alts"] + [{"model": p["model"]["id"], "quant": p["quant"]["id"],
                                          "tok_s": round(p["fit"]["tok_s"], 1)}
                                         for p in picks if p is not best])[:4],
    }
