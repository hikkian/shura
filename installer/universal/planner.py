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
    "target_tok_s": 35.0,         # what a plan aims for, judged with the window half full (see fill_fraction)
    "min_tok_s": 15.0,            # below this a model is not recommended at all
    "comfort_tok_s": 20.0,        # if the target is out of reach: the best that reaches this, then min_tok_s
    "upgrade_margin": 1.15,       # a quant above the catalog default needs target * margin predicted speed
    "fill_fraction": 0.5,         # speed is judged with this share of the window filled (a working session, not an empty chat)
    "window_floors": (100000, 65536, 32768, 16384),   # the window is given up step by step before the quant is
    "useful_window": 65536,       # below this a window is hardly enough for an agent: only chosen when nothing larger works
    # "attention": efficiency of reading the KV cache for every generated token (attention kernels use a fraction of the
    # bandwidth; ~0.2 measured on the reference GPU at 187k context). It is what makes a long window cost speed.
    "efficiency": {"hybrid": 0.30, "gpu": 0.50, "cpu": 0.40, "unified": 0.45, "attention": 0.25},
    "os_reserve_min": 6 * GiB,    # RAM kept for the OS, desktop and apps: max(min, frac * total)
    "os_reserve_frac": 0.15,
    "vram_reserve_display": int(0.8 * GiB),   # VRAM kept free on top of what the desktop already uses (display GPU)
    "vram_reserve_headless": int(0.3 * GiB),
    "unified_fraction": 0.70,     # share of unified memory the GPU may use (macOS wired limit is about this)
    "compute_buffer": 1 * GiB,    # GPU compute buffers and runtime context
    "host_base": int(1.5 * GiB),  # the server process itself
    "min_context": 16384,
    "contexts": (262144, 200000, 131072, 65536, 32768, 16384),
    "kv_types": (("q8_0", 0.53), ("q4_0", 0.28)),   # (name, size relative to f16)
    "bandwidth_fallback_gbs": {"ram": 20.0, "gpu": 150.0},
    "uncertainty": (0.6, 1.35),   # prediction range relative to the mid estimate
    # The "fork" tier: NVIDIA + Linux with the Shura CUDA fork (turbo3 KV, MoE expert cache, MTP). It is fitted to ONE
    # measured machine (docs/HARDWARE.md, "The fork tier"), so it is anchored on those measurements instead of pure bandwidth.
    "fork": {
        "enabled": True,
        "byte_time_scale": 2.59,      # seconds per token = scale * (bytes / bandwidth) + sync * CPU layers ...
        "sync_s_per_cpu_layer": 0.0004,
        "hit_half_slots": 12.0,       # expert-cache hit rate = slots / (slots + this)
        "mtp_speedup": 1.32,          # ... divided by this when the model has an MTP head
        "attention_efficiency": 0.205,  # KV reads at depth (turbo3), measured 7.2 ms/token at 187k
        "gpu_overhead": 885 * 1024 ** 2 + 400 * 1024 ** 2,   # compute buffers, recurrent state, CUDA context
        "draft_kv_bytes_per_token": 2048,   # f16 KV of the MTP draft
        "unverified_headroom": 0.75,
        "max_slots": 24,                # measured: 24 best, 28 degraded at the time; beyond that is unmeasured
        "validated_min_cpu_layers": 20,     # layouts with more layers in VRAM than measured are flagged as extrapolated
    },
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
        "os": hw["os"]["family"],
        "fork": False,
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
            res["fork"] = bool(cfg["fork"]["enabled"] and g["vendor"] == "nvidia" and res["os"] == "linux")
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
        base = time_s / eta                                        # seconds per token with an empty window
        kv_bw = (res["gpu_bw"] if mode in ("hybrid", "gpu", "unified") else res["ram_bw"]) * eff["attention"]
        typical = cfg["fill_fraction"]                             # share of the window in use during a working session
        speed = {fill: 1.0 / (base + fill * kv / kv_bw) if base else 0.0 for fill in (0.0, typical, 1.0)}
        pool = (res["pool"] if g and g.get("unified") else res.get("vram_budget", 0)) \
            if mode in ("hybrid", "gpu", "unified") else res["ram_budget"]
        return {"mode": mode, "tok_s": speed[typical], "tok_empty": speed[0.0], "tok_full": speed[1.0],
                "fill_tokens": int(ctx * typical), "kv_bytes": kv, "kv_share": kv / pool if pool > 0 else 1.0,
                "ram_need": ram_need, "vram_need": vram_need, "settings": s}, ""

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


def _fit_fork(res, cfg, m, q, ctx):
    """Fork tier (NVIDIA + Linux, our CUDA fork): turbo3 KV, expert layers on the CPU plus a cache of hot experts in VRAM.

    Searches the number of expert layers left in RAM (`--n-cpu-moe`) together with the cache slots that the rest of the VRAM
    pays for, because the two compete for the same bytes. A model needs a `fork` block in the catalog (the turbo3 size of
    its KV and the window it was verified at). Returns (fit, reason) like `_fit`."""
    fk, f = m.get("fork"), cfg["fork"]
    if not fk or m["kind"] != "moe" or not res["fork"]:
        return None, "not available"
    n = m["n_layers"]
    budget, kv_main = res["vram_budget"], ctx * m["kv_bytes_per_token_f16"] * fk["kv_factor"]
    if ctx > fk.get("verified_context", m["context_max"]):
        budget *= f["unverified_headroom"]          # a window nobody has run: keep 25% of the VRAM budget free
    mtp = "mtp" in m.get("features", ())
    kv_draft = ctx * f["draft_kv_bytes_per_token"] if mtp else 0
    expert_bytes = q["file_bytes"] - q["nonexpert_bytes"]
    e_layer, e_slot = expert_bytes / n, expert_bytes / m["n_experts"] / n          # bytes of one layer / of one slot per CPU layer
    fixed = q["nonexpert_bytes"] + kv_main + kv_draft + f["gpu_overhead"]
    if fixed > budget:
        return None, f"needs {fixed / GiB:.1f} GiB of VRAM before any expert, only {budget / GiB:.1f} GiB is free"
    active = _active_bytes(m, q)
    routed = active * m["expert_active_fraction"]
    best, why = None, ""
    for n_cpu in range(n + 1):
        on_gpu = n - n_cpu
        left = budget - fixed - on_gpu * e_layer
        if left < 0:
            continue
        slots = min(f["max_slots"], int(left // (e_slot * n_cpu))) if n_cpu else 0
        ram_need = n_cpu * e_layer + cfg["host_base"]
        if ram_need > res["ram_budget"]:
            why = (f"needs {ram_need / GiB:.1f} GiB of RAM, only {res['ram_budget'] / GiB:.1f} GiB is available after "
                   f"keeping memory for the system")
            continue
        share = on_gpu / n
        hit = slots / (slots + f["hit_half_slots"]) if n_cpu else 0.0
        gpu_b = active - routed + routed * (share + (1 - share) * hit)
        cpu_b = routed * (1 - share) * (1 - hit)
        t = f["byte_time_scale"] * (gpu_b / res["gpu_bw"] + cpu_b / res["ram_bw"]) + f["sync_s_per_cpu_layer"] * n_cpu
        t /= f["mtp_speedup"] if mtp else 1.0
        if best is None or t < best["t"] * 0.99 or (t <= best["t"] * 1.01 and n_cpu > best["n_cpu"]):
            best = {"t": t, "n_cpu": n_cpu, "slots": slots, "vram": fixed + on_gpu * e_layer + slots * e_slot * n_cpu,
                    "ram": ram_need}
    if best is None:
        return None, why or "no layout fits"
    kv_bw = res["gpu_bw"] * f["attention_efficiency"]
    speed = {fill: 1.0 / (best["t"] + fill * kv_main / kv_bw) for fill in (0.0, cfg["fill_fraction"], 1.0)}
    pool = res["vram_budget"]
    s = {"context": ctx, "kv_type": fk.get("kv_type", "turbo3"), "flash_attn": True, "parallel": 1,
         "n_cpu_moe": best["n_cpu"], "moe_cache_slots": best["slots"], "mtp": mtp, "ngl": "all",
         "threads": min(res["cores"], 8), "numa": None}
    return {"mode": "hybrid" if best["n_cpu"] else "gpu", "tier": "fork",
            "tok_s": speed[cfg["fill_fraction"]], "tok_empty": speed[0.0], "tok_full": speed[1.0],
            "fill_tokens": int(ctx * cfg["fill_fraction"]), "kv_bytes": kv_main + kv_draft,
            "kv_share": (kv_main + kv_draft) / pool, "ram_need": best["ram"], "vram_need": best["vram"], "settings": s,
            "verified": ctx <= fk.get("verified_context", m["context_max"]),
            "extrapolated": best["n_cpu"] < f["validated_min_cpu_layers"]}, ""


def _best_fit_at(res, cfg, m, q, ctx):
    """The best placement of one (quant, window): the fork tier when this machine has it, else the standard one, with the
    KV cache as precise as fits (`q8_0`, then `q4_0`). Returns (fit, reason)."""
    fit, reason = _fit_fork(res, cfg, m, q, ctx)
    if fit:
        return fit, ""
    if res["fork"] and m.get("fork"):             # the fork tier is the plan on this machine; upstream's would be slower
        return None, reason
    for name, factor in cfg["kv_types"]:
        fit, why = _fit(res, cfg, m, q, ctx, name, factor)
        if fit:
            fit["tier"] = "standard"
            return fit, ""
        reason = why or reason
    return None, reason or "no context size fits"


def _windows(res, cfg, m, q):
    """{window: fit} for every window this quant can run. On a machine with a GPU, windows that would push the work off the
    GPU are dropped as long as any window keeps it in use."""
    out, reasons = {}, []
    for ctx in cfg["contexts"]:
        if ctx > m["context_max"] or ctx < cfg["min_context"]:
            continue
        fit, why = _best_fit_at(res, cfg, m, q, ctx)
        if fit:
            out[ctx] = fit
        else:
            reasons.append(why)
    if res["gpu"] and any(f["mode"] != "cpu" for f in out.values()):
        out = {c: f for c, f in out.items() if f["mode"] != "cpu"}
    return out, (reasons[0] if reasons else "")


def _select(res, cfg, m, forced=None):
    """Quant and window of one model.

    Three rungs of speed are tried in turn (target 35, comfortable 20, minimum 15 tok/s, judged with the window half full).
    On each rung the window is given up before the quality is: first the tested default quant with the biggest window that
    reaches the speed, but not smaller than 100k; then the next quants down (never below the catalog's `min_quant`) at
    100k or more; then the same with 64k, 32k and 16k. A quant above the default is taken only if it keeps the same window
    and a clear speed margin."""
    qs = sorted(m["quants"], key=lambda q: q["quality"])
    by_id = {q["id"]: q for q in qs}
    default = by_id[m["default_quant"]]
    floor_q = by_id.get(m.get("min_quant"), qs[0])
    if forced:
        if forced not in by_id:
            return None
        allowed = [by_id[forced]]
    else:
        allowed = [q for q in qs if q["quality"] >= floor_q["quality"]]
    win, reasons = {}, []
    for q in allowed:
        win[q["id"]], why = _windows(res, cfg, m, q)
        reasons.append(why)
    if not any(win.values()):
        return {"model": m, "fit": None, "reason": next((r for r in reasons if r), "nothing fits"), "quant": None}
    order = ([default] if default in allowed else []) + [q for q in reversed(allowed) if q["quality"] < default["quality"]]
    order = order or allowed
    above = [q for q in reversed(allowed) if q["quality"] > default["quality"]]

    def reaches(f, thr):                            # a window nobody has run needs a clear margin on top
        return f["tok_s"] >= thr * (1.0 if f.get("verified", True) else cfg["upgrade_margin"])

    def best_window(q, thr, lo):
        ok = [c for c, f in win[q["id"]].items() if reaches(f, thr) and c >= min(lo, m["context_max"])]
        return max(ok) if ok else None

    chosen = None
    useful = tuple(lo for lo in cfg["window_floors"] if lo >= cfg["useful_window"])
    small = tuple(lo for lo in cfg["window_floors"] if lo < cfg["useful_window"])
    target, comfort, minimum = cfg["target_tok_s"], cfg["comfort_tok_s"], cfg["min_tok_s"]
    rungs = ([(t, "target" if t == target else "comfort", lo) for lo in useful for t in (target, comfort)]
             + [(t, "target" if t == target else "comfort", lo) for lo in small for t in (target, comfort)]
             + [(minimum, "minimum", lo) for lo in cfg["window_floors"]])
    for thr, label, lo in rungs:
        for q in order:
            ctx = best_window(q, thr, lo)
            if ctx:
                chosen = (q, ctx, thr, label, lo)
                break
        if chosen:
            break
    if chosen:
        q, ctx, thr, label, lo = chosen
        if label != "target":                       # the target is out of reach: the window is capacity, as big as memory
            ctx = best_window(q, minimum, lo)       # allows while the speed stays usable
        if not forced and q is default and label == "target":
            for up in above:
                f = win[up["id"]].get(ctx)
                if f and reaches(f, thr * cfg["upgrade_margin"]):
                    q = up
                    break
        why = {"target": "reaches the target speed", "comfort": "the target speed is out of reach; this is the best that "
               "still feels responsive", "minimum": "nothing reaches the comfortable speed; this is the best that is still "
               "usable"}[label]
        if q is default:
            why += " with the catalog's tested quant" if forced is None else ""
        elif q["quality"] > default["quality"]:
            why += " and a higher-quality quant still keeps a clear speed margin"
        else:
            why += (" with a smaller quant: the tested default does not reach it with a useful window on this machine"
                    if not forced else "")
        fit, thr_met = win[q["id"]][ctx], thr
    else:                                           # nothing reaches even the minimum: the fastest that fits
        cands = [(f["tok_s"], c, q) for q in allowed for c, f in win[q["id"]].items()]
        _, ctx, q = max(cands, key=lambda t: (t[0], t[1]))
        fit, thr_met = win[q["id"]][ctx], 0.0
        why = "nothing reaches the minimum speed; this is the fastest that fits"
    alts = []
    for a in reversed(allowed):
        if a is not q and win[a["id"]]:
            c, f = max(win[a["id"]].items())
            if f["tok_s"] >= cfg["min_tok_s"]:
                alts.append({"model": m["id"], "quant": a["id"], "context": c, "tok_s": round(f["tok_s"], 1)})
    return {"model": m, "quant": q, "fit": fit, "window": fit, "why": why, "alts": alts, "reason": "", "thr": thr_met,
            "rung": label if chosen else "none"}


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
        p = _select(res, cfg, m, quant if model else None)
        (picks if p and p["fit"] else refusals).append(p or {"model": m, "reason": f"unknown quant {quant!r}"})
    floor = cfg["min_tok_s"]
    # between models the comfortable speed decides (the target only steers quant and window inside one model), so a
    # far more capable model is not given up for a weaker one just because the weaker one is faster
    pool = ([p for p in picks if p["thr"] >= cfg["comfort_tok_s"]] or [p for p in picks if p["thr"] >= floor]
            or sorted(picks, key=lambda p: -p["fit"]["tok_s"])[:1])
    if not pool:
        why = "; ".join(f"{r['model']['name']}: {r['reason']}" for r in refusals) or "the catalog is empty"
        return {"ok": False, "reasons": [f"no model in the catalog fits this machine ({why})"],
                "backend_candidates": backends, "needs_probe": probe}
    best = max(pool, key=lambda p: (p["model"]["capability"], p["quant"]["quality"]))
    fit = best["window"]
    mid = fit["tok_s"]
    lo, hi = cfg["uncertainty"]
    warnings = []
    if mid < floor:
        warnings.append(f"predicted {mid:.0f} tok/s is below the {floor:.0f} tok/s minimum; this is the best "
                        f"available on this machine")
    if not res["ram_bw_measured"] and fit["mode"] != "unified":
        warnings.append("RAM bandwidth was not measured, a conservative estimate was used")
    if res["numa_nodes"] > 1:
        warnings.append(f"{res['numa_nodes']} NUMA nodes: calibration should compare --numa modes")
    if fit.get("extrapolated"):
        warnings.append("this layout keeps more of the model in VRAM than the measured machine did; the speed is an "
                        "extrapolation and calibration decides")
    confidence = "low" if (res["numa_nodes"] > 1 or fit["mode"] == "unified" or not res["ram_bw_measured"]) else "medium"
    reasons = [best["why"], f"mode: {fit['mode']} ({fit.get('tier', 'standard')} tier), "
                            f"{fit['settings']['context']} tokens of context, {fit['settings']['kv_type']} KV cache"]
    return {
        "ok": True,
        "model": best["model"]["id"], "model_name": best["model"]["name"], "quant": best["quant"]["id"],
        "backend_candidates": backends, "needs_probe": probe,
        "speed_rung": best["rung"],
        "settings": fit["settings"], "mode": fit["mode"], "tier": fit.get("tier", "standard"),
        "predicted_tok_s": {"low": round(mid * lo, 1), "mid": round(mid, 1), "high": round(mid * hi, 1)},
        "speed_by_fill": {"empty": round(fit["tok_empty"], 1), "typical": round(mid, 1),
                          "full": round(fit["tok_full"], 1), "typical_tokens": fit["fill_tokens"]},
        "confidence": confidence,
        "memory": {"ram_need": int(fit["ram_need"]), "ram_budget": int(res["ram_budget"]),
                   "vram_need": int(fit["vram_need"]), "kv_bytes": int(fit["kv_bytes"]),
                   "kv_share": round(fit["kv_share"], 3),
                   "vram_budget": int(res.get("vram_budget", res.get("pool", 0)))},
        "reasons": reasons, "warnings": warnings,
        "alternatives": (best["alts"] + [{"model": p["model"]["id"], "quant": p["quant"]["id"],
                                          "context": p["fit"]["settings"]["context"], "tok_s": round(p["fit"]["tok_s"], 1)}
                                         for p in picks if p is not best])[:4],
    }
