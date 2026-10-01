"""The universal planner: hardware profile + model catalog -> plan (model, quant, settings, prediction).

No I/O, stdlib only, deterministic. The decision rests on one idea: token generation is limited by memory
bandwidth, so speed ~ efficiency * bandwidth / bytes of active weights read per token, where the bytes are split
between fast memory (VRAM or unified memory) and system RAM. The efficiency constants below were calibrated on ONE
machine (RTX 4070 SUPER + DDR4-3200, upstream-style `--n-cpu-moe` layout) and are deliberately conservative: the
prediction is for choosing and ranking, never a promise. A short measurement on the user's machine (calibration)
corrects it. See docs/HARDWARE.md.
"""
import copy
import math

from .schema import validate_catalog, validate_hardware

GiB = 1024 ** 3
GB = 10 ** 9

DEFAULTS = {
    "target_tok_s": 35.0,         # what a plan aims for, judged with the window half full (see fill_fraction)
    "weights": {"quality": 1.0, "window": 0.4, "speed": 0.3},   # exponents of the score: what matters how much (see `score`)
    "mtp_standard": True,         # use the model's own MTP head (speculative decoding) with upstream llama.cpp when there is a GPU
    "mtp_speedup": 1.15,          # unmeasured outside the CUDA fork (there x1.32), so conservative; calibration corrects it
    "mtp_draft_kv_per_token": 2048,   # the MTP draft keeps its own f16 KV cache (measured in the fork: 2 KiB per token)
    "mtp_compute": 236 * 1024 ** 2,   # and a compute buffer
    "default_bonus": 1.03,        # the catalog's tested quant is worth 2% more than an untested one of the same bits
    "attention_by_kv": {"turbo3": 0.16},   # KV read efficiency per type where it differs (turbo3 dequantises on the fly)
    "tqp_enabled": True,          # may the installer use the third-party TurboQuant+ llama.cpp build (turbo KV, expert cache)?
    "tqp_kv_factor": 0.3625,      # K stays q8_0 (auto-asymmetric on GQA 8:1) and V is turbo3: (0.53 + 0.195) / 2 of f16, seen in its log
    "tqp_time_penalty": 1.42,     # its Vulkan kernels vs our CUDA fork's: measured 34.4 vs 54 tok/s on an RTX 4070 SUPER (empty window)
    "tqp_trust": 1.0,             # measured end to end on one machine (NVIDIA over Vulkan); AMD is still unmeasured
    "kv_quality": {"f16": 1.0, "q8_0": 1.0, "turbo3": 0.985, "turbo3_tqp": 0.99, "q5_0": 0.99, "q4_0": 0.975},   # share of quality kept by each KV cache type
    "kv_exotic": ("q5_0",),       # KV types the common builds may not run with flash attention (CUDA/HIP prebuilts: only q8_0/q4_0 pairs)
    "kv_unavailable": (),         # KV types the engine on this machine cannot run
    "unverified_margin": 1.15,    # a window nobody has run (beyond the verified one) is taken only with 15% more speed than the target
    "target_tolerance": 0.95,     # within 5% of the target counts as reaching it (measured run-to-run spread is 5-8%)
    "speed_scale": 1.0,           # measured speed / predicted speed on this machine (set by the installer after a real run)
    "min_tok_s": 15.0,            # below this a model is not recommended at all
    "comfort_tok_s": 20.0,        # if the target is out of reach: the best that reaches this, then min_tok_s
    "fill_fraction": 0.5,         # speed is judged with this share of the window filled (a working session, not an empty chat)
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
    "kv_types": (("q8_0", 0.53), ("q5_0", 0.34), ("q4_0", 0.28), ("turbo3", 0.3625)),   # (name, size relative to f16); turbo3 needs a TurboQuant+ build, which keeps K at q8_0 on GQA 8:1 models (measured)
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

# What the user can ask for instead of the balanced default. They are only different weights on the same selection rules.
PROFILES = {
    "balanced": {},
    "fast": {"weights": {"speed": 1.2, "window": 0.25}, "target_tok_s": 50.0},
    "long": {"weights": {"window": 0.9, "speed": 0.15}},
    "quality": {"weights": {"quality": 4.0}},
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
        "arch": hw["os"]["arch"],
        "fork": False,
        "tqp": None,
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
    res["tqp"] = None if (res["fork"] or not cfg["tqp_enabled"]) else _tqp_backend(res, gpus[0] if gpus else None)
    return res


def _tqp_backend(res, gpu):
    """Backend of the prebuilt TurboQuant+ llama.cpp that fits this machine, or None: Vulkan on Linux x86_64 (any GPU), Metal on
    Apple Silicon, CUDA on Windows with an NVIDIA card. CPU-only machines and other combinations use upstream llama.cpp."""
    if not gpu:
        return None
    if res["os"] == "linux" and res["arch"] == "x86_64" and not gpu.get("unified"):
        return "vulkan"
    if res["os"] == "macos" and res["arch"] == "arm64" and gpu.get("unified"):
        return "metal"
    if res["os"] == "windows" and res["arch"] == "x86_64" and gpu["vendor"] == "nvidia":
        return "cuda"
    return None


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
    mtp = bool(g and cfg["mtp_standard"] and "mtp" in m.get("features", ()))
    draft = (ctx * cfg["mtp_draft_kv_per_token"] + cfg["mtp_compute"]) if mtp else 0      # VRAM of the MTP draft

    def finish(mode, time_s, ram_need, vram_need, ngl, eta):
        if ram_need > res["ram_budget"]:
            return None, (f"needs {ram_need / GiB:.1f} GiB of RAM, only {res['ram_budget'] / GiB:.1f} GiB is "
                          f"available after keeping memory for the system")
        s["ngl"] = ngl
        s["threads"] = min(res["cores"], 8) if mode in ("hybrid", "gpu", "unified") else res["cores"]
        s["numa"] = "distribute" if res["numa_nodes"] > 1 and mode in ("cpu", "hybrid") else None
        base = time_s / eta                                        # seconds per token with an empty window
        if mtp and mode != "cpu":
            base /= cfg["mtp_speedup"]
            s["mtp"] = True
        kv_bw = (res["gpu_bw"] if mode in ("hybrid", "gpu", "unified") else res["ram_bw"]) * eff["attention"]
        typical = cfg["fill_fraction"]                             # share of the window in use during a working session
        speed = {fill: cfg["speed_scale"] / (base + fill * kv / kv_bw) if base else 0.0 for fill in (0.0, typical, 1.0)}
        pool = (res["pool"] if g and g.get("unified") else res.get("vram_budget", 0)) \
            if mode in ("hybrid", "gpu", "unified") else res["ram_budget"]
        return {"mode": mode, "tok_s": speed[typical], "tok_empty": speed[0.0], "tok_full": speed[1.0],
                "fill_tokens": int(ctx * typical), "kv_bytes": kv, "kv_share": kv / pool if pool > 0 else 1.0,
                "ram_need": ram_need, "vram_need": vram_need, "settings": s}, ""

    if g and g.get("unified"):                                    # Apple-style unified memory
        need = file + kv + draft + cfg["compute_buffer"]
        if need > res["pool"]:
            return None, f"needs {need / GiB:.1f} GiB of unified memory, the GPU may use {res['pool'] / GiB:.1f} GiB"
        return finish("unified", active / res["gpu_bw"], cfg["host_base"], need, "all", eff["unified"])

    budget = res.get("vram_budget", 0) if g else 0
    if g and m["kind"] == "moe":
        expert_bytes = file - q["nonexpert_bytes"]
        fixed = q["nonexpert_bytes"] + kv + draft + cfg["compute_buffer"]
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
        fixed = kv + draft + cfg["compute_buffer"]
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


def _fit_fork(res, cfg, m, q, ctx, tier="fork"):
    """Fork tier (NVIDIA + Linux, our CUDA fork) or `tier="tqp"` (the TurboQuant+ build on a discrete GPU): turbo3 KV, expert
    layers on the CPU plus a cache of hot experts in VRAM. Both are the same idea; the TurboQuant+ numbers are the fork's with
    a time penalty, because nobody has measured that build on other hardware yet.

    Searches the number of expert layers left in RAM (`--n-cpu-moe`) together with the cache slots that the rest of the VRAM
    pays for, because the two compete for the same bytes. A model needs a `fork` block in the catalog (the turbo3 size of
    its KV and the window it was verified at). Returns (fit, reason) like `_fit`."""
    fk, f = m.get("fork"), dict(cfg["fork"])
    if tier == "fork":
        usable = res["fork"]
    else:
        usable = bool(res["tqp"]) and not (res["gpu"] or {}).get("unified") and "turbo3" not in cfg["kv_unavailable"]
        f["byte_time_scale"] *= cfg["tqp_time_penalty"]
        f["mtp_speedup"] = cfg["mtp_speedup"]
    if not fk or m["kind"] != "moe" or not usable:
        return None, "not available"
    n = m["n_layers"]
    kv_factor = fk["kv_factor"] if tier == "fork" else cfg["tqp_kv_factor"]
    budget, kv_main = res["vram_budget"], ctx * m["kv_bytes_per_token_f16"] * kv_factor
    if ctx > fk.get("verified_context", m["context_max"]):
        budget *= f["unverified_headroom"]          # a window nobody has run: keep 25% of the VRAM budget free
    mtp = "mtp" in m.get("features", ()) and (tier == "fork" or cfg["mtp_standard"])
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
    speed = {fill: cfg["speed_scale"] / (best["t"] + fill * kv_main / kv_bw) for fill in (0.0, cfg["fill_fraction"], 1.0)}
    pool = res["vram_budget"]
    s = {"context": ctx, "kv_type": fk.get("kv_type", "turbo3"), "flash_attn": True, "parallel": 1,
         "n_cpu_moe": best["n_cpu"], "moe_cache_slots": best["slots"], "mtp": mtp, "ngl": "all",
         "threads": min(res["cores"], 8), "numa": None}
    if tier == "tqp":                       # the TurboQuant+ build takes a VRAM budget for the cache, not a slot count
        s["moe_cache_mib"] = int(best["slots"] * e_slot * best["n_cpu"] / (1024 ** 2))
    extra = {"engine": "turboquant-plus", "backend": res["tqp"]} if tier == "tqp" else {}
    return {"mode": "hybrid" if best["n_cpu"] else "gpu", "tier": tier, **extra,
            "tok_s": speed[cfg["fill_fraction"]], "tok_empty": speed[0.0], "tok_full": speed[1.0],
            "fill_tokens": int(ctx * cfg["fill_fraction"]), "kv_bytes": kv_main + kv_draft,
            "kv_share": (kv_main + kv_draft) / pool, "ram_need": best["ram"], "vram_need": best["vram"], "settings": s,
            "verified": ctx <= fk.get("verified_context", m["context_max"]),
            "extrapolated": best["n_cpu"] < f["validated_min_cpu_layers"]}, ""


def _options_at(res, cfg, m, q, ctx):
    """Every placement worth scoring for one (quant, window): the fork tier when this machine has it; else the TurboQuant+ build
    (if allowed and available) and upstream llama.cpp with each KV cache type that fits. Returns (list of fits, reason)."""
    fit, reason = _fit_fork(res, cfg, m, q, ctx)
    if fit:
        return [fit], ""
    if res["fork"] and m.get("fork"):             # the fork tier is the plan on this machine; upstream's would be slower
        return [], reason
    fits = []
    discrete_moe = bool(m.get("fork")) and m["kind"] == "moe" and not (res["gpu"] or {}).get("unified")
    if res["tqp"] and discrete_moe:
        t, why = _fit_fork(res, cfg, m, q, ctx, tier="tqp")
        if t:
            fits.append(t)
        else:
            reason = why or reason
    for name, factor in cfg["kv_types"]:
        if name in cfg["kv_unavailable"] or (name == "turbo3" and (not res["tqp"] or discrete_moe)):
            continue
        f, why = _fit(res, cfg, m, q, ctx, name, factor)
        if f:
            f["tier"] = "tqp" if name == "turbo3" else "standard"
            if name == "turbo3":
                f["engine"], f["backend"] = "turboquant-plus", res["tqp"]
            fits.append(f)
        else:
            reason = why or reason
    return fits, ("" if fits else reason or "no context size fits")


def _bits_per_weight(m, q):
    return q["file_bytes"] * 8 / (m["params_total_b"] * 1e9)


def quality_of(m, q, cfg):
    """Prior for how much of the full model's quality a quant keeps (1 = lossless), from its bits per weight. Quality falls off
    quickly below about 4 bits. A prior, not a measurement: the catalog's tested default also gets a small bonus."""
    bpw = _bits_per_weight(m, q)
    base = 1.0 - 0.5 * math.exp(-(bpw - 2.0) * 1.1)
    return base * (cfg["default_bonus"] if q["id"] == m["default_quant"] else 1.0)


def speed_credit(tok, cfg):
    """How much a predicted speed is worth (1 at the target). Predictions are only good to about +-40%, so a margin over the
    target keeps counting up to 1.3x the target; far above that more speed is a windfall and counts for almost nothing. Below
    the target the slope is soft, below the comfortable speed steeper. No cliff: 34 tok/s is worth 99% of 35."""
    target, comfort = cfg["target_tok_s"], cfg["comfort_tok_s"]
    cap = 1.3 * target
    credit = (min(tok, cap) / target) ** cfg["weights"]["speed"]
    if tok > cap:
        credit *= 1.0 + 0.02 * math.log(tok / cap)
    if tok < comfort:
        credit *= (tok / comfort) ** 0.5
    return credit


def score(m, q, fit, cfg):
    """One number for a candidate: quality of the quant and of the KV cache, the window, the speed. Multiplicative, so a
    collapse in one of them cannot be bought back by the others. Weights are exponents (cfg['weights'])."""
    w = cfg["weights"]
    window = (fit["settings"]["context"] / m["context_max"]) ** w["window"]
    kvt = fit["settings"]["kv_type"]
    kv = cfg["kv_quality"].get("turbo3_tqp" if (fit.get("tier") == "tqp" and kvt == "turbo3") else kvt, 1.0)
    trust = cfg["tqp_trust"] if fit.get("tier") == "tqp" else 1.0
    return (quality_of(m, q, cfg) ** w["quality"]) * kv * window * speed_credit(fit["tok_s"], cfg) * trust


def _select(res, cfg, m, forced=None):
    """Quant, window and KV cache of one model: the candidate with the best score among everything that fits.

    Hard rules: the quant is not below the catalog's `min_quant` unless forced, and nothing slower than the minimum speed is
    taken while a faster option exists. Everything else is one smooth objective (see `score`), so there are no thresholds
    that flip a plan for a single token per second."""
    qs = sorted(m["quants"], key=lambda q: q["quality"])
    by_id = {q["id"]: q for q in qs}
    floor_q = by_id.get(m.get("min_quant"), qs[0])
    if forced:
        if forced not in by_id:
            return None
        allowed = [by_id[forced]]
    else:
        allowed = [q for q in qs if q["quality"] >= floor_q["quality"]]
    cands, reasons = [], []
    for q in allowed:
        got = []
        for ctx in cfg["contexts"]:
            if ctx > m["context_max"] or ctx < cfg["min_context"]:
                continue
            fits, why = _options_at(res, cfg, m, q, ctx)
            got += [(q, f) for f in fits]
            if not fits:
                reasons.append(why)
        if res["gpu"] and any(f["mode"] != "cpu" for _, f in got):     # keep the GPU in use if any window allows it
            got = [(qq, f) for qq, f in got if f["mode"] != "cpu"]
        cands += got
    if not cands:
        return {"model": m, "fit": None, "reason": next((r for r in reasons if r), "nothing fits"), "quant": None}
    # evidence first: a window beyond the verified one needs a clear speed margin (it leaves room, so it is not a gamble)
    cands = [c for c in cands if c[1].get("verified", True) or c[1]["tok_s"] >= cfg["target_tok_s"] * cfg["unverified_margin"]
             ] or cands
    usable = [c for c in cands if c[1]["tok_s"] >= cfg["min_tok_s"]]
    pool = usable or cands
    q, fit = max(pool, key=lambda c: (score(m, c[0], c[1], cfg), c[1]["settings"]["context"]))
    best_score = score(m, q, fit, cfg)
    tok = fit["tok_s"]
    rung = ("target" if tok >= cfg["target_tok_s"] * cfg["target_tolerance"] else "comfort" if tok >= cfg["comfort_tok_s"]
            else "minimum" if tok >= cfg["min_tok_s"] else "none")
    why = {"target": "reaches the target speed", "comfort": "the target speed is out of reach with this window; the speed is "
           "still comfortable", "minimum": "the speed is usable but low", "none": "nothing reaches the minimum speed; this "
           "is the fastest that fits"}[rung]
    if not usable:
        fit = max(cands, key=lambda c: c[1]["tok_s"])[1]
        q = max(cands, key=lambda c: c[1]["tok_s"])[0]
        best_score = score(m, q, fit, cfg)
    elif q["id"] != m["default_quant"] and not forced:
        why += (" with a higher-quality quant than the tested default" if q["quality"] > by_id[m["default_quant"]]["quality"]
                else " with a smaller quant than the tested default (it does not fit or is too slow here)")
    alts = []
    for a in reversed(allowed):
        if a is not q:
            rest = [(aa, ff) for aa, ff in pool if aa is a]
            if rest:
                aq, af = max(rest, key=lambda c: (score(m, c[0], c[1], cfg), c[1]["settings"]["context"]))
                alts.append({"model": m["id"], "quant": a["id"], "context": af["settings"]["context"],
                             "tok_s": round(af["tok_s"], 1)})
    return {"model": m, "quant": q, "fit": fit, "window": fit, "why": why, "alts": alts, "reason": "", "thr": tok,
            "rung": rung, "score": round(best_score, 4)}


def plan(hw, catalog, *, config=None, model=None, quant=None, profile="balanced", menu=True):
    """Return a plan dict. Raises ValueError if the hardware profile or the catalog is malformed.

    `profile` is what to optimise for (see PROFILES). With `menu` the plan also carries `profiles`: what each profile would
    choose on this machine, so that the user sees the alternatives instead of one hidden decision."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose one of {', '.join(PROFILES)}")
    result = _plan(hw, catalog, {**(config or {}), **PROFILES[profile]}, model, quant)
    result["profile"] = profile
    if menu and result["ok"]:
        table = {}
        for name in PROFILES:
            p = result if name == profile else _plan(hw, catalog, {**(config or {}), **PROFILES[name]}, model, quant)
            if p["ok"]:
                table[name] = {"quant": p["quant"], "context": p["settings"]["context"], "kv_type": p["settings"]["kv_type"],
                               "tok_s": p["speed_by_fill"]["typical"], "tok_s_full": p["speed_by_fill"]["full"],
                               "speed_rung": p["speed_rung"]}
        result["profiles"] = table
    return result


def _plan(hw, catalog, config, model, quant):
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
        "engine": fit.get("engine", "llama.cpp"),
        "backend_candidates": [fit["backend"]] if fit.get("tier") == "tqp" else backends,
        "needs_probe": False if fit.get("tier") == "tqp" else probe,
        "speed_rung": best["rung"], "score": best["score"],
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
