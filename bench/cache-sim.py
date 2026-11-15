#!/usr/bin/env python3
"""Offline expert-cache replay. Reads GGUF metadata and routing CSVs only; never starts a model."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import struct
import tarfile
import tempfile
from pathlib import Path


# GGML type enum -> (short name, block elements, encoded bytes/block), from ggml.c and ggml-common.h.
QUANT_LAYOUT = {
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    14: ("Q6_K", 256, 210),
    23: ("IQ4_XS", 256, 136),
}
CPU_LAYER_COUNT = 26
GPU_LAYERS = tuple(range(26, 40))
ALL_LAYERS = tuple(range(40))
TOP_K_FALLBACK = 8


def quantized_bytes_per_expert(dimensions, expert_count, quant_type):
    if quant_type not in QUANT_LAYOUT:
        raise ValueError(f"unsupported GGUF quant type {quant_type}")
    _, block_elements, block_bytes = QUANT_LAYOUT[quant_type]
    if not dimensions or dimensions[-1] != expert_count:
        raise ValueError(f"expert axis {expert_count} not found in tensor shape {dimensions}")
    elements = math.prod(dimensions[:-1])
    if elements % block_elements:
        raise ValueError(f"expert tensor element count {elements} is not block aligned")
    return (elements // block_elements) * block_bytes


def _u32(f):
    raw = f.read(4)
    if len(raw) != 4:
        raise ValueError("truncated GGUF uint32")
    return struct.unpack("<I", raw)[0]


def _u64(f):
    raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("truncated GGUF uint64")
    return struct.unpack("<Q", raw)[0]


def _string(f):
    size = _u64(f)
    raw = f.read(size)
    if len(raw) != size:
        raise ValueError("truncated GGUF string")
    return raw.decode("utf-8", errors="replace")


SCALAR = {0: ("B", "u8"), 1: ("b", "i8"), 2: ("H", "u16"), 3: ("h", "i16"),
          4: ("I", "u32"), 5: ("i", "i32"), 6: ("f", "f32"), 7: ("B", "bool"),
          10: ("Q", "u64"), 11: ("q", "i64"), 12: ("d", "f64")}


def _metadata_value(f, kind, keep=False):
    if kind == 8:
        value = _string(f)
        return value if keep else None
    if kind == 9:
        element_kind, count = _u32(f), _u64(f)
        if element_kind == 8:
            values = [] if keep else None
            for _ in range(count):
                value = _string(f)
                if keep:
                    values.append(value)
            return values
        if element_kind == 9:
            values = [_metadata_value(f, _u32(f), keep=keep) for _ in range(count)] if keep else None
            if not keep:
                for _ in range(count):
                    nested = _u32(f)
                    _metadata_value(f, nested, keep=False)
            return values
        fmt = SCALAR.get(element_kind, (None,))[0]
        if fmt is None:
            raise ValueError(f"unsupported GGUF array type {element_kind}")
        width = struct.calcsize("<" + fmt)
        if not keep:
            f.seek(count * width, 1)
            return None
        return [struct.unpack("<" + fmt, f.read(width))[0] for _ in range(count)]
    fmt = SCALAR.get(kind, (None,))[0]
    if fmt is None:
        raise ValueError(f"unsupported GGUF metadata type {kind}")
    width = struct.calcsize("<" + fmt)
    raw = f.read(width)
    if len(raw) != width:
        raise ValueError("truncated GGUF metadata")
    return struct.unpack("<" + fmt, raw)[0] if keep else None


def read_gguf_experts(path):
    """Read only GGUF header/tensor table. No tensor payload is copied or mmap'd."""
    path = Path(path)
    metadata = {}
    tensors = {}
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        version = _u32(f)
        tensor_count, metadata_count = _u64(f), _u64(f)
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        for _ in range(metadata_count):
            key = _string(f)
            kind = _u32(f)
            keep = key == "general.architecture" or key.endswith((".expert_count", ".expert_used_count", ".block_count"))
            value = _metadata_value(f, kind, keep)
            if keep:
                metadata[key] = value
        arch = metadata.get("general.architecture")
        if not arch:
            raise ValueError("GGUF has no general.architecture metadata")
        def meta_suffix(suffix):
            return next((v for k, v in metadata.items() if k.endswith("." + suffix)), None)
        expert_count = int(meta_suffix("expert_count") or 0)
        experts_per_token = int(meta_suffix("expert_used_count") or TOP_K_FALLBACK)
        block_count = int(meta_suffix("block_count") or 0)
        if expert_count <= 0:
            raise ValueError("GGUF is missing expert_count")
        rx = re.compile(r"blk\.(\d+)\.ffn_(up|down|gate)_exps\.weight$")
        for _ in range(tensor_count):
            name = _string(f)
            dimensions = tuple(_u64(f) for _ in range(_u32(f)))
            quant_type = _u32(f)
            offset = _u64(f)
            match = rx.fullmatch(name)
            if not match:
                continue
            layer, projection = int(match.group(1)), match.group(2)
            # Some MTP GGUFs contain an extra expert layer (for example blk.40); the routing captures stop at the 40 base layers.
            if layer >= len(ALL_LAYERS):
                continue
            if quant_type not in QUANT_LAYOUT:
                raise ValueError(f"unsupported GGUF quant type {quant_type} for {name}")
            qname, block_elements, block_bytes = QUANT_LAYOUT[quant_type]
            per_expert_elements = math.prod(dimensions[:-1])
            byte_count = quantized_bytes_per_expert(dimensions, expert_count, quant_type)
            nblocks = per_expert_elements // block_elements
            tensors.setdefault(layer, {})[projection] = {
                "name": name, "shape": dimensions, "type_id": quant_type, "type": qname,
                "expert_elements": per_expert_elements, "blocks_per_expert": nblocks,
                "bytes_per_expert": byte_count, "offset": offset,
            }
        expected_layers = set(range(min(block_count, len(ALL_LAYERS)))) if block_count else set(tensors)
        missing = [(layer, projection) for layer in expected_layers for projection in ("up", "down", "gate")
                   if projection not in tensors.get(layer, {})]
        if missing:
            raise ValueError(f"GGUF missing routed expert tensor(s), first: {missing[:3]}")
    expert_bytes = {layer: sum(tensors[layer][p]["bytes_per_expert"] for p in ("up", "down", "gate"))
                    for layer in tensors}
    return {
        "path": str(path), "file_bytes": path.stat().st_size, "version": version,
        "architecture": arch, "expert_count": expert_count,
        "experts_per_token": experts_per_token, "block_count": block_count,
        "layers": tensors, "expert_bytes": expert_bytes,
    }


def read_trace_csv(path, name=None, expected_layers=ALL_LAYERS, include_negative=False):
    """Read token,layer,8-expert rows and preserve token-index reset boundaries."""
    path = Path(path)
    sequences = []
    current_tokens = []
    pending_index = None
    pending_layers = {}
    previous_index = None
    start_index = None
    negative_positions = None

    def flush_token():
        nonlocal pending_index, pending_layers, previous_index
        if pending_index is None:
            return
        if set(pending_layers) != set(expected_layers):
            raise ValueError(f"{path}: token {pending_index} has {len(pending_layers)} layers, expected {len(expected_layers)}")
        current_tokens.append(tuple(pending_layers[layer] for layer in expected_layers))
        previous_index = pending_index
        pending_index, pending_layers = None, {}

    def flush_sequence():
        nonlocal current_tokens, previous_index, start_index
        flush_token()
        if current_tokens:
            sequences.append({"name": name or path.stem, "start_index": start_index,
                              "positions": current_tokens})
        current_tokens, previous_index, start_index = [], None, None

    with path.open(newline="", encoding="utf-8") as f:
        for row_number, row in enumerate(csv.reader(f), 1):
            if not row:
                continue
            if len(row) != 10:
                raise ValueError(f"{path}:{row_number}: expected token, layer, 8 experts; got {len(row)} columns")
            token, layer, *experts = map(int, row)
            # llama.cpp's runtime profile parser skips negative positions (prefill rows).
            if token < 0 and not include_negative:
                continue
            if layer not in expected_layers or any(e < 0 for e in experts):
                raise ValueError(f"{path}:{row_number}: invalid layer or expert id")
            if token < 0:
                if negative_positions is None:
                    negative_positions = {}
                layer_map = negative_positions.setdefault(token, {})
                if layer in layer_map:
                    raise ValueError(f"{path}:{row_number}: duplicate negative-position layer")
                layer_map[layer] = tuple(experts)
                continue
            if negative_positions is not None:
                for negative_index in sorted(negative_positions):
                    layer_map = negative_positions[negative_index]
                    if set(layer_map) == set(expected_layers):
                        current_tokens.append(tuple(layer_map[x] for x in expected_layers))
                negative_positions = None
            if layer == expected_layers[0]:
                if pending_index is not None:
                    flush_token()
                if previous_index is not None and token <= previous_index:
                    flush_sequence()
                if start_index is None:
                    start_index = token
                pending_index = token
            if pending_index != token:
                raise ValueError(f"{path}:{row_number}: token/layer rows are not ordered")
            if layer in pending_layers:
                raise ValueError(f"{path}:{row_number}: duplicate layer {layer} for token {token}")
            pending_layers[layer] = tuple(experts)
            if layer == expected_layers[-1]:
                flush_token()
                previous_index = token
    if negative_positions is not None:
        for negative_index in sorted(negative_positions):
            layer_map = negative_positions[negative_index]
            if set(layer_map) == set(expected_layers):
                current_tokens.append(tuple(layer_map[x] for x in expected_layers))
    flush_sequence()
    if not sequences:
        raise ValueError(f"empty routing trace: {path}")
    return sequences


def flatten(sequences):
    return [position for sequence in sequences for position in sequence["positions"]]


def route_statistics(sequences, experts_per_token=8):
    positions = flatten(sequences)
    duplicate_assignments = 0
    for token in positions:
        for route in token:
            duplicate_assignments += len(route) - len(set(route))
            if len(route) != experts_per_token:
                raise ValueError(f"route has {len(route)} experts, expected {experts_per_token}")
    return {"positions": len(positions), "sequences": len(sequences),
            "layers": len(positions[0]) if positions else 0,
            "duplicate_route_assignments": duplicate_assignments}


def exact_route_overlap(left, right):
    """Count identical complete per-token/per-layer routing vectors in two splits."""
    a=collections.Counter(tuple(tuple(route) for route in p) for p in flatten(left))
    b=collections.Counter(tuple(tuple(route) for route in p) for p in flatten(right))
    return sum(min(count,b.get(vector,0)) for vector,count in a.items())


def raw_trace_counts(path, layer_count=len(ALL_LAYERS)):
    rows = negative = 0
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            rows += 1
            negative += int(int(row[0]) < 0)
    return {"rows": rows, "raw_positions": rows//layer_count,
            "negative_rows": negative, "negative_positions": negative//layer_count}


def profile_counts(sequences, gpu_layers=GPU_LAYERS):
    counts = {layer: collections.Counter() for layer in gpu_layers}
    for position in flatten(sequences):
        for layer in gpu_layers:
            counts[layer].update(position[layer])
    return counts


def top_cache(counts, capacity, gpu_layers=GPU_LAYERS):
    return {layer: {expert for expert, _ in counts[layer].most_common(capacity)} for layer in gpu_layers}


def _empty_stats(layers):
    return {"hits": collections.Counter(), "calls": collections.Counter(),
            "misses": collections.Counter(), "cpu_bytes": 0, "pcie_bytes": 0,
            "replacements": 0, "replacement_bytes": 0, "tokens": 0, "steps": 0,
            "replacement_by_layer": collections.Counter(), "replacements_each_token": []}


def evaluate_static(cache, sequences, layer_bytes, cpu_layers=range(CPU_LAYER_COUNT), gpu_layers=GPU_LAYERS):
    stats = _empty_stats(gpu_layers)
    positions = flatten(sequences)
    stats["tokens"] = len(positions)
    gpu_set = set(gpu_layers)
    for position in positions:
        for layer, route in enumerate(position):
            if layer not in gpu_set:
                if layer in cpu_layers:
                    stats["cpu_bytes"] += len(route) * layer_bytes[layer]
                continue
            resident = cache[layer]
            for expert in route:
                stats["calls"][layer] += 1
                if expert in resident:
                    stats["hits"][layer] += 1
                else:
                    stats["misses"][layer] += 1
                    stats["pcie_bytes"] += layer_bytes[layer]
                    stats["cpu_bytes"] += layer_bytes[layer]
    return summarize(stats, layer_bytes)


def _train_state(sequences, cache, gpu_layers, mode):
    counts = profile_counts(sequences, gpu_layers)
    last_seen = {layer: {} for layer in gpu_layers}
    tick = 0
    for position in flatten(sequences):
        for layer in gpu_layers:
            for expert in set(position[layer]):
                last_seen[layer][expert] = tick
        tick += 1
    if mode == "lfu":
        scores = {layer: dict(counts[layer]) for layer in gpu_layers}
        last_update = {layer: {expert: -1 for expert in scores[layer]} for layer in gpu_layers}
    else:
        # Only recency order is needed. Normalize the final training touch to age zero.
        scores = None
        last_update = {layer: {expert: seen-tick for expert, seen in last_seen[layer].items()}
                       for layer in gpu_layers}
    return scores, last_update


def simulate_adaptive(train, warmup, evaluation, layer_bytes, *, capacity=26, policy="lfu",
                      half_life=1000, replacements_per_layer=1,
                      cpu_layers=range(CPU_LAYER_COUNT), gpu_layers=GPU_LAYERS,
                      static_pinned_slots=0):
    counts = profile_counts(train, gpu_layers)
    if static_pinned_slots < 0 or static_pinned_slots > capacity:
        raise ValueError("static_pinned_slots must be between zero and capacity")
    cache = {}
    pinned = {}
    for layer in gpu_layers:
        ranking = [expert for expert, _ in counts[layer].most_common()]
        pinned[layer] = set(ranking[:static_pinned_slots])
        dynamic_count = capacity-static_pinned_slots
        dynamic = [expert for expert in ranking[static_pinned_slots:static_pinned_slots+dynamic_count]]
        # Preserve the full capacity even if the profile observed fewer distinct experts.
        if len(dynamic) < dynamic_count:
            dynamic.extend(e for e in range(256) if e not in pinned[layer] and e not in dynamic)
            dynamic = dynamic[:dynamic_count]
        cache[layer] = pinned[layer] | set(dynamic)
    scores, last_seen = _train_state(train, cache, gpu_layers, policy)
    stats = _empty_stats(gpu_layers)
    step = 0
    decay = 2.0 ** (-1.0 / half_life) if policy == "lfu" else None
    gpu_set, cpu_set = set(gpu_layers), set(cpu_layers)

    def consume(sequences, measure):
        nonlocal step
        start_replacements = stats["replacements"]
        positions = flatten(sequences)
        stats["steps"] += len(positions)
        if measure:
            stats["tokens"] += len(positions)
        for position in positions:
            step += 1
            token_replacements = 0
            for layer, route in enumerate(position):
                if layer in cpu_set and layer not in gpu_set and measure:
                    stats["cpu_bytes"] += len(route) * layer_bytes[layer]
                if layer not in gpu_set:
                    continue
                if static_pinned_slots == capacity:
                    continue
                resident = cache[layer]
                if measure:
                    for expert in route:
                        stats["calls"][layer] += 1
                        if expert in resident:
                            stats["hits"][layer] += 1
                        else:
                            stats["misses"][layer] += 1
                            stats["pcie_bytes"] += layer_bytes[layer]
                            stats["cpu_bytes"] += layer_bytes[layer]
                active = set(route)
                if policy == "lfu":
                    sc = scores[layer]
                    lu = last_seen[layer]
                    for expert, access_count in collections.Counter(route).items():
                        previous = sc.get(expert, 0.0)
                        last = lu.get(expert, -1)
                        sc[expert] = previous * (decay ** max(0, step-last)) + access_count
                        lu[expert] = step
                    def score_now(expert):
                        return sc.get(expert, 0.0) * (decay ** max(0, step-lu.get(expert, -1)))
                    candidates = sorted((e for e in active if e not in resident and e not in pinned[layer]), key=lambda e: (-score_now(e), e))
                    victims = sorted((e for e in resident if e not in active and e not in pinned[layer]), key=lambda e: (score_now(e), e))
                    swaps = 0
                    while candidates and victims and swaps < replacements_per_layer:
                        cand, victim = candidates[0], victims[0]
                        if score_now(cand) <= score_now(victim):
                            break
                        candidates.pop(0); victims.pop(0)
                        resident.remove(victim); resident.add(cand)
                        swaps += 1
                elif policy == "lru":
                    lu = last_seen[layer]
                    for expert in active:
                        lu[expert] = step
                    # Stable input-route order breaks ties when experts first appear in the same token.
                    candidates = list(dict.fromkeys(e for e in route if e not in resident and e not in pinned[layer]))
                    victims = sorted((e for e in resident if e not in active and e not in pinned[layer]), key=lambda e: (lu.get(e, -1), e))
                    swaps = 0
                    while candidates and victims and swaps < replacements_per_layer:
                        cand, victim = candidates.pop(0), victims.pop(0)
                        resident.remove(victim); resident.add(cand)
                        swaps += 1
                else:
                    raise ValueError(f"unknown policy {policy}")
                if swaps:
                    token_replacements += swaps
                    stats["replacements"] += swaps
                    stats["replacement_bytes"] += swaps * layer_bytes[layer]
                    stats["replacement_by_layer"][layer] += swaps
            if measure:
                stats["replacements_each_token"].append(token_replacements)
        return stats["replacements"]-start_replacements

    warm_swaps = consume(warmup, False) if warmup else 0
    eval_swaps = consume(evaluation, True)
    result = summarize(stats, layer_bytes)
    result["warmup_replacements"] = warm_swaps
    result["evaluation_replacements"] = eval_swaps
    result["policy"] = policy
    result["half_life"] = half_life if policy == "lfu" else None
    result["k"] = replacements_per_layer
    result["static_pinned_slots"] = static_pinned_slots
    return result


def summarize(stats, layer_bytes):
    calls = sum(stats["calls"].values())
    hits = sum(stats["hits"].values())
    tokens = stats["tokens"]
    ram_mb = stats["cpu_bytes"] / tokens / 1_000_000 if tokens else 0.0
    pcie_mb = stats["pcie_bytes"] / tokens / 1_000_000 if tokens else 0.0
    measured_steps = len(stats["replacements_each_token"])
    repl_mb = stats["replacement_bytes"] / measured_steps / 1_000_000 if measured_steps else 0.0
    replacement_samples = sorted(stats["replacements_each_token"])
    p99_replacements = replacement_samples[max(0, math.ceil(0.99*len(replacement_samples))-1)] if replacement_samples else 0
    layers = {layer: {"hits": stats["hits"][layer], "calls": stats["calls"][layer],
                      "hit_rate": stats["hits"][layer]/stats["calls"][layer] if stats["calls"][layer] else 0.0,
                      "replacements": stats["replacement_by_layer"][layer]}
              for layer in sorted(stats["calls"] or stats["replacement_by_layer"])}
    return {"tokens": tokens, "hits": hits, "calls": calls,
            "hit_rate": hits/calls if calls else 0.0,
            "gpu_misses": sum(stats["misses"].values()),
            "ram_mb_per_token": ram_mb, "pcie_mb_per_token": pcie_mb,
            "replacements": stats["replacements"],
            "replacements_per_token": stats["replacements"]/measured_steps if measured_steps else 0.0,
            "max_replacements_per_token": max(replacement_samples, default=0),
            "p99_replacements_per_token": p99_replacements,
            "replacement_mb_per_token": repl_mb,
            "layers": layers}


def cache_capacity_bytes(expert_bytes, gpu_layers, capacity):
    return sum(expert_bytes[layer] * capacity for layer in gpu_layers)


def fit_tiel_speed(model, training_trace):
    """Two-point diagnostic fit from the prior clean 187k, 8/16-slot block."""
    counts = profile_counts(training_trace)
    x = {}
    for slots in (8, 16):
        measured = evaluate_static(top_cache(counts, slots), training_trace, model["expert_bytes"])
        x[slots] = measured["ram_mb_per_token"]
    # Medians from the prior same-context Tiel block in REPORT-ELASTIC-VRAM-B-RESTART.
    speed = {8: 34.927, 16: 36.736}
    y = {slots: 1000.0 / speed[slots] for slots in speed}  # milliseconds/token
    slope = (y[16]-y[8])/(x[16]-x[8])
    intercept = y[8]-slope*x[8]
    def predict(read_mb):
        ms = intercept+slope*read_mb
        return {"milliseconds_per_token": ms, "tokens_per_second": 1000/ms if ms>0 else None}
    pred = {"static_N8": predict(x[8]), "static_N16": predict(x[16]),
            "static_N26": predict(evaluate_static(top_cache(counts,26),training_trace,model["expert_bytes"])["ram_mb_per_token"])}
    return {"source":"/home/alisher/Projects/shura/bench/results/elastic-vram-b-restart-20260930/report.md",
            "context_tokens":187186,"observed_decode_tps":speed,
            "ram_mb_per_token":x,"time_ms_per_token":y,
            "slope_ms_per_mb":slope,"intercept_ms_per_token":intercept,
            "predictions":pred,
            "caveat":"Two points only; slot profiles were later shown to produce different token IDs, so this is a rough association, not a causal or quality-equivalent speed claim."}


def _round(value, places=3):
    return round(value, places)


def _read_occamy_tasks(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    traces = []
    with tarfile.open(archive, "r:gz") as tf:
        members = sorted((m for m in tf.getmembers() if m.isfile() and m.name.startswith("traces/") and m.name.endswith(".csv")), key=lambda m: m.name)
        for member in members:
            safe_name = Path(member.name).name
            target = destination / safe_name
            src = tf.extractfile(member)
            if src is None:
                raise ValueError(f"could not read archive member {member.name}")
            with target.open("wb") as out:
                while True:
                    chunk = src.read(1024*1024)
                    if not chunk:
                        break
                    out.write(chunk)
            traces.append((safe_name.removesuffix(".csv"), read_trace_csv(target, safe_name.removesuffix(".csv"))))
    if not traces:
        raise ValueError(f"no route CSVs found in {archive}")
    return traces


def _model_result(name, model, profile_sequences, holdout_sequences, occamy_tasks=None,
                  legacy_profile_source=None, training_sequences=None, profile_overlap_removed=0):
    n_experts, top_k = model["expert_count"], model["experts_per_token"]
    expert_bytes, tensors = model["expert_bytes"], model["layers"]
    if n_experts != 256 or top_k != 8:
        # The replay follows what the file says, but preserve a machine-readable warning.
        model["trace_metadata_warning"] = f"GGUF says {n_experts} experts/top-k {top_k}"
    profile_stats = route_statistics(profile_sequences, top_k)
    holdout_stats = route_statistics(holdout_sequences, top_k)
    full_counts = profile_counts(profile_sequences)
    train_counts = full_counts
    if occamy_tasks is not None:
        train_sequences = [seq for _, group in occamy_tasks[:8] for seq in group]
        test_sequences = [seq for _, group in occamy_tasks[8:] for seq in group]
        train_counts = profile_counts(train_sequences)
        holdout_sequences = test_sequences
        holdout_stats = route_statistics(holdout_sequences, top_k)
    else:
        train_sequences = training_sequences if training_sequences is not None else profile_sequences
        train_counts = profile_counts(train_sequences)
    static_in_sample = []
    static_holdout = []
    for capacity in (16, 20, 24, 26, 32):
        static_in_sample.append({"capacity": capacity, **evaluate_static(top_cache(full_counts, capacity), profile_sequences, expert_bytes)})
        static_holdout.append({"capacity": capacity, **evaluate_static(top_cache(train_counts, capacity), holdout_sequences, expert_bytes)})
    base_cache = top_cache(train_counts, 26)
    fixed = evaluate_static(base_cache, holdout_sequences, expert_bytes)
    policies = []
    for half_life in (200, 1000, 5000):
        for k in (1, 2, 4):
            result = simulate_adaptive(train_sequences, [], holdout_sequences, expert_bytes,
                policy="lfu", half_life=half_life, replacements_per_layer=k)
            policies.append(result)
    for k in (1, 2, 4):
        result = simulate_adaptive(train_sequences, [], holdout_sequences, expert_bytes,
            policy="lru", replacements_per_layer=k)
        policies.append(result)
    for result in policies:
        result["gain_pp"] = (result["hit_rate"]-fixed["hit_rate"])*100
        result["ram_saved_mb_per_token"] = fixed["ram_mb_per_token"]-result["ram_mb_per_token"]
        saved = result["ram_saved_mb_per_token"]
        result["replacement_percent_of_saved_read"] = (100*result["replacement_mb_per_token"]/saved) if saved > 0 else None
        result["go_candidate"] = result["gain_pp"] >= 3.0 and saved > 0 and result["replacement_percent_of_saved_read"] <= 15.0
    best = max(policies, key=lambda r: (r["gain_pp"], -((r["replacement_percent_of_saved_read"] or 1e9))))
    robustness = None
    if occamy_tasks is not None and len(occamy_tasks) >= 10:
        adapt_seq = [seq for _, group in [occamy_tasks[8]] for seq in group]
        check_seq = [seq for _, group in [occamy_tasks[9]] for seq in group]
        robustness = []
        static_check = evaluate_static(base_cache, check_seq, expert_bytes)
        for policy in policies:
            name_policy = policy["policy"]
            hl = policy.get("half_life") or 1000
            k = policy["k"]
            trans = simulate_adaptive(train_sequences, adapt_seq, check_seq, expert_bytes,
                policy=name_policy, half_life=hl, replacements_per_layer=k)
            trans["static_check_hit_rate"] = static_check["hit_rate"]
            trans["gain_pp_vs_static_check"] = (trans["hit_rate"]-static_check["hit_rate"])*100
            robustness.append(trans)
        best_transfer = max(robustness, key=lambda r:r["gain_pp_vs_static_check"])
    elif name == "tiel" and len(holdout_sequences) >= 2:
        adapt_seq, check_seq = holdout_sequences[0:1], holdout_sequences[1:2]
        static_check = evaluate_static(base_cache, check_seq, expert_bytes)
        robustness = []
        for policy in policies:
            name_policy = policy["policy"]
            hl = policy.get("half_life") or 1000
            k = policy["k"]
            trans = simulate_adaptive(train_sequences, adapt_seq, check_seq, expert_bytes,
                policy=name_policy, half_life=hl, replacements_per_layer=k)
            trans["static_check_hit_rate"] = static_check["hit_rate"]
            trans["gain_pp_vs_static_check"] = (trans["hit_rate"]-static_check["hit_rate"])*100
            robustness.append(trans)
        best_transfer = max(robustness, key=lambda r:r["gain_pp_vs_static_check"])
    else:
        best_transfer = None
    in_sample_26 = next(r for r in static_in_sample if r["capacity"] == 26)
    legacy_reproduction = None
    if legacy_profile_source is not None:
        legacy_all = read_trace_csv(Path(legacy_profile_source), "tiel-profile-all-rows", include_negative=True)
        legacy_counts = profile_counts(legacy_all)
        legacy_reproduction = evaluate_static(top_cache(legacy_counts, 26), legacy_all, expert_bytes)
    return {"name": name, "model": {k:v for k,v in model.items() if k not in ("layers", "expert_bytes")},
            "expert_bytes_by_layer": expert_bytes, "expert_bytes_min": min(expert_bytes.values()),
            "expert_bytes_max": max(expert_bytes.values()), "profile_stats": profile_stats,
            "holdout_stats": holdout_stats, "static_in_sample": static_in_sample,
            "static_holdout": static_holdout, "static_holdout_26": fixed,
            "static_in_sample_26": in_sample_26, "adaptive": policies, "best": best,
            "legacy_all_rows_reproduction": legacy_reproduction,
            "robustness": robustness, "best_transfer": best_transfer,
            "exact_route_overlap_with_holdout": exact_route_overlap(train_sequences,holdout_sequences),
            "profile_overlap_removed": profile_overlap_removed,
            "profile_tokens": len(flatten(profile_sequences)),
            "train_sequences": len(train_sequences), "holdout_sequences": len(holdout_sequences),
            "train_tokens": len(flatten(train_sequences)), "holdout_tokens": len(flatten(holdout_sequences))}


def exclude_holdout_vectors(profile_sequences, holdout_sequences):
    holdout_vectors={tuple(tuple(route) for route in position) for position in flatten(holdout_sequences)}
    filtered=[]
    removed=0
    for sequence in profile_sequences:
        kept=[position for position in sequence["positions"]
              if tuple(tuple(route) for route in position) not in holdout_vectors]
        removed += len(sequence["positions"])-len(kept)
        if kept:
            filtered.append({**sequence,"positions":kept})
    return filtered,removed


def _format_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def _policy_specs():
    return [
        ("LFU 200 K=2", {"policy":"lfu", "half_life":200, "replacements_per_layer":2}),
        ("LRU K=1", {"policy":"lru", "replacements_per_layer":1}),
        ("LRU K=2", {"policy":"lru", "replacements_per_layer":2}),
    ]


def _adaptive_comparison(train, check, model, capacity, policy_options, pinned=0):
    layer_bytes=model["expert_bytes"]
    fixed=evaluate_static(top_cache(profile_counts(train),capacity),check,layer_bytes)
    out=[]
    for label,opts in policy_options:
        sim=simulate_adaptive(train,[],check,layer_bytes,capacity=capacity,
            static_pinned_slots=pinned,**opts)
        saved=fixed["ram_mb_per_token"]-sim["ram_mb_per_token"]
        sim.update({"label":label,"capacity":capacity,"static_hit_rate":fixed["hit_rate"],
            "static_ram_mb_per_token":fixed["ram_mb_per_token"],
            "gain_pp":(sim["hit_rate"]-fixed["hit_rate"])*100,
            "ram_saved_mb_per_token":saved,
            "net_saved_mb_per_token":saved-sim["replacement_mb_per_token"]})
        out.append(sim)
    return fixed,out


def _fold_summary(folds):
    if not folds:
        return {"count":0,"mean_hit_rate":0.0,"worst":None,"best":None}
    mean=sum(f["hit_rate"] for f in folds)/len(folds)
    return {"count":len(folds),"mean_hit_rate":mean,
            "worst":min(folds,key=lambda f:f["hit_rate"]),
            "best":max(folds,key=lambda f:f["hit_rate"])}


def _static_fold(train, check, model, capacities, label):
    train,removed=exclude_holdout_vectors(train,check)
    counts=profile_counts(train)
    out=[]
    for n in capacities:
        result=evaluate_static(top_cache(counts,n),check,model["expert_bytes"])
        out.append({"label":label,"capacity":n,"hit_rate":result["hit_rate"],
            "ram_mb_per_token":result["ram_mb_per_token"],
            "test_tokens":result["tokens"],"duplicate_vectors_removed":removed})
    return out


def run_o4_analysis(tiel_profile_segments, tiel_train, tiel_windows, occamy_tasks,
                    tiel_model, occamy_model, tiel_speed_fit):
    """Small-slot comparison, task-level static CV, cost metrics, and optional hybrid."""
    policies=_policy_specs()
    model_data={"tiel":(tiel_train,tiel_windows,tiel_model),
                "occamy":([seq for _,group in occamy_tasks[:8] for seq in group],
                          [seq for _,group in occamy_tasks[8:] for seq in group],occamy_model)}
    adaptive_rows={}
    static_rows={}
    direct_18_vs_26={}
    for name,(train,check,model) in model_data.items():
        static_rows[name]=[]
        for n in (18,26,32):
            result=evaluate_static(top_cache(profile_counts(train),n),check,model["expert_bytes"])
            static_rows[name].append({"capacity":n,"hit_rate":result["hit_rate"],
                "ram_mb_per_token":result["ram_mb_per_token"]})
        adaptive_rows[name]=[]
        by_capacity={}
        for n in (16,18,22,26):
            fixed,results=_adaptive_comparison(train,check,model,n,policies)
            by_capacity[n]=fixed
            for row in results:
                adaptive_rows[name].append(row)
        fixed26=next(x for x in static_rows[name] if x["capacity"]==26)
        direct_18_vs_26[name]=[]
        for row in (x for x in adaptive_rows[name] if x["capacity"]==18):
            direct_18_vs_26[name].append({"policy":row["label"],"hit_rate":row["hit_rate"],
                "ram_mb_per_token":row["ram_mb_per_token"],
                "delta_hit_pp":(row["hit_rate"]-fixed26["hit_rate"])*100,
                "ram_saved_vs_static26":fixed26["ram_mb_per_token"]-row["ram_mb_per_token"],
                "replacements_per_token":row["replacements_per_token"],
                "replacement_mb_per_token":row["replacement_mb_per_token"],
                "net_saved_vs_static26":fixed26["ram_mb_per_token"]-row["ram_mb_per_token"]-row["replacement_mb_per_token"],
                "max_replacements_per_token":row["max_replacements_per_token"],
                "p99_replacements_per_token":row["p99_replacements_per_token"]})

    # Static leave-one-task-out on each of Occamy's ten labeled tasks.
    occamy_folds=[]
    for held_index,(task_name,held_group) in enumerate(occamy_tasks):
        train=[seq for index,(_,group) in enumerate(occamy_tasks) if index!=held_index for seq in group]
        for row in _static_fold(train,held_group,occamy_model,(18,26),task_name):
            occamy_folds.append(row)
    occamy_cv={n:_fold_summary([f for f in occamy_folds if f["capacity"]==n]) for n in (18,26)}

    # Tiel code/chat: two external windows; training includes the other window and profile,
    # with exact held-out route vectors removed. Also leave each source-profile chunk out.
    tiel_window_folds=[]
    for i,test in enumerate(tiel_windows):
        train=list(tiel_train)+[seq for j,seq in enumerate(tiel_windows) if j!=i]
        for row in _static_fold(train,[test],tiel_model,(18,26),f"code-chat-window-{i+1}"):
            tiel_window_folds.append(row)
    tiel_chunk_folds=[]
    for i,test in enumerate(tiel_profile_segments):
        train=[seq for j,seq in enumerate(tiel_profile_segments) if j!=i]
        for row in _static_fold(train,[test],tiel_model,(18,26),f"profile-chunk-{i+1}"):
            tiel_chunk_folds.append(row)
    tiel_window_cv={n:_fold_summary([f for f in tiel_window_folds if f["capacity"]==n]) for n in (18,26)}
    tiel_chunk_cv={n:_fold_summary([f for f in tiel_chunk_folds if f["capacity"]==n]) for n in (18,26)}

    # Hybrid pinned/adaptive split: half the N most frequent experts stay pinned; LFU/LRU
    # operate only on a disjoint half, with K=1 replacement per layer per token.
    hybrid_rows={}
    hybrid_policies=[("LFU 200 K=1",{"policy":"lfu","half_life":200,"replacements_per_layer":1}),
                     ("LRU K=1",{"policy":"lru","replacements_per_layer":1})]
    for name,(train,check,model) in model_data.items():
        hybrid_rows[name]=[]
        for n in (18,26):
            fixed,rows=_adaptive_comparison(train,check,model,n,hybrid_policies,pinned=n//2)
            for row in rows:
                hybrid_rows[name].append(row)

    # Apply the pre-existing Tiel two-point RAM-read fit only as a cautious illustration.
    speed={}
    slope=tiel_speed_fit["slope_ms_per_mb"]
    intercept=tiel_speed_fit["intercept_ms_per_token"]
    for row in direct_18_vs_26["tiel"]:
        fit_ms=intercept+slope*row["ram_mb_per_token"]
        row["rough_fit_tps"]=1000/fit_ms if fit_ms>0 else None
    fixed26=next(x for x in static_rows["tiel"] if x["capacity"]==26)
    base_ms=intercept+slope*fixed26["ram_mb_per_token"]
    tiel_lfu26=next(x for x in adaptive_rows["tiel"] if x["capacity"]==26 and x["label"]=="LFU 200 K=2")
    lfu26_ms=intercept+slope*tiel_lfu26["ram_mb_per_token"]
    tiel_lfu26["rough_fit_tps"]=1000/lfu26_ms if lfu26_ms>0 else None
    speed={"source":tiel_speed_fit["source"],"equation":f"time_ms/token = {intercept:.5f} + {slope:.5f} × RAM_MB/token",
           "static26_tps":1000/base_ms if base_ms>0 else None}
    return {"adaptive":adaptive_rows,"static":static_rows,"direct_18_vs_26":direct_18_vs_26,
            "occamy_loo":occamy_folds,"occamy_cv":occamy_cv,
            "tiel_window_folds":tiel_window_folds,"tiel_window_cv":tiel_window_cv,
            "tiel_chunk_folds":tiel_chunk_folds,"tiel_chunk_cv":tiel_chunk_cv,
            "hybrid":hybrid_rows,"speed":speed,
            "tiel_profile_chunk_count":len(tiel_profile_segments),
            "tiel_window_count":len(tiel_windows),"occamy_task_count":len(occamy_tasks)}


def render_o4_section(o4, tiel_model, occamy_model):
    lines=["# O4 — Small-slot adaptive cache simulation", "",
        "Offline trace replay only. No GPU, model, or llama-server was started.", "",
        "## Input files", "",
        "- Tiel profile: `/home/alisher/Projects/shura-cachesim2/config/moe-trace/tiel-coder-agentic.csv`; Tiel code/chat holdout windows: `/home/alisher/Desktop/AI/config/moe-trace-tiel-code+chat.csv`.",
        "- Occamy task traces: `/home/alisher/Projects/shura/bench/results/occamy-s3-20261003/raw.tar.gz` (`traces/00-python-code.csv` through `traces/09-real-shuracode-agent.csv`).",
        f"- GGUF tensor headers only: `{tiel_model['path']}` and `{occamy_model['path']}`. Tensor payloads were not read.", "",
        "## A. Small-slot policies and static baselines", "",
        "Adaptive runs use the train-only profile to initialize N slots, then process the same held-out routes. LFU decay half-life is 200 generated tokens. K is the replacement cap per GPU layer per token. RAM reads include CPU-resident layers 0–25 and GPU-cache misses in layers 26–39. Replacement MB is separately reported and subtracted for net benefit as requested.", ""]
    for name in ("tiel","occamy"):
        stat=o4["static"][name]
        static_map={x["capacity"]:x for x in stat}
        lines += [f"### {name.title()}", "",
            _format_table(["Policy", "N", "Hit %", "RAM MB/token", "Repl MB/token", "RAM saved vs static same N", "Net saved", "Max repl/token", "P99 repl/token"],
                [["Static",x["capacity"],f"{100*x['hit_rate']:.3f}",f"{x['ram_mb_per_token']:.3f}","0.000","0.000","0.000","0","0"] for x in stat]+
                [[r["label"],r["capacity"],f"{100*r['hit_rate']:.3f}",f"{r['ram_mb_per_token']:.3f}",
                  f"{r['replacement_mb_per_token']:.3f}",f"{r['ram_saved_mb_per_token']:+.3f}",
                  f"{r['net_saved_mb_per_token']:+.3f}",r["max_replacements_per_token"],r["p99_replacements_per_token"]]
                 for r in o4["adaptive"][name]]), ""]
        f26=static_map[26]
        slot_bytes=(sum((tiel_model if name=="tiel" else occamy_model)["expert_bytes"][layer] for layer in GPU_LAYERS)*8/1e6)
        lines += [f"Adaptive N=18 uses 8 fewer resident experts/layer than static N=26, saving {slot_bytes:.3f} MB (decimal) of expert-cache VRAM across the 14 GPU layers, before allocator overhead. Direct comparison with static N=26:",
            _format_table(["Policy", "Hit %", "Δ hit pp", "RAM MB/token", "RAM saved vs static26", "Repl MB/token", "Net saved vs static26", "Max repl/token", "P99 repl/token"],
                [[r["policy"],f"{100*r['hit_rate']:.3f}",f"{r['delta_hit_pp']:+.3f}",f"{r['ram_mb_per_token']:.3f}",
                  f"{r['ram_saved_vs_static26']:+.3f}",f"{r['replacement_mb_per_token']:.3f}",
                  f"{r['net_saved_vs_static26']:+.3f}",r["max_replacements_per_token"],r["p99_replacements_per_token"]]
                 for r in o4["direct_18_vs_26"][name]]), ""]

    lines += ["## B. Static-profile cross-validation", "",
        "Occamy uses leave-one-task-out: each fold learns from the other nine labeled tasks and tests the held-out task. Tiel source traces have no task labels, so the two code/chat windows are evaluated by leaving one window out (the source profile plus the other window build the profile), and source-profile chunks are separately evaluated leave-one-chunk-out. Exact held-out route vectors are removed from fold training before building top-N.", "",
        "Occamy leave-one-task-out summary (macro mean across tasks; best/worst task shown):",
        _format_table(["N", "Folds", "Mean hit %", "Worst task / %", "Best task / %", "Test tokens range"], [
            [n,o4["occamy_cv"][n]["count"],f"{100*o4['occamy_cv'][n]['mean_hit_rate']:.3f}",
             f"{o4['occamy_cv'][n]['worst']['label']} / {100*o4['occamy_cv'][n]['worst']['hit_rate']:.3f}",
             f"{o4['occamy_cv'][n]['best']['label']} / {100*o4['occamy_cv'][n]['best']['hit_rate']:.3f}",
             f"{min(x['test_tokens'] for x in o4['occamy_loo'] if x['capacity']==n)}–{max(x['test_tokens'] for x in o4['occamy_loo'] if x['capacity']==n)}"] for n in (18,26)]), "",
        _format_table(["Task", "N", "Hit %", "Test tokens", "Duplicate route vectors removed"], [
            [f["label"],f["capacity"],f"{100*f['hit_rate']:.3f}",f["test_tokens"],f["duplicate_vectors_removed"]]
            for f in o4["occamy_loo"]]), ""]
    for label,key,summary_key in (("Tiel two code/chat windows","tiel_window_folds","tiel_window_cv"),
                                  ("Tiel source profile chunks","tiel_chunk_folds","tiel_chunk_cv")):
        lines += [f"{label}:",_format_table(["N","Folds","Mean hit %","Worst fold / %","Best fold / %","Test tokens range"],[
            [n,o4[summary_key][n]["count"],f"{100*o4[summary_key][n]['mean_hit_rate']:.3f}",
             f"{o4[summary_key][n]['worst']['label']} / {100*o4[summary_key][n]['worst']['hit_rate']:.3f}",
             f"{o4[summary_key][n]['best']['label']} / {100*o4[summary_key][n]['best']['hit_rate']:.3f}",
             f"{min(x['test_tokens'] for x in o4[key] if x['capacity']==n)}–{max(x['test_tokens'] for x in o4[key] if x['capacity']==n)}"] for n in (18,26)]), "",
            _format_table(["Fold", "N", "Hit %", "Test tokens", "Duplicate route vectors removed"], [
                [f["label"],f["capacity"],f"{100*f['hit_rate']:.3f}",f["test_tokens"],f["duplicate_vectors_removed"]]
                for f in o4[key]]), ""]

    lines += ["## C. Replacement traffic and net benefit", "",
        "Net benefit = RAM reads saved against static at the same N minus bytes staged for replacements. Maximum and p99 count replacements summed across all 14 GPU-cache layers for one generated token; p99 uses nearest-rank selection.", "",
        "The A tables contain these values for every policy. Summary of each model's best positive net benefit:",
        _format_table(["Model", "Policy", "N", "Net saved MB/token", "Max repl/token", "P99 repl/token"], [
            [name,(lambda r:r['label'])(max((x for x in o4['adaptive'][name] if x['net_saved_mb_per_token']>0),key=lambda x:x['net_saved_mb_per_token'])),
             max((x for x in o4['adaptive'][name] if x['net_saved_mb_per_token']>0),key=lambda x:x['net_saved_mb_per_token'])['capacity'],
             f"{max((x for x in o4['adaptive'][name] if x['net_saved_mb_per_token']>0),key=lambda x:x['net_saved_mb_per_token'])['net_saved_mb_per_token']:.3f}",
             max((x for x in o4['adaptive'][name] if x['net_saved_mb_per_token']>0),key=lambda x:x['net_saved_mb_per_token'])['max_replacements_per_token'],
             max((x for x in o4['adaptive'][name] if x['net_saved_mb_per_token']>0),key=lambda x:x['net_saved_mb_per_token'])['p99_replacements_per_token']]
            for name in ("tiel","occamy")]), ""]

    lines += ["## G. Half-static, half-adaptive hybrid", "",
        "The top half of each layer's profile ranking is pinned; the remaining half starts from the next-ranked experts and can adapt with K=1 per layer/token.", ""]
    for name in ("tiel","occamy"):
        lines += [f"### {name.title()}",_format_table(["Policy","N","Pinned slots/layer","Hit %","RAM MB/token","Repl MB/token","Net saved vs static same N","Max/token","P99/token"], [
            [r["label"],r["capacity"],r["static_pinned_slots"],f"{100*r['hit_rate']:.3f}",f"{r['ram_mb_per_token']:.3f}",
             f"{r['replacement_mb_per_token']:.3f}",f"{r['net_saved_mb_per_token']:+.3f}",r["max_replacements_per_token"],r["p99_replacements_per_token"]]
            for r in o4["hybrid"][name]]),""]
    tiel_lfu26=next(r for r in o4["adaptive"]["tiel"] if r["capacity"]==26 and r["label"]=="LFU 200 K=2")
    occ_lfu18=next(r for r in o4["direct_18_vs_26"]["occamy"] if r["policy"]=="LFU 200 K=2")
    tiel_lfu18=next(r for r in o4["direct_18_vs_26"]["tiel"] if r["policy"]=="LFU 200 K=2")
    lines += ["## Speed illustration", "",
        f"For Tiel only, the existing two-point fit from `{o4['speed']['source']}` is `{o4['speed']['equation']}`. Applying it to held-out RAM-read totals gives only a rough correlation: static N=26 is {o4['speed']['static26_tps']:.2f} tok/s; Tiel LFU N=26 is {tiel_lfu26['rough_fit_tps']:.2f} tok/s. LFU/LRU N=18 predictions are listed below. These are not measured speeds and prior slot-count tests did not preserve token IDs.",
        _format_table(["Tiel policy at N=18","RAM MB/token","Fit-only tok/s"],[[r["policy"],f"{r['ram_mb_per_token']:.3f}",f"{r['rough_fit_tps']:.2f}"] for r in o4["direct_18_vs_26"]["tiel"]]), "",
        "## Recommendation", "",
        f"Prototype LFU with 200-token half-life and K=2 replacements per layer/token. For Occamy, N=18 versus static N=26 gives {100*occ_lfu18['hit_rate']:.1f}% versus {100*next(x for x in o4['static']['occamy'] if x['capacity']==26)['hit_rate']:.1f}% hits, saves {occ_lfu18['ram_saved_vs_static26']:.1f} MB RAM/token before replacement traffic and {occ_lfu18['net_saved_vs_static26']:.1f} MB/token after subtracting about 1.0 MB/token of replacements, while freeing about 187.2 MB of cache storage. For Tiel, keep N=26: LFU improves from {100*next(x for x in o4['static']['tiel'] if x['capacity']==26)['hit_rate']:.1f}% to {100*tiel_lfu26['hit_rate']:.1f}% and nets {tiel_lfu26['net_saved_mb_per_token']:.1f} MB/token; at N=18, LFU's net versus static N=26 is {tiel_lfu18['net_saved_vs_static26']:.1f} MB/token, so shrinking Tiel to 18 slots is not supported by the net-traffic measure. The per-layer K=2 is a hard cap; for these selected configurations LFU peaks at 7 replacements/token (Tiel) and 4 (Occamy), with p99 of 4 and 3 respectively, so the data do not justify an additional global cap. This is a replay-based prototype recommendation, not proof of speed or response quality.", "",
        "Limitations: only 1,024 Tiel and 768 Occamy tokens are in the main held-out checks; leave-one-task/window folds remain small and do not establish production workload coverage. The MTP layer is not routed in the traces, asynchronous prefetch and runtime overhead are not modeled, and no actual speed or quality response was measured. These values describe replayed routes and traffic only.", ""]
    return "\n".join(lines)


def render_report(results, trace_paths, ggml_source):
    lines = ["# Offline expert-cache simulation", "",
        "This report replays generated-token routing only. It does not run llama-server, load a model, or query GPU hardware.", "",
        "## Inputs and data split", "",
        f"- Tiel profile trace: `{trace_paths['tiel_profile']}` ({trace_paths['tiel_raw_positions']} raw token positions; {trace_paths['tiel_negative_positions']} negative prefill positions filtered by runtime; {results['tiel']['profile_stats']['positions']} generated positions). Training uses {results['tiel']['train_tokens']} profile positions after removing {results['tiel']['profile_overlap_removed']} exact route-vector duplicates of holdout.",
        f"- Tiel held-out trace: `{trace_paths['tiel_holdout']}` ({trace_paths['tiel_holdout_raw_positions']} raw positions, {trace_paths['tiel_holdout_negative_positions']} negative prefill positions; {results['tiel']['holdout_stats']['positions']} generated positions). Rows lack task labels; the two complete token-index-reset segments are used as non-overlapping windows. Exact full-route vector overlap after filtering: {results['tiel']['exact_route_overlap_with_holdout']}.",
        f"- Occamy trace archive: `{trace_paths['occamy_archive']}`. Task CSVs `00`–`07` ({results['occamy']['train_tokens']} tokens) build the initial profile; `08-russian-explain` and `09-real-shuracode-agent` ({results['occamy']['holdout_tokens']} tokens) are held out. Exact full-route vector overlap with training: {results['occamy']['exact_route_overlap_with_holdout']}.",
        f"- Historical estimates are in `{trace_paths['historical_estimates']}`. The tracked Tiel file contains 4384 raw positions including 79 negative prefill positions; llama.cpp's runtime profile parser filters negative positions. The all-row replay below reproduces the previous ~37.3% result; the generated-only profile replay is also shown.",
        "- The Occamy archive's 10 task files are verified byte-for-byte as an exact concatenation of `config/moe-trace/occamy-agentic.csv`; the full-file N=26 in-sample calculation is shown separately only to compare with the old 37.0% estimate.",
        "- Both routing formats have 40 layers and eight expert IDs per generated-token/layer row. GPU-cache layers are 26–39; CPU layers are 0–25, matching the previous RAM-read estimate. The untraced MTP layer 40 is excluded.", "",
        "## Expert byte sizes from GGUF", "",
        f"Tensor metadata read from `{results['tiel']['model']['path']}` and `{results['occamy']['model']['path']}`; quant block layouts checked against `{ggml_source}`.",
        "Formula per expert: `sum_over(up, gate, down)((tensor_elements / expert_count / block_elements) * block_bytes)`. IQ3_S is 256 elements / 110 bytes; IQ4_XS is 256 / 136 bytes. MB below are decimal.",
        _format_table(["Model", "Experts", "Routed experts/token", "Expert bytes", "MB/expert"], [
            [name, r['model']['expert_count'], r['model']['experts_per_token'], r['expert_bytes_min'], f"{r['expert_bytes_min']/1e6:.6f}"]
            for name,r in results.items()]), "",
        "## Static top-N", "",
        "The first table is the full profile evaluated against itself (an in-sample replay to reproduce the prior number, not a validation claim). The second uses the disjoint train/holdout split.", ""]
    for name,result in results.items():
        lines += [f"### {name.title()}", "", "Full-profile in-sample:",
                  _format_table(["N", "Hit %", "Hits", "GPU route calls", "RAM MB/token"], [
                    [x['capacity'], f"{100*x['hit_rate']:.3f}", x['hits'], x['calls'], f"{x['ram_mb_per_token']:.3f}"] for x in result['static_in_sample']]), "",
                  "Disjoint holdout:",
                  _format_table(["N", "Hit %", "Hits", "GPU route calls", "RAM MB/token", "PCIe MB/token"], [
                    [x['capacity'], f"{100*x['hit_rate']:.3f}", x['hits'], x['calls'], f"{x['ram_mb_per_token']:.3f}", f"{x['pcie_mb_per_token']:.3f}"] for x in result['static_holdout']]), "",
                  f"N=26 generated-only in-sample hit rate: {100*result['static_in_sample_26']['hit_rate']:.3f}% (previous estimate: {37.3 if name=='tiel' else 37.0:.1f}%).", "",
                  (f"Legacy all-row N=26 replay: {100*result['legacy_all_rows_reproduction']['hit_rate']:.3f}% over {result['legacy_all_rows_reproduction']['tokens']} positions. This includes negative prefill rows; the runtime profile reader skips them, so the generated-only rate above is the applicable decode result." if result.get("legacy_all_rows_reproduction") else ""), "",
                  "N=26 static per-GPU-layer heldout hit rates:",
                  _format_table(["Layer", "Hit %", "Hits", "Calls"], [[layer, f"{100*x['hit_rate']:.2f}", x['hits'], x['calls']]
                    for layer,x in result['static_holdout_26']['layers'].items()]), ""]
    lines += ["## Adaptive policies at N=26", "", "Each run starts with the static N=26 membership learned only from its training/profile data. Hits are measured before that token's update; replacements are limited independently per GPU layer and generated token.", ""]
    for name,result in results.items():
        lines += [f"### {name.title()}", "", _format_table(
            ["Policy", "Hit %", "Δpp vs heldout static", "Repl/token", "PCIe MB/token", "RAM MB/token", "RAM saved MB/token", "Repl MB/saved MB"],
            [[(f"LFU half-life {x['half_life']} K={x['k']}" if x['policy']=='lfu' else f"LRU K={x['k']}"),
              f"{100*x['hit_rate']:.3f}", f"{x['gain_pp']:+.3f}", f"{x['replacements_per_token']:.4f}",
              f"{x['pcie_mb_per_token']:.3f}", f"{x['ram_mb_per_token']:.3f}", f"{x['ram_saved_mb_per_token']:+.3f}",
              "—" if x['replacement_percent_of_saved_read'] is None else f"{x['replacement_percent_of_saved_read']:.2f}%"] for x in result['adaptive']]), ""]
        if result["best_transfer"]:
            tr=result["best_transfer"]
            lines += [f"Robustness split (adapt on one window/task, check another): best gain `{tr['gain_pp_vs_static_check']:+.3f} pp`; static check hit `{100*tr['static_check_hit_rate']:.3f}%`, adapted check hit `{100*tr['hit_rate']:.3f}%`." , ""]
        best=result["best"]
        policy_label=(f"LFU half-life {best['half_life']} K={best['k']}" if best['policy']=="lfu" else f"LRU K={best['k']}")
        lines += [f"Per-layer heldout hit rate, highest-hit adaptive policy ({policy_label}; may fail the replacement-cost gate):",
                  _format_table(["Layer", "Static N=26 %", "Adaptive %", "Replacements"], [
                    [layer, f"{100*result['static_holdout_26']['layers'][layer]['hit_rate']:.2f}",
                     f"{100*x['hit_rate']:.2f}", x['replacements']]
                    for layer,x in best['layers'].items()]), ""]
    candidates=[(name,x) for name,r in results.items() for x in r['adaptive'] if x['go_candidate']]
    best_all=max(((name,x) for name,r in results.items() for x in r['adaptive']), key=lambda pair:pair[1]['gain_pp'])
    qualified=max(candidates,key=lambda pair:pair[1]['gain_pp']) if candidates else None
    outcome="GO" if qualified else "NO-GO"
    if qualified:
        chosen_name,chosen=qualified
        chosen_label=(f"LFU half-life {chosen['half_life']}" if chosen['policy']=='lfu' else "LRU")
        decision_text=f"Best qualifying candidate: {chosen_name} {chosen_label} K={chosen['k']}, gain {chosen['gain_pp']:+.3f} pp, replacement traffic {chosen['replacement_percent_of_saved_read']:.2f}% of saved RAM reads."
    else:
        decision_text="No policy met both gates."
    raw_label=(f"LFU half-life {best_all[1]['half_life']}" if best_all[1]['policy']=='lfu' else "LRU")
    raw_ratio = best_all[1]['replacement_percent_of_saved_read']
    raw_ratio_text = f"{raw_ratio:.2f}%" if raw_ratio is not None else "undefined"
    lines += [f"**Decision: {outcome}.** {decision_text} Highest raw hit gain was {best_all[0]} {raw_label} K={best_all[1]['k']} at {best_all[1]['gain_pp']:+.3f} pp with replacement traffic ratio {raw_ratio_text}; a high-hit policy that exceeds 15% is not accepted.", ""]
    speed_fit=results["tiel"]["speed_fit"]
    speed_rows=[[f"Observed {slots}-slot", f"{speed_fit['ram_mb_per_token'][slots]:.3f}",
                 f"{speed_fit['observed_decode_tps'][slots]:.3f}", f"{speed_fit['time_ms_per_token'][slots]:.4f}"] for slots in (8,16)]
    speed_rows += [["Static N=26 fit estimate", f"{results['tiel']['static_in_sample_26']['ram_mb_per_token']:.3f}",
                    f"{speed_fit['predictions']['static_N26']['tokens_per_second']:.3f}",
                    f"{speed_fit['predictions']['static_N26']['milliseconds_per_token']:.4f}"],
                   ["Tiel max-hit policy (raw fit)", f"{speed_fit['predictions']['max_hit_policy']['ram_mb_per_token']:.3f}",
                    f"{speed_fit['predictions']['max_hit_policy']['tokens_per_second']:.3f}",
                    f"{speed_fit['predictions']['max_hit_policy']['milliseconds_per_token']:.4f}"]]
    qualified_speed=speed_fit["predictions"].get("best_qualifying_policy")
    speed_rows.append(["Tiel best GO-qualified policy", f"{qualified_speed['ram_mb_per_token']:.3f}" if qualified_speed else "—",
                       f"{qualified_speed['tokens_per_second']:.3f}" if qualified_speed else "—",
                       f"{qualified_speed['milliseconds_per_token']:.4f}" if qualified_speed else "—"])
    lines += ["## Speed estimate", "",
        f"Only Tiel has two eligible paired cache-size measurements in the prior evidence: 187,186-token context, 8 slots at 34.927 tok/s and 16 slots at 36.736 tok/s (source: `{speed_fit['source']}`). The two-point line fits time/token against simulated RAM reads; it is a rough association, not a causal or quality-equivalent speed claim.",
        f"Fit: `time_ms/token = {speed_fit['intercept_ms_per_token']:.5f} + {speed_fit['slope_ms_per_mb']:.5f} × RAM_MB/token`.",
        _format_table(["Point", "RAM MB/token", "Decode tok/s", "Time ms/token"], speed_rows),
        "Occamy has no eligible paired same-model/cache-size speed data, so its speed is not estimated. The prior 8/16-slot report also documented different token-ID outputs across slot counts; the fitted line does not establish equal quality.", "",
        "## Limits", "",
        "This is a trace replay, not an implementation benchmark. It does not model asynchronous prefetch, CUDA scheduling, cache-line fragmentation, shared experts, or MTP layer 40 (no route trace was recorded). PCIe bytes count routed expert misses; replacement bytes count newly staged expert weights. Hits are weighted equally per routed expert, not by compute duration.", ""]
    return "\n".join(lines)


def run(args):
    repo = Path(__file__).resolve().parents[1]
    tiel_profile = Path(args.tiel_profile or repo / "config/moe-trace/tiel-coder-agentic.csv")
    occamy_profile = Path(args.occamy_profile or repo / "config/moe-trace/occamy-agentic.csv")
    tiel_holdout_path = Path(args.tiel_holdout or "/home/alisher/Desktop/AI/config/moe-trace-tiel-code+chat.csv")
    tiel_gguf = Path(args.tiel_gguf or "/home/alisher/Desktop/AI/models/Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf")
    occamy_gguf = Path(args.occamy_gguf or "/home/alisher/Desktop/AI/models/occamy/occamy-1.0-IQ4_XS-MTP.gguf")
    archive = Path(args.occamy_archive or "/home/alisher/Projects/shura/bench/results/occamy-s3-20261003/raw.tar.gz")
    for path in (tiel_profile, occamy_profile, tiel_holdout_path, tiel_gguf, occamy_gguf, archive):
        if not path.is_file():
            raise FileNotFoundError(path)
    tiel_profile_raw=raw_trace_counts(tiel_profile)
    tiel_holdout_raw=raw_trace_counts(tiel_holdout_path)
    scratch_parent = Path(args.tmp_root)
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shura-cache-sim-", dir=scratch_parent) as tmp:
        work = Path(tmp)
        def copy_input(path, name):
            target=work/name
            target.write_bytes(path.read_bytes())
            return target
        tiel_profile_tmp=copy_input(tiel_profile,"tiel-profile.csv")
        occamy_profile_tmp=copy_input(occamy_profile,"occamy-profile.csv")
        tiel_holdout_tmp=copy_input(tiel_holdout_path,"tiel-code-chat.csv")
        tiel_profile_trace=read_trace_csv(tiel_profile_tmp,"tiel-agentic")
        tiel_holdout_all=read_trace_csv(tiel_holdout_tmp,"tiel-code-chat")
        tiel_holdout=[seq for seq in tiel_holdout_all if seq["start_index"] == 0]
        if not tiel_holdout:
            raise ValueError("no complete token-index-reset window in Tiel code+chat holdout")
        tiel_train, tiel_overlap_removed = exclude_holdout_vectors(tiel_profile_trace, tiel_holdout)
        if exact_route_overlap(tiel_train,tiel_holdout) != 0:
            raise ValueError("Tiel train/holdout route vectors overlap after exclusion")
        occamy_tasks=_read_occamy_tasks(archive,work/"occamy-tasks")
        concatenated=b"".join((work/"occamy-tasks"/(task_name+".csv")).read_bytes() for task_name,_ in occamy_tasks)
        if concatenated != occamy_profile_tmp.read_bytes():
            raise ValueError("Occamy task archive does not match the configured profile CSV byte-for-byte")
        occamy_profile_trace=read_trace_csv(occamy_profile_tmp,"occamy-agentic")
        tiel_model=read_gguf_experts(tiel_gguf)
        occamy_model=read_gguf_experts(occamy_gguf)
        tiel_result=_model_result("tiel",tiel_model,tiel_profile_trace,tiel_holdout,
            legacy_profile_source=tiel_profile_tmp,training_sequences=tiel_train,
            profile_overlap_removed=tiel_overlap_removed)
        tiel_result["speed_fit"] = fit_tiel_speed(tiel_model,tiel_profile_trace)
        candidate_policies=[x for x in tiel_result["adaptive"] if x["go_candidate"]]
        predictions_to_add=[("max_hit_policy",tiel_result["best"])]
        if candidate_policies:
            predictions_to_add.append(("best_qualifying_policy",max(candidate_policies,key=lambda x:x["gain_pp"])))
        for label,policy in predictions_to_add:
            read_mb=policy["ram_mb_per_token"]
            predicted_ms=tiel_result["speed_fit"]["intercept_ms_per_token"]+tiel_result["speed_fit"]["slope_ms_per_mb"]*read_mb
            tiel_result["speed_fit"]["predictions"][label]={"ram_mb_per_token":read_mb,"milliseconds_per_token":predicted_ms,
                "tokens_per_second":1000/predicted_ms if predicted_ms>0 else None,"policy":policy["policy"],"k":policy["k"],"half_life":policy.get("half_life")}
        occamy_result=_model_result("occamy",occamy_model,occamy_profile_trace,[],occamy_tasks)
        results={"tiel":tiel_result,"occamy":occamy_result}
        o4_report=None
        if args.o4:
            o4=run_o4_analysis(tiel_profile_trace,tiel_train,tiel_holdout,occamy_tasks,
                               tiel_model,occamy_model,tiel_result["speed_fit"])
            o4_report=render_o4_section(o4,tiel_model,occamy_model)
        trace_paths={"tiel_profile":str(tiel_profile),"tiel_holdout":str(tiel_holdout_path),"occamy_archive":str(archive),
                     "historical_estimates":"/home/alisher/Desktop/AI/MORNING.md and /home/alisher/Desktop/AI/MORNING-P3LITE.md",
                     "tiel_raw_positions":tiel_profile_raw["raw_positions"],
                     "tiel_negative_positions":tiel_profile_raw["negative_positions"],
                     "tiel_holdout_raw_positions":tiel_holdout_raw["raw_positions"],
                     "tiel_holdout_negative_positions":tiel_holdout_raw["negative_positions"]}
        return {"results":results,"trace_paths":trace_paths,"o4_report":o4_report,
                "ggml_source":"/home/alisher/Desktop/AI/llama.cpp-perf/ggml/src/ggml.c and ggml/src/ggml-common.h"}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiel-profile")
    parser.add_argument("--tiel-holdout")
    parser.add_argument("--occamy-profile")
    parser.add_argument("--occamy-archive")
    parser.add_argument("--tiel-gguf")
    parser.add_argument("--occamy-gguf")
    parser.add_argument("--tmp-root",default="/dev/shm")
    parser.add_argument("--json-out",help="optional result JSON path; keep under /dev/shm")
    parser.add_argument("--o4",action="store_true",help="run and render the O4 small-slot/cross-validation analysis")
    args=parser.parse_args(argv)
    result=run(args)
    report=result["o4_report"] if args.o4 else render_report(result["results"],result["trace_paths"],result["ggml_source"])
    print(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result,indent=2),encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
