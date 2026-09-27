#!/usr/bin/env python3
"""Shura setup helper: detect hardware, pick a Tiel-Coder quant and layout, fetch the model, auto-tune.

Standard library only. Called by setup.sh; every step can also be run by hand:

    shura_setup.py detect                         hardware summary (JSON)
    shura_setup.py plan [--quant Q]               quant + initial GPU/CPU layout, from GGUF headers on Hugging Face
    shura_setup.py fetch --quant Q --dest DIR     download (resumable) and verify SHA-256, or reuse --existing FILE
    shura_setup.py autotune --model F --llama BIN measure candidates on the real GPU and write the config
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HF_REPO = "peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"
MMPROJ = "mmproj-BF16.gguf"
QUANT_FILE = "Tiel-Coder-35B-A3B-MTP-UD-{}.gguf"
DOWNGRADE = ["IQ4_XS", "Q3_K_XL", "IQ3_XXS", "Q2_K_XL"]  # IQ4_XS is the benchmarked default
UPGRADE = ["Q5_K_XL", "Q4_K_XL"]                        # only when almost everything fits in VRAM

MB = 1024 * 1024
GB = 1024 * MB
# Calibrated on the reference machine (RTX 4070 SUPER, 200k ctx, turbo3 KV, MTP): see docs/INSTALLER.md.
GPU_OVERHEAD = 2040 * MB   # KV cache + MTP draft context + compute buffers + CUDA context
VRAM_RESERVE = 600 * MB    # desktop spikes + CUDA pool growth at full context
RAM_BASE = 1500 * MB       # llama-server process without the prompt cache
VISION_EXTRA = 1000 * MB   # mmproj projector + its buffers
MAX_SLOTS = 64
CACHE_RAM_MAX = 4096       # MB of host prompt cache; more brings little

REPO_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.path.expanduser("~/.cache/shura"))


def log(msg):
    print(f"  {msg}", flush=True)


# ----------------------------------------------------------------------------------------- hardware
def detect():
    hw = {"gpus": [], "ram_total": 0, "ram_available": 0, "physical_cores": os.cpu_count() or 1}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,compute_cap,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
        for line in out.strip().splitlines():
            name, total, used, cap, driver = [x.strip() for x in line.split(",")]
            hw["gpus"].append({"name": name, "vram_total": int(total) * MB, "vram_used": int(used) * MB,
                               "cuda_arch": cap.replace(".", ""), "driver": driver})
    except (OSError, subprocess.SubprocessError):
        pass
    with open("/proc/meminfo") as f:
        mem = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f}
    hw["ram_total"], hw["ram_available"] = mem["MemTotal"], mem["MemAvailable"]
    try:
        out = subprocess.run(["lscpu", "-p=Core,Socket"], capture_output=True, text=True).stdout
        cores = {line for line in out.splitlines() if line and not line.startswith("#")}
        hw["physical_cores"] = max(1, len(cores))
    except OSError:
        pass
    return hw


# ------------------------------------------------------------------------------------- GGUF headers
class _Buf:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, n):
        if self.pos + n > len(self.data):
            raise EOFError
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8", "replace")


_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _value(b, t):
    if t == 8:
        return b.string()
    if t == 9:
        et, n = b.u32(), b.u64()
        items = [_value(b, et) for _ in range(n)]
        return items if n <= 16 else None
    fmt = _SCALAR[t]
    return struct.unpack(fmt, b.take(struct.calcsize(fmt)))[0]


def parse_gguf_header(data, file_size):
    """Per-layer routed-expert bytes and the rest, from the GGUF header only (tensor sizes = offset deltas)."""
    b = _Buf(data)
    if b.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    b.u32()
    n_tensors, n_kv = b.u64(), b.u64()
    kv = {}
    for _ in range(n_kv):
        key = b.string()
        kv[key] = _value(b, b.u32())
    tensors = []
    for _ in range(n_tensors):
        name = b.string()
        n_dims = b.u32()
        b.take(8 * n_dims)
        b.u32()
        tensors.append((b.u64(), name))
    align = kv.get("general.alignment", 32)
    data_start = (b.pos + align - 1) // align * align
    tensors.sort()
    arch = kv["general.architecture"]
    n_layers = kv[f"{arch}.block_count"]
    experts, other = [0] * n_layers, 0
    for i, (off, name) in enumerate(tensors):
        size = (tensors[i + 1][0] if i + 1 < len(tensors) else file_size - data_start) - off
        if "_exps." in name and name.startswith("blk."):
            experts[int(name.split(".")[1])] += size
        else:
            other += size
    return {"arch": arch, "n_layers": n_layers, "experts": experts, "other": other, "file_size": file_size,
            "n_experts": kv.get(f"{arch}.expert_count", 0)}


def read_local_layout(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        data = f.read(64 * MB)
    return parse_gguf_header(data, size)


def hf_files():
    """name -> (size, sha256) for every file in the model repo."""
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{HF_REPO}/tree/main", timeout=30) as r:
        items = json.load(r)
    return {i["path"]: (i.get("lfs", {}).get("size", i.get("size", 0)), i.get("lfs", {}).get("oid", ""))
            for i in items if i["path"].endswith(".gguf")}


def remote_layout(quant, files):
    """Parse a quant's layout from its first megabytes on Hugging Face (no full download)."""
    name = QUANT_FILE.format(quant)
    size = files[name][0]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{name}.header"
    want = 16 * MB
    while True:
        if cached.exists() and cached.stat().st_size >= want:
            data = cached.read_bytes()
        else:
            req = urllib.request.Request(f"{HF_BASE}/{name}", headers={"Range": f"bytes=0-{want - 1}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            cached.write_bytes(data)
        try:
            return parse_gguf_header(data, size)
        except EOFError:
            want *= 2
            if want > 128 * MB:
                raise


# ----------------------------------------------------------------------------------------- planning
def place(layout, gpu, ram_total, desktop_ram, ncmoe=None):
    """GPU/CPU split for one quant. Full expert layers go to the GPU first (last layers first), then the
    remaining VRAM becomes hot-expert cache slots for the CPU-resident layers."""
    E, n = layout["experts"], layout["n_layers"]
    budget = gpu["vram_total"] - gpu["vram_used"] - VRAM_RESERVE - layout["other"] - GPU_OVERHEAD
    if budget < 0:
        return None
    if ncmoe is None:
        ncmoe, used = n, 0
        while ncmoe > 0 and used + E[ncmoe - 1] <= budget:
            ncmoe -= 1
            used += E[ncmoe]
    used = sum(E[ncmoe:])
    if used > budget:
        return None
    slot_cost = sum(E[:ncmoe]) / 256 if ncmoe else 0
    slots = min(MAX_SLOTS, int((budget - used) // slot_cost)) if slot_cost else 0
    cpu_bytes = sum(E[:ncmoe])
    ram_for_model = ram_total - desktop_ram
    cache_ram = int(min(CACHE_RAM_MAX * MB, ram_for_model - cpu_bytes - RAM_BASE) // MB)
    vision_slots, vision_ncmoe, freed = slots, ncmoe, 0
    while freed < VISION_EXTRA and vision_slots > 0:
        vision_slots -= 1
        freed += slot_cost
    while freed < VISION_EXTRA and vision_ncmoe < n:
        freed += E[vision_ncmoe]
        vision_ncmoe += 1
    return {
        "ncmoe": ncmoe, "slots": slots, "cpu_expert_bytes": cpu_bytes, "gpu_expert_bytes": used,
        "vram_llama": layout["other"] + GPU_OVERHEAD + used + slots * slot_cost,
        "ram_llama": cpu_bytes + RAM_BASE + max(cache_ram, 0) * MB,
        "cache_ram_mb": cache_ram, "fits_ram": cache_ram >= 1024,
        "vision_ncmoe": vision_ncmoe, "vision_slots": vision_slots,
    }


def choose(hw, get_layout, desktop_ram, forced=None):
    """Best quant for this machine. get_layout(q) returns a layout or None. Returns (quant, placement, reason)."""
    if not hw["gpus"]:
        return None, None, "no NVIDIA GPU found (nvidia-smi); Shura needs an NVIDIA GPU with CUDA"
    gpu = hw["gpus"][0]
    if forced:
        p = place(get_layout(forced), gpu, hw["ram_total"], desktop_ram)
        return forced, p, "forced with --quant" if p else f"{forced} does not fit this GPU"
    for q in UPGRADE:
        layout = get_layout(q)
        p = layout and place(layout, gpu, hw["ram_total"], desktop_ram)
        if p and p["fits_ram"] and p["cpu_expert_bytes"] <= 2 * GB:
            return q, p, "large GPU: higher-quality quant fits almost entirely in VRAM"
    for q in DOWNGRADE:
        layout = get_layout(q)
        p = layout and place(layout, gpu, hw["ram_total"], desktop_ram)
        if p and p["fits_ram"]:
            why = "benchmarked default" if q == "IQ4_XS" else "smaller quant so the model fits next to your desktop apps"
            return q, p, why
    return None, None, (f"not enough memory: even {DOWNGRADE[-1]} would need more RAM than "
                        f"{hw['ram_total'] / GB:.0f} GB minus {desktop_ram / GB:.0f} GB kept for the desktop")


def threads_for(hw):
    return max(1, min(hw["physical_cores"], 8))  # CPU side is memory-bandwidth bound; more threads only steal CPU


# ------------------------------------------------------------------------------------------- fetch
def sha256_file(path):
    h = hashlib.sha256()
    total, done, last = os.path.getsize(path), 0, 0
    with open(path, "rb") as f:
        while chunk := f.read(16 * MB):
            h.update(chunk)
            done += len(chunk)
            if done - last > total / 10:
                last = done
                log(f"verifying {os.path.basename(path)}: {100 * done // total}%")
    return h.hexdigest()


def download(name, size, sha, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and dest.stat().st_size == size:
        log(f"{name} already present")
    else:
        log(f"downloading {name} ({size / GB:.1f} GB, resumable - rerun setup if interrupted)")
        rc = subprocess.run(["curl", "-fL", "-C", "-", "--retry", "5", "--retry-delay", "5", "-o", str(part),
                             f"{HF_BASE}/{name}"]).returncode
        if rc != 0 or part.stat().st_size != size:
            sys.exit(f"download of {name} failed (curl exit {rc}); rerun setup to resume")
        part.rename(dest)
    if sha and sha256_file(dest) != sha:
        dest.rename(dest.with_suffix(".corrupt"))
        sys.exit(f"{name}: SHA-256 mismatch - file moved aside as .corrupt, rerun setup to download again")
    log(f"{name}: OK")


# ---------------------------------------------------------------------------------------- autotune
class Server:
    def __init__(self, llama, model, mmproj, threads, ncmoe, slots, cache_type, port, slot_dir, log_path):
        self.url = f"http://127.0.0.1:{port}"
        args = ["nice", "-n", "10", llama, "--host", "127.0.0.1", "--port", str(port), "-m", model,
                "-c", "200000", "-np", "1", "-t", str(threads), "-tb", str(threads), "-b", "2048", "-ub", "512",
                "-ngl", "99", "-ncmoe", str(ncmoe), "-ctk", cache_type, "-ctv", cache_type, "-fa", "on",
                "--load-mode", "none", "--jinja", "--cache-ram", "1024", "--slot-save-path", str(slot_dir),
                "--no-mmproj-auto", "--spec-type", "draft-mtp", "--spec-draft-n-max", "1", "--spec-draft-p-min", "0.2"]
        trace = REPO_DIR / "config/moe-trace/tiel-coder-agentic.csv"
        if slots > 0:
            args += ["--moe-cache-profile", str(trace), "--moe-cache-slots", str(slots)]
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def wait(self, timeout=300):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=3) as r:
                    if r.status == 200:
                        return True
            except OSError:
                pass
            time.sleep(1)
        return False

    def post(self, path, payload, timeout=3600):
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def vram_mb(self):
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        for line in out.strip().splitlines():
            pid, mem = [x.strip() for x in line.split(",")]
            if int(pid) == self.proc.pid:
                return int(mem)
        return -1

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()
        time.sleep(3)


PROBES = [("What is 17 + 26?", "43"), ("What is 123 * 4?", "492"), ("What is 100 - 37?", "63"),
          ("What is 9 * 9?", "81"), ("What is 256 / 4?", "64"), ("What is the capital of France?", "Paris"),
          ("What is the capital of Japan?", "Tokyo"), ("What color do you get mixing blue and yellow?", "green"),
          ("How many days are in a week?", "7")]


def correctness(srv):
    ok = 0
    for q, want in PROBES:
        out = srv.post("/completion", {"prompt": f"Q: {q}\nA:", "n_predict": 40, "temperature": 0})
        ok += want.lower() in out.get("content", "").lower()
    return ok


def filler(llama_src, chars):
    files = sorted(p for sub in ("src", "ggml/src", "common", "tools")
                   for ext in ("*.cpp", "*.h", "*.cu") for p in (Path(llama_src) / sub).rglob(ext))
    text, total = [], 0
    for p in files:
        t = p.read_text(errors="ignore")
        text.append(t)
        total += len(t)
        if total >= chars:
            break
    return "".join(text)[:chars]


def decode_speed(srv, turn, runs):
    """Median decode tok/s over `runs` requests after one discarded warm-up (first run after a load or a
    long prefill is not representative)."""
    speeds = []
    for i in range(runs + 1):
        srv.post("/slots/0?action=restore", {"filename": "tune.bin"})
        t = srv.post("/v1/chat/completions", {"messages": turn, "max_tokens": 128, "ignore_eos": True,
                                              "temperature": 0.6}).get("timings", {})
        if i:
            speeds.append(t.get("predicted_per_second", 0))
    return sorted(speeds)[len(speeds) // 2]


def autotune(args):
    hw = detect()
    gpu = hw["gpus"][0]
    layout = read_local_layout(args.model)
    desktop_ram = args.desktop_ram_gb * GB
    base = place(layout, gpu, hw["ram_total"], desktop_ram)
    if not base:
        sys.exit("this model does not fit the GPU even with all experts on the CPU")
    threads = threads_for(hw)
    work = CACHE_DIR / "autotune"
    work.mkdir(parents=True, exist_ok=True)
    port = 18090

    # Candidates: the planned split, then trade one or two GPU expert layers for more cache slots.
    cands = []
    for extra in range(0, 1 if args.quick else 3):
        p = place(layout, gpu, hw["ram_total"], desktop_ram, ncmoe=min(layout["n_layers"], base["ncmoe"] + extra))
        if p and (not cands or (p["ncmoe"], p["slots"]) != (cands[-1]["ncmoe"], cands[-1]["slots"])):
            cands.append(p)

    cache_type = "turbo3"
    # Compare at real depth: at ~32k a layout can win by ~10% and then tie at 187k (measured).
    ctx_text = filler(args.llama_src, 40_000 if args.quick else 300_000)  # ~12k / ~100k tokens
    turn1 = [{"role": "user", "content": "Here is a codebase:\n\n" + ctx_text + "\n\nReply with just: OK"}]
    results, assistant = [], None
    for i, c in enumerate(cands):
        for attempt in range(4):
            log(f"candidate {i + 1}/{len(cands)}: ncmoe {c['ncmoe']}, {c['slots']} cache slots "
                f"(attempt {attempt + 1})")
            srv = Server(args.llama, args.model, None, threads, c["ncmoe"], c["slots"], cache_type, port, work,
                         work / f"cand{i}.log")
            try:
                if not srv.wait():
                    raise RuntimeError("failed to load")
                if assistant is None:
                    if correctness(srv) < len(PROBES):
                        if cache_type == "turbo3":
                            log("turbo3 KV cache failed the correctness probes - falling back to q4_0")
                            cache_type = "q4_0"
                            continue
                        sys.exit("model output failed the correctness probes even with q4_0 KV - aborting")
                    r = srv.post("/v1/chat/completions", {"messages": turn1, "max_tokens": 16})
                    msg = r["choices"][0]["message"]
                    assistant = {"role": "assistant", "content": msg.get("content", "")}
                    if msg.get("reasoning_content"):
                        assistant["reasoning_content"] = msg["reasoning_content"]
                    srv.post("/slots/0?action=save", {"filename": "tune.bin"})
                turn2 = turn1 + [assistant, {"role": "user",
                                             "content": "Write a short function to reverse a linked list in Python."}]
                speed = decode_speed(srv, turn2, 2 if args.quick else 3)
                vram = srv.vram_mb()
                log(f"  -> {speed:.1f} tok/s, llama-server VRAM {vram} MB")
                results.append((speed, c, vram))
                break
            except (RuntimeError, OSError) as e:
                log(f"  failed ({e}); reducing cache slots")
                c = dict(c, slots=max(0, int(c["slots"] * 0.75)))
            finally:
                srv.stop()
    if not results:
        sys.exit("no configuration could be loaded - see ~/.cache/shura/autotune/*.log")
    top = max(r[0] for r in results)
    # Within 5% of the fastest counts as a tie: prefer the one that leaves the desktop more VRAM.
    speed, best, vram = min((r for r in results if r[0] >= 0.95 * top), key=lambda r: r[2])
    log(f"chosen: ncmoe {best['ncmoe']}, {best['slots']} slots ({speed:.1f} tok/s, {vram} MB VRAM)")

    if not args.quick:  # full-context stress test: catches OOM that only appears at a full window
        big = filler(args.llama_src, 1_500_000)  # plenty; trimmed below to ~185k real tokens
        for _ in range(3):
            log(f"full-context check: ~185k tokens with ncmoe {best['ncmoe']}, {best['slots']} slots "
                "(prefill takes a few minutes)")
            srv = Server(args.llama, args.model, None, threads, best["ncmoe"], best["slots"], cache_type, port, work,
                         work / "full.log")
            try:
                if not srv.wait():
                    raise RuntimeError("failed to load")
                n = len(srv.post("/tokenize", {"content": big})["tokens"])
                if n < 185_000:
                    log(f"  note: only {n} tokens of filler text available")
                text = big[: int(len(big) * min(1.0, 185_000 / n))]
                msgs = [{"role": "user", "content": "Here is a codebase:\n\n" + text + "\n\nSummarize it briefly."}]
                t0 = time.time()
                t = srv.post("/v1/chat/completions", {"messages": msgs, "max_tokens": 64, "ignore_eos": True})["timings"]
                full_speed = t.get("predicted_per_second", 0)
                log(f"  OK: {t.get('prompt_n')} tokens prefilled in {time.time() - t0:.0f}s, "
                    f"decode {full_speed:.1f} tok/s at full depth")
                best = dict(best, full_ctx_speed=full_speed)
                break
            except (RuntimeError, OSError) as e:
                log(f"  failed at full context ({e}); reducing cache slots")
                best = dict(best, slots=max(0, int(best["slots"] * 0.8)))
            finally:
                srv.stop()

    write_config(args, layout, hw, best, cache_type, threads, desktop_ram)
    print(json.dumps({"ncmoe": best["ncmoe"], "slots": best["slots"], "cache_type": cache_type,
                      "threads": threads, "decode_compare": round(speed, 1),
                      "decode_full": round(best.get("full_ctx_speed", 0), 1)}))


def write_config(args, layout, hw, best, cache_type, threads, desktop_ram):
    gpu = hw["gpus"][0]
    p = place(layout, gpu, hw["ram_total"], desktop_ram, ncmoe=best["ncmoe"])
    cfg_dir = REPO_DIR / "config"
    model = {
        "exePath": args.llama, "cacheRamMB": max(512, p["cache_ram_mb"]), "defaultModel": "tiel-coder",
        "models": {"tiel-coder": {
            "modelPath": args.model, "mmprojPath": args.mmproj, "ctxSize": 200000, "parallelSlots": 1,
            "threads": threads, "threadsBatch": threads, "batchSize": 2048, "ubatchSize": 512, "nGpuLayers": 99,
            "nCpuMoe": best["ncmoe"], "cacheTypeK": cache_type, "cacheTypeV": cache_type, "flashAttn": "on",
            "loadMode": "none", "moeCacheProfile": "moe-trace/tiel-coder-agentic.csv", "moeCacheSlots": best["slots"],
            "temperature": 0.6, "topK": 20, "topP": 0.95, "specType": "draft-mtp", "specDraftNMax": 1,
            "specDraftPMin": 0.2, "reasoning": "auto", "niceLevel": 10,
            "vramFreeMinGBToLoad": round(p["vram_llama"] / GB, 1),
            "visionOverrides": {"nCpuMoe": p["vision_ncmoe"], "moeCacheSlots": min(best["slots"], p["vision_slots"]),
                                "vramFreeMinGBToLoad": round((p["vram_llama"]) / GB, 1)},
        }},
    }
    guardian = json.loads((cfg_dir / "guardian.example.json").read_text())
    guardian["ramFreeMinGBToLoad"] = round(p["ram_llama"] / GB + 1.0, 1)
    for name, data in (("model-launch.json", model), ("guardian.json", guardian)):
        target = cfg_dir / name
        if target.exists():
            shutil.copy(target, target.with_suffix(f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}"))
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(target)
    log(f"wrote {cfg_dir}/model-launch.json and guardian.json")


# ---------------------------------------------------------------------------------------------- CLI
def cmd_plan(args):
    hw = detect()
    files = hf_files()
    cache = {}

    def get_layout(q):
        if q not in cache:
            cache[q] = remote_layout(q, files) if QUANT_FILE.format(q) in files else None
        return cache[q]

    quant, p, why = choose(hw, get_layout, args.desktop_ram_gb * GB, args.quant)
    name = QUANT_FILE.format(quant) if quant else None
    out = {"quant": quant, "reason": why, "file": name, "size": files[name][0] if name else 0,
           "cuda_arch": hw["gpus"][0]["cuda_arch"] if hw["gpus"] else None, "threads": threads_for(hw),
           "placement": {k: v for k, v in (p or {}).items() if not k.endswith("_bytes")}}
    print(json.dumps(out))


def cmd_fetch(args):
    """Model into --dest (or verify --existing), mmproj next to the model. Prints the final paths."""
    files = hf_files()
    name = QUANT_FILE.format(args.quant)
    if args.existing:
        model = Path(args.existing).resolve()
        size, sha = files[name]
        if model.stat().st_size != size or sha256_file(model) != sha:
            sys.exit(f"{model} is not an intact {name} (size/SHA-256 mismatch)")
        log(f"reusing existing {model}")
    else:
        model = Path(args.dest).resolve() / name
        download(name, *files[name], model)
    mmproj = model.parent / MMPROJ
    download(MMPROJ, *files[MMPROJ], mmproj)
    print(json.dumps({"model": str(model), "mmproj": str(mmproj)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("detect")
    p = sub.add_parser("plan")
    p.add_argument("--quant")
    p.add_argument("--desktop-ram-gb", type=float, default=6)
    f = sub.add_parser("fetch")
    f.add_argument("--quant", required=True)
    f.add_argument("--dest", required=True)
    f.add_argument("--existing")
    t = sub.add_parser("autotune")
    t.add_argument("--model", required=True)
    t.add_argument("--mmproj", required=True)
    t.add_argument("--llama", required=True)
    t.add_argument("--llama-src", required=True)
    t.add_argument("--desktop-ram-gb", type=float, default=6)
    t.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.cmd == "detect":
        print(json.dumps(detect(), indent=2))
    elif a.cmd == "plan":
        cmd_plan(a)
    elif a.cmd == "fetch":
        cmd_fetch(a)
    elif a.cmd == "autotune":
        autotune(a)


if __name__ == "__main__":
    main()
