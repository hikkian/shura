"""`shura install`: detect -> plan -> engine -> model -> verified launch -> ready to use.

Everything that can fail is checked *before* the expensive part (disk space, a plan that fits) and the result is
verified by really starting the server with the planned settings. If that fails (almost always: out of memory at
load), the plan is made again with more room left free and tried again, so the user ends with a configuration that
was seen to work instead of an error. Nothing is written outside the Shura home folder.
"""
import copy
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from . import backends, engine, launch, modelstore, planner, report

GiB = 1024 ** 3
OOM = re.compile(r"out of memory|failed to allocate|cudaMalloc|cannot allocate|ErrorOutOfDeviceMemory|bad_alloc|"
                 r"unable to allocate|insufficient memory", re.I)
MAX_ATTEMPTS = 3
WARMUP_RUNS, MEASURE_RUNS = 2, 3         # speed is the median of 3 runs after 2 warm-up runs
NATIVE = {"nvidia": "cuda", "amd": "rocm", "intel": "sycl", "apple": "metal"}   # each vendor's own stack: the main engine there
NATIVE_BONUS = 1.10                  # anything else must be 10% faster, measured on this machine, to replace it


class Out:
    """Terminal output; tests replace it."""
    def say(self, text=""):
        print(text, flush=True)

    def ask(self, question, default=True):
        if not sys.stdin or not sys.stdin.isatty():       # nobody to ask: never start a big download silently
            self.say(f"  ({question} not asked: no terminal; pass --yes to agree)")
            return False
        reply = input(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
        return default if not reply else reply.startswith("y")


def relaxed_config(base, attempt, hw):
    """Planner overrides for retry number `attempt` (0 = the first plan): leave more VRAM free, then a smaller window."""
    cfg = copy.deepcopy(base or {})
    if attempt:
        has_gpu = bool(hw["gpus"])
        if has_gpu:
            cfg["vram_reserve_display"] = planner.DEFAULTS["vram_reserve_display"] + attempt * GiB
            cfg["vram_reserve_headless"] = planner.DEFAULTS["vram_reserve_headless"] + attempt * GiB
        cfg["os_reserve_min"] = planner.DEFAULTS["os_reserve_min"] + attempt * GiB
        if attempt >= 2:
            cfg["contexts"] = tuple(c for c in planner.DEFAULTS["contexts"] if c <= 131072)
    return cfg


def preview(plan, model, quant, dirs, server_bytes=120 * 1024 ** 2, turbo=None, hip=None):
    q = next(x for x in model["quants"] if x["id"] == quant)
    lines = [f"model file : {modelstore.file_name(model, quant)}  ({q['file_bytes'] / GiB:.1f} GiB)"]
    if hip:
        lines.append("our fork  : built on this machine for ROCm with scripts/build-llama.sh --hip (tens of minutes, ~1 GiB of "
                     "disk; skip with --no-build): its expert cache is what makes our NVIDIA setup fast")
    if turbo:
        lines.append(f"engine     : TurboQuant+ llama.cpp {engine.TQP_TAG} (third-party fork, {turbo['backend_candidates'][0]}, "
                     f"pinned SHA-256; skip with --no-turbo), then llama.cpp {engine.PINNED_TAG}; the fastest measured is kept")
    else:
        lines.append(f"engine     : llama.cpp {engine.PINNED_TAG}, backend {', '.join(plan['backend_candidates'])}  "
                     f"(~{server_bytes // 1024 ** 2} MB each)")
    lines += [f"folder     : {dirs}",
              f"disk needed: about {(q['file_bytes'] + server_bytes * (len(plan['backend_candidates']) + (1 if turbo else 0))) / GiB + 2 + (1 if hip else 0):.0f} GiB"]
    return lines


def build_rocm_fork(home, out, arch=None):
    """Build our fork (the expert cache) for ROCm with scripts/build-llama.sh --hip. Takes tens of minutes, runs at a low
    priority, logs to <home>/logs/build-hip.log. Returns the path of llama-server or raises EngineError."""
    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "build-llama.sh"
    dest = Path(home) / "llama.cpp-rocm"
    server = dest / "build" / "bin" / "llama-server"
    if server.exists():
        return server
    log_path = Path(home) / "logs" / "build-hip.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["nice", "-n", "10", "bash", str(script), "--hip"] + ([arch] if arch else []) + [str(dest)]
    out.say("  Building our fork for ROCm: this takes tens of minutes (log: " + str(log_path) + ")")
    t0 = time.monotonic()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env={**os.environ, "SHURA_BUILD_JOBS": "6"})
        while proc.poll() is None:
            time.sleep(1)
            if int(time.monotonic() - t0) % 300 == 0 and time.monotonic() - t0 > 1:
                out.say(f"  ... still building ({int((time.monotonic() - t0) / 60)} min)")
                time.sleep(1)
    if proc.returncode != 0 or not server.exists():
        raise engine.EngineError(f"the ROCm build of our fork failed ({log_path}); attach the end of that log to a hardware report: "
                                 + launch.tail(log_path, 400).replace("\n", " "))
    return server


def _choose_hipfork(hw, plan, root, out):
    server = build_rocm_fork(root, out)
    res = engine.selftest(server, engine.smoke_model(root / "models"), backend="rocm")
    if not res["ok"]:
        raise engine.EngineError(f"our ROCm build did not pass the self-test: {res.get('error', '?')}")
    out.say(f"  rocm: works ({res['tok_s']} tok/s on the test model)")
    return "rocm", server, [res]


def _choose_tqp(hw, plan, root, out):
    """The pinned TurboQuant+ build for this machine: its archive must match the SHA-256 recorded in `engine.TQP_SHA256`."""
    release = engine.fetch_release(engine.TQP_TAG, repo=engine.TQP_REPO)
    backend = plan["backend_candidates"][0]
    asset = backends.pick_tqp_asset(release["assets"], hw["os"]["family"], hw["os"]["arch"], backend)
    if not asset:
        raise engine.EngineError(f"the TurboQuant+ release {release['tag']} has no {backend} build for this machine")
    pinned = engine.TQP_SHA256.get(asset["name"])
    if not pinned:
        raise engine.EngineError(f"{asset['name']} is not one of the builds this installer has pinned")
    out.say(f"  using {asset['name']}")
    server = engine.install_build(release, asset, root / "engines", label=f"tqp-{backend}", sha256=pinned)
    res = engine.selftest(server, engine.smoke_model(root / "models"), backend=backend)
    if not res["ok"]:
        raise engine.EngineError(f"the TurboQuant+ build did not pass the self-test: {res.get('error', '?')}")
    out.say(f"  {backend}: works ({res['tok_s']} tok/s on the test model)")
    return backend, server, [res]


def choose_engine(hw, plan, root, tag, out, release=None, only_backend=None):
    """Download the candidate builds and keep the fastest one that passes the self-test (or just `only_backend`).
    Returns (backend, server, results)."""
    if plan.get("engine") == "turboquant-plus":
        return _choose_tqp(hw, plan, root, out)
    if plan.get("engine") == "shura-fork-rocm":
        return _choose_hipfork(hw, plan, root, out)
    release = release or engine.fetch_release(tag)
    builds = backends.builds_for(hw, plan["backend_candidates"], release["assets"])
    if only_backend:
        builds = [b for b in builds if b[0] == only_backend]
    if not builds:
        raise engine.EngineError(f"llama.cpp {release['tag']} has no build for this machine "
                                 f"({', '.join(plan['backend_candidates'])})")
    smoke = engine.smoke_model(root / "models")
    tested, servers = [], {}
    for backend, asset in builds:
        if backend == "cpu" and any(r["ok"] for r in tested):          # the CPU build is only the fallback
            break
        out.say(f"  trying {backend} build ({asset['name']}) ...")
        try:
            server = engine.install_build(release, asset, root / "engines")
            res = engine.selftest(server, smoke, backend=backend)
        except (engine.EngineError, OSError) as e:
            res = {"backend": backend, "ok": False, "error": str(e)}
        tested.append(res)
        servers[backend] = server if res["ok"] else None
        out.say(f"    {backend}: " + (f"works ({res['tok_s']} tok/s on the test model)" if res["ok"]
                                     else "does not work here (" + str(res.get("error", "?"))[:100] + ")"))
        if res["ok"] and (only_backend or not plan["needs_probe"]):
            break
    order = [b for b, _ in builds]
    best, why = engine.choose_backend(tested, order)
    if not best:
        raise engine.EngineError("no llama.cpp build worked on this machine: " + why)
    out.say(f"  using {best}: {why}")
    return best, servers[best], tested


def reconcile(hw, server, backend, out, tolerance=0.10):
    """Trust what the downloaded build reports about the GPU over our own guess (Windows cannot read AMD/Intel video memory
    reliably, and the desktop's share is only an estimate there). Returns a corrected copy of `hw`, or `hw` itself."""
    if backend == "cpu" or not hw["gpus"]:
        return hw
    devices = [d for d in engine.list_devices(server) if not d["id"].upper().startswith("CPU")]
    if not devices:
        return hw
    dev = max(devices, key=lambda d: d["total"])
    gpu = max(hw["gpus"], key=lambda g: g["vram_total"])
    used = max(0, dev["total"] - dev["free"])
    if abs(dev["total"] - gpu["vram_total"]) <= tolerance * dev["total"] and gpu["vram_used"] >= used * 0.8:
        return hw
    fixed = copy.deepcopy(hw)
    g = max(fixed["gpus"], key=lambda x: x["vram_total"])
    g.update(vram_total=dev["total"], vram_used=used, display=g.get("display", True) or used > 256 * 1024 ** 2)
    out.say(f"  The build reports {dev['name']} with {dev['total'] / GiB:.1f} GiB ({dev['free'] / GiB:.1f} GiB free): "
            f"planning with that.")
    return fixed


def refine(hw, catalog, plan, argv, verdict, server, model_path, args, base_cfg, out, home):
    """Close the loop: the plan was predicted, the server was really run. If the measured speed is far from the prediction,
    plan again with the predictions scaled by the measured/predicted ratio (same model file, so nothing is downloaded) and
    keep the new plan only if it really starts and answers. Returns (plan, argv, verdict, calibration info)."""
    predicted = plan["speed_by_fill"]["empty"]
    ratio = max(0.25, min(1.6, verdict["tok_s"] / predicted)) if predicted else 1.0
    info = {"ratio": round(ratio, 2), "replanned": False, "adopted": False}
    if abs(ratio - 1.0) <= 0.15:
        return plan, argv, verdict, info
    cfg = {**relaxed_config(base_cfg, 0, hw), "speed_scale": ratio}
    new = planner.plan(hw, catalog, config=cfg, model=plan["model"], quant=plan["quant"], menu=False,
                       profile=getattr(args, "profile_name", "balanced"))
    key = lambda p: (p["settings"]["context"], p["settings"]["kv_type"], p["settings"].get("n_cpu_moe"))   # noqa: E731
    if not new["ok"] or key(new) == key(plan):
        return plan, argv, verdict, info
    info["replanned"] = True
    out.say(f"  Measured {verdict['tok_s']} tok/s, the plan said {predicted:.0f} (x{ratio:.2f}). Planning again with the "
            f"measured speed: {new['settings']['context']} tokens of context, {new['settings']['kv_type']} KV cache.")
    new_argv = launch.server_args(new, server, model_path, port=args.port, alias=new["model"])
    check = verify(new_argv, args.port, home=home, wait_s=args.wait)
    if check["ok"]:
        info["adopted"] = True
        return new, new_argv, check, info
    out.say("  The new settings did not start; keeping the ones that worked.")
    return plan, argv, verdict, info


def _complete(port, prompt, n_predict):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", method="POST",
                                 data=json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0, "seed": 1}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def verify(argv, port, *, home, wait_s=900, n_predict=64):
    """Start the planned server, generate some tokens, stop it. Returns {ok, load_s, tok_s, prompt_tok_s, error, oom}."""
    res = {"ok": False}
    t0 = time.monotonic()
    try:
        launch.start(home, argv, port, wait_s=wait_s, log_name="verify.log")
    except launch.LaunchError as e:
        res.update(error=str(e), oom=bool(OOM.search(str(e) + launch.tail(Path(home) / "logs" / "verify.log", 8000))))
        launch.stop(home)
        return res
    try:
        res["load_s"] = round(time.monotonic() - t0, 1)
        prompt = "Write a short Python function that returns the n-th Fibonacci number, with a docstring.\n"
        for _ in range(WARMUP_RUNS):    # shader compilation, and an adaptive expert cache that fills as it goes (measured: the first
            _complete(port, prompt, 48)  # requests ran at half speed), would otherwise read as a slow machine
        runs = [_complete(port, prompt, n_predict) for _ in range(MEASURE_RUNS)]
        runs.sort(key=lambda b: b.get("timings", {}).get("predicted_per_second", 0.0))
        body = runs[len(runs) // 2]                                      # the median run
        tm = body.get("timings", {})
        res.update(ok=bool(body.get("content", "").strip()), tokens=tm.get("predicted_n", 0),
                   tok_s=round(tm.get("predicted_per_second", 0.0), 1),
                   prompt_tok_s=round(tm.get("prompt_per_second", 0.0), 1))
        if not res["ok"]:
            res["error"] = "the server answered with an empty text"
    except (OSError, ValueError) as e:
        res.update(error=f"the server stopped answering: {e}", oom=bool(OOM.search(launch.tail(Path(home) / "logs" / "verify.log", 8000))))
    finally:
        launch.stop(home)
    return res


def _ensure_model(args, catalog, plan, home, out, cache):
    """The model file for the plan's quant, downloaded and verified once per install (`cache` remembers it)."""
    model = next(m for m in catalog["models"] if m["id"] == plan["model"])
    name = modelstore.file_name(model, plan["quant"])
    if name in cache:
        return cache[name]
    out.say("\n  Model")
    files = modelstore.listing(model["source"]["hf_repo"], **({"base": args.hf_base, "allow_local": True} if args.hf_base else {}))
    if name not in files:
        raise modelstore.StoreError(f"{name} is not in the {model['source']['hf_repo']} repository")
    size, sha = files[name]
    url = f"{args.hf_base or 'https://huggingface.co'}/{model['source']['hf_repo']}/resolve/main/{name}"
    last = [0.0]

    def progress(done, total):
        if time.monotonic() - last[0] > 5:
            out.say(f"  {done / GiB:.1f} / {total / GiB:.1f} GiB ({done * 100 // total}%)")
            last[0] = time.monotonic()
    path = modelstore.fetch(url, home / "models" / name, size, sha, progress=progress, allow_local=bool(args.hf_base))
    out.say(f"  {name}: OK (size and SHA-256 verified)")
    cache[name] = path
    return path


def _attempt(args, hw, catalog, out, release, base_cfg, home, cache, label, only_backend=None):
    """One complete try with one engine: plan, engine, model, test launch (with the out-of-memory and no-MTP fallbacks),
    calibration. Returns a result dict ({ok: True, plan, argv, verdict, ...}) or {ok: False, errors, plan}. Never raises for
    the problems it knows (a failed download, a build that does not work)."""
    first_hw = hw
    plan = planner.plan(hw, catalog, config=relaxed_config(base_cfg, 0, hw), model=args.model_id, quant=args.quant,
                        profile=getattr(args, "profile_name", "balanced"), menu=False)
    if not plan["ok"]:
        return {"ok": False, "errors": plan["reasons"], "plan": plan}
    errors = []
    try:
        out.say(f"\n  Engine ({label})")
        backend, server, _ = choose_engine(hw, plan, home, None if args.latest else args.tag, out, release,
                                           only_backend=only_backend)
        hw = reconcile(hw, server, backend, out)
        if backend == "cpu" and hw["gpus"]:
            out.say("  No GPU build works here: planning for the CPU instead.")
            hw = {**hw, "gpus": []}
        if hw is not first_hw:
            plan = planner.plan(hw, catalog, model=plan["model"], quant=plan["quant"],       # same file: never a surprise download
                                config=relaxed_config(base_cfg, 0, hw), menu=False)
            if not plan["ok"]:
                return {"ok": False, "errors": plan["reasons"], "plan": plan}
        path = _ensure_model(args, catalog, plan, home, out, cache)
    except (engine.EngineError, modelstore.StoreError, OSError) as e:
        return {"ok": False, "errors": [str(e)], "plan": plan, "stopped": True}

    out.say(f"\n  Test launch with the planned settings ({label})")
    verdict, final = None, None
    attempt, replan = 0, False
    while attempt < MAX_ATTEMPTS:
        if attempt or replan:
            plan = planner.plan(hw, catalog, model=plan["model"], quant=plan["quant"], menu=False,
                                config=relaxed_config(base_cfg, attempt, hw))
            if not plan["ok"]:
                break
            if attempt:
                out.say(f"  Trying again with more room left free (attempt {attempt + 1}): "
                        f"{plan['settings']['context']} tokens of context")
            replan = False
        argv = launch.server_args(plan, server, path, port=args.port, alias=plan["model"])
        verdict = verify(argv, args.port, home=home, wait_s=args.wait)
        if verdict["ok"]:
            final = (plan, argv)
            break
        errors.append(f"{label}, attempt {attempt + 1}: {verdict.get('error', '?')[:300]}")
        out.say("  did not work" + (" (out of memory)" if verdict.get("oom") else "") + ": "
                + str(verdict.get("error", ""))[:200].replace("\n", " "))
        kv = plan["settings"]["kv_type"]
        if kv in planner.DEFAULTS["kv_exotic"] and not verdict.get("oom") and kv not in base_cfg.get("kv_unavailable", ()):
            base_cfg = {**base_cfg, "kv_unavailable": tuple(base_cfg.get("kv_unavailable", ())) + (kv,)}
            out.say(f"  This build may not run a {kv} KV cache with flash attention: trying q8_0 / q4_0 instead.")
            replan = True
            continue
        if plan["settings"].get("mtp") and not verdict.get("oom") and base_cfg.get("mtp_standard", True):
            base_cfg = {**base_cfg, "mtp_standard": False}                  # this build may not run the MTP head: try without
            out.say("  Trying without the MTP draft (speculative decoding).")
            replan = True
            continue
        if not verdict.get("oom") and attempt:
            break
        attempt += 1
    if not final:
        return {"ok": False, "errors": errors, "plan": plan}
    plan, argv = final
    first_predicted, first_measured = plan["speed_by_fill"]["empty"], verdict["tok_s"]
    plan, argv, verdict, calibration = refine(hw, catalog, plan, argv, verdict, server, path, args, base_cfg, out, home)
    return {"ok": True, "plan": plan, "argv": argv, "verdict": verdict, "calibration": calibration, "backend": backend,
            "server": str(server), "path": str(path), "hw": hw, "label": label, "first_predicted": round(first_predicted, 1),
            "first_measured": first_measured, "errors": errors}


def run(args, hw, catalog, out=None, *, release=None):
    """The whole installation. Returns the process exit code: 0 ready, 1 nothing fits / failed, 2 bad input.

    Where the TurboQuant+ build can run (Vulkan, Metal, CUDA on Windows) it is tried first, and then upstream llama.cpp is
    tried too; the faster of the two, measured on this machine, is kept (the TurboQuant+ build wins a near tie, it is the
    main engine). `--no-turbo` skips it. A build that does not start never stops the install: the other one is used."""
    out = out or Out()
    home = Path(args.dir) if args.dir else launch.home_dir()
    base_cfg = {"contexts": tuple(c for c in planner.DEFAULTS["contexts"] if c <= args.context)} if args.context else {}
    no_turbo = bool(getattr(args, "no_turbo", False))
    no_build = bool(getattr(args, "no_build", False))
    profile = getattr(args, "profile_name", "balanced")
    up_cfg = {**base_cfg, "tqp_enabled": False, "hipfork_enabled": False}
    tq_cfg = {**base_cfg, "kv_unavailable": ("q8_0", "q5_0", "q4_0"), "hipfork_enabled": False}   # only TurboQuant+ candidates
    hip_cfg = {**base_cfg, "only_tier": "hipfork", "tqp_enabled": False}                       # only our fork built for ROCm
    first_cfg = {**base_cfg, **({"tqp_enabled": False} if no_turbo else {}), **({"hipfork_enabled": False} if no_build else {})}
    plan = planner.plan(hw, catalog, config=relaxed_config(first_cfg, 0, hw),
                        model=args.model_id, quant=args.quant, profile=profile)
    out.say(report.card(hw, plan))
    if not plan["ok"]:
        return 1
    model = next(m for m in catalog["models"] if m["id"] == plan["model"])
    if plan.get("tier") == "fork":
        setup = Path(__file__).resolve().parent.parent.parent / "setup.sh"
        out.say("\nThis machine gets the fast tier (Shura's CUDA fork: turbo3 KV cache, expert cache, MTP).")
        out.say(f"Run:  bash {setup} --quant {plan['quant']}" + ("  --yes" if args.yes else ""))
        if args.dry_run or not setup.exists() or not out.ask("Run it now?", default=False):
            return 0
        return subprocess.call(["bash", str(setup), "--quant", plan["quant"]] + (["--yes"] if args.yes else []))
    turbo_plan = None
    if not no_turbo:
        t = planner.plan(hw, catalog, config=relaxed_config(tq_cfg, 0, hw), model=args.model_id, quant=args.quant,
                         profile=profile, menu=False)
        turbo_plan = t if t["ok"] and t.get("tier") == "tqp" else None
    hip_plan = None
    if not no_build:
        h = planner.plan(hw, catalog, config=relaxed_config(hip_cfg, 0, hw), model=args.model_id, quant=args.quant,
                         profile=profile, menu=False)
        hip_plan = h if h["ok"] and h.get("tier") == "hipfork" else None
    out.say("\nWhat will be downloaded")
    for line in preview(plan, model, plan["quant"], home, turbo=turbo_plan, hip=hip_plan):
        out.say("  " + line)
    if args.dry_run:
        out.say("\nDry run: nothing was downloaded or changed.")
        return 0
    if not args.yes and not out.ask("Continue?"):
        out.say("Cancelled. Nothing was changed.")
        return 0

    cache, results = {}, []
    if hip_plan:
        out.say("\nTrying our fork built for ROCm (its expert cache keeps the hottest experts in VRAM)")
        r = _attempt(args, hw, catalog, out, release, hip_cfg, home, cache, "Shura fork (ROCm build)")
        if r["ok"]:
            results.append(r)
        else:
            out.say("  Our ROCm build did not work here (" + "; ".join(r["errors"])[:300] + ").")
    if turbo_plan:
        out.say("\nTrying the TurboQuant+ build (third-party llama.cpp fork: turbo KV cache, expert cache, MTP)")
        r = _attempt(args, hw, catalog, out, release, tq_cfg, home, cache, "TurboQuant+")
        if r["ok"]:
            results.append(r)
        else:
            out.say("  The TurboQuant+ build did not work here (" + "; ".join(r["errors"])[:200] + ").")
    up_plan = planner.plan(hw, catalog, config=relaxed_config(up_cfg, 0, hw), model=args.model_id, quant=args.quant,
                           profile=profile, menu=False)
    vendor = max(hw["gpus"], key=lambda g: g["vram_total"])["vendor"] if hw["gpus"] else None
    gpu_backends = sorted((b for b in up_plan["backend_candidates"] if b != "cpu"),
                          key=lambda b: b != NATIVE.get(vendor)) if up_plan["ok"] else []   # the vendor's own stack first
    r = {"ok": False, "errors": []}
    for backend in gpu_backends or ["cpu"]:                       # every build that can drive the GPU, on the real model:
        out.say(f"\nTrying standard llama.cpp, {backend} build")  # the tiny test model does not predict a big MoE's speed
        r = _attempt(args, hw, catalog, out, release, up_cfg, home, cache, f"llama.cpp ({backend})", only_backend=backend)
        if r["ok"]:
            results.append(r)
        else:
            out.say(f"  {backend} did not work here (" + "; ".join(r["errors"])[:160] + ").")
    if not results and gpu_backends:                              # no GPU build worked: the CPU is the last resort
        out.say("\nNo GPU build worked: trying the CPU.")
        r = _attempt(args, hw, catalog, out, release, up_cfg, home, cache, "llama.cpp (cpu)", only_backend="cpu")
        if r["ok"]:
            results.append(r)
    if not results and r.get("stopped"):                                    # a download or engine problem, not a failed launch
        out.say(f"\nInstall stopped: {'; '.join(r['errors'])}")
        out.say("Nothing is broken: run the same command again, downloads resume and finished files are kept.")
        return 1
    if not results:
        out.say("\nNo configuration passed the test launch. Downloaded files are kept, so a rerun does not start over.")
        errors = [e for rr in ([r] if not r["ok"] else []) for e in rr.get("errors", [])]
        _, md = report.make_report(hw, plan, measured={"verified": False}, errors=errors, scrub_args=_scrub())
        (home / "shura-hardware-report.md").write_text(md + "\n")
        out.say(f"A report without personal data is in {home / 'shura-hardware-report.md'}. Posting it as a "
                f"'Hardware report' issue helps everyone with this hardware.")
        return 1
    def rank(x):                                                  # measured speed, with the vendor's own stack favoured
        return x["verdict"]["tok_s"] * (NATIVE_BONUS if x["backend"] == NATIVE.get(vendor) else 1.0)
    best = max(results, key=rank)
    if len(results) > 1:
        shown = ", ".join(f"{x['label']} {x['verdict']['tok_s']} tok/s" for x in results)
        out.say(f"\n  Measured here: {shown}. Keeping {best['label']}"
                + (f" ({NATIVE.get(vendor)} is this GPU's own stack: another build must be {int((NATIVE_BONUS - 1) * 100)}% faster "
                   f"to replace it)." if vendor else "."))
    plan, verdict, hw = best["plan"], best["verdict"], best["hw"]
    predicted = plan["speed_by_fill"]["empty"]
    measured = {"verified": True, "tok_s_short_prompt": verdict["tok_s"], "predicted_tok_s_empty": predicted,
                "first_plan_predicted": best["first_predicted"], "first_plan_measured": best["first_measured"],
                "calibration": best["calibration"], "load_s": verdict["load_s"], "backend": best["backend"],
                "engine": plan.get("engine", "llama.cpp"), "llama_cpp": args.tag,
                "engine_comparison": {x["label"]: x["verdict"]["tok_s"] for x in results}, "chosen": best["label"]}
    launch.save_state(home, {"schema": 1, "plan": plan, "model_path": best["path"], "server": best["server"],
                             "backend": best["backend"], "argv": best["argv"], "port": args.port, "measured": measured})
    _, md = report.make_report(hw, plan, measured=measured, scrub_args=_scrub())
    (home / "shura-hardware-report.md").write_text(md + "\n")
    out.say(f"\nReady. Measured {verdict['tok_s']} tok/s (plan predicted about {predicted:.0f} for a short prompt), "
            f"loaded in {verdict['load_s']} s, engine: {plan.get('engine', 'llama.cpp')}.")
    out.say(f"\n  shura start     start the server ({plan['settings']['context']} tokens of context)")
    out.say("  shura status    is it running?      shura stop    stop it and free the memory")
    out.say(f"  API: http://127.0.0.1:{args.port}/v1  (OpenAI-compatible; any client works, model name \"{plan['model']}\")")
    out.say(f"\nPlease share how it went: {home / 'shura-hardware-report.md'} has no personal data; "
            f"`shura report --issue` prints a pre-filled GitHub link.")
    return 0


def _scrub():
    import getpass
    import socket
    return {"home": str(Path.home()), "user": getpass.getuser(), "host": socket.gethostname()}
