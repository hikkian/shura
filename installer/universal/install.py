"""`shura install`: detect -> plan -> engine -> model -> verified launch -> ready to use.

Everything that can fail is checked *before* the expensive part (disk space, a plan that fits) and the result is
verified by really starting the server with the planned settings. If that fails (almost always: out of memory at
load), the plan is made again with more room left free and tried again, so the user ends with a configuration that
was seen to work instead of an error. Nothing is written outside the Shura home folder.
"""
import copy
import json
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
            cfg["window_floors"] = (65536, 32768, 16384)
            cfg["contexts"] = tuple(c for c in planner.DEFAULTS["contexts"] if c <= 131072)
    return cfg


def preview(plan, model, quant, dirs, server_bytes=120 * 1024 ** 2):
    q = next(x for x in model["quants"] if x["id"] == quant)
    return [f"model file : {modelstore.file_name(model, quant)}  ({q['file_bytes'] / GiB:.1f} GiB)",
            f"engine     : llama.cpp {engine.PINNED_TAG}, backend {', '.join(plan['backend_candidates'])}  (~{server_bytes // 1024 ** 2} MB each)",
            f"folder     : {dirs}",
            f"disk needed: about {(q['file_bytes'] + server_bytes * len(plan['backend_candidates'])) / GiB + 2:.0f} GiB"]


def choose_engine(hw, plan, root, tag, out, release=None):
    """Download the candidate builds and keep the fastest one that passes the self-test. Returns (backend, server, results)."""
    release = release or engine.fetch_release(tag)
    builds = backends.builds_for(hw, plan["backend_candidates"], release["assets"])
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
        if res["ok"] and not plan["needs_probe"]:
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
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", method="POST",
                                     data=json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0,
                                                      "seed": 1}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            body = json.load(r)
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


def run(args, hw, catalog, out=None, *, release=None):
    """The whole installation. Returns the process exit code: 0 ready, 1 nothing fits / failed, 2 bad input."""
    out = out or Out()
    home = Path(args.dir) if args.dir else launch.home_dir()
    base_cfg = {"contexts": tuple(c for c in planner.DEFAULTS["contexts"] if c <= args.context)} if args.context else {}
    plan = planner.plan(hw, catalog, config=relaxed_config(base_cfg, 0, hw), model=args.model_id, quant=args.quant)
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
    out.say("\nWhat will be downloaded")
    for line in preview(plan, model, plan["quant"], home):
        out.say("  " + line)
    if args.dry_run:
        out.say("\nDry run: nothing was downloaded or changed.")
        return 0
    if not args.yes and not out.ask("Continue?"):
        out.say("Cancelled. Nothing was changed.")
        return 0
    first_hw = hw
    try:
        out.say("\n1/4  Engine (llama.cpp)")
        backend, server, tested = choose_engine(hw, plan, home, None if args.latest else args.tag, out, release)
        hw = reconcile(hw, server, backend, out)
        if backend == "cpu" and hw["gpus"]:
            out.say("  No GPU build works here: planning for the CPU instead.")
            cpu_hw = {**hw, "gpus": []}
            plan = planner.plan(cpu_hw, catalog, model=plan["model"], quant=args.quant,
                                config=relaxed_config(base_cfg, 0, cpu_hw))
            out.say(report.card(cpu_hw, plan))
            if not plan["ok"]:
                return 1
        if hw is not first_hw and backend != "cpu":
            plan = planner.plan(hw, catalog, model=plan["model"], quant=args.quant, config=relaxed_config(base_cfg, 0, hw))
            out.say(report.card(hw, plan))
            if not plan["ok"]:
                return 1
            model = next(m for m in catalog["models"] if m["id"] == plan["model"])
        out.say("\n2/4  Model")
        files = modelstore.listing(model["source"]["hf_repo"], **({"base": args.hf_base, "allow_local": True} if args.hf_base else {}))
        name = modelstore.file_name(model, plan["quant"])
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
    except (engine.EngineError, modelstore.StoreError, OSError) as e:
        out.say(f"\nInstall stopped: {e}")
        out.say("Nothing is broken: run the same command again, downloads resume and finished files are kept.")
        return 1

    out.say("\n3/4  Test launch with the planned settings")
    measured, errors, final, verdict = {}, [], None, None
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            plan = planner.plan(hw if backend != "cpu" else {**hw, "gpus": []}, catalog, model=plan["model"],
                                quant=plan["quant"], config=relaxed_config(base_cfg, attempt, hw))
            if not plan["ok"]:
                break
            out.say(f"  Trying again with more room left free (attempt {attempt + 1}): "
                    f"{plan['settings']['context']} tokens of context")
        argv = launch.server_args(plan, server, path, port=args.port, alias=plan["model"])
        verdict = verify(argv, args.port, home=home, wait_s=args.wait)
        if verdict["ok"]:
            final = (plan, argv)
            break
        errors.append(f"attempt {attempt + 1}: {verdict.get('error', '?')[:300]}")
        out.say("  did not work" + (" (out of memory)" if verdict.get("oom") else "") + ": "
                + str(verdict.get("error", ""))[:200].replace("\n", " "))
        if not verdict.get("oom") and attempt:
            break
    if not final:
        out.say("\nNo configuration passed the test launch. The model and engine are downloaded and kept.")
        _, md = report.make_report(hw, plan, measured={"verified": False}, errors=errors, scrub_args=_scrub())
        (home / "shura-hardware-report.md").write_text(md + "\n")
        out.say(f"A report without personal data is in {home / 'shura-hardware-report.md'}. Posting it as a "
                f"'Hardware report' issue helps everyone with this hardware.")
        return 1
    plan, argv = final
    predicted = plan["speed_by_fill"]["empty"]
    measured = {"verified": True, "tok_s_short_prompt": verdict["tok_s"], "predicted_tok_s_empty": predicted,
                "load_s": verdict["load_s"], "backend": backend, "llama_cpp": args.tag}
    launch.save_state(home, {"schema": 1, "plan": plan, "model_path": str(path), "server": str(server), "backend": backend,
                             "argv": argv, "port": args.port, "measured": measured})
    _, md = report.make_report(hw, plan, measured=measured, scrub_args=_scrub())
    (home / "shura-hardware-report.md").write_text(md + "\n")
    out.say(f"\n4/4  Ready. Measured {verdict['tok_s']} tok/s (plan predicted about {predicted:.0f} for a short prompt), "
            f"loaded in {verdict['load_s']} s.")
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
