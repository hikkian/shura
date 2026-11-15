"""Shura on your hardware: look, install, run.

    shura check   [--no-measure] [--profile FILE] [--json]     what this machine can run (no download, no root)
    shura doctor  [--dir DIR]                                  diagnose hardware, model, gateway, service and versions
    shura install [--dry-run] [--yes] [--quant Q] [--context N] download, test and set up the best configuration
    shura start | stop | status                                 run the installed server in the background
    shura report  [--issue]                                     a report without personal data for a GitHub issue
    shura selftest [--backend cpu|vulkan|cuda|...]              does a llama.cpp build work here? (downloads ~100 MB)
    shura uninstall [--all] [--yes] [--force] [--dir D]         delete downloaded files or all Shura-owned user data

(`shura` is the script in the repository root; `python3 installer/universal/cli.py` is the same thing.)
check and report write nothing except the report file. install asks before it downloads, resumes interrupted downloads,
verifies every file, and tests the result by really starting the server; if that fails it retries with more room left
free. Exit code: 0 ok, 1 nothing fits / failed, 2 invalid input.
"""
import argparse
import getpass
import http.client
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from universal import backends, bandwidth, detect, engine, install, journal, launch, planner, report, schema  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent


def disk_sample_path():
    """A model file (64 MiB or more) in the Shura folder to read for the disk speed, or None."""
    try:
        for f in sorted((launch.home_dir() / "models").glob("*.gguf")):
            if f.is_file() and f.stat().st_size >= 64 * 1024 ** 2:
                return str(f)
    except OSError:
        pass
    return None


def load(args):
    if args.profile:
        hw = json.loads(Path(args.profile).read_text())
    else:
        if getattr(args, "disk_benchmark", False):
            sample = disk_sample_path()
            hw = detect.detect(disk_sample=sample, disk_benchmark=True)
            if not sample:
                hw["detection_notes"].append("скорость диска: не измерено (в папке Шуры нет файла-образца)")
        else:
            hw = detect.detect()
        if args.ram_gbs:                       # the user knows better than a missing compiler (typical: dual-channel DDR4-3200 ~ 40)
            hw["memory"]["bandwidth_gbs"] = float(args.ram_gbs)
            hw["detection_notes"].append(f"RAM bandwidth set by the user: {args.ram_gbs} GB/s")
        elif not args.no_measure:
            gbs = bandwidth.measure(hw["cpu"]["physical_cores"], hw["memory"]["available"])
            if gbs:
                hw["memory"]["bandwidth_gbs"] = round(gbs, 1)
            else:
                hw["detection_notes"].append("RAM bandwidth could not be measured (no C compiler or too little free RAM)")
    catalog = json.loads(Path(args.catalog or REPO / "catalog" / "models.json").read_text())
    return hw, catalog


def selftest(args):
    """Download one prebuilt backend, run it on a 19 MB test model, print the result as JSON. Used by CI and by people who
    want to know whether a backend works on their machine. Downloads about 100 MB into --dir (default: a temp folder)."""
    import tempfile
    root = Path(args.dir) if args.dir else Path(tempfile.mkdtemp(prefix="shura-selftest-"))
    backend = args.backend or "cpu"
    try:
        release = engine.fetch_release(None if args.latest else args.tag)
        hw = detect.detect()
        arch, osf = hw["os"]["arch"], hw["os"]["family"]
        driver = next((g.get("driver") for g in hw["gpus"] if g["vendor"] == "nvidia"), None)
        asset = backends.pick_asset(release["assets"], osf, arch, backend, driver)
        if not asset:
            print(json.dumps({"backend": backend, "ok": False, "tag": release["tag"],
                              "error": f"release {release['tag']} has no {backend} build for {osf} {arch}"}))
            return 1
        server = engine.install_build(release, asset, root / "engines")
        result = engine.selftest(server, engine.smoke_model(root / "models"), backend=backend)
        result.update(tag=release["tag"], asset=asset["name"])
    except (engine.EngineError, OSError, ValueError) as e:
        result = {"backend": backend, "ok": False, "error": str(e)}
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def _doctor_model_path(home, state):
    path = (state or {}).get("model_path")
    if path:
        return Path(path)
    models = Path(home) / "models"
    if models.is_dir():
        found = sorted(models.glob("*.gguf"))
        if found:
            return found[0]
    config_dir = Path(os.environ.get("AI_GATEWAY_CONFIG") or REPO / "config")
    try:
        data = json.loads((config_dir / "model-launch.json").read_text())
    except (OSError, ValueError):
        return None
    candidates = []

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, str) and ".gguf" in child.lower() and ("model" in key.lower() or "path" in key.lower()):
                    candidates.append(child)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(data)
    return Path(candidates[0]).expanduser() if candidates else None


def doctor(args):
    """Read-only preflight diagnostics. The gateway probe is only GET /guardian/status."""
    home = Path(args.dir).expanduser() if args.dir else launch.home_dir()
    state = launch.load_state(home)
    warnings = 0

    def show(level, label, detail):
        nonlocal warnings
        warnings += level in ("WARN", "FAIL")
        print(f"[{level}] {label}: {detail}")

    try:
        hw = detect.detect()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
        hw = None
        show("FAIL", "Hardware", f"could not detect this machine ({type(e).__name__})")
    if hw:
        gpus = hw.get("gpus", [])
        if gpus:
            names = ", ".join(f"{g['name']} ({g.get('driver') or 'driver version unknown'}; "
                               f"builds: {', '.join(g.get('backends') or ['unknown'])})" for g in gpus)
            backend = (state or {}).get("backend")
            show("OK", "GPU/backend", names + (f"; installed backend {backend}" if backend else ""))
            free = max(max(0, g.get("vram_total", 0) - g.get("vram_used", 0)) for g in gpus)
            show("WARN" if free < 1024 ** 3 else "OK", "VRAM free", f"{free / 1024 ** 3:.1f} GiB estimated from detection")
        else:
            show("INFO", "GPU/backend", "no GPU detected; CPU inference is available")
        memory = hw.get("memory", {})
        avail = memory.get("available", 0)
        total = memory.get("total", 0)
        show("WARN" if avail < 2 * 1024 ** 3 else "OK", "RAM", f"{avail / 1024 ** 3:.1f} GiB available of {total / 1024 ** 3:.1f} GiB")

    probe = home if home.exists() else home.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
        show("WARN" if free < 20 * 1024 ** 3 else "OK", "Disk", f"{free / 1024 ** 3:.1f} GiB free near the Shura home")
    except OSError as e:
        show("FAIL", "Disk", f"could not read free space ({e.strerror or type(e).__name__})")

    model_path = _doctor_model_path(home, state)
    if model_path and model_path.is_file():
        show("OK", "Model", f"present ({model_path.stat().st_size / 1024 ** 3:.1f} GiB)")
    elif model_path:
        show("FAIL", "Model", "configured model file is missing; rerun shura install or restore the model")
    else:
        show("WARN", "Model", "no installed model was found")

    service_file = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "systemd/user/ai-gateway.service"
    try:
        result = subprocess.run(["systemctl", "--user", "is-active", "ai-gateway.service"],
                                capture_output=True, text=True, timeout=5)
        active = result.returncode == 0 and result.stdout.strip() == "active"
        show("OK" if active else "WARN" if service_file.exists() else "INFO", "systemd service",
             "ai-gateway.service is active" if active else
             "ai-gateway.service is not active; try `shura restart`" if service_file.exists() else
             "ai-gateway.service is not installed (the universal server may be used instead)")
    except (OSError, subprocess.SubprocessError):
        show("WARN" if service_file.exists() else "INFO", "systemd service",
             "could not query ai-gateway.service" if service_file.exists() else "systemd user service was not found")

    gateway_base = os.environ.get("SHURA_GATEWAY", "http://127.0.0.1:8080")
    try:
        parsed = urllib.parse.urlsplit(gateway_base)
        gateway_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        parsed, gateway_port = None, None
    if not parsed or parsed.scheme not in ("http", "https") or not parsed.hostname:
        show("FAIL", "Gateway", "SHURA_GATEWAY must be an http(s) URL")
    else:
        conn_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        conn = None
        try:
            conn = conn_type(parsed.hostname, gateway_port, timeout=3)
            conn.request("GET", "/guardian/status", headers={"Connection": "close"})
            response = conn.getresponse()
            if not 200 <= response.status < 300:
                raise OSError(f"HTTP {response.status}")
            body = json.loads(response.read(64 * 1024).decode("utf-8"))
            status = str(body.get("status", "responding")) if isinstance(body, dict) else "responding"
            show("OK", "Gateway", f"GET /guardian/status answered ({status})")
        except (OSError, http.client.HTTPException, ValueError, TypeError):
            if state:
                show("WARN", "Gateway", "GET /guardian/status did not answer; the installed local server can still work")
            else:
                show("INFO", "Gateway", "GET /guardian/status is not available")
        finally:
            if conn:
                conn.close()

    server = (state or {}).get("server")
    if not server:
        candidate = Path(os.environ.get("SHURA_HOME") or Path.home() / ".local/share/shura") / "llama.cpp-perf/build/bin/llama-server"
        server = str(candidate) if candidate.is_file() else None
    if not server:
        config_dir = Path(os.environ.get("AI_GATEWAY_CONFIG") or REPO / "config")
        try:
            setup_config = json.loads((config_dir / "model-launch.json").read_text())
            configured_server = setup_config.get("exePath") or setup_config.get("server")
            server = str(Path(configured_server).expanduser()) if configured_server else None
        except (OSError, ValueError):
            pass
    if server and Path(server).is_file():
        try:
            result = subprocess.run([str(server), "--version"], capture_output=True, text=True, timeout=8,
                                    env=launch.server_env(server))
            version = next((line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()), "unknown")
            version = version.replace(str(Path.home()), "~").replace(str(home), "<shura-home>")[:120]
            show("OK" if result.returncode == 0 else "WARN", "llama.cpp version", version)
        except (OSError, subprocess.SubprocessError):
            show("WARN", "llama.cpp version", "server binary could not report its version")
    else:
        show("WARN", "llama.cpp version", "server binary was not found")
    show("OK", "Python", platform.python_version())
    print("Doctor: " + ("issues found; see WARN/FAIL above" if warnings else "checks look good"))
    return 1 if warnings else 0


OWN_ENTRIES = (journal.NAME, journal.TMP_NAME, "engines", "models", "logs", "llama.cpp-rocm", "state.json", "state.json.tmp", "server.pid",
               "shura-hardware-report.md", launch.HOME_MARKER)     # everything `shura install` creates directly in the Shura home


def _tree_size(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def uninstall_plan(home, state, force=False):
    """What `shura uninstall` may delete in `home`: (paths, kept names, problem). Never the folder as a whole unless it holds
    nothing but our own entries: the Linux+NVIDIA setup.sh keeps the gateway repository (config!), the llama.cpp build and
    the 20 GB model in this same folder, and a user may point --dir anywhere."""
    home = Path(home)
    user = Path.home().resolve()
    if home.parent == home or home == user or home in user.parents:
        return [], [], f"{home} is your home folder or a parent of it: refusing to delete there"
    if (home / ".git").exists():
        return [], [], f"{home} is a git repository: refusing to delete there"
    entries = sorted(e.name for e in home.iterdir())
    foreign = [e for e in entries if e not in OWN_ENTRIES]
    marked = (home / launch.HOME_MARKER).exists() or (home / "state.json").exists()
    if foreign and not marked and not force:
        return [], [], (f"{home} does not look like a Shura folder (no state.json or {launch.HOME_MARKER}, and it holds "
                        f"{', '.join(foreign[:4])}): nothing was deleted. If you are sure, add --force "
                        f"(only the entries `shura install` creates are ever deleted)")
    paths = [home / e for e in entries if e in OWN_ENTRIES and e != "models"]
    if "models" in entries:
        models = home / "models"
        if not foreign:
            paths.append(models)
        else:                                  # a shared folder: only the files the universal installer put there
            mine = {engine.SMOKE_MODEL["name"], Path((state or {}).get("model_path") or "?").name}
            paths += [f for f in models.iterdir() if f.name in mine or f.name in {m + ".part" for m in mine}]
    return paths, foreign, None


def _is_model(home, path):
    """Downloaded model files (not the tiny self-test model): `shura uninstall` keeps them unless told otherwise."""
    models = Path(home) / "models"
    return models in Path(path).parents and Path(path).name.split(".gguf")[0] + ".gguf" != engine.SMOKE_MODEL["name"]


def uninstall_journal(args, home, jr):
    """Remove exactly what the installation journal lists; models only on an explicit yes."""
    home = Path(home)
    models = sorted(f for f in (home / "models").glob("*") if _is_model(home, f)) if (home / "models").is_dir() else []
    msize = sum(_tree_size(f) for f in models)
    print(f"The installation journal ({jr.path}) lists {len(jr.paths())} item(s) made by `shura install`"
          + ("" if jr.committed else " (the installation never finished)") + ".")
    print("This removes them and nothing else; files that are not in the journal are not touched.")
    delete_models = bool(getattr(args, "models", False))
    if models:
        print(f"Models in {home / 'models'} ({msize / 1024 ** 3:.1f} GiB): " + ("will be deleted (--models)." if delete_models
              else "kept unless you say otherwise."))
    if args.dry_run:
        print("Dry run: nothing was deleted.")
        return 0
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("No terminal is available; rerun with --yes to confirm (models are kept unless --models is added).",
                  file=sys.stderr)
            return 1
        if input("Remove what the installer made? [y/N] ").strip().lower() != "y":
            print("Nothing was deleted.")
            return 0
        if models and not delete_models:
            delete_models = input(f"Also delete the downloaded models ({msize / 1024 ** 3:.1f} GiB)? [y/N] ").strip().lower() == "y"
    launch.stop(home)
    res = jr.rollback(keep=None if delete_models else (lambda p: _is_model(home, p)))
    print("Removed: " + res.summary() + ".")
    if res.kept:
        print(f"Kept the models: {', '.join(Path(k).name for k in res.kept)}. Delete them yourself, or run "
              f"`shura uninstall --models` later.")
    if res.problems:
        print("Some items could not be removed (the journal keeps them; fix the cause and run the command again).", file=sys.stderr)
        return 1
    return 0


def uninstall(args, home):
    import shutil
    home = Path(home).expanduser()
    if getattr(args, "all", False):
        return uninstall_all(args, home)
    jr = journal.Journal.load(home) if home.is_dir() else None
    if jr is not None:
        return uninstall_journal(args, home, jr)
    if home.is_dir() and ((home / "state.json").exists() or (home / launch.HOME_MARKER).exists()):
        print(f"No installation journal in {home}: this folder was made before `shura install` kept one; "
              f"falling back to the older, stricter rules (only the entries the installer creates).")
    if not home.exists():
        print(f"Nothing to remove ({home} does not exist).")
        return 0
    home = home.resolve()
    paths, kept, problem = uninstall_plan(home, launch.load_state(home), force=args.force)
    if problem:
        print(problem, file=sys.stderr)
        return 1
    size = sum(_tree_size(p) for p in paths)
    print(f"This deletes from {home} ({size / 1024 ** 3:.1f} GiB): " + (", ".join(p.name for p in paths) or "nothing") + ".")
    if kept:
        print(f"It keeps everything else in that folder: {', '.join(kept)}.")
    if args.dry_run:
        print("Dry run: nothing was deleted.")
        return 0
    if not paths:
        return 0
    if not args.yes and input("Delete it? [y/N] ").strip().lower() != "y":
        print("Nothing was deleted.")
        return 0
    launch.stop(home)
    for p in paths:
        shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink(missing_ok=True)
    models = home / "models"
    if models.is_dir() and not any(models.iterdir()):
        models.rmdir()
    if not kept and not any(home.iterdir()):
        home.rmdir()
    print("Removed.")
    return 0


def _owned_systemd_unit(path):
    try:
        return (REPO / "gateway/ai_gateway.py").as_posix() in Path(path).read_text(errors="replace")
    except OSError:
        return False


def uninstall_all_plan(home):
    """Owned user-level paths for `uninstall --all`; never includes a checkout or the home directory itself."""
    user = Path.home().resolve()
    config = Path(os.environ.get("XDG_CONFIG_HOME") or user / ".config").expanduser()
    share = Path(os.environ.get("XDG_DATA_HOME") or user / ".local/share").expanduser()
    cache = Path(os.environ.get("XDG_CACHE_HOME") or user / ".cache").expanduser()
    data_homes = []
    for candidate in (Path(home).expanduser(), Path(os.environ.get("SHURA_HOME") or launch.home_dir()).expanduser(),
                      user / ".local/share/shura"):
        resolved = candidate.resolve()
        if (resolved != user and user in resolved.parents and resolved not in data_homes
                and not (resolved / ".git").exists()):
            data_homes.append(resolved)
    items, seen = [], set()

    def add(path, label, *, allow_symlink=False):
        path = Path(path)
        if not path.exists() and not path.is_symlink():
            return
        if path.is_symlink() and not allow_symlink:
            return
        resolved_parent = path.parent.resolve()
        if resolved_parent != user and user not in resolved_parent.parents:
            return
        key = (str(path), label)
        if key not in seen:
            items.append({"path": path, "label": label})
            seen.add(key)

    for data_home in data_homes:
        for name, label in (("engines", "downloaded inference engines"), ("models", "model files"),
                            ("logs", "installer logs"), ("llama.cpp-rocm", "local ROCm build"),
                            ("llama.cpp-perf", "local llama.cpp build"), ("state.json", "installer state"), (journal.NAME, "installation journal"),
                            ("state.json.tmp", "temporary installer state"), ("server.pid", "server pid file"),
                            ("shura-hardware-report.md", "local hardware report"), (launch.HOME_MARKER, "installer marker"),
                            ("sessions", "session cache"), ("cache", "Shura cache")):
            add(data_home / name, label)
    custom_llama = os.environ.get("SHURA_LLAMA_DIR")
    if custom_llama and Path(custom_llama).expanduser().resolve() != REPO.resolve():
        add(Path(custom_llama).expanduser(), "custom llama.cpp build")
    for path, label in ((config / "shuracode", "ShuraCode configuration"),
                        (config / "shura", "Shura configuration"),
                        (share / "shuracode", "ShuraCode data and session memory"),
                        (cache / "shura", "Shura build/download cache"),
                        (cache / "shuracode", "ShuraCode cache"),
                        (config / "systemd/user/ai-gateway.service", "ai-gateway user service")):
        if label == "ai-gateway user service" and not _owned_systemd_unit(path):
            continue
        add(path, label)

    command = user / ".local/bin/shura"
    if command.is_symlink() and command.resolve() == (REPO / "scripts/shura").resolve():
        add(command, "shura command symlink", allow_symlink=True)
    desktop = share / "applications/io.github.hikkian.Shura.desktop"
    try:
        if desktop.is_file() and str(command) in desktop.read_text(errors="replace"):
            add(desktop, "Shura app-menu entry")
    except OSError:
        pass
    icon_root = share / "icons/hicolor"
    for size in (16, 24, 32, 48, 64, 128, 256, 512):
        add(icon_root / f"{size}x{size}/apps/shura.png", f"Shura {size}x{size} icon")
    autostart = config / "autostart/shura-monitor.desktop"
    try:
        if autostart.is_file() and "X-Shura-Managed=true" in autostart.read_text(errors="replace"):
            add(autostart, "Shura Monitor autostart")
    except OSError:
        pass

    kwin = False
    try:
        from . import monitor_window
        kwin = bool(monitor_window._owned_rule(monitor_window._read_config(monitor_window._config_path())))
    except (OSError, ValueError):
        pass
    return items, {"service": config / "systemd/user/ai-gateway.service"
                   if _owned_systemd_unit(config / "systemd/user/ai-gateway.service") else None,
                   "kwin_rule": kwin}


def uninstall_all(args, home):
    """Show and confirm all Shura-owned user data cleanup; operations run only after confirmation."""
    import shutil

    items, actions = uninstall_all_plan(home)
    print("Full Shura removal will delete:")
    for item in items:
        path = item["path"]
        try:
            size = _tree_size(path)
        except OSError:
            size = 0
        print(f"  - {item['label']}: {path} ({size / 1024 ** 3:.1f} GiB)")
    if actions["service"]:
        print(f"  - stop, disable and remove systemd user service: {actions['service']}")
    if actions["kwin_rule"]:
        print("  - remove the Shura Monitor KWin rule (other KWin rules are preserved)")
    if not items and not actions["service"] and not actions["kwin_rule"]:
        print("Nothing owned by this Shura installation was found.")
        return 0
    print("  Kept: this repository checkout and all unrelated files.")
    if args.dry_run:
        print("Dry run: nothing was removed.")
        return 0
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("No terminal is available; rerun `shura uninstall --all --yes` to confirm.", file=sys.stderr)
            return 1
        if input("Remove all listed Shura files and settings? [y/N] ").strip().lower() != "y":
            print("Nothing was removed.")
            return 0

    if actions["service"]:
        try:
            stopped = subprocess.run(["systemctl", "--user", "disable", "--now", "ai-gateway.service"],
                                     capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"Could not stop the gateway service; no files were removed ({e}).", file=sys.stderr)
            return 1
        if stopped.returncode:
            print("Could not stop the gateway service; no files were removed: " +
                  (stopped.stderr.strip() or stopped.stdout.strip() or "systemctl failed"), file=sys.stderr)
            return 1

    if actions["kwin_rule"]:
        try:
            from . import monitor_window
            if monitor_window.remove_rule() != 0:
                print("Could not remove the Shura Monitor KWin rule; stopping before deleting files.", file=sys.stderr)
                return 1
            if monitor_window._owned_rule(monitor_window._read_config(monitor_window._config_path())):
                print("The KWin rule is still present; stopping before deleting files.", file=sys.stderr)
                return 1
        except (OSError, subprocess.SubprocessError) as e:
            print(f"Could not remove the KWin rule; stopping before deleting files ({e}).", file=sys.stderr)
            return 1

    for item in items:
        path = item["path"]
        try:
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path)
        except OSError as e:
            print(f"Could not remove {path}: {e}", file=sys.stderr)
            return 1
    if actions["service"]:
        try:
            subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
    for directory in (Path.home() / ".local/bin", Path.home() / ".local/share/applications"):
        try:
            directory.rmdir()
        except OSError:
            pass
    print("Full Shura removal completed.")
    return 0


def manage(args):
    home = Path(args.dir) if args.dir else launch.home_dir()
    if args.command == "uninstall":
        return uninstall(args, home)
    state = launch.load_state(home)
    if not state:
        print("Shura is not installed here yet: run `shura install`.", file=sys.stderr)
        return 1
    port = state["port"]
    if args.command == "status":
        up = launch.alive(launch.read_pid(home)) and launch.health(port)
        backend_info = state.get("backend_info", {})
        if backend_info.get("experimental"):
            backend_note = (f" · experimental {backend_info['requested']} failed; fallback {state['backend']}"
                            if backend_info.get("fallback") else f" · experimental backend {backend_info['requested']}")
        else:
            backend_note = ""
        print(("running" if up else "stopped") + f" · http://127.0.0.1:{port}/v1 · {state['plan']['settings']['context']} tokens "
              f"· {state['plan']['quant']} · backend {state['backend']}{backend_note}")
        return 0 if up else 1
    if args.command == "stop":
        print("stopped" if launch.stop(home) else "it was not running")
        return 0
    try:
        print("Loading the model (can take a minute) ...", flush=True)
        pid = launch.start(home, state["argv"], port, wait_s=args.wait)
    except launch.LaunchError as e:
        print(f"Could not start: {e}", file=sys.stderr)
        return 1
    print(f"Running (pid {pid}). API: http://127.0.0.1:{port}/v1  ·  `shura stop` frees the memory.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shura", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "report", "selftest", "install", "doctor", "start", "stop", "status", "uninstall"])
    ap.add_argument("--dry-run", action="store_true", help="show what install would do and download nothing")
    ap.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation (install)")
    ap.add_argument("--quant", help="force a quant (install)")
    ap.add_argument("--optimize", dest="profile_name", choices=sorted(planner.PROFILES), default="balanced",
                    help="what to optimise for: balanced (default), fast, long (biggest window), quality")
    ap.add_argument("--no-turbo", action="store_true", help="never use the third-party TurboQuant+ llama.cpp build (install)")
    ap.add_argument("--no-build", action="store_true", help="never build anything from source (our fork for ROCm) (install)")
    ap.add_argument("--model-id", help="model from the catalog (install; default: the best one that fits)")
    ap.add_argument("--context", type=int, help="do not use a window larger than this (install)")
    ap.add_argument("--port", type=int, default=launch.DEFAULT_PORT, help="server port (install, start)")
    ap.add_argument("--wait", type=int, default=900, help="seconds to wait for the model to load (install)")
    ap.add_argument("--hf-base", help=argparse.SUPPRESS)
    ap.add_argument("--issue", action="store_true", help="print a pre-filled GitHub issue link (report)")
    ap.add_argument("--backend", choices=("vulkan", "cpu", "cuda", "rocm", "hip", "sycl", "metal"),
                    help="try this backend first during install; failed experimental builds fall back to Vulkan, then CPU. "
                         "For selftest, run only this backend (default: cpu).")
    ap.add_argument("--dir", help="where to keep downloaded builds and the test model (selftest); the Shura folder to remove (uninstall)")
    ap.add_argument("--force", action="store_true", help="uninstall: also clean a folder that has no state.json (never your home folder)")
    ap.add_argument("--models", action="store_true", help="uninstall: also delete the downloaded model files (kept by default)")
    ap.add_argument("--all", action="store_true", help="uninstall: also remove Shura service, user settings, cache and KWin rule (asks first)")
    ap.add_argument("--tag", default=engine.PINNED_TAG, help="llama.cpp release tag (selftest)")
    ap.add_argument("--latest", action="store_true", help="use the newest llama.cpp release (selftest)")
    ap.add_argument("--profile", help="use this hardware profile (JSON) instead of detecting the machine")
    ap.add_argument("--catalog", help="use this model catalog (JSON) instead of catalog/models.json")
    ap.add_argument("--disk-benchmark", action="store_true", help="measure the disk read speed on a model file in the Shura folder (check; off by default)")
    ap.add_argument("--no-measure", action="store_true", help="skip the few-second RAM bandwidth measurement")
    ap.add_argument("--ram-gbs", type=float, help="RAM read speed in GB/s when it cannot be measured (no C compiler, as on most "
                                                  "Windows PCs): dual-channel DDR4-3200 is about 40, DDR5-6000 about 80")
    ap.add_argument("--json", action="store_true", help="print the plan as JSON (check)")
    ap.add_argument("-o", "--output", default="shura-hardware-report.md", help="report file (report)")
    args = ap.parse_args(argv)
    if args.command == "selftest":
        return selftest(args)
    if args.command == "doctor":
        return doctor(args)
    if args.backend and args.command != "install":
        print("shura: --backend is available for install and selftest, not this command", file=sys.stderr)
        return 2
    if args.command in ("start", "stop", "status", "uninstall"):
        return manage(args)
    try:
        hw, catalog = load(args)
        problems = schema.validate_hardware(hw) + schema.validate_catalog(catalog)
        if problems:
            print("invalid input:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 2
        plan = planner.plan(hw, catalog, profile=args.profile_name)
    except (OSError, ValueError, KeyError) as e:
        print(f"shura: {e}", file=sys.stderr)
        return 2
    if args.command == "install":
        if args.quant and not args.model_id:
            args.model_id = catalog["models"][0]["id"]
        return install.run(args, hw, catalog)
    if args.command == "check":
        print(json.dumps(plan, indent=2) if args.json else report.card(hw, plan))
    else:
        scrub_args = {"home": str(Path.home()), "user": getpass.getuser(), "host": socket.gethostname(),
                      "shura_home": str(Path(args.dir) if args.dir else launch.home_dir())}
        state = launch.load_state(Path(args.dir) if args.dir else launch.home_dir())
        _, md = report.make_report(hw, plan, scrub_args=scrub_args, measured=(state or {}).get("measured"))
        Path(args.output).write_text(md + "\n")
        print(f"Report written to {args.output}\nOpen a 'Hardware report' issue on GitHub and paste it. "
              f"Nothing was sent anywhere.")
        if args.issue:
            print("\nOr open this link (nothing is sent until you press the button on GitHub):\n" + report.issue_url(md))
    return 0 if plan["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
