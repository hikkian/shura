"""Shura on your hardware: look, install, run.

    shura check   [--no-measure] [--profile FILE] [--json]     what this machine can run (no download, no root)
    shura install [--dry-run] [--yes] [--quant Q] [--context N] download, test and set up the best configuration
    shura start | stop | status                                 run the installed server in the background
    shura report  [--issue]                                     a report without personal data for a GitHub issue
    shura selftest [--backend cpu|vulkan|cuda|...]              does a llama.cpp build work here? (downloads ~100 MB)
    shura uninstall                                             delete everything Shura downloaded

(`shura` is the script in the repository root; `python3 installer/universal/cli.py` is the same thing.)
check and report write nothing except the report file. install asks before it downloads, resumes interrupted downloads,
verifies every file, and tests the result by really starting the server; if that fails it retries with more room left
free. Exit code: 0 ok, 1 nothing fits / failed, 2 invalid input.
"""
import argparse
import getpass
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from universal import backends, bandwidth, detect, engine, install, launch, planner, report, schema  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent


def load(args):
    if args.profile:
        hw = json.loads(Path(args.profile).read_text())
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
    try:
        release = engine.fetch_release(None if args.latest else args.tag)
        hw = detect.detect()
        arch, osf = hw["os"]["arch"], hw["os"]["family"]
        driver = next((g.get("driver") for g in hw["gpus"] if g["vendor"] == "nvidia"), None)
        asset = backends.pick_asset(release["assets"], osf, arch, args.backend, driver)
        if not asset:
            print(json.dumps({"backend": args.backend, "ok": False, "tag": release["tag"],
                              "error": f"release {release['tag']} has no {args.backend} build for {osf} {arch}"}))
            return 1
        server = engine.install_build(release, asset, root / "engines")
        result = engine.selftest(server, engine.smoke_model(root / "models"), backend=args.backend)
        result.update(tag=release["tag"], asset=asset["name"])
    except (engine.EngineError, OSError, ValueError) as e:
        result = {"backend": args.backend, "ok": False, "error": str(e)}
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def manage(args):
    home = Path(args.dir) if args.dir else launch.home_dir()
    if args.command == "uninstall":
        import shutil
        if not home.exists():
            print(f"Nothing to remove ({home} does not exist).")
            return 0
        size = sum(f.stat().st_size for f in home.rglob("*") if f.is_file())
        print(f"This deletes {home} ({size / 1024 ** 3:.1f} GiB: engines, models, logs, state).")
        if not args.yes and input("Delete it? [y/N] ").strip().lower() != "y":
            print("Nothing was deleted.")
            return 0
        launch.stop(home)
        shutil.rmtree(home)
        print("Removed.")
        return 0
    state = launch.load_state(home)
    if not state:
        print("Shura is not installed here yet: run `shura install`.", file=sys.stderr)
        return 1
    port = state["port"]
    if args.command == "status":
        up = launch.alive(launch.read_pid(home)) and launch.health(port)
        print(("running" if up else "stopped") + f" · http://127.0.0.1:{port}/v1 · {state['plan']['settings']['context']} tokens "
              f"· {state['plan']['quant']} · backend {state['backend']}")
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
    ap.add_argument("command", choices=["check", "report", "selftest", "install", "start", "stop", "status", "uninstall"])
    ap.add_argument("--dry-run", action="store_true", help="show what install would do and download nothing")
    ap.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation (install)")
    ap.add_argument("--quant", help="force a quant (install)")
    ap.add_argument("--optimize", dest="profile_name", choices=sorted(planner.PROFILES), default="balanced",
                    help="what to optimise for: balanced (default), fast, long (biggest window), quality")
    ap.add_argument("--model-id", help="model from the catalog (install; default: the best one that fits)")
    ap.add_argument("--context", type=int, help="do not use a window larger than this (install)")
    ap.add_argument("--port", type=int, default=launch.DEFAULT_PORT, help="server port (install, start)")
    ap.add_argument("--wait", type=int, default=900, help="seconds to wait for the model to load (install)")
    ap.add_argument("--hf-base", help=argparse.SUPPRESS)
    ap.add_argument("--issue", action="store_true", help="print a pre-filled GitHub issue link (report)")
    ap.add_argument("--backend", default="cpu", help="backend to self-test (selftest)")
    ap.add_argument("--dir", help="where to keep downloaded builds and the test model (selftest)")
    ap.add_argument("--tag", default=engine.PINNED_TAG, help="llama.cpp release tag (selftest)")
    ap.add_argument("--latest", action="store_true", help="use the newest llama.cpp release (selftest)")
    ap.add_argument("--profile", help="use this hardware profile (JSON) instead of detecting the machine")
    ap.add_argument("--catalog", help="use this model catalog (JSON) instead of catalog/models.json")
    ap.add_argument("--no-measure", action="store_true", help="skip the few-second RAM bandwidth measurement")
    ap.add_argument("--ram-gbs", type=float, help="RAM read speed in GB/s when it cannot be measured (no C compiler, as on most "
                                                  "Windows PCs): dual-channel DDR4-3200 is about 40, DDR5-6000 about 80")
    ap.add_argument("--json", action="store_true", help="print the plan as JSON (check)")
    ap.add_argument("-o", "--output", default="shura-hardware-report.md", help="report file (report)")
    args = ap.parse_args(argv)
    if args.command == "selftest":
        return selftest(args)
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
        scrub_args = {"home": str(Path.home()), "user": getpass.getuser(), "host": socket.gethostname()}
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
