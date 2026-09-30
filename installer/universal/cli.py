"""`shura check` and `shura report`: describe this machine and show what Shura would do, without installing anything.

    python3 installer/universal/cli.py check  [--no-measure] [--profile FILE] [--catalog FILE] [--json]
    python3 installer/universal/cli.py report [--no-measure] [--profile FILE] [--catalog FILE] [-o FILE]
    python3 installer/universal/cli.py selftest [--backend cpu|vulkan|cuda|...] [--dir DIR] [--tag TAG | --latest]

check and report need no root, no downloads and no network, and write nothing except the report file you ask for.
selftest downloads one llama.cpp build and a 19 MB test model. Exit code: 0 a plan was made / the backend works,
1 nothing fits / the backend failed, 2 invalid input.
"""
import argparse
import getpass
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from universal import backends, bandwidth, detect, engine, planner, report, schema  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent


def load(args):
    if args.profile:
        hw = json.loads(Path(args.profile).read_text())
    else:
        hw = detect.detect()
        if not args.no_measure:
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shura", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "report", "selftest"])
    ap.add_argument("--backend", default="cpu", help="backend to self-test (selftest)")
    ap.add_argument("--dir", help="where to keep downloaded builds and the test model (selftest)")
    ap.add_argument("--tag", default=engine.PINNED_TAG, help="llama.cpp release tag (selftest)")
    ap.add_argument("--latest", action="store_true", help="use the newest llama.cpp release (selftest)")
    ap.add_argument("--profile", help="use this hardware profile (JSON) instead of detecting the machine")
    ap.add_argument("--catalog", help="use this model catalog (JSON) instead of catalog/models.json")
    ap.add_argument("--no-measure", action="store_true", help="skip the few-second RAM bandwidth measurement")
    ap.add_argument("--json", action="store_true", help="print the plan as JSON (check)")
    ap.add_argument("-o", "--output", default="shura-hardware-report.md", help="report file (report)")
    args = ap.parse_args(argv)
    if args.command == "selftest":
        return selftest(args)
    try:
        hw, catalog = load(args)
        problems = schema.validate_hardware(hw) + schema.validate_catalog(catalog)
        if problems:
            print("invalid input:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 2
        plan = planner.plan(hw, catalog)
    except (OSError, ValueError, KeyError) as e:
        print(f"shura: {e}", file=sys.stderr)
        return 2
    if args.command == "check":
        print(json.dumps(plan, indent=2) if args.json else report.card(hw, plan))
    else:
        scrub_args = {"home": str(Path.home()), "user": getpass.getuser(), "host": socket.gethostname()}
        _, md = report.make_report(hw, plan, scrub_args=scrub_args)
        Path(args.output).write_text(md + "\n")
        print(f"Report written to {args.output}\nOpen a 'Hardware report' issue on GitHub and paste it. "
              f"Nothing was sent anywhere.")
    return 0 if plan["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
