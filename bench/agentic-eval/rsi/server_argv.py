"""RSIL0c: the llama-server command line of the executor, built by the GATEWAY's own build_args (gateway/ai_gateway.py), not by hand.

The gateway reads its settings at import from $AI_GATEWAY_CONFIG (guardian.json, model-launch.json). This module points that variable at
a temporary folder: guardian.json is a copy of the live file with the stand's port (8091) and model-launch.json is the candidate layout
(reports/layout1/model-launch.candidate.json, the profile A of the model). The live config is only read, never written; the gateway's
state goes to a temporary folder too. Nothing is started: the command line is only computed.

    python3 -B -m rsi.server_argv --model tiel-coder --out FILE       # writes a JSON list: the argv for the stand
    python3 -B -m rsi.server_argv --model tiel-coder --compare REF     # also lists the differences to a reference command line
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GATEWAY = Path("/home/alisher/Projects/shura-integration/gateway")
CANDIDATE = Path("/home/alisher/Desktop/AI/reports/layout1/model-launch.candidate.json")
GUARDIAN_LIVE = Path("/home/alisher/Projects/shura/config/guardian.json")
STAND_PORT = 8091
REAL_CONFIG = Path("/home/alisher/Projects/shura-integration/config")
STAND_SLOTS = "/dev/shm/RSIL0c/slots"
HERE = Path(__file__).resolve().parent


def _argv_in_this_process(model, vision):
    """Imports the gateway with the temporary config (the environment is set by the caller) and returns build_args' list."""
    sys.path.insert(0, str(GATEWAY))
    import ai_gateway  # noqa: E402  (module-level config is read from AI_GATEWAY_CONFIG)
    ai_gateway.server_slot_dir = lambda: Path(STAND_SLOTS)   # the live gateway's slot folder is never created or used here
    mc = ai_gateway.M["models"][model]
    return ai_gateway.build_args(mc, vision)


def build(model="tiel-coder", vision=False):
    """The argv for `model` (profile A of the candidate layout), as a list. Runs the gateway's code in its own process."""
    out = Path(tempfile.mkdtemp(prefix="rsil0c-argv-", dir="/dev/shm"))
    try:
        cfg = out / "config"
        cfg.mkdir()
        guardian = json.loads(GUARDIAN_LIVE.read_text(encoding="utf-8"))
        guardian["llamaPort"] = STAND_PORT
        (cfg / "guardian.json").write_text(json.dumps(guardian, ensure_ascii=False, indent=1), encoding="utf-8")
        shutil.copy2(CANDIDATE, cfg / "model-launch.json")
        env = {**os.environ, "AI_GATEWAY_CONFIG": str(cfg), "AI_GATEWAY_STATE": str(out / "state")}
        dump = out / "argv.json"
        r = subprocess.run([sys.executable, "-B", "-m", "rsi.server_argv", "--model", model, "--dump", str(dump),
                            *(["--vision"] if vision else [])],
                           cwd=str(HERE.parent), env=env, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"the gateway's build_args failed: {r.stderr[-800:]}")
        return _stand_paths(json.loads(dump.read_text(encoding="utf-8")), cfg)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _stand_paths(argv, tmp_cfg):
    """Two paths the gateway writes relative to its own folders are moved: the MoE profile is read from the real config folder (the
    temporary one holds no data files), and the slot files go to the stand's folder under /dev/shm, never into the live gateway state."""
    out = list(argv)
    if "--moe-cache-profile" in out:
        i = out.index("--moe-cache-profile") + 1
        out[i] = out[i].replace(str(tmp_cfg), str(REAL_CONFIG))
    if "--slot-save-path" in out:
        out[out.index("--slot-save-path") + 1] = STAND_SLOTS
    Path(STAND_SLOTS).mkdir(parents=True, exist_ok=True)
    return out


def differences(argv, reference_text):
    """The tokens of the reference command line that the gateway's argv lacks, and the ones it adds. Flags with values are compared as
    pairs; the first token (the executable) is not compared."""
    ref = shlex.split(reference_text)[1:]
    ours = argv[1:]

    def flags(tokens):
        out, i = {}, 0
        while i < len(tokens):
            t = tokens[i]
            if t.startswith("--") or t.startswith("-"):
                nxt = tokens[i + 1] if i + 1 < len(tokens) and not tokens[i + 1].startswith("-") else None
                out.setdefault(t, []).append(nxt)
                i += 2 if nxt is not None else 1
            else:
                i += 1
        return out

    fr, fo = flags(ref), flags(ours)
    missing = {k: v for k, v in fr.items() if k not in fo}
    added = {k: v for k, v in fo.items() if k not in fr}
    changed = {k: {"reference": fr[k], "gateway": fo[k]} for k in fr if k in fo and fr[k] != fo[k]}
    return {"missing_in_gateway": missing, "only_in_gateway": added, "changed": changed}


def _dump(model, vision, path):
    """Runs inside the process whose environment points at the temporary config."""
    argv = [str(x) for x in _argv_in_this_process(model, vision)]
    Path(path).write_text(json.dumps(argv, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rsi.server_argv")
    ap.add_argument("--model", default="tiel-coder")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--out", help="write the argv as a JSON list here")
    ap.add_argument("--compare", help="a reference command-line file to compare with")
    ap.add_argument("--dump", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.dump:
        _dump(args.model, args.vision, args.dump)
        return 0
    lst = build(args.model, args.vision)
    if args.out:
        Path(args.out).write_text(json.dumps(lst, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.compare:
        print(json.dumps(differences(lst, Path(args.compare).read_text(encoding="utf-8")), ensure_ascii=False, indent=1))
    if not args.out and not args.compare:
        print(shlex.join(lst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
