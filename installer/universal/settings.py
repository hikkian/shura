"""INSTALL1c step 4: from the chosen mode (autofit_api.choose) to the settings the installed Shura runs on.

Three things, all written ONLY inside the Shura home, each through the installation journal (journal.py):
  (a) config/model-launch.json  gateway settings for engine 1.0: one model profile (engine "v1", exePath = the wrapper from the
      release, vramChunkMiB, the VRAM governor on). No short / long / lite layouts: the 1.0 engine decides by itself.
  (b) config/guardian.json + config/shuracode.json  the guardian (vramPolicy governor) and the ShuraCode client settings.
  (c) a systemd --user unit ai-gateway.service in $XDG_CONFIG_HOME/systemd/user (default ~/.config/systemd/user; the only place
      outside the Shura home the installer writes), enabled and started
      through the injected `runner`. The journal entry for the service is written BEFORE the first systemctl call.

A failure at any step rolls back everything (files restored from their journal copies, new ones removed, the service disabled
with `disable --now` from journal.ACTIONS). Nothing here touches /sys/fs/cgroup: memory.low is a separate step with its own
consent. Without consent (`consent=False`) nothing is written at all.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from . import journal as journal_mod

SERVICE = "ai-gateway.service"
DEFAULT_CHUNK_MIB = 52             # vramChunkMiB of the DEPLOY3 sample (reports/deploy3/model-launch.v1.json)
GATEWAY_PORT = 8080
LLAMA_PORT = 8090
LAYOUT_KEYS = ("longContextProfile", "liteProfile", "longLiteProfile", "shortProfile", "longProfile")


class SettingsError(Exception):
    pass


def model_profile(choice, model_path, exe_path, *, chunk_mib=DEFAULT_CHUNK_MIB, mmproj_path=None):
    """The single model profile for engine 1.0. No layout switching keys."""
    prof = {
        "engine": "v1",
        "exePath": str(exe_path),
        "modelPath": str(model_path),
        "ctxSize": int(choice.context),
        "parallelSlots": 1,
        "nGpuLayers": 99,
        "flashAttn": "on",
        "loadMode": "dio",
        "reasoning": "auto",
        "vramChunkMiB": int(chunk_mib),
    }
    if mmproj_path:
        prof["mmprojPath"] = str(mmproj_path)
    return prof


def launch_settings(choice, model_path, exe_path, *, chunk_mib=DEFAULT_CHUNK_MIB, mmproj_path=None):
    name = choice.model_id
    return {
        "exePath": str(exe_path),
        "defaultModel": name,
        "vramChunkMiB": int(chunk_mib),
        "models": {name: model_profile(choice, model_path, exe_path, chunk_mib=chunk_mib, mmproj_path=mmproj_path)},
    }


def guardian_settings(choice):
    """Guardian settings of the GWV1 draft: the governor watches VRAM. Values are the tested defaults (guardian example)."""
    return {
        "gatewayHost": "127.0.0.1", "gatewayPort": GATEWAY_PORT,
        "llamaHost": "127.0.0.1", "llamaPort": LLAMA_PORT,
        "vramGuard": True, "vramPolicy": "governor",
        "vramIdleSeconds": 90, "vramEmergencyMiB": 60, "vramEmergencySeconds": 10,
        "vramFreeCriticalMiB": 120, "vramPressureSeconds": 5, "vramReloadWaitSeconds": 30,
        "vramPressureSlotReserveMiB": 512, "vramPressureSlotDir": "/dev/shm",
        "ramPolicy": "honest",
    }


def shuracode_settings(choice):
    """ShuraCode (OpenCode config) in the shape of the live file: one local provider through the gateway, no secrets, no plugins."""
    mid = choice.model_id
    return {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "provider": {"tiel-local": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Shura (local)",
            "options": {"baseURL": f"http://127.0.0.1:{GATEWAY_PORT}/v1"},
            "models": {mid: {
                "name": f"{mid} (local)",
                "attachment": False,
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": int(choice.context), "output": 16384},
                "options": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                "settings": {"reasoningEffort": "medium"},
            }},
        }},
        "model": f"tiel-local/{mid}",
        "small_model": f"tiel-local/{mid}",
        "permission": {
            "edit": "allow",
            "bash": {"*": "ask", "git status*": "allow", "git diff*": "allow", "git log*": "allow",
                     "python3 -m unittest*": "allow", "rm -rf*": "deny", "sudo *": "ask"},
            "webfetch": "deny",
            "external_directory": "allow",
        },
    }


def unit_text(home, gateway_script, python=None):
    home = Path(home)
    py = python or sys.executable or "/usr/bin/python3"
    return (
        journal_mod.UNIT_MARKER + "\n"
        "[Unit]\nDescription=Shura gateway (ai-gateway)\nAfter=default.target\n\n"
        "[Service]\nType=simple\n"
        f"Environment=AI_GATEWAY_CONFIG={home / 'config'}\n"
        f"Environment=AI_GATEWAY_STATE={home / 'state'}\n"
        f"ExecStart={py} {gateway_script}\n"
        "Restart=on-failure\nRestartSec=5\n\n"
        "[Install]\nWantedBy=default.target\n")


def _dump(obj):
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_file(path, data):
    """Atomic write (temporary file, then replace)."""
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)


def _inside(home, path):
    """Only the Shura home or the fixed unit file; compared after resolving symlinks (a symlinked ~/.config gets refused)."""
    home = Path(home).resolve()
    try:
        p = Path(path).resolve()
    except (OSError, RuntimeError):
        raise SettingsError(f"refusing to write {path}: cannot resolve it")
    if p == home or home in p.parents:
        return
    if p == journal_mod.unit_path():
        return
    raise SettingsError(f"refusing to write {path}: outside the Shura home {home}")


def _check_unit_ours(unit):
    if unit.is_symlink() or unit.exists():
        try:
            first = unit.read_text(encoding="utf-8", errors="replace").splitlines()[:1] if not unit.is_symlink() else []
        except OSError:
            first = []
        if first != [journal_mod.UNIT_MARKER]:
            raise SettingsError(f"{unit} already exists and was not made by the Shura installer: not overwriting it. "
                                "Move or remove it yourself and run the installer again.")


def _systemctl(runner, *args):
    cmd = ["systemctl", "--user", *args]
    proc = runner(cmd, capture_output=True, text=True, timeout=60)
    if getattr(proc, "returncode", 0):
        raise SettingsError("`" + " ".join(cmd) + "` failed: " +
                            ((getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip() or f"exit {proc.returncode}"))


def apply(home, choice, *, model_path, exe_path, gateway_script=None, runner=subprocess.run, jr=None,
          consent=False, chunk_mib=DEFAULT_CHUNK_MIB, mmproj_path=None, python=None):
    """Write the settings and start the service. Returns the list of written paths. Nothing happens without `consent`.
    With `jr=None` this call owns the journal: any failure rolls everything back (and re-raises); with a journal passed in,
    the caller's rollback does it (install.run already does)."""
    if not consent:
        return []
    if not getattr(choice, "ok", True) or not choice.model_id:
        raise SettingsError("no mode was chosen: nothing to configure")
    home = Path(home)
    owns = jr is None
    jr = jr or journal_mod.Journal(home)
    gateway_script = gateway_script or home / "gateway" / "ai_gateway.py"
    unit = journal_mod.unit_path()
    unit_dir = unit.parent
    cfg = home / "config"
    files = [
        (cfg / "model-launch.json", _dump(launch_settings(choice, model_path, exe_path, chunk_mib=chunk_mib, mmproj_path=mmproj_path))),
        (cfg / "guardian.json", _dump(guardian_settings(choice))),
        (cfg / "shuracode.json", _dump(shuracode_settings(choice))),
        (unit, unit_text(home, gateway_script, python).encode("utf-8")),
    ]
    written = []
    try:
        for path, _ in files:
            _inside(home, path)
        _check_unit_ours(unit)                        # a foreign unit: refuse BEFORE the journal or any file is touched
        jr.mkdir_p(cfg)
        jr.mkdir_p(unit_dir)
        for path, data in files:
            jr.track_file(path)                       # created, or replaced with a copy of the old bytes
            write_file(path, data)
            written.append(str(path))
        jr.record_service(SERVICE, "user-service-disable")      # BEFORE the first systemctl call
        _systemctl(runner, "daemon-reload")
        _systemctl(runner, "enable", SERVICE)
        _systemctl(runner, "start", SERVICE)
    except BaseException:
        if owns:
            jr.rollback(runner=runner)
        raise
    if owns:
        jr.commit()
    return written
