"""Create a scratch-only OpenCode config from the current ShuraCode config.

The source config is read, never changed. Credentials are scrubbed, external MCP
servers and personal memory are omitted, and the caller's test proxy replaces only
the model endpoint. The resulting file belongs in /dev/shm.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

SHURA_REPO = Path.home() / "Projects/shuracode"
SENSITIVE = ("apikey", "api_key", "token", "secret", "password", "authorization", "credential")


def _scrub(value):
    if isinstance(value, dict):
        return {k: ("<redacted>" if any(tag in k.lower() for tag in SENSITIVE) else _scrub(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    return value


def snapshot(source: Path, dest: Path, model_id: str, proxy_url: str,
             plugins: list[str] | None = None, *, shura_repo: Path = SHURA_REPO) -> dict:
    original = json.loads(source.read_text(encoding="utf-8"))
    cfg = _scrub(copy.deepcopy(original))
    provider_id = next((pid for pid, provider in cfg.get("provider", {}).items()
                        if model_id in (provider.get("models") or {})), None)
    if not provider_id:
        raise ValueError(f"model {model_id!r} is not present in the source config")
    provider = cfg["provider"][provider_id]
    provider.setdefault("options", {})["baseURL"] = proxy_url
    provider["options"]["apiKey"] = "sandbox-placeholder"
    cfg["model"] = f"{provider_id}/{model_id}"
    cfg["small_model"] = f"{provider_id}/{model_id}"
    model = provider["models"][model_id]
    model.setdefault("limit", {})["output"] = 32768
    model.setdefault("limit", {}).setdefault("context", 200000)

    plugins_out = [str(shura_repo / "plugin/server.ts")]
    paths = {
        "completion-gate": shura_repo / "plugin/reliability/completion-gate.ts",
        "soft-edit": shura_repo / "plugin/reliability/soft-edit.ts",
        "loop-detector": shura_repo / "plugin/reliability/loop-detector.ts",
        "task-card": shura_repo / "plugin/reliability/task-card.ts",
    }
    for feature in plugins or []:
        path = paths[feature]
        option_key = {"completion-gate": "completionGate", "soft-edit": "softEdit",
                      "loop-detector": "loopDetector", "task-card": "taskCard"}[feature]
        plugins_out.append([str(path), {option_key: True}])
    cfg["plugin"] = plugins_out
    # MCP servers can perform network or personal-memory I/O. This public-task profile is offline.
    cfg["mcp"] = {}

    scratch_home = dest.parent / "sandbox-home"
    rule_src = Path.home() / ".config/shuracode/AGENTS.md"
    scratch_rules = scratch_home / ".config/shuracode/AGENTS.md"
    scratch_rules.parent.mkdir(parents=True, exist_ok=True)
    if rule_src.is_file():
        scratch_rules.write_bytes(rule_src.read_bytes())
        cfg["instructions"] = [str(scratch_rules)]
    cfg["autoupdate"] = False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"model": cfg["model"], "provider": provider_id, "output_limit": 32768,
            "external_mcp_disabled": True, "personal_memory_included": False,
            "plugins": [p if isinstance(p, str) else p[0] for p in plugins_out]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path.home() / ".config/shuracode/shuracode.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-id", choices=("tiel-coder", "occamy"), required=True)
    parser.add_argument("--proxy-url", default="http://127.0.0.1:19091/v1")
    parser.add_argument("--plugins", default="")
    parser.add_argument("--shura-repo", type=Path, default=SHURA_REPO)
    args = parser.parse_args()
    features = [x.strip() for x in args.plugins.split(",") if x.strip()]
    print(json.dumps(snapshot(args.source, args.out, args.model_id, args.proxy_url, features,
                              shura_repo=args.shura_repo), ensure_ascii=False))


if __name__ == "__main__":
    main()
