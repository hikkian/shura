"""QLEV0: configure QLEV1 arms with one controlled difference from base.

Run from bench/agentic-eval:
    python3 -m qlev.arms base
    python3 -m qlev.arms P

The CLI writes a throwaway config under /dev/shm/QLEV0 and prints the proxy arm fragment.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

ARM_IDS = ("base", "S", "P", "T", "G", "L", "C", "D", "TB")
QUALITY_ARMS = frozenset(("S", "G", "L", "C"))
SPEED_ARMS = frozenset(("T", "P", "D", "TB"))
UNAVAILABLE = {
    "L": "LSP arm unavailable: QLSP1 has not provided a verified LSP setup; keep L off until QLSP1.",
    "TB": "ThinkBrake arm unavailable: THINK3 has not provided a verified ThinkBrake setup; keep TB off until THINK3.",
}
DEFAULT_EXTRA_PLUGIN_SRC = Path("/home/alisher/Projects/shuracode-integration/plugin")
CONTRACT_RULE = Path("/home/alisher/Projects/shuracode-integration/config/rules/contract-first.md")


class UnavailableArm(ValueError):
    pass


def normalize_arm(arm):
    value = str(arm)
    if value not in ARM_IDS:
        raise ValueError(f"unknown QLEV arm {value!r}; choose one of {', '.join(ARM_IDS)}")
    if value in UNAVAILABLE:
        raise UnavailableArm(UNAVAILABLE[value])
    return value


def is_arm(arm):
    return str(arm) in ARM_IDS


def reliability_for_arm(arm):
    arm = normalize_arm(arm)
    return {"completion-gate"} if arm == "G" else set()


def proxy_extra_for_arm(arm):
    """Per-arm fields merged by run_experiment's measuring proxy into each request body."""
    arm = normalize_arm(arm)
    extra = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}
    if arm == "S":
        extra.update({"temperature": 1.0, "presence_penalty": 1.5})
    if arm == "P":
        extra["chat_template_kwargs"] = {"preserve_thinking": False}
    return extra


def experiment_arm(arm, model):
    """Fragment for run_experiment.py arms.json; the caller supplies the chosen model path."""
    arm = normalize_arm(arm)
    return {"name": arm, "model": str(model), "extra": proxy_extra_for_arm(arm)}


def _plugin_source():
    return Path(os.environ.get("AGENTIC_EVAL_QLEV_PLUGIN_SRC", str(DEFAULT_EXTRA_PLUGIN_SRC)))


def _copy_plugin(cfg_dir, source_rel, destination_rel):
    source = _plugin_source() / source_rel
    destination = Path(cfg_dir).parent / "plugin" / destination_rel
    if not source.is_file():
        raise FileNotFoundError(f"QLEV plugin source missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _add_plugin(conf, entry):
    path = entry[0] if isinstance(entry, list) else entry
    existing = {x[0] if isinstance(x, list) else x for x in conf.get("plugin", [])}
    if path not in existing:
        conf.setdefault("plugin", []).append(entry)


def apply_arm(conf, arm, cfg_dir, workdir=None, task=None):
    """Mutate a harness config for one arm. `P` is carried by run_experiment's proxy extra."""
    arm = normalize_arm(arm)
    model = conf["provider"]["eval"]["models"]["m"]
    options = model.setdefault("options", {})
    if arm == "S":
        options.update({"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5})
        conf.setdefault("agent", {}).setdefault("build", {}).update({"temperature": 1.0, "top_p": 0.95})
    elif arm == "T":
        model.setdefault("settings", {})["reasoningEffort"] = "low"
        model.setdefault("variants", {})["low"] = {"settings": {"reasoningEffort": "low"}}
        conf.setdefault("agent", {}).setdefault("build", {})["variant"] = "low"
    elif arm == "G":
        gate = next((p for p in conf.get("plugin", [])
                     if (p[0] if isinstance(p, list) else p).endswith("/reliability/completion-gate.ts")), None)
        if gate is None:
            raise RuntimeError("completion-gate plugin was not installed in the scratch config")
        if isinstance(gate, list):
            gate[1]["completionGate"] = True
        else:
            conf["plugin"][conf["plugin"].index(gate)] = [gate, {"completionGate": True}]
    elif arm == "C":
        if task and task.get("kind") not in ("implement", "refactor"):
            raise ValueError("C arm is only valid for implement/refactor tasks")
        if not workdir:
            raise ValueError("C arm needs a scratch workdir for its project instruction")
        rule = Path(workdir) / ".qlev" / "contract-first.md"
        rule.parent.mkdir(parents=True, exist_ok=True)
        if not CONTRACT_RULE.is_file():
            raise FileNotFoundError(f"contract-first rule missing: {CONTRACT_RULE}")
        shutil.copy2(CONTRACT_RULE, rule)
        conf["instructions"] = [str(rule)]
        visible = _copy_plugin(cfg_dir, "visible-tests.ts", "visible-tests.ts")
        _copy_plugin(cfg_dir, "lib/visible-tests.ts", "lib/visible-tests.ts")
        _add_plugin(conf, [str(visible), {"visibleTests": True}])
    elif arm == "D":
        diet = _copy_plugin(cfg_dir, "context-diet.ts", "context-diet.ts")
        _copy_plugin(cfg_dir, "lib/context-diet.ts", "lib/context-diet.ts")
        _add_plugin(conf, [str(diet), {"contextDiet": True}])
    # P is the llama.cpp request field chat_template_kwargs. The measuring proxy injects it
    # through arm.extra (see proxy_extra_for_arm); it is not an AI SDK model option.
    return conf


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARM_IDS)
    parser.add_argument("--scratch", default="/dev/shm/QLEV0/arms")
    parser.add_argument("--port", type=int, default=18991, help="local fake/proxy port written into the config")
    parser.add_argument("--task-kind", default="implement", help="used to validate the C arm")
    parser.add_argument("--model", default="/path/to/tiel-model.gguf", help="model path for the run_experiment arm fragment")
    args = parser.parse_args(argv)
    try:
        arm = normalize_arm(args.arm)
    except UnavailableArm as error:
        parser.error(str(error))
    import harness
    scratch = Path(args.scratch) / arm
    work = scratch / "work"
    work.mkdir(parents=True, exist_ok=True)
    harness.write_project_rules(work)
    task = {"id": "qlev-config-probe", "kind": args.task_kind}
    cfg = harness.write_config(scratch / "cfg", args.port, arm=arm, workdir=work, task=task)
    print(json.dumps({"arm": arm, "config": str(Path(cfg) / "shuracode.json"),
                      "run_experiment_arm": experiment_arm(arm, args.model)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
