"""Run the Aider Polyglot exercises through ShuraCode/OpenCode and the existing measurement helpers.

Invoke this module inside `bwrap --unshare-net`. It imports the protected v2 harness/proxy,
but never edits them. Large task data, sessions and logs belong in the caller's /dev/shm root.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import harness
import proxy

SHURA_REPO = Path("/home/alisher/Projects/shuracode")
MCP_MODULES = Path("/home/alisher/.config/shuracode/node_modules")
FEATURE_FLAGS = {
    "completion-gate": "completionGate",
    "soft-edit": "softEdit",
    "loop-detector": "loopDetector",
    "task-card": "taskCard",
}


def sampling(model: str) -> dict:
    if model == "tiel-coder":
        return {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 32768}
    if model == "occamy":
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 32768, "presence_penalty": 1.5}
    raise ValueError("model must be tiel-coder or occamy")


def _wait_healthy(url: str, proc: subprocess.Popen, timeout: int = 600) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"model server exited ({proc.returncode}) during load")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"model server health check timed out: {url}")


def _stop_owned(proc: subprocess.Popen | None) -> None:
    if not proc or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _model_server(args, out: Path, stub_server=None):
    if stub_server is not None:
        return stub_server, "http://127.0.0.1:%d" % stub_server.server_port, None
    if not args.server_argv_json:
        raise SystemExit("real-model mode requires --server-argv-json; this command never reads or changes a live config")
    argv = json.loads(Path(args.server_argv_json).read_text())
    argv = [str(part).replace("{PORT}", str(args.server_port)) for part in argv]
    if not argv:
        raise ValueError("server argv is empty")
    if Path(argv[0]).name != "nice":
        argv = ["nice", "-n", "10", *argv]
    log = (out / "llama-server.log").open("wb")
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, preexec_fn=harness._expendable)
    try:
        _wait_healthy(f"http://127.0.0.1:{args.server_port}/health", proc)
    except Exception:
        _stop_owned(proc)
        log.close()
        raise
    return None, f"http://127.0.0.1:{args.server_port}", (proc, log)


def _stage_plugins(scratch: Path, features: list[str]) -> Path:
    """Stage exact ShuraCode plugin sources in tmpfs so package imports resolve."""
    root = scratch / "plugins"
    reliability = root / "reliability"
    reliability.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SHURA_REPO / "plugin/server.ts", root / "server.ts")
    feature_files = {
        "completion-gate": "completion-gate.ts",
        "soft-edit": "soft-edit.ts",
        "loop-detector": "loop-detector.ts",
        "task-card": "task-card.ts",
    }
    for name in features:
        if name not in feature_files:
            raise ValueError(f"unknown reliability plugin: {name}")
        shutil.copy2(SHURA_REPO / "plugin/reliability/core.ts", reliability / "core.ts")
        shutil.copy2(SHURA_REPO / "plugin/reliability" / feature_files[name], reliability / feature_files[name])
    modules_link = root / "node_modules"
    if not modules_link.exists():
        modules_link.symlink_to(MCP_MODULES, target_is_directory=True)
    return root


def _prepare_plugin_dependencies(directory: Path) -> None:
    # OpenCode waits for config-directory plugin dependencies before starting a
    # session. Mirror only the non-secret dependency manifest and point its modules
    # to the preinstalled read-only package tree so startup never attempts npm/network.
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ("package.json", "package-lock.json"):
        source = Path.home() / ".config/shuracode" / filename
        if source.is_file():
            shutil.copy2(source, directory / filename)
    modules_dir = directory / "node_modules"
    if not modules_dir.exists():
        modules_dir.symlink_to(MCP_MODULES, target_is_directory=True)


def _config_for_task(snapshot: Path, cfg_dir: Path, proxy_url: str,
                     model: str, features: list[str], plugin_root: Path):
    _prepare_plugin_dependencies(cfg_dir)
    cfg = json.loads(snapshot.read_text(encoding="utf-8"))
    provider_id = next((name for name, provider in cfg["provider"].items() if model in (provider.get("models") or {})), None)
    if provider_id is None:
        raise ValueError(f"model {model!r} not present in scratch config snapshot")
    provider = cfg["provider"][provider_id]
    provider.setdefault("options", {})["baseURL"] = proxy_url.rstrip("/") + "/v1"
    provider["options"]["apiKey"] = "sandbox-placeholder"
    cfg["model"] = f"{provider_id}/{model}"
    cfg["small_model"] = cfg["model"]
    model_cfg = provider["models"][model]
    model_cfg.setdefault("limit", {})["output"] = 32768
    model_cfg.setdefault("limit", {}).setdefault("context", 200000)

    cfg["plugin"] = [str(plugin_root / "server.ts")]
    for name in features:
        flag = FEATURE_FLAGS[name]
        cfg["plugin"].append([str(plugin_root / "reliability" / f"{name}.ts"), {flag: True}])
    rules_src = snapshot.parent / "sandbox-home" / ".config/shuracode/AGENTS.md"
    rules_dest = cfg_dir / "AGENTS.md"
    if rules_src.is_file():
        rules_dest.write_bytes(rules_src.read_bytes())
        cfg["instructions"] = [str(rules_dest)]
    cfg.setdefault("agent", {}).setdefault("build", {}).update({"temperature": sampling(model)["temperature"],
                                                                 "top_p": 0.95, "max_output_tokens": 32768})
    cfg["autoupdate"] = False
    path = cfg_dir / "shuracode.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _runner(snapshot: Path, model: str, proxy_url: str, features: list[str]):
    def run(workdir, prompt, cfg_dir, data_dir, timeout_s, log_path):
        scratch = Path(workdir).parent
        # OpenCode 1.18.32 waits for dependency checks for both config and project
        # plugin directories. Seed the scratch project directory with the same
        # lockfile and a read-only module symlink to prevent any attempted install.
        _prepare_plugin_dependencies(Path(workdir) / ".opencode")
        _prepare_plugin_dependencies(scratch / "home" / ".config" / "opencode")
        plugin_root = _stage_plugins(scratch, features)
        config = _config_for_task(snapshot, Path(cfg_dir), proxy_url, model, features, plugin_root)
        env = {**os.environ, **harness.ENGINE_ENV,
               "OPENCODE_CONFIG": str(config), "OPENCODE_CONFIG_DIR": str(cfg_dir),
               "HOME": str(scratch / "home"), "XDG_CONFIG_HOME": str(scratch / "home/.config"),
               "XDG_DATA_HOME": str(scratch / "home/.local/share"),
               "XDG_CACHE_HOME": str(scratch / "home/.cache"),
               "NO_PROXY": "*", "no_proxy": "*", "PWD": str(workdir)}
        argv = harness.agent_argv(workdir, prompt)
        if os.environ.get("POLYGLOT_DIRECT_ENGINE") == "1":
            wrapped = argv
        else:
            wrapped = harness.sandboxed(argv, scratch, workdir, env={
                **harness.ENGINE_ENV,
                "OPENCODE_CONFIG": str(config),
                "OPENCODE_CONFIG_DIR": str(cfg_dir),
                "NO_PROXY": "*", "no_proxy": "*",
                "XDG_CONFIG_HOME": str(scratch / "home/.config"),
                "XDG_DATA_HOME": str(scratch / "home/.local/share"),
                "XDG_CACHE_HOME": str(scratch / "home/.cache"),
            })
            # The parent runner is already in a no-network namespace. Plugin sources
            # are staged under this task's scratch tree; only the SDK package is bound.
            insert_at = wrapped.index("--chdir")
            binds = ["--ro-bind", str(MCP_MODULES), str(MCP_MODULES)]
            wrapped[insert_at:insert_at] = binds
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with Path(log_path).open("wb") as log:
            proc = subprocess.Popen(wrapped, cwd=workdir, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True, preexec_fn=harness._expendable)
            try:
                code = proc.wait(timeout=timeout_s)
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out, code = True, None
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        return code, timed_out, time.monotonic() - start
    return run


def run(args) -> list[dict]:
    if os.environ.get("SHURA_POLYGLOT_NETNS") != "1":
        raise SystemExit("refusing to run agent unless SHURA_POLYGLOT_NETNS=1 is set inside bwrap --unshare-net")
    task_dir = Path(args.tasks)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = "tiel-coder" if args.model == "tiel" else "occamy"
    sample = sampling(model)
    stub, upstream, owned_server = None, None, None
    if args.stub:
        from stub_openai_server import Handler, ThreadingHTTPServer
        Handler.profile = "text"
        Handler.log_path = out_dir / "stub-requests.jsonl"
        stub = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        import threading
        threading.Thread(target=stub.serve_forever, daemon=True).start()
        upstream = f"http://127.0.0.1:{stub.server_port}"
    else:
        stub, upstream, owned_server = _model_server(args, out_dir)

    tasks = harness.all_tasks(task_dir)
    selected_languages = {x.strip() for x in args.languages.split(",") if x.strip()}
    if selected_languages:
        tasks = [t for t in tasks if t["lang"] in selected_languages]
    if args.task_ids:
        wanted = set(args.task_ids.split(","))
        tasks = [t for t in tasks if t["id"] in wanted]
    if args.task_ids_file:
        wanted = {line.strip() for line in Path(args.task_ids_file).read_text().splitlines() if line.strip()}
        tasks = [t for t in tasks if t["id"] in wanted]
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise SystemExit("no tasks selected")

    records = []
    try:
        for idx, task in enumerate(tasks, 1):
            task_path = Path(task["dir"])
            # The ordinary harness runner copies hidden tests after the agent. Restore
            # them under their public Exercism paths inside this offline runner adapter.
            task["test_cmd"] = [sys.executable, str(BENCH / "external" / "run_aider_tests.py"), "--task-dir", str(task_path)]
            if args.timeout_s > 0:
                task["timeout_s"] = args.timeout_s
            proxy_log = out_dir / "proxy" / f"{task['id']}__{args.model}__0.jsonl"
            proxy_log.parent.mkdir(parents=True, exist_ok=True)
            proxy_log.unlink(missing_ok=True)
            server = proxy.serve(0, upstream, str(proxy_log), sample)
            import threading
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = harness.run_one(task, args.model, 0, server.server_port, proxy_log,
                                         out_dir / "agent-logs", context=200000,
                                         runner=_runner(Path(args.config_snapshot), model,
                                                        f"http://127.0.0.1:{server.server_port}",
                                                        args.plugins.split(",") if args.plugins else []),
                                         scratch_root=args.scratch)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
            records.append(result)
            with (out_dir / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            print(f"{idx}/{len(tasks)} {task['id']} resolved={result['resolved']} requests={result['requests']}", flush=True)
    finally:
        if stub:
            stub.shutdown()
            stub.server_close()
        if owned_server:
            proc, log = owned_server
            _stop_owned(proc)
            log.close()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config-snapshot", required=True)
    parser.add_argument("--scratch", default="/dev/shm/agentic-eval-external")
    parser.add_argument("--model", choices=("tiel", "occamy"), default="tiel")
    parser.add_argument("--languages", default="python,javascript")
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--task-ids-file", default="", help="newline-separated IDs from validate_polyglot_tasks.py")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-s", type=int, default=0,
                        help="override per-task deadline (useful for endpoint smoke tests)")
    parser.add_argument("--plugins", default="", help="comma list of completion-gate,soft-edit,loop-detector,task-card")
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--server-port", type=int, default=8090)
    parser.add_argument("--server-argv-json", default="")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
