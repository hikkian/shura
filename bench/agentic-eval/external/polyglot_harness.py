"""Run the Aider Polyglot exercises through ShuraCode/OpenCode and the existing measurement helpers.

Invoke this module inside `bwrap --unshare-net`. It imports the protected v2 harness/proxy,
but never edits them. Large task data, sessions and logs belong in the caller's /dev/shm root.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import harness
import proxy as measurement_proxy
import safety
import capture_proxy
import system_pressure

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


def _preexec_safety():
    harness._expendable()
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (AttributeError, OSError, ValueError):
        pass


def _wait_healthy(url: str, proc: subprocess.Popen, timeout: int = 600, abort_event=None,
                  abort_reason=None) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if abort_event is not None and abort_event.is_set():
            detail = abort_reason() if abort_reason else ""
            raise RuntimeError(f"S6 safety watchdog tripped while waiting for the test server: {detail}")
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


def _require_draft_mtp(argv: list[str]) -> None:
    enabled = "--spec-type=draft-mtp" in argv or any(
        argv[index] == "--spec-type" and index + 1 < len(argv) and argv[index + 1] == "draft-mtp"
        for index in range(len(argv))
    )
    if not enabled:
        raise ValueError("test server argv must explicitly enable --spec-type draft-mtp")


def _server_command(argv: list[str]) -> list[str]:
    """Run llama-server on all allowed CPUs at nice 10; never inherit client affinity."""
    command = list(argv)
    while command and Path(command[0]).name in {"nice", "taskset"}:
        if Path(command[0]).name == "nice":
            command = command[3:] if len(command) >= 4 and command[1] == "-n" else command[1:]
        elif "-c" in command:
            command = command[command.index("-c") + 2:]
        else:
            command = command[1:]
    return ["/usr/bin/nice", "-n", "10", *command]


def _client_command(argv: list[str], cpus: str) -> list[str]:
    """Apply taskset/nice only to OpenCode and its child tool processes."""
    return ["/usr/bin/taskset", "-c", cpus, "/usr/bin/nice", "-n", "19", *argv]


def _server_log_decode_rate(log_path: Path, start_offset: int) -> tuple[int, float, str] | None:
    try:
        with log_path.open("rb") as stream:
            stream.seek(start_offset)
            text = stream.read().decode(errors="replace")
    except OSError:
        return None
    pattern = re.compile(
        r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
        r"\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second"
    )
    for line in reversed(text.splitlines()):
        match = pattern.search(line)
        if match:
            return int(match.group(2)), float(match.group(4)), line.strip()
    return None


def _server_log_decode_average(log_path: Path, start_offset: int) -> float | None:
    try:
        with log_path.open("rb") as stream:
            stream.seek(start_offset)
            text = stream.read().decode(errors="replace")
    except OSError:
        return None
    pattern = re.compile(r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
    samples = []
    for line in text.splitlines():
        match = pattern.search(line)
        if match and float(match.group(1)) > 0 and int(match.group(2)) > 0:
            samples.append((float(match.group(1)), int(match.group(2))))
    elapsed_ms = sum(ms for ms, _ in samples)
    return 1000.0 * sum(tokens for _, tokens in samples) / elapsed_ms if elapsed_ms else None


def _warmup_200(upstream: str, out: Path, server_log: Path, abort_event, tag: str = "start") -> float:
    safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)
    body = {
        "model": "tiel-coder",
        "messages": [{"role": "user", "content": (
            "Output a comma-separated sequence of consecutive positive integers beginning at 1. "
            "Continue without explanation until the response token limit stops you."
        )}],
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 200, "stream": False,
    }
    (out / f"warmup-{safe_tag}-request.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    start_offset = server_log.stat().st_size
    request = urllib.request.Request(
        upstream.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read())
    elapsed = time.monotonic() - t0
    (out / f"warmup-{safe_tag}-response.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if abort_event.is_set():
        raise RuntimeError("S6 safety watchdog tripped during the 200-token warm-up")
    logged = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        logged = _server_log_decode_rate(server_log, start_offset)
        if logged:
            break
        time.sleep(0.1)
    if not logged:
        raise RuntimeError("200-token warm-up completed but decode rate was not found in llama-server.log")
    predicted_n, tokens_per_second, log_line = logged
    usage_tokens = (payload.get("usage") or {}).get("completion_tokens")
    record = {"requested_tokens": 200, "usage_completion_tokens": usage_tokens,
              "server_log_predicted_tokens": predicted_n,
              "server_log_decode_tokens_per_second": tokens_per_second,
              "warmup_wall_seconds": round(elapsed, 3), "server_log_line": log_line}
    (out / f"warmup-{safe_tag}-result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"warmup: requested=200 usage={usage_tokens} server_log={predicted_n} tokens tps={tokens_per_second:.2f}", flush=True)
    if usage_tokens != 200 or predicted_n != 200:
        raise RuntimeError(f"warm-up produced usage={usage_tokens}, server_log={predicted_n}; expected exactly 200 tokens")
    if tokens_per_second < 35.0:
        raise RuntimeError(f"warm-up decode speed {tokens_per_second:.2f} tok/s is below 35 tok/s")
    return tokens_per_second


class ServerHealthMonitor(threading.Thread):
    """Detect an exited or unresponsive test server without waiting out the task cap."""

    def __init__(self, proc, health_url: str, *, period: float = 5.0, failures: int = 6,
                 health_check=None):
        super().__init__(daemon=True)
        self.proc = proc
        self.health_url = health_url
        self.period = period
        self.failure_limit = failures
        self.health_check = health_check or self._check_health
        self.stop_event = threading.Event()
        self.failed = threading.Event()
        self.reason = ""

    def _check_health(self) -> bool:
        try:
            with urllib.request.urlopen(self.health_url, timeout=2) as response:
                return response.status == 200
        except OSError:
            return False

    def run(self):
        missed = 0
        while not self.stop_event.wait(self.period):
            code = self.proc.poll()
            if code is not None:
                self.reason = f"llama-server exited with code {code}"
                self.failed.set()
                return
            try:
                healthy = self.health_check()
            except Exception as exc:
                healthy = False
                self.reason = f"health probe raised {type(exc).__name__}: {exc}"
            missed = 0 if healthy else missed + 1
            if missed >= self.failure_limit:
                if not self.reason:
                    self.reason = f"/health missed {missed} probes at {self.period:g}s intervals"
                self.failed.set()
                return

    def stop(self):
        self.stop_event.set()


class CombinedAbortEvent:
    def __init__(self, *events):
        self.events = events

    def is_set(self):
        return any(event.is_set() for event in self.events)


def _model_server(args, out: Path, stub_server=None, safety_watchdog=None, instance: int = 0):
    if stub_server is not None:
        return stub_server, "http://127.0.0.1:%d" % stub_server.server_port, None
    if not args.server_argv_json:
        raise SystemExit("real-model mode requires --server-argv-json; this command never reads or changes a live config")
    argv = json.loads(Path(args.server_argv_json).read_text())
    argv = [str(part).replace("{PORT}", str(args.server_port)) for part in argv]
    if not argv:
        raise ValueError("server argv is empty")
    _require_draft_mtp(argv)
    argv = _server_command(argv)
    log_name = "llama-server.log" if instance == 0 else f"llama-server-restart-{instance}.log"
    log = (out / log_name).open("wb")
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, preexec_fn=_preexec_safety)
    try:
        _wait_healthy(f"http://127.0.0.1:{args.server_port}/health", proc,
                      abort_event=safety_watchdog.tripped if safety_watchdog else None,
                      abort_reason=lambda: safety_watchdog.reason if safety_watchdog else "")
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
                     model: str, features: list[str], plugin_root: Path,
                     context: int = 131072):
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
    model_cfg["limit"]["context"] = context

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


def _runner(snapshot: Path, model: str, proxy_url: str, features: list[str],
            context: int, cpus: str, abort_event):
    def run(workdir, prompt, cfg_dir, data_dir, timeout_s, log_path):
        scratch = Path(workdir).parent
        # OpenCode 1.18.32 waits for dependency checks for both config and project
        # plugin directories. Seed the scratch project directory with the same
        # lockfile and a read-only module symlink to prevent any attempted install.
        _prepare_plugin_dependencies(Path(workdir) / ".opencode")
        _prepare_plugin_dependencies(scratch / "home" / ".config" / "opencode")
        plugin_root = _stage_plugins(scratch, features)
        config = _config_for_task(snapshot, Path(cfg_dir), proxy_url, model, features, plugin_root, context)
        env = {**os.environ, **harness.ENGINE_ENV,
               "OPENCODE_CONFIG": str(config), "OPENCODE_CONFIG_DIR": str(cfg_dir),
               "HOME": str(scratch / "home"), "XDG_CONFIG_HOME": str(scratch / "home/.config"),
               "XDG_DATA_HOME": str(scratch / "home/.local/share"),
               "XDG_CACHE_HOME": str(scratch / "home/.cache"),
               "NO_PROXY": "*", "no_proxy": "*", "PWD": str(workdir)}
        argv = _client_command(harness.agent_argv(workdir, prompt), cpus)
        if os.environ.get("POLYGLOT_DIRECT_ENGINE") == "1":
            wrapped = argv
        else:
            wrapped = harness.sandboxed(argv, scratch, workdir, env={
                **harness.ENGINE_ENV,
                "OPENCODE_CONFIG": str(config),
                "OPENCODE_CONFIG_DIR": str(cfg_dir),
                "NO_PROXY": "*", "no_proxy": "*",
                "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "32768",
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
                                    start_new_session=True, preexec_fn=_preexec_safety)
            start_deadline = time.monotonic() + timeout_s
            timed_out = False
            code = None
            while proc.poll() is None:
                if abort_event.is_set():
                    timed_out = True
                    break
                if time.monotonic() >= start_deadline:
                    timed_out = True
                    break
                time.sleep(0.2)
            if timed_out:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            else:
                code = proc.returncode
        return code, timed_out, time.monotonic() - start
    return run


def _require_tmpfs(*paths: str | Path) -> None:
    root = Path("/dev/shm").resolve()
    for value in paths:
        if not value:
            continue
        path = Path(value).resolve()
        if path != root and root not in path.parents:
            raise SystemExit(f"large run input/output must stay in /dev/shm: {path}")


def _completed(results_path: Path) -> set[tuple[str, str, int]]:
    """Read a resumable JSONL journal and discard only a torn trailing record."""
    if not results_path.is_file():
        return set()
    good_lines = []
    completed = set()
    for line in results_path.read_bytes().splitlines(keepends=True):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            break
        if isinstance(row, dict) and row.get("task") and row.get("arm"):
            completed.add((str(row["task"]), str(row["arm"]), int(row.get("rep", 0))))
        good_lines.append(line)
    if len(good_lines) != len(results_path.read_bytes().splitlines(keepends=True)):
        recovered = results_path.with_suffix(".recovered")
        recovered.write_bytes(b"".join(good_lines))
        os.replace(recovered, results_path)
    return completed


def _checkpoint_warmup_due(completed_count: int, last_warmup_count: int) -> bool:
    return completed_count > 0 and completed_count % 20 == 0 and last_warmup_count != completed_count


def _save_result(path: Path, record: dict) -> None:
    payload = (json.dumps(record, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _run_one(task, arm, port, proxy_log, out_dir, scratch_root, runner, context,
             safety_watchdog, server_log, server_monitor=None, attempt=0):
    scratch = Path(tempfile.mkdtemp(prefix=f"{task['id']}-{arm}-0-", dir=scratch_root))
    try:
        work = scratch / "work"
        shutil.copytree(task["dir"] / "start", work)
        cfg = harness.write_config(scratch / "cfg", port, context=context, output=32768)
        task["test_cmd"] = [sys.executable, str(BENCH / "external" / "run_aider_tests.py"),
                            "--task-dir", str(task["dir"])]
        attempt_suffix = "" if attempt == 0 else f"__attempt{attempt}"
        log_path = Path(out_dir) / "agent-logs" / f"{task['id']}__{arm}__0{attempt_suffix}.agent.log"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        server_log_offset = server_log.stat().st_size if server_log.is_file() else 0
        code, timed_out, seconds = runner(work, task["prompt"], cfg, scratch / "data",
                                          task.get("timeout_s", 900), log_path)
        decode_rate = _server_log_decode_average(server_log, server_log_offset)
        server_failed = bool(server_monitor and server_monitor.failed.is_set())
        if safety_watchdog.tripped.is_set():
            passed, tail = False, f"S6 watchdog stopped run: {safety_watchdog.reason}"
        elif server_failed:
            passed, tail = False, f"server failed during task: {server_monitor.reason}"
        else:
            test_log = Path(out_dir) / "test-logs" / f"{task['id']}__{arm}__0{attempt_suffix}.log"
            passed, tail = harness.run_tests(work, task["dir"] / "hidden", task["test_cmd"], timeout=30,
                                             log_path=test_log)
        totals = measurement_proxy.totals(proxy_log)
        record = {
            "arm": arm, "task": task["id"], "family": task["family"], "kind": task["kind"],
            "rep": 0, "resolved": bool(passed), "timed_out": timed_out,
            "context_limit_tokens": context, "output_limit_tokens": 32768,
            "agent_exit": code, "seconds": round(seconds, 1),
            "tokens": totals["completion_tokens"], "prompt_tokens": totals["prompt_tokens"],
            "requests": totals["requests"], "request_errors": totals["errors"],
            "decode_tokens_per_second": round(decode_rate, 3) if decode_rate else None,
            "server_failed": server_failed,
            "server_failure_reason": server_monitor.reason if server_failed else None,
            "attempt": attempt,
            **harness.count_events(log_path),
        }
        if not passed:
            record["test_tail"] = tail[-600:]
        if safety_watchdog.tripped.is_set():
            record["safety_stop"] = safety_watchdog.reason
        return record
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run(args) -> list[dict]:
    if os.environ.get("SHURA_POLYGLOT_NETNS") != "1":
        raise SystemExit("refusing to run agent unless SHURA_POLYGLOT_NETNS=1 is set inside bwrap --unshare-net")
    _require_tmpfs(args.tasks, args.out, args.config_snapshot, args.scratch, args.task_ids_file,
                   args.server_argv_json)
    task_dir = Path(args.tasks)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(args.scratch)
    scratch_root.mkdir(parents=True, exist_ok=True)
    model = "tiel-coder" if args.model == "tiel" else "occamy"
    features = [name for name in args.plugins.split(",") if name]
    arm = args.arm_name or (f"{args.model}-with-reliability" if features else f"{args.model}-no-reliability")
    sample = sampling(model)
    tasks = harness.all_tasks(task_dir)
    selected_languages = {x.strip() for x in args.languages.split(",") if x.strip()}
    if selected_languages:
        tasks = [t for t in tasks if t["lang"] in selected_languages]
    if args.task_ids:
        wanted = [item.strip() for item in args.task_ids.split(",") if item.strip()]
        by_id = {t["id"]: t for t in tasks}
        tasks = [by_id[item] for item in wanted if item in by_id]
    if args.task_ids_file:
        wanted = [line.strip() for line in Path(args.task_ids_file).read_text().splitlines() if line.strip()]
        by_id = {t["id"]: t for t in tasks}
        tasks = [by_id[item] for item in wanted if item in by_id]
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise SystemExit("no tasks selected")
    results_path = out_dir / "results.jsonl"
    done = _completed(results_path) if args.resume else set()
    body_arm = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in arm)
    body_log = out_dir / f"request-bodies-{body_arm}.jsonl.gz"
    stub, upstream, owned_server = None, None, None
    if args.stub:
        stub_safety_samples = {"count": 0}
        def stub_reader():
            stub_safety_samples["count"] += 1
            return {"gpu_c": 45, "cpu_c": 55, "vram_free": 8192, "vram_used": 0,
                    "ram": 8.0, "disk": 100.0}
        safety_watchdog = safety.Watchdog("/", period=1.0, gpu=object(), reader=stub_reader)
    else:
        stub_safety_samples = None
        # In bwrap, `/` is the sandbox mount tree and its apparent free bytes are
        # not the host SSD. BENCH is a read-only bind mount on the real project
        # filesystem, so this checks the actual disk headroom without writing to it.
        safety_watchdog = safety.Watchdog(str(BENCH), period=5.0)
    safety_watchdog.start()
    pressure_guard = system_pressure.SystemPressureGuard(safety_watchdog._trip, period=5.0)
    pressure_guard.start()
    records = []
    restart_count = 0
    server_instance = 0
    server_log_path = out_dir / "llama-server.log"
    restart_log = out_dir / "server-restarts.jsonl"
    interrupted_log = out_dir / "interrupted-attempts.jsonl"

    def start_server():
        nonlocal stub, upstream, owned_server, server_log_path
        stub, upstream, owned_server = _model_server(
            args, out_dir, safety_watchdog=safety_watchdog, instance=server_instance)
        server_log_path = out_dir / ("llama-server.log" if server_instance == 0
                                     else f"llama-server-restart-{server_instance}.log")

    def warmup(tag: str, completed: int):
        if args.stub or not args.warmup_200:
            return
        rate = _warmup_200(upstream, out_dir, server_log_path,
                           safety_watchdog.tripped, tag=tag)
        _save_result(out_dir / "warmups.jsonl", {
            "ts": time.time(), "tag": tag, "completed_tasks": completed,
            "decode_tokens_per_second": rate,
        })

    def restart_server(reason: str, task_id: str | None):
        nonlocal restart_count, server_instance, owned_server, upstream, server_log_path
        while restart_count < args.max_server_restarts:
            if owned_server:
                proc, log = owned_server
                _stop_owned(proc)
                log.close()
                owned_server = None
            restart_count += 1
            server_instance = restart_count
            _save_result(restart_log, {
                "ts": time.time(), "restart_count": restart_count,
                "task": task_id, "reason": reason, "action": "restarting",
            })
            try:
                start_server()
                warmup(f"restart-{restart_count}", len(done))
                return
            except Exception as exc:
                reason = f"restart {restart_count} failed: {type(exc).__name__}: {exc}"
                _save_result(restart_log, {
                    "ts": time.time(), "restart_count": restart_count,
                    "task": task_id, "reason": reason, "action": "restart-failed",
                })
                if safety_watchdog.tripped.is_set() or "below 35 tok/s" in str(exc):
                    raise
                if owned_server:
                    proc, log = owned_server
                    _stop_owned(proc)
                    log.close()
                    owned_server = None
        _save_result(restart_log, {
            "ts": time.time(), "restart_count": restart_count,
            "task": task_id, "reason": reason, "action": "limit-reached",
        })
        raise RuntimeError(f"server restart limit ({args.max_server_restarts}) reached: {reason}")

    try:
        if args.stub:
            from stub_openai_server import Handler, ThreadingHTTPServer
            Handler.profile = "text"
            Handler.log_path = out_dir / f"stub-requests-{body_arm}.jsonl"
            stub = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            import threading
            threading.Thread(target=stub.serve_forever, daemon=True).start()
            upstream = f"http://127.0.0.1:{stub.server_port}"
        else:
            if safety_watchdog.gpu is None:
                raise RuntimeError("NVML unavailable; refusing real-model run without GPU safety telemetry")
            while True:
                try:
                    start_server()
                    warmup("start", len(done))
                    break
                except Exception as exc:
                    if safety_watchdog.tripped.is_set():
                        raise
                    if "below 35 tok/s" in str(exc) or "expected exactly 200 tokens" in str(exc):
                        raise
                    if restart_count >= args.max_server_restarts:
                        raise RuntimeError(f"server restart limit ({args.max_server_restarts}) reached during startup: {exc}")
                    if owned_server:
                        failed_proc, failed_log = owned_server
                        _stop_owned(failed_proc)
                        failed_log.close()
                        owned_server = None
                    restart_count += 1
                    server_instance = restart_count
                    _save_result(restart_log, {
                        "ts": time.time(), "restart_count": restart_count, "task": None,
                        "reason": f"startup/warmup failure: {type(exc).__name__}: {exc}",
                        "action": "retry-startup",
                    })

        last_warmup_count = len(done)
        for idx, task in enumerate(tasks, 1):
            if safety_watchdog.tripped.is_set():
                break
            if shutil.disk_usage(scratch_root).free < int(args.tmpfs_min_free_gib * (1024 ** 3)):
                raise RuntimeError("tmpfs free-space guard stopped the run")
            if (task["id"], arm, 0) in done:
                print(f"resume: skip {task['id']} ({arm})", flush=True)
                continue
            if _checkpoint_warmup_due(len(done), last_warmup_count):
                try:
                    warmup(f"checkpoint-{len(done)}", len(done))
                except Exception as exc:
                    if safety_watchdog.tripped.is_set() or "below 35 tok/s" in str(exc):
                        raise
                    restart_server(f"checkpoint warmup failure: {type(exc).__name__}: {exc}", task["id"])
                last_warmup_count = len(done)
            task_path = Path(task["dir"])
            # The ordinary harness runner copies hidden tests after the agent. Restore
            # them under their public Exercism paths inside this offline runner adapter.
            task["test_cmd"] = [sys.executable, str(BENCH / "external" / "run_aider_tests.py"), "--task-dir", str(task_path)]
            if args.timeout_s > 0:
                task["timeout_s"] = args.timeout_s
            attempt = 0
            while True:
                if safety_watchdog.tripped.is_set():
                    break
                proxy_suffix = "" if attempt == 0 else f"__attempt{attempt}"
                proxy_log = out_dir / "proxy" / f"{task['id']}__{arm}__0{proxy_suffix}.jsonl"
                proxy_log.parent.mkdir(parents=True, exist_ok=True)
                proxy_log.unlink(missing_ok=True)
                server = capture_proxy.serve(0, upstream, str(proxy_log), str(body_log), sample)
                import threading
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                monitor = None
                try:
                    if owned_server:
                        monitor = ServerHealthMonitor(owned_server[0],
                                                      f"http://127.0.0.1:{args.server_port}/health")
                        monitor.start()
                    abort_event = (CombinedAbortEvent(safety_watchdog.tripped, monitor.failed)
                                   if monitor else safety_watchdog.tripped)
                    result = _run_one(task, arm, server.server_port, proxy_log,
                                      out_dir, args.scratch,
                                      runner=_runner(Path(args.config_snapshot), model,
                                                     f"http://127.0.0.1:{server.server_port}",
                                                     features, args.context, args.cpus, abort_event),
                                      context=args.context, safety_watchdog=safety_watchdog,
                                      server_log=server_log_path, server_monitor=monitor, attempt=attempt)
                finally:
                    if monitor:
                        monitor.stop()
                        monitor.join(timeout=3)
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)

                if safety_watchdog.tripped.is_set():
                    _save_result(interrupted_log, {
                        "ts": time.time(), "task": task["id"], "attempt": attempt,
                        "reason": safety_watchdog.reason, "action": "stopped-by-safety",
                        "partial_result": result,
                    })
                    print(f"S6 safety stop: {safety_watchdog.reason}", flush=True)
                    break
                if result.get("server_failed") or (monitor and monitor.failed.is_set()):
                    reason = (monitor.reason if monitor and monitor.failed.is_set()
                              else result.get("server_failure_reason") or "server failed during task")
                    _save_result(interrupted_log, {
                        "ts": time.time(), "task": task["id"], "attempt": attempt,
                        "reason": reason, "action": "task-retry-after-server-restart",
                        "partial_result": result,
                    })
                    restart_server(reason, task["id"])
                    attempt += 1
                    continue
                records.append(result)
                _save_result(results_path, result)
                done.add((task["id"], arm, 0))
                print(f"{idx}/{len(tasks)} {task['id']} resolved={result['resolved']} requests={result['requests']}", flush=True)
                if len(done) % 20 == 0:
                    try:
                        warmup(f"checkpoint-{len(done)}", len(done))
                    except Exception as exc:
                        if safety_watchdog.tripped.is_set() or "below 35 tok/s" in str(exc):
                            raise
                        restart_server(f"checkpoint warmup failure: {type(exc).__name__}: {exc}", task["id"])
                    last_warmup_count = len(done)
                break
            if safety_watchdog.tripped.is_set():
                break
    finally:
        pressure_guard.stop()
        pressure_guard.join(timeout=2)
        pressure_guard.sample_once()
        safety_watchdog.stop()
        safety_watchdog.join(timeout=2)
        safety_path = out_dir / f"safety-{body_arm}.json"
        safety_path.write_text(json.dumps({
                "mode": "stub-synthetic" if args.stub else "S6-real",
                "samples": stub_safety_samples["count"] if stub_safety_samples is not None else None,
                "reason": safety_watchdog.reason or None,
                "tripped": safety_watchdog.tripped.is_set(),
                "server_restarts": restart_count,
                "limits": safety_watchdog.limits,
                "peak": safety_watchdog.peak,
                "minimum": safety_watchdog.minimum,
                "system_pressure": {
                    "baseline": pressure_guard.baseline,
                    "peak_zram_pct": pressure_guard.peak_zram_pct,
                    "peak_psi_memory_full_avg10": pressure_guard.peak_psi_pct,
                    "samples": pressure_guard.samples,
                },
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    parser.add_argument("--arm-name", default="", help="stable resume key, e.g. tiel-no-plugins")
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--tmpfs-min-free-gib", type=float, default=1.5)
    parser.add_argument("--cpus", default="0,1", help="CPU affinity for OpenCode client and child tools only")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--server-port", type=int, default=8090)
    parser.add_argument("--server-argv-json", default="")
    parser.add_argument("--warmup-200", action="store_true")
    parser.add_argument("--max-server-restarts", type=int, default=3)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
