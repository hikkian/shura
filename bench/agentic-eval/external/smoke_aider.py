"""One-task offline smoke for the official Aider benchmark CLI using stub_openai_server."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

from stub_openai_server import Handler, ThreadingHTTPServer


def smoke(aider_repo: Path, polyglot: Path, work_root: Path, model: str = "openai/shura-tiel",
          temperature: float = 0.6, presence_penalty: float | None = None) -> dict:
    work_root.mkdir(parents=True, exist_ok=True)
    bench_root = work_root / "benchmark-root"
    bench_root.mkdir(exist_ok=True)
    link = bench_root / "polyglot-benchmark"
    if link.is_symlink() or link.exists():
        link.unlink() if link.is_symlink() else None
    link.symlink_to(polyglot, target_is_directory=True)
    log = work_root / "stub-calls.jsonl"
    Handler.profile = "aider"
    Handler.log_path = log
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings = work_root / "model-settings.yml"
    extra = {"temperature": temperature, "top_p": 0.95, "top_k": 20, "max_tokens": 32768}
    if presence_penalty is not None:
        extra["presence_penalty"] = presence_penalty
    settings.write_text(f"""- name: {model}
  edit_format: whole
  use_temperature: true
  streaming: false
  extra_params: {json.dumps(extra)}
""", encoding="utf-8")
    metadata = work_root / "model-metadata.json"
    metadata.write_text(json.dumps({model: {
        "max_tokens": 32768, "max_input_tokens": 32768, "max_output_tokens": 32768,
        "input_cost_per_token": 0, "output_cost_per_token": 0,
        "litellm_provider": "openai", "mode": "chat",
    }}), encoding="utf-8")
    env = {
        **os.environ,
        "AIDER_DOCKER": "1",  # the official safety gate; this process is inside bubblewrap --unshare-net
        "AIDER_BENCHMARK_DIR": str(bench_root),
        "AIDER_MODEL_METADATA_FILE": str(metadata),
        "OPENAI_API_KEY": "stub-only",
        "OPENAI_API_BASE": f"http://127.0.0.1:{server.server_port}/v1",
        "NO_PROXY": "*", "no_proxy": "*",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    command = [
        sys.executable, str(aider_repo / "benchmark" / "benchmark.py"), "stub-smoke",
        "--model", model, "--edit-format", "whole",
        "--languages", "python", "--keywords", "proverb", "--num-tests", "1",
        "--tries", "1", "--threads", "1", "--read-model-settings", str(settings),
        "--exercises-dir", "polyglot-benchmark", "--new",
    ]
    try:
        result = subprocess.run(command, cwd=aider_repo, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120, check=False)
    finally:
        server.shutdown()
        server.server_close()
    runs = sorted(bench_root.glob("*stub-smoke*/python/exercises/practice/proverb/.aider.results.json"))
    rec = json.loads(runs[0].read_text()) if runs else None
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return {
        "command": command,
        "exit_code": result.returncode,
        "result_record_found": rec is not None,
        "resolved": bool(rec and any(rec.get("tests_outcomes", []))),
        "requests_seen_by_stub": len(calls),
        "request_parameters": [{k: row.get(k) for k in ("temperature", "top_p", "top_k", "max_tokens")} for row in calls],
        "stdout_tail": result.stdout[-6000:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aider-repo", type=Path, required=True)
    parser.add_argument("--polyglot", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--model", default="openai/shura-tiel")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--presence-penalty", type=float)
    args = parser.parse_args()
    result = smoke(args.aider_repo, args.polyglot, args.work_root, args.model, args.temperature, args.presence_penalty)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["exit_code"] or not result["result_record_found"] or not result["requests_seen_by_stub"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
