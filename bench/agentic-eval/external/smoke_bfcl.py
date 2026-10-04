"""BFCL CLI transport/evaluation smoke using two public sample IDs and a local stub."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from stub_openai_server import Handler, ThreadingHTTPServer


def smoke(venv: Path, site_packages: Path, work_root: Path, model: str = "shura-tiel",
          temperature: float = 0.6, presence_penalty: float | None = None) -> dict:
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / ".env").write_text("", encoding="utf-8")
    ids = {"simple_python": ["simple_python_0"], "multi_turn_base": ["multi_turn_base_0"]}
    (work_root / "test_case_ids_to_generate.json").write_text(json.dumps(ids), encoding="utf-8")
    call_log = work_root / "stub-calls.jsonl"
    Handler.profile = "bfcl"
    Handler.log_path = call_log
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {
        **os.environ,
        "BFCL_PROJECT_ROOT": str(work_root),
        "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
        "OPENAI_API_KEY": "stub-only",
        "SHURA_TOP_P": "0.95",
        "SHURA_TOP_K": "20",
        "SHURA_MAX_TOKENS": "32768",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_PROXY": "*", "no_proxy": "*",
    }
    if presence_penalty is not None:
        env["SHURA_PRESENCE_PENALTY"] = str(presence_penalty)
    generate = [str(venv / "bin/bfcl"), "generate", "--model", model, "--run-ids",
                "--skip-server-setup", "--num-threads", "1", "--temperature", str(temperature), "--include-input-log"]
    evaluate = [str(venv / "bin/bfcl"), "evaluate", "--model", model,
                "--test-category", "simple_python,multi_turn_base", "--partial-eval"]
    try:
        gen = subprocess.run(generate, cwd=work_root, env=env, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=120, check=False)
        ev = subprocess.run(evaluate, cwd=work_root, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=120, check=False) if gen.returncode == 0 else None
    finally:
        server.shutdown()
        server.server_close()
    calls = [json.loads(line) for line in call_log.read_text().splitlines()] if call_log.exists() else []
    result_files = sorted((work_root / "result" / model).rglob("*.json")) if (work_root / "result" / model).exists() else []
    scores = sorted((work_root / "score" / model).rglob("*.json")) if (work_root / "score" / model).exists() else []
    return {
        "generate_exit": gen.returncode,
        "evaluate_exit": ev.returncode if ev else None,
        "result_files": [p.name for p in result_files],
        "score_files": [p.name for p in scores],
        "request_count": len(calls),
        "request_parameters": [{k: row.get(k) for k in ("temperature", "top_p", "top_k", "max_tokens", "presence_penalty")} for row in calls],
        "generate_output_tail": gen.stdout[-4000:],
        "evaluate_output_tail": ev.stdout[-4000:] if ev else "not run",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--model", default="shura-tiel")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--presence-penalty", type=float)
    args = parser.parse_args()
    result = smoke(args.venv, args.site_packages, args.work_root, args.model, args.temperature, args.presence_penalty)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["generate_exit"] or result["evaluate_exit"] or not result["request_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
