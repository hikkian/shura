"""No-GPU stage-1 proof: exercise each QLEV arm through the real serve harness against a scripted fake HTTP model."""
import argparse
import copy
import http.server
import json
import os
import shutil
import threading
import time
from pathlib import Path

import harness
import proxy
from qlev.arms import ARM_IDS, proxy_extra_for_arm

PROBE_ARMS = ("base", "S", "P", "T", "G", "C", "D")


class FakeModel(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, mode):
        self.mode = mode
        self.requests = []
        self.main_calls = 0
        self.gate_seen = False
        super().__init__(address, FakeHandler)


class FakeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        server = self.server
        server.requests.append(copy.deepcopy(body))
        messages = body.get("messages") or []
        server.gate_seen = server.gate_seen or any("[Shura completion gate]" in str(m.get("content", "")) for m in messages)
        tools = body.get("tools") or []
        main = bool(tools)
        calls = []
        text = "probe complete"
        if main:
            server.main_calls += 1
            if server.mode in ("G", "G-green") and server.main_calls == 1:
                names = {t.get("function", t).get("name") for t in tools if isinstance(t, dict)}
                if "edit" in names:
                    calls = [{"name": "edit", "arguments": {"filePath": "probe.txt", "oldString": "before", "newString": "after"}}]
            elif server.mode == "G-green" and server.main_calls == 2:
                names = {t.get("function", t).get("name") for t in tools if isinstance(t, dict)}
                if "bash" in names:
                    calls = [{"name": "bash", "arguments": {"command": "python3 -m unittest test_probe -v"}}]
            elif server.mode == "D" and server.main_calls == 1:
                names = {t.get("function", t).get("name") for t in tools if isinstance(t, dict)}
                if "read" in names:
                    calls = [{"name": "read", "arguments": {"filePath": f"large-{i}.txt"}} for i in range(30)]
            if server.mode == "G" and server.gate_seen:
                text = "gate follow-up received; no test command was called"
            elif server.mode == "G-green" and server.main_calls >= 3:
                text = "project tests passed"
        if not body.get("stream"):
            message = {"role": "assistant", "content": text}
            if calls:
                message["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                                          "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])} }
                                         for i, c in enumerate(calls)]
            self._json({"id": "qlev-probe", "object": "chat.completion", "created": int(time.time()), "model": "fake",
                        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(delta, finish=None, extra=None):
            obj = {"id": "qlev-probe", "object": "chat.completion.chunk", "created": int(time.time()), "model": "fake",
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if extra:
                obj.update(extra)
            data = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        try:
            chunk({"role": "assistant", "content": ""})
            if calls:
                for i, call in enumerate(calls):
                    chunk({"tool_calls": [{"index": i, "id": f"call_{i}", "type": "function",
                                           "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}]})
            else:
                chunk({"content": text})
            chunk({}, "tool_calls" if calls else "stop", {"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
            end = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(end):x}\r\n".encode() + end + b"\r\n0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _system_text(body):
    return "\n".join(str(m.get("content", "")) for m in body.get("messages", []) if m.get("role") == "system")


def run_arm(arm, root, gate_green=False):
    label = "G-green" if arm == "G" and gate_green else arm
    arm_dir = Path(root) / label
    work = arm_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    harness.write_project_rules(work)
    (work / "probe.txt").write_text("before\n", encoding="utf-8")
    if gate_green:
        (work / "test_probe.py").write_text(
            "import unittest\n\nclass ProbeTest(unittest.TestCase):\n    def test_project_verification(self):\n        self.assertEqual(1 + 1, 2)\n\nif __name__ == '__main__':\n    unittest.main()\n",
            encoding="utf-8")
    for index in range(30):
        content = "\n".join(f"{index:02d}-{line:04d}-" + ("0123456789abcdef" * 4) for line in range(100)) + "\n"
        (work / f"large-{index}.txt").write_text(content, encoding="utf-8")
    fake = FakeModel(("127.0.0.1", 0), label)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    proxy_log = arm_dir / "proxy.jsonl"
    gateway = proxy.serve(0, f"http://127.0.0.1:{fake.server_port}", str(proxy_log), extra=proxy_extra_for_arm(arm))
    threading.Thread(target=gateway.serve_forever, daemon=True).start()
    try:
        cfg = harness.write_config(arm_dir / "cfg", gateway.server_port, arm=arm, workdir=work,
                                   task={"id": "qlev-probe", "kind": "implement"})
        if arm == "C" and os.environ.get("QLEV_DEBUG_PLUGIN") == "1":
            plugin_file = Path(cfg).parent / "plugin" / "visible-tests.ts"
            plugin_text = plugin_file.read_text(encoding="utf-8")
            marker = plugin_file.parent / "visible-plugin-debug.json"
            plugin_text = plugin_text.replace('import type { PluginModule } from "@opencode-ai/plugin"',
                                              'import type { PluginModule } from "@opencode-ai/plugin"\nimport { writeFileSync } from "node:fs"\nwriteFileSync(' + json.dumps(str(marker)) + ', "loaded")', 1)
            plugin_text = plugin_text.replace("server: async (input, options) => {", 'server: async (input, options) => {\n    writeFileSync(' + json.dumps(str(marker)) + ', JSON.stringify(options))', 1)
            plugin_file.write_text(plugin_text, encoding="utf-8")
        log_path = arm_dir / "agent.log"
        if arm == "G" and not gate_green:
            prompt = "Change probe.txt from before to after. Do not run tests; stop after the edit."
        elif gate_green:
            prompt = "Change probe.txt from before to after, then run the project unittest suite."
        elif arm == "D":
            prompt = "Read large.txt repeatedly and summarize it."
        else:
            prompt = "Say probe complete."
        result = harness.run_agent_serve(work, prompt, cfg, arm_dir / "data", 150, log_path,
                                         settle_s=0.7, variant="low" if arm == "T" else None)
        requests = fake.requests
        main_requests = [r for r in requests if r.get("tools")]
        first = main_requests[0] if main_requests else {}
        tool_names = sorted({t.get("function", t).get("name") for t in first.get("tools", []) if isinstance(t, dict)})
        raw_path = arm_dir / "raw-requests.jsonl"
        raw_path.write_text("".join(json.dumps(body, ensure_ascii=False) + "\n" for body in fake.requests), encoding="utf-8")
        event_counts = harness.count_events(log_path)
        agent_events = [json.loads(line) for line in log_path.read_text(errors="replace").splitlines() if line.startswith("{")]
        bash_commands = [((row.get("part") or {}).get("state") or {}).get("input", {}).get("command")
                         for row in agent_events if row.get("type") == "tool_use" and (row.get("part") or {}).get("tool") == "bash"]
        evidence = {"arm": label, "engine_exit": result[0], "timed_out": result[1], "seconds": round(result[2], 2),
                    "main_requests": len(main_requests), "tool_names": tool_names,
                    "sampling": {k: first.get(k) for k in ("temperature", "top_p", "top_k", "presence_penalty") if k in first},
                    "reasoning_effort": first.get("reasoning_effort"),
                    "chat_template_kwargs": first.get("chat_template_kwargs"),
                    "contract_first": "Contract first" in _system_text(first),
                    "visible_tests": "visible_tests" in tool_names,
                    "completion_gate_message": any("[Shura completion gate]" in str(m.get("content", ""))
                                                    for r in main_requests for m in r.get("messages", [])),
                    "user_messages": event_counts["user_messages"], "reasoning_tokens": event_counts["reasoning_tokens"],
                    "bash_commands": bash_commands, "edited_file": (work / "probe.txt").read_text(encoding="utf-8").strip(),
                    "masked_tool_messages": sum(1 for r in main_requests for m in r.get("messages", [])
                                                 if m.get("role") == "tool" and "[tool output elided:" in str(m.get("content", ""))),
                    "mask_probe_threshold": 60000 if arm == "D" else None,
                    "plugin_paths_inside_scratch": all(str(arm_dir) in (p[0] if isinstance(p, list) else p)
                                                       for p in json.loads((Path(cfg) / "shuracode.json").read_text()).get("plugin", [])),
                    "runtime_zod_copied": (Path(cfg).parent / "node_modules" / "zod").is_dir(),
                    "fable3_gate_product_source": "const VERIFICATION =" in
                        (harness.RELIABILITY_PLUGIN_SRC / "core.ts").read_text(encoding="utf-8"),
                    "fable3_gate_copied_from_product": (Path(cfg).parent / "plugin" / "reliability" / "core.ts").read_text(encoding="utf-8") ==
                        (harness.RELIABILITY_PLUGIN_SRC / "core.ts").read_text(encoding="utf-8"),
                    "raw_requests": str(raw_path)}
        if os.environ.get("QLEV_DEBUG_PLUGIN") == "1":
            engine_log = arm_dir / "home" / ".local" / "share" / "opencode" / "log" / "opencode.log"
            evidence["plugin_debug"] = [line for line in engine_log.read_text(errors="replace").splitlines()
                                        if "failed to load plugin" in line] if engine_log.exists() else []
            marker = arm_dir / "plugin" / "visible-plugin-debug.json"
            evidence["visible_plugin_debug"] = marker.read_text(errors="replace") if marker.exists() else None
        return evidence
    finally:
        gateway.shutdown(); gateway.server_close()
        fake.shutdown(); fake.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", default="/dev/shm/QLEV0/probes")
    parser.add_argument("--out", help="write sanitized evidence JSON")
    parser.add_argument("--raw-dir", help="copy raw fake HTTP requests per arm to this directory")
    parser.add_argument("--arms", nargs="*", choices=PROBE_ARMS, help="run only selected fake-server probes")
    parser.add_argument("--gate-green", action="store_true", help="also prove G after a successful project unittest run")
    args = parser.parse_args(argv)
    root = Path(args.scratch); root.mkdir(parents=True, exist_ok=True)
    selected = args.arms or PROBE_ARMS
    evidence = [run_arm(arm, root) for arm in selected]
    if args.gate_green and "G" in selected:
        evidence.append(run_arm("G", root, gate_green=True))
    if args.raw_dir:
        raw_dir = Path(args.raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
        for row in evidence:
            shutil.copy2(row["raw_requests"], raw_dir / (row["arm"] + ".jsonl"))
    if args.out:
        path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in evidence:
        print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
