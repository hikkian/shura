"""Behavior metrics for ordinary v2/Polyglot harness JSONL and agent event logs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(value)


def _walk(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk(value)


def parse_agent_log(path: Path) -> dict[str, Any]:
    tool_calls = []
    tool_errors = 0
    length_finishes = 0
    assistant_finishes = 0
    seen_calls = set()
    seen_errors = set()
    seen_length_messages = set()
    if not path.is_file():
        return {"tool_calls": [], "tool_errors": 0, "finish_length": 0, "assistant_finishes": 0, "log_missing": True}
    for line in path.read_text(errors="replace").splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        # OpenCode message.updated and message.part.updated events carry finish metadata.
        for item in _walk(event):
            info = item.get("info") if isinstance(item.get("info"), dict) else item
            if info.get("role") == "assistant" and info.get("finish") is not None:
                message_id = str(info.get("id") or item.get("messageID") or id(info))
                if message_id not in seen_length_messages:
                    seen_length_messages.add(message_id)
                    assistant_finishes += 1
                    if str(info.get("finish")).lower() in {"length", "max_tokens", "max_output_tokens"}:
                        length_finishes += 1
            reason = item.get("finish_reason") or item.get("finishReason")
            if reason is not None and str(reason).lower() == "length":
                message_id = str(item.get("messageID") or item.get("id") or id(item))
                if message_id not in seen_length_messages:
                    seen_length_messages.add(message_id)
                    length_finishes += 1
            # OpenCode 1.18.32 writes model output truncation on the enclosing
            # step_finish event, not always as message.finish/finish_reason.
            if item.get("type") == "step_finish":
                part = item.get("part") if isinstance(item.get("part"), dict) else {}
                step_reason = part.get("reason")
                if str(step_reason).lower() in {"length", "max_tokens", "max_output_tokens"}:
                    message_id = str(part.get("messageID") or part.get("id") or item.get("id") or id(item))
                    if message_id not in seen_length_messages:
                        seen_length_messages.add(message_id)
                        length_finishes += 1
            status = item.get("status")
            state = item.get("state") if isinstance(item.get("state"), dict) else {}
            failed = status == "error" or state.get("status") == "error" or item.get("error") not in (None, False, "")
            call_id = str(item.get("callID") or item.get("id") or state.get("callID") or "")
            if failed:
                if item.get("type") in {"tool", "tool_use", "tool_result"} or "tool" in item:
                    if not call_id or call_id not in seen_errors:
                        tool_errors += 1
                        if call_id:
                            seen_errors.add(call_id)
            if item.get("type") == "tool" and item.get("tool"):
                tool = item.get("tool")
                args = state.get("input") or item.get("args") or item.get("input") or {}
                if not call_id or call_id not in seen_calls:
                    tool_calls.append({"tool": str(tool), "args": args, "call_id": call_id})
                    if call_id:
                        seen_calls.add(call_id)
            elif item.get("type") in {"tool_use", "tool_call"}:
                tool = item.get("tool") or item.get("name")
                if tool and (not call_id or call_id not in seen_calls):
                    tool_calls.append({
                        "tool": str(tool),
                        "args": item.get("args") or item.get("input") or item.get("arguments") or {},
                        "call_id": call_id,
                    })
                    if call_id:
                        seen_calls.add(call_id)
    return {"tool_calls": tool_calls, "tool_errors": tool_errors,
            "finish_length": length_finishes, "assistant_finishes": assistant_finishes, "log_missing": False}


def summarize(results_path: Path, agent_log_dir: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    aggregate = Counter()
    resolved_requests = []
    resolved_tools = []
    repeats = []
    per_arm = defaultdict(Counter)
    for row in rows:
        task_id = row.get("task", row.get("id", "unknown"))
        log_path = Path(row.get("agent_log") or
                        agent_log_dir / f"{task_id}__{row.get('arm','stub')}__{row.get('rep',0)}.agent.log")
        parsed = parse_agent_log(log_path)
        requests = int(row.get("requests", row.get("requests_count", 0)) or 0)
        resolved = bool(row.get("resolved", row.get("passed", False)))
        arm = str(row.get("arm", "unknown"))
        calls = parsed["tool_calls"]
        loop_repeats = 0
        previous = None
        for call in calls:
            signature = (call["tool"], _canonical(call["args"]))
            if signature == previous:
                loop_repeats += 1
            previous = signature
        # Existing harness result records already summarize some tool errors; avoid double counting the same event.
        tool_errors = max(int(row.get("tool_errors", 0) or 0), parsed["tool_errors"])
        aggregate["runs"] += 1
        aggregate["resolved"] += int(resolved)
        aggregate["early_stops"] += int((not resolved) and requests <= 3)
        aggregate["requests"] += requests
        aggregate["tool_calls"] += len(calls)
        aggregate["loop_repeats"] += loop_repeats
        aggregate["tool_errors"] += tool_errors
        aggregate["finish_length_runs"] += int(parsed["finish_length"] > 0)
        aggregate["finish_length_events"] += parsed["finish_length"]
        aggregate["agent_logs_missing"] += int(parsed["log_missing"])
        if resolved:
            resolved_requests.append(requests)
            resolved_tools.append(len(calls))
        repeats.append(loop_repeats)
        bucket = per_arm[arm]
        bucket["runs"] += 1
        bucket["resolved"] += int(resolved)
        bucket["early_stops"] += int((not resolved) and requests <= 3)
        bucket["requests"] += requests
        bucket["tool_errors"] += tool_errors
        bucket["loops"] += loop_repeats
        bucket["finish_length_runs"] += int(parsed["finish_length"] > 0)
    n = aggregate["runs"] or 1
    output = {
        "runs": aggregate["runs"],
        "resolved_rate_pct": round(100 * aggregate["resolved"] / n, 2),
        "early_stop_failures_le_3_requests": aggregate["early_stops"],
        "early_stop_failure_rate_pct": round(100 * aggregate["early_stops"] / max(1, aggregate["runs"] - aggregate["resolved"]), 2),
        "tool_calls": aggregate["tool_calls"],
        "consecutive_loop_repeats": aggregate["loop_repeats"],
        "tool_errors": aggregate["tool_errors"],
        "mean_requests_to_postrun_solution": round(sum(resolved_requests) / len(resolved_requests), 2) if resolved_requests else None,
        "median_requests_to_postrun_solution": median(resolved_requests) if resolved_requests else None,
        "mean_tool_calls_on_solved_runs": round(sum(resolved_tools) / len(resolved_tools), 2) if resolved_tools else None,
        "finish_reason_length_runs": aggregate["finish_length_runs"],
        "finish_reason_length_pct": round(100 * aggregate["finish_length_runs"] / n, 2),
        "finish_reason_length_events": aggregate["finish_length_events"],
        "agent_logs_missing": aggregate["agent_logs_missing"],
        "per_arm": {},
    }
    for arm, bucket in sorted(per_arm.items()):
        count = bucket["runs"] or 1
        output["per_arm"][arm] = {
            "runs": bucket["runs"],
            "resolved_rate_pct": round(100 * bucket["resolved"] / count, 2),
            "early_stop_failures_le_3_requests": bucket["early_stops"],
            "mean_requests": round(bucket["requests"] / count, 2),
            "tool_errors": bucket["tool_errors"],
            "consecutive_loop_repeats": bucket["loops"],
            "finish_reason_length_runs": bucket["finish_length_runs"],
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path, help="harness results.jsonl")
    parser.add_argument("--agent-logs", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = summarize(args.results, args.agent_logs)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
