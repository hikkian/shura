import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("external_behavior_metrics", HERE / "external" / "behavior_metrics.py")
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


class BehaviorMetricsTests(unittest.TestCase):
    def test_step_finish_length_is_counted_for_opencode_events(self):
        with tempfile.TemporaryDirectory(prefix="behavior-step-length-", dir="/dev/shm") as tmp:
            event_path = Path(tmp) / "task.agent.log"
            event_path.write_text(json.dumps({
                "type": "step_finish",
                "part": {"reason": "length", "messageID": "message-1", "tokens": {"output": 32768}},
            }))
            parsed = metrics.parse_agent_log(event_path)
            self.assertEqual(parsed["finish_length"], 1)
            self.assertEqual(parsed["assistant_finishes"], 0)

    def test_events_count_loops_errors_and_output_truncation(self):
        with tempfile.TemporaryDirectory(prefix="behavior-metrics-", dir="/dev/shm") as tmp:
            logs = Path(tmp) / "agent-logs"
            logs.mkdir()
            event_path = logs / "task-a__tiel__0.agent.log"
            events = [
                {"type": "message.part.updated", "properties": {"part": {
                    "type": "tool", "tool": "bash", "callID": "c1",
                    "state": {"status": "completed", "input": {"command": "pytest"}, "metadata": {"exit": 1}},
                }}},
                {"type": "message.part.updated", "properties": {"part": {
                    "type": "tool", "tool": "bash", "callID": "c2",
                    "state": {"status": "error", "input": {"command": "pytest"}, "error": "failed"},
                }}},
                {"type": "message.updated", "properties": {"info": {"role": "assistant", "finish": "length"}}},
            ]
            event_path.write_text("\n".join(json.dumps(e) for e in events))
            parsed = metrics.parse_agent_log(event_path)
            self.assertEqual(len(parsed["tool_calls"]), 2)
            self.assertEqual(parsed["tool_errors"], 1)
            self.assertEqual(parsed["finish_length"], 1)
            self.assertEqual(parsed["assistant_finishes"], 1)

    def test_summary_measures_early_failure_and_steps_on_solved_runs(self):
        with tempfile.TemporaryDirectory(prefix="behavior-summary-", dir="/dev/shm") as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "failed__tiel__0.agent.log").write_text(
                json.dumps({"type": "tool_use", "tool": "bash", "args": {"command": "pytest"}}) + "\n" +
                json.dumps({"type": "tool_use", "tool": "bash", "args": {"command": "pytest"}})
            )
            (logs / "passed__tiel__0.agent.log").write_text(json.dumps({"type": "tool_use", "tool": "edit", "args": {"filePath": "a.py"}}))
            results = root / "results.jsonl"
            results.write_text("\n".join([
                json.dumps({"task": "failed", "arm": "tiel", "rep": 0, "resolved": False, "requests": 3}),
                json.dumps({"task": "passed", "arm": "tiel", "rep": 0, "resolved": True, "requests": 5}),
            ]))
            row = metrics.summarize(results, logs)
            self.assertEqual(row["early_stop_failures_le_3_requests"], 1)
            self.assertEqual(row["consecutive_loop_repeats"], 1)
            self.assertEqual(row["mean_requests_to_postrun_solution"], 5)
            self.assertEqual(row["finish_reason_length_runs"], 0)
            self.assertEqual(row["agent_logs_missing"], 0)


if __name__ == "__main__":
    unittest.main()
