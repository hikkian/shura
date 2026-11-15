import json
import os
import tempfile
import unittest
from pathlib import Path

import harness
from qlev.arms import UnavailableArm, normalize_arm, proxy_extra_for_arm
from qlev.selection import c_tasks, dev_code_tasks, d_required_tasks, p_train_candidates, select_d, select_p


class ArmConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path("/dev/shm/QLEV0/test-units")
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="arm-config-", dir=str(self.tmp_root))
        self.root = Path(self.tmp.name)
        self.old_plugin_src = os.environ.get("AGENTIC_EVAL_QLEV_PLUGIN_SRC")
        os.environ["AGENTIC_EVAL_QLEV_PLUGIN_SRC"] = "/home/alisher/Projects/shuracode-integration/plugin"

    def tearDown(self):
        self.tmp.cleanup()
        if self.old_plugin_src is None:
            os.environ.pop("AGENTIC_EVAL_QLEV_PLUGIN_SRC", None)
        else:
            os.environ["AGENTIC_EVAL_QLEV_PLUGIN_SRC"] = self.old_plugin_src

    def config(self, arm, kind="implement"):
        path = self.root / arm
        work = path / "work"
        work.mkdir(parents=True)
        harness.write_project_rules(work)
        return json.loads(harness.write_config(path / "cfg", 19001, arm=arm, workdir=work,
                                               task={"id": "probe", "kind": kind}).joinpath("shuracode.json").read_text())

    def test_sampling_and_think_arms_change_only_their_settings(self):
        base = self.config("base")
        sample = self.config("S")
        think = self.config("T")
        base_model = base["provider"]["eval"]["models"]["m"]
        s_model = sample["provider"]["eval"]["models"]["m"]
        self.assertEqual(s_model["options"], {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5})
        self.assertEqual(base_model["options"]["temperature"], 0.6)
        self.assertEqual(sample["agent"]["build"]["temperature"], 1.0)
        self.assertEqual(think["provider"]["eval"]["models"]["m"]["settings"]["reasoningEffort"], "low")
        self.assertEqual(base_model["settings"]["reasoningEffort"], "medium")

    def test_preserve_thinking_is_a_proxy_request_field(self):
        self.config("P")
        extra = proxy_extra_for_arm("P")
        self.assertEqual(extra["chat_template_kwargs"], {"preserve_thinking": False})
        self.assertEqual((extra["temperature"], extra["top_p"], extra["top_k"], extra["presence_penalty"]),
                         (0.6, 0.95, 20, 0.0))

    def test_g_uses_the_product_gate_source_without_scratch_patch(self):
        data = self.config("G")
        source = harness.RELIABILITY_PLUGIN_SRC / "core.ts"
        copied = self.root / "G" / "plugin" / "reliability" / "core.ts"
        product_core = source.read_text(encoding="utf-8")
        self.assertIn("const VERIFICATION =", product_core)
        self.assertIn("node\\s+--test", product_core)
        self.assertIn("# fail [1-9]", product_core)
        self.assertEqual(copied.read_text(encoding="utf-8"), product_core)
        gate = next(entry for entry in data["plugin"] if isinstance(entry, list) and entry[0].endswith("completion-gate.ts"))
        self.assertTrue(gate[1]["completionGate"])

    def test_gate_is_enabled_and_plugins_are_copied_with_runtime_dependencies(self):
        cfg = self.root / "G"
        work = cfg / "work"; work.mkdir(parents=True)
        config_dir = harness.write_config(cfg / "cfg", 19001, arm="G", workdir=work, task={"kind": "implement"})
        data = json.loads((config_dir / "shuracode.json").read_text())
        gate = next(entry for entry in data["plugin"] if isinstance(entry, list) and entry[0].endswith("completion-gate.ts"))
        self.assertTrue(gate[1]["completionGate"])
        self.assertTrue((config_dir.parent / "node_modules" / "@opencode-ai" / "plugin").is_dir())
        self.assertTrue((config_dir.parent / "node_modules" / "zod").is_dir())
        self.assertTrue((config_dir / "node_modules").is_symlink())

    def test_contract_arm_has_rule_and_visible_tool(self):
        data = self.config("C")
        self.assertIn("Contract first", Path(data["instructions"][0]).read_text())
        tool = next(entry for entry in data["plugin"] if isinstance(entry, list) and entry[0].endswith("visible-tests.ts"))
        self.assertTrue(tool[1]["visibleTests"])
        self.assertTrue(Path(tool[0]).is_file())

    def test_diet_arm_is_enabled(self):
        data = self.config("D")
        tool = next(entry for entry in data["plugin"] if isinstance(entry, list) and entry[0].endswith("context-diet.ts"))
        self.assertTrue(tool[1]["contextDiet"])
        self.assertTrue(Path(tool[0]).is_file())

    def test_lsp_and_thinkbrake_are_clear_unavailable_stubs(self):
        for arm, source in (("L", "QLSP1"), ("TB", "THINK3")):
            with self.assertRaises(UnavailableArm) as error:
                normalize_arm(arm)
            self.assertIn(source, str(error.exception))

    def test_p_and_d_selectors_use_base_logs_without_test_tasks(self):
        results = self.root / "base-results.jsonl"
        dev = dev_code_tasks()
        train = p_train_candidates()
        rows = [
            {"arm": "base", "task": dev[0]["id"], "family": dev[0]["family"], "rep": 1, "user_messages": 2,
             "max_prompt_tokens": 21001},
            {"arm": "base", "task": dev[0]["id"], "family": dev[0]["family"], "rep": 2, "user_messages": 2,
             "max_prompt_tokens": 19000},
            {"arm": "base", "task": dev[1]["id"], "family": dev[1]["family"], "rep": 1, "user_messages": 1,
             "max_prompt_tokens": 8000},
            {"arm": "base", "task": train[0]["id"], "family": train[0]["family"], "rep": 1, "user_messages": 2,
             "max_prompt_tokens": 30000},
        ]
        results.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        p_result = select_p(results)
        self.assertIn(dev[0]["id"], {row["id"] for row in p_result["selected"]})
        self.assertIn(train[0]["id"], {row["id"] for row in p_result["selected"]})
        d_result = select_d(results)
        d_ids = {row["id"] for row in d_result["selected"]}
        self.assertIn(dev[0]["id"], d_ids)  # peak > 20k in one request, not the sum
        self.assertIn("js_event_journal-implement", d_ids)  # mandatory QLEV1 context case

    def test_harness_records_turn_count_reasoning_tokens_and_peak_prompt(self):
        events = self.root / "agent.jsonl"
        events.write_text("\n".join((
            json.dumps({"type": "user", "text": "first"}),
            json.dumps({"type": "token_usage", "tokens": {"reasoning": 31}}),
            json.dumps({"type": "user", "text": "gate follow-up"}),
        )), encoding="utf-8")
        counts = harness.count_events(events)
        self.assertEqual(counts["user_messages"], 2)
        self.assertEqual(counts["reasoning_tokens"], 31)
        proxy_log = self.root / "proxy.jsonl"
        proxy_log.write_text("\n".join((json.dumps({"usage": {"prompt_tokens": 12000}}),
                                          json.dumps({"usage": {"prompt_tokens": 21001}}))), encoding="utf-8")
        self.assertEqual(harness.request_prompt_peak(proxy_log), 21001)

    def test_dataset_subsets_are_dev_only_and_expected_size(self):
        dev = dev_code_tasks()
        c = c_tasks()
        required = d_required_tasks()
        candidates = p_train_candidates()
        self.assertEqual(len(dev), 56)
        self.assertEqual(len(c), 28)
        self.assertTrue(all(row["split"] == "dev" for row in dev + c))
        self.assertEqual({row["family"] for row in required}, {"js_event_journal", "js_headers", "js_deps_plan"})
        self.assertTrue(all(row["split"] == "train" for row in candidates))
        self.assertEqual(len(candidates), 8)


if __name__ == "__main__":
    unittest.main()
