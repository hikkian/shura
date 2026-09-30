"""`shura check` / `shura report`: the card, the anonymous report and the exit codes. Offline; uses fixture profiles."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
from universal import cli, planner, report  # noqa: E402

HWDIR = ROOT / "tests/fixtures/hardware"
REAL = json.loads((ROOT / "catalog/models.json").read_text())


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class Card(unittest.TestCase):
    def test_check_prints_the_plan_for_a_profile(self):
        code, out, _ = run("check", "--profile", str(HWDIR / "ref_rtx4070s_12g_32g.json"))
        self.assertEqual(code, 0)
        for text in ("Tiel-Coder 35B-A3B (MTP)", "IQ4_XS", "hybrid", "cuda, vulkan, cpu", "confidence"):
            self.assertIn(text, out)

    def test_check_json_is_the_plan(self):
        code, out, _ = run("check", "--json", "--profile", str(HWDIR / "rx7900xtx_24g_64g.json"))
        plan = json.loads(out)
        self.assertEqual((code, plan["ok"], plan["backend_candidates"][:2]), (0, True, ["rocm", "vulkan"]))

    def test_nothing_fits_exits_with_1_and_says_why(self):
        code, out, _ = run("check", "--profile", str(HWDIR / "laptop_cpu_16g.json"))
        self.assertEqual(code, 1)
        self.assertIn("no model fits", out)
        self.assertIn("RAM", out)

    def test_invalid_profile_exits_with_2_and_lists_problems(self):
        with tempfile.TemporaryDirectory() as d:                 # not NamedTemporaryFile: Windows cannot reopen an open file
            path = Path(d) / "bad.json"
            path.write_text(json.dumps({"schema": 1, "os": {"family": "amiga"}}))
            code, _, err = run("check", "--profile", str(path))
        self.assertEqual(code, 2)
        self.assertIn("os.family", err)

    def test_missing_file_is_a_clean_error(self):
        code, _, err = run("check", "--profile", "/nonexistent.json")
        self.assertEqual(code, 2)
        self.assertIn("shura:", err)


class Report(unittest.TestCase):
    def hw_plan(self):
        hw = json.loads((HWDIR / "ref_rtx4070s_12g_32g.json").read_text())
        hw["detection_notes"] = ["could not read /home/alice/.config/x on host box-7"]
        return hw, planner.plan(hw, REAL)

    def test_scrub_removes_home_user_and_host(self):
        t = report.scrub("/home/alice/x failed for alice on box-7", home="/home/alice", user="alice", host="box-7")
        self.assertNotIn("alice", t)
        self.assertNotIn("box-7", t)
        self.assertIn("~/x", t)

    def test_report_is_allow_listed_and_scrubbed(self):
        hw, plan = self.hw_plan()
        data, md = report.make_report(hw, plan, errors=["crash in /home/alice/.local/share/shura for alice"],
                                      scrub_args={"home": "/home/alice", "user": "alice", "host": "box-7"},
                                      measured={"decode_tok_s": 27.4})
        text = json.dumps(data) + md
        for secret in ("alice", "box-7", "/home"):
            self.assertNotIn(secret, text)
        self.assertEqual(data["kind"], "shura-hardware-report")
        self.assertNotIn("disk", data["hardware"])
        self.assertIn("decode_tok_s", md)
        self.assertEqual(set(data), {"kind", "shura_version", "hardware", "plan", "measured", "errors"})

    def test_report_command_writes_a_file_and_sends_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            out_file = Path(d) / "r.md"
            code, out, _ = run("report", "--profile", str(HWDIR / "arc_a770_16g_32g.json"), "-o", str(out_file))
            self.assertEqual(code, 0)
            self.assertIn("Nothing was sent", out)
            self.assertIn("Arc A770", out_file.read_text())


if __name__ == "__main__":
    unittest.main()
