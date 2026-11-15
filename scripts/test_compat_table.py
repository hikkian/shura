"""Tests for scripts/compat_table.py (PUB8). Offline: reports are written into a temporary folder.
Run: python3 -m unittest discover -s scripts -p 'test_compat_table.py'"""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compat_table as ct  # noqa: E402

PLAN = {"ok": True, "model": "tiel-coder-35b-a3b-mtp", "model_name": "Tiel-Coder 35B-A3B (MTP)", "quant": "Q4_K_XL",
        "backend_candidates": ["vulkan", "cpu"], "predicted_tok_s": {"low": 24.0, "mid": 34.0, "high": 46.0}}
VERIFIED = {"verified": True, "tok_s_short_prompt": 36.0, "predicted_tok_s_empty": 34.0, "backend": "vulkan"}
TABLE = """# Compatibility

<!-- compat:start -->
| Карта | Драйвер | ОС | Бэкенд | Модель | Скорость, tok/s | Статус | Кто проверил | Дата |
|---|---|---|---|---|---|---|---|---|
| RX 7800 XT (16 GB) | ROCm 7.1.1 | Fedora 44 | hip | Tiel | 74 | ⚠ | hikkian | 2026-10-20 |
<!-- compat:end -->

tail
"""


def report(plan=PLAN, measured=None, gpu="RX 9070 XT", driver="25.10", kind="shura-hardware-report"):
    data = {"kind": kind, "shura_version": "test", "hardware": {"os": {"family": "linux", "arch": "x86_64"},
            "gpus": [{"name": gpu, "driver": driver}]}, "plan": plan, "measured": measured or {}, "errors": []}
    return "# Shura hardware report\n\n```text\ncard\n```\n\n```json\n" + json.dumps(data, indent=2) + "\n```\n"


class Status(unittest.TestCase):
    def status_of(self, plan=PLAN, measured=None):
        return ct.status(ct.parse_report(report(plan, measured)))

    def test_no_fit_is_a_cross(self):
        self.assertEqual(self.status_of({**PLAN, "ok": False}), "✗")

    def test_a_failed_verification_is_a_cross(self):
        self.assertEqual(self.status_of(measured={"verified": False}), "✗")

    def test_a_speed_within_the_band_is_a_tick(self):
        self.assertEqual(self.status_of(measured=VERIFIED), "✓")

    def test_a_speed_far_from_the_estimate_is_a_warning(self):
        self.assertEqual(self.status_of(measured={**VERIFIED, "tok_s_short_prompt": 80.0}), "⚠")

    def test_a_plan_without_a_measurement_is_a_warning(self):
        self.assertEqual(self.status_of(measured={}), "⚠")


class Rows(unittest.TestCase):
    def test_the_row_is_built_from_the_report(self):
        row = ct.row_from_report(ct.parse_report(report(measured=VERIFIED)), "tester", "2026-10-20")
        self.assertEqual(row, ["RX 9070 XT", "25.10", "linux x86_64", "vulkan", "Tiel-Coder 35B-A3B (MTP) Q4_K_XL",
                               "36 (замер)", "✓", "tester", "2026-10-20"])

    def test_an_estimate_is_marked_as_one(self):
        row = ct.row_from_report(ct.parse_report(report()), "tester", "2026-10-20")
        self.assertEqual(row[5], "~34 (оценка)")
        self.assertEqual(row[3], "vulkan")

    def test_a_non_report_is_refused(self):
        with self.assertRaises(ValueError):
            ct.parse_report(report(kind="something-else"))


class Table(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "COMPATIBILITY.md"
        self.path.write_text(TABLE, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, text):
        report_path = Path(self.tmp.name) / "r.md"
        report_path.write_text(text, encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            return ct.main(["add", str(report_path), "--who", "tester", "--table", str(self.path), "--date", "2026-10-21"])

    def rows(self):
        return ct.parse_table(self.path.read_text(encoding="utf-8").split(ct.START, 1)[1].split(ct.END, 1)[0].splitlines())

    def test_a_new_machine_is_appended_and_the_seed_row_stays(self):
        self.assertEqual(self.add(report(measured=VERIFIED)), 0)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "RX 7800 XT (16 GB)")
        self.assertEqual(rows[1][0], "RX 9070 XT")

    def test_the_same_machine_again_replaces_its_row(self):
        self.add(report(measured=VERIFIED))
        self.add(report(measured={**VERIFIED, "tok_s_short_prompt": 35.0}))
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][5], "35 (замер)")

    def test_another_backend_on_the_same_card_is_its_own_row(self):
        self.add(report(measured=VERIFIED))
        self.add(report(plan={**PLAN, "backend_candidates": ["cpu"]}, measured={}))
        self.assertEqual(len(self.rows()), 3)

    def test_a_table_without_markers_is_refused(self):
        self.path.write_text("no markers here\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.add(report(measured=VERIFIED))


if __name__ == "__main__":
    unittest.main()
