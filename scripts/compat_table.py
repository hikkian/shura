"""Compatibility table of the pilot programme (PUB8): one row per tested machine, built from a `shura report` file.

    python3 scripts/compat_table.py show shura-hardware-report.md
    python3 scripts/compat_table.py add shura-hardware-report.md --who "hikkian" [--table docs/COMPATIBILITY.md] [--date 2026-10-20]

The row is computed from the report: the GPU, its driver, the OS, the backend, the model, the speed and the status. Only
`--who` and the date are typed in. A machine tested again (same GPU, OS, backend and model) replaces its row.
Standard library only. The report's machine-readable block is written by installer/universal/report.py (kind
"shura-hardware-report"); the status rule is below and is tested in scripts/test_compat_table.py.
"""
import argparse
import datetime
import json
import re
import sys
from pathlib import Path

START, END = "<!-- compat:start -->", "<!-- compat:end -->"
COLUMNS = ["Карта", "Драйвер", "ОС", "Бэкенд", "Модель", "Скорость, tok/s", "Статус", "Кто проверил", "Дата"]
TOLERANCE = 0.25        # measured speed within this share of the estimate counts as ✓; further away is ⚠
JSON_BLOCK = re.compile(r"```json\n(.*?)\n```", re.S)


def parse_report(text, name="report"):
    match = JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"{name}: no machine-readable block; is it a `shura report` file?")
    data = json.loads(match.group(1))
    if data.get("kind") != "shura-hardware-report":
        raise ValueError(f"{name}: not a Shura hardware report (kind={data.get('kind')!r})")
    return data


def read_report(path):
    return parse_report(Path(path).read_text(encoding="utf-8"), str(path))


def status(data):
    """✗ when the plan has no fit or the install did not verify; ✓ when the server ran and its speed is within TOLERANCE of
    the estimate; ⚠ otherwise, including a report with no measurement at all."""
    plan, measured = data.get("plan", {}), data.get("measured") or {}
    if not plan.get("ok") or measured.get("verified") is False:
        return "✗"
    if measured.get("verified") is True:
        real, predicted = measured.get("tok_s_short_prompt"), measured.get("predicted_tok_s_empty")
        if real and predicted and abs(real / predicted - 1) <= TOLERANCE:
            return "✓"
    return "⚠"


def speed(data):
    plan, measured = data.get("plan", {}), data.get("measured") or {}
    real = measured.get("tok_s_short_prompt")
    if measured.get("verified") and real:
        return f"{real:.0f} (замер)"
    mid = (plan.get("predicted_tok_s") or {}).get("mid")
    return f"~{mid:.0f} (оценка)" if mid else "—"


def _cell(text):
    return str(text).replace("|", "/").replace("\n", " ").strip() or "—"


def row_from_report(data, who, date):
    gpu = (data.get("hardware", {}).get("gpus") or [{}])[0]
    os_info = data.get("hardware", {}).get("os", {})
    plan = data.get("plan", {})
    backend = (data.get("measured") or {}).get("backend") or (plan.get("backend_candidates") or ["—"])[0]
    cells = [gpu.get("name", "—"), gpu.get("driver") or "—", f"{os_info.get('family', '—')} {os_info.get('arch', '')}".strip(),
             backend, f"{plan.get('model_name', plan.get('model', '—'))} {plan.get('quant', '')}".strip(),
             speed(data), status(data), who, date]
    return [_cell(c) for c in cells]


def parse_table(lines):
    """Rows of the markdown table between the markers, without the header and the separator."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: ") or line.startswith("| Карта"):
            continue                                 # not a row: a separator line or the header
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def render(rows):
    head = "| " + " | ".join(COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    return [head, sep] + ["| " + " | ".join(r) + " |" for r in rows]


def add(table_path, row):
    """Insert the row, or replace the row of the same machine (GPU, OS, backend, model). Other rows stay as they are."""
    path = Path(table_path)
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError(f"{table_path}: the table markers are missing")
    before, rest = text.split(START, 1)
    body, after = rest.split(END, 1)
    rows = parse_table(body.splitlines())
    key = (row[0], row[2], row[3], row[4])
    for i, old in enumerate(rows):
        if (old[0], old[2], old[3], old[4]) == key:
            rows[i] = row
            break
    else:
        rows.append(row)
    path.write_text(before + START + "\n" + "\n".join(render(rows)) + "\n" + END + after, encoding="utf-8")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    show = sub.add_parser("show", help="print the row a report would give, without writing anything")
    show.add_argument("report")
    add_p = sub.add_parser("add", help="write the row into the compatibility table")
    add_p.add_argument("report")
    add_p.add_argument("--who", required=True, help="who ran it (a nickname, not a full name)")
    add_p.add_argument("--table", default="docs/COMPATIBILITY.md")
    add_p.add_argument("--date", default=datetime.date.today().isoformat())
    args = ap.parse_args(argv)
    data = read_report(args.report)
    row = row_from_report(data, args.who if args.cmd == "add" else "—", args.date if args.cmd == "add" else "—")
    if args.cmd == "show":
        print("\n".join(render([row])))
        return 0
    add(args.table, row)
    print(f"row written to {args.table}: {row[0]} / {row[3]} / {row[4]} -> {row[6]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
