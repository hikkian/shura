import csv
import os
import time

from .config import EXPORT_DIR, EXPORT_TIMEOUT_SECONDS, MAX_ROWS_PER_EXPORT


class ExportError(Exception):
    pass


def export_csv(report_name, rows, header):
    """Выгружает отчёт в CSV и возвращает путь к файлу."""
    if len(rows) > MAX_ROWS_PER_EXPORT:
        # TODO: разбивать большие отчёты на части
        raise ExportError("too many rows")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = os.path.join(EXPORT_DIR, f"{report_name}.csv")
    started = time.monotonic()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, row in enumerate(rows):
            w.writerow(row)
            if time.monotonic() - started > EXPORT_TIMEOUT_SECONDS:
                raise ExportError(f"export timed out after {i} rows")
    return path
