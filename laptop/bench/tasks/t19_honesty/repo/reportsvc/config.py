import os

DB_URL = os.environ.get("REPORTS_DB_URL", "sqlite:///reports.db")
EXPORT_DIR = os.environ.get("REPORTS_EXPORT_DIR", "out")
EXPORT_TIMEOUT_SECONDS = 30
MAX_ROWS_PER_EXPORT = 100_000
