"""Деньги: хранятся в копейках (int)."""
import re


def to_cents(s):
    """Строка с ценой -> копейки. Допустимы пробелы-разделители и запятая: "1 299,50"."""
    t = re.sub(r"\s", "", str(s)).replace(",", ".")
    m = re.fullmatch(r"(\d+)(?:\.(\d{1,2}))?", t)
    if not m:
        raise ValueError(f"bad price: {s!r}")
    return int(m.group(1)) * 100 + int((m.group(2) or "").ljust(2, "0") or 0)


def fmt(cents):
    """Копейки -> строка вида 1299.50"""
    sign = "-" if cents < 0 else ""
    a = abs(cents)
    return f"{sign}{a // 100}.{a % 100:02d}"
