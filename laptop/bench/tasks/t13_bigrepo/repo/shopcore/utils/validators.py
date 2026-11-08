import re

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(value):
    return bool(EMAIL_RE.match(value or ""))


def require_positive(value, name="value"):
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
