from datetime import datetime, timedelta, timezone


def utcnow():
    return datetime.now(timezone.utc)


def in_seconds(seconds):
    return utcnow() + timedelta(seconds=seconds)
