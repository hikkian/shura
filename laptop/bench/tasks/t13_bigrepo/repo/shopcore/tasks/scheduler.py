from .retry_payments import SCHEDULED


def due_jobs(now):
    return [j for j in SCHEDULED if j["run_at"] <= now]
