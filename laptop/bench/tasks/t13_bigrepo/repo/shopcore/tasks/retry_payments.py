from ..utils.dates import in_seconds

SCHEDULED = []


def schedule_retry(order_id, delay_seconds):
    SCHEDULED.append({"order_id": order_id, "run_at": in_seconds(delay_seconds)})
