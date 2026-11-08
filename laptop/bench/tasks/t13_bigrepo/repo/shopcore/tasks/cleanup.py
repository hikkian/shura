from ..models.order import STATUS_PAYMENT_FAILED
from ..repositories.order_repo import order_repo


def stale_failed_orders():
    return order_repo.find_by_status(STATUS_PAYMENT_FAILED)
