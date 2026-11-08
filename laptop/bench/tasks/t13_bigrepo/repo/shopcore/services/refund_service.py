from ..models.order import STATUS_CANCELLED
from ..repositories.order_repo import order_repo
from . import inventory_service


def cancel_order(order_id):
    order = order_repo.get(order_id)
    if order is None:
        raise KeyError(order_id)
    inventory_service.release(order.items)
    order.status = STATUS_CANCELLED
    return order
